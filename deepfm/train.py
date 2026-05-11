"""train.py – main entry point for DeepFM training.

Usage example
-------------
python train.py \\
    --data_dir      /data/pcvr/train \\
    --schema_path   /data/pcvr/schema.json \\
    --save_dir      checkpoints \\
    --num_epochs    10 \\
    --batch_size    2048 \\
    --embed_dim     16 \\
    --dnn_hidden    256,128,64 \\
    --lr            1e-3 \\
    --num_buckets   10000 \\
    --use_amp
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Optional, Tuple

import torch
import torch.optim as optim

# Local modules
from pcvr_parquet_dataset import PCVRParquetDataset, get_pcvr_data
from model import DeepFM
from trainer import Trainer


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('train')


# ──────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train DeepFM on PCVR Parquet data')

    # Data
    p.add_argument('--data_dir',    required=True, help='Directory with *.parquet files')
    p.add_argument('--schema_path', required=True, help='Path to schema.json')
    p.add_argument('--save_dir',    default='checkpoints', help='Output directory')
    p.add_argument('--valid_ratio', type=float, default=0.1)
    p.add_argument('--train_ratio', type=float, default=1.0)
    p.add_argument('--num_workers', type=int,   default=8)
    p.add_argument('--buffer_batches', type=int, default=20)

    # Model
    p.add_argument('--embed_dim',   type=int,   default=16)
    p.add_argument('--dnn_hidden',  type=str,   default='256,128,64',
                   help='Comma-separated DNN hidden layer sizes')
    p.add_argument('--dnn_dropout', type=float, default=0.1)
    p.add_argument('--num_buckets', type=int,   default=None,
                   help='Sub-table size for compositional embedding; '
                        'defaults to ceil(sqrt(num_items))')
    p.add_argument('--num_items',   type=int,   default=None,
                   help='Vocab size for item_id.  If omitted, inferred from '
                        'schema item_int_feats with fid==0 or set to 10_000_000.')

    # Training
    p.add_argument('--batch_size',  type=int,   default=2048)
    p.add_argument('--num_epochs',  type=int,   default=10)
    p.add_argument('--lr',          type=float, default=1e-3)
    p.add_argument('--weight_decay',type=float, default=1e-5)
    p.add_argument('--grad_clip',   type=float, default=1.0)
    p.add_argument('--grad_accum',  type=int,   default=1)
    p.add_argument('--use_amp',     action='store_true')
    p.add_argument('--log_interval',type=int,   default=200)
    p.add_argument('--seed',        type=int,   default=42)

    # Resume / transfer
    p.add_argument('--resume_checkpoint', type=str, default=None,
                   help='Resume training from a checkpoint file')
    p.add_argument('--pretrained_model',  type=str, default=None,
                   help='Load model weights only (no optimizer state)')

    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_hidden(s: str) -> List[int]:
    return [int(x) for x in s.split(',') if x.strip()]


def _build_int_fields(
    schema: PCVRParquetDataset,
    group: str,  # 'user' or 'item'
) -> List[Tuple[int, int]]:
    """Extract ``[(fid, vocab_size), ...]`` for scalar (dim==1) int fields."""
    raw = schema.user_int_schema if group == 'user' else schema.item_int_schema
    cols = schema._user_int_cols if group == 'user' else schema._item_int_cols
    fields = []
    for fid, vs, dim in cols:
        if dim == 1:
            fields.append((fid, vs))
    return fields


def _dense_dim_and_fids(schema: PCVRParquetDataset) -> Tuple[int, List[int]]:
    """Return (dense_dim, ordered fid list) for scalar (dim==1) user dense features."""
    fids = [fid for fid, dim in schema._user_dense_cols if dim == 1]
    return len(fids), fids


def _dense_dim(schema: PCVRParquetDataset) -> int:
    """Total dimension of scalar (dim==1) user dense features."""
    return _dense_dim_and_fids(schema)[0]


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    os.makedirs(args.save_dir, exist_ok=True)

    # ── DataLoaders ──────────────────────────────────────────────────────
    logger.info("Building DataLoaders …")
    train_loader, valid_loader, train_ds = get_pcvr_data(
        data_dir=args.data_dir,
        schema_path=args.schema_path,
        batch_size=args.batch_size,
        valid_ratio=args.valid_ratio,
        train_ratio=args.train_ratio,
        num_workers=args.num_workers,
        buffer_batches=args.buffer_batches,
        shuffle_train=True,
        seed=args.seed,
    )

    # ── Steps per epoch (needed by OneCycleLR) ───────────────────────────
    # len(train_dataset) returns an upper-bound batch count (ceil per RG).
    # Divide by grad_accum so the LR cycle matches actual optimizer steps.
    steps_per_epoch = (len(train_loader.dataset) + args.grad_accum - 1) // args.grad_accum
    total_steps = steps_per_epoch * args.num_epochs
    logger.info(f"steps_per_epoch≈{steps_per_epoch}, total_steps≈{total_steps}")

    # ── Feature field descriptors ─────────────────────────────────────────
    user_int_fields = _build_int_fields(train_ds, 'user')
    item_int_fields = _build_int_fields(train_ds, 'item')
    dense_dim, dense_fids = _dense_dim_and_fids(train_ds)

    # item_id vocab size
    num_items = args.num_items or 10_000_000
    logger.info(
        f"user_int fields: {len(user_int_fields)}  "
        f"item_int fields: {len(item_int_fields)}  "
        f"dense_dim: {dense_dim}  num_items: {num_items}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = DeepFM(
        user_int_fields=user_int_fields,
        item_int_fields=item_int_fields,
        dense_dim=dense_dim,
        num_items=num_items,
        embed_dim=args.embed_dim,
        dnn_hidden_dims=_parse_hidden(args.dnn_hidden),
        dnn_dropout=args.dnn_dropout,
        num_buckets=args.num_buckets,
    ).to(device)

    # Register dense fid order so forward() reads them in the exact same
    # sequence that determined dnn_input_dim at construction time.
    model.register_dense_fids(dense_fids)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    # OneCycleLR: linear warmup (30% of steps) → cosine decay, step-level.
    # pct_start controls the warmup fraction; div_factor sets the initial LR
    # to peak_lr / div_factor, final_div_factor sets the floor LR.
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        total_steps=total_steps,
        pct_start=0.1,           # 10% warmup
        anneal_strategy='cos',
        div_factor=25.0,         # initial lr = max_lr / 25
        final_div_factor=1e4,    # final lr  = initial_lr / 1e4
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        use_amp=args.use_amp,
        grad_clip=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        log_interval=args.log_interval,
    )

    # Resume or load pretrained weights
    if args.resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {args.resume_checkpoint}")
        trainer.load_checkpoint(args.resume_checkpoint)
    elif args.pretrained_model:
        logger.info(f"Loading pretrained weights: {args.pretrained_model}")
        trainer.load_model(args.pretrained_model)

    # ── Training loop ─────────────────────────────────────────────────────
    trainer.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epochs=args.num_epochs,
        save_dir=args.save_dir,
        save_best=True,
    )

    # Save final model weights and config under save_dir/global_step100/
    output_dir = os.path.join(args.save_dir, 'global_step100')
    os.makedirs(output_dir, exist_ok=True)

    final_path = os.path.join(output_dir, 'final_model.pt')
    trainer.save_model(final_path)

    config = {
        'num_items':       num_items,
        'embed_dim':       args.embed_dim,
        'dnn_hidden_dims': _parse_hidden(args.dnn_hidden),
        'dnn_dropout':     args.dnn_dropout,
        'num_buckets':     args.num_buckets,
    }
    config_path = os.path.join(output_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Training complete.  Final model → {final_path}")
    logger.info(f"Model config       → {config_path}")


if __name__ == '__main__':
    main()