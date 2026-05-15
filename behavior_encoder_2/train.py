"""
train.py  –  Train BehaviorEncoder (standalone, no GatedFusion / DeepFM).

Usage example
-------------
python train.py \\
    --data_dir   /data/pcvr/train \\
    --schema_path /data/pcvr/schema.json \\
    --save_dir   checkpoints \\
    --static_fids "10:user:100,20:user:50,5:item:200" \\
    --static_array_fids "30:user:500:10,7:item:200:5"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import torch

from pcvr_parquet_dataset import PCVRParquetDataset, get_pcvr_data
from behavior_encoder_model import BehaviorEncoder
from trainer import BehaviorEncoderTrainer


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('train')


# ──────────────────────────────────────────────
# Args
# ──────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train BehaviorEncoder')

    # ── Data ──────────────────────────────────
    p.add_argument('--data_dir',       required=True)
    p.add_argument('--schema_path',    required=True)
    p.add_argument('--save_dir',       default='checkpoints')
    p.add_argument('--valid_ratio',    type=float, default=0.1)
    p.add_argument('--train_ratio',    type=float, default=1.0)
    p.add_argument('--num_workers',    type=int,   default=8)
    p.add_argument('--buffer_batches', type=int,   default=20)
    p.add_argument('--seed',           type=int,   default=42)

    # ── BehaviorEncoder ───────────────────────
    p.add_argument('--d_model',           type=int,   default=64)
    p.add_argument('--emb_dim',           type=int,   default=64)
    p.add_argument('--num_heads',         type=int,   default=4)
    p.add_argument('--num_domain_layers', type=int,   default=2)
    p.add_argument('--num_cross_layers',  type=int,   default=2)
    p.add_argument('--hidden_mult',       type=int,   default=4)
    p.add_argument('--num_time_buckets',  type=int,   default=65)
    p.add_argument('--action_num',        type=int,   default=1)
    p.add_argument('--use_rope',          action='store_true', default=True)
    p.add_argument('--no_rope',           dest='use_rope', action='store_false')
    p.add_argument('--rope_base',         type=float, default=10000.0)
    p.add_argument('--emb_skip_threshold',type=int,   default=0)
    p.add_argument('--seq_id_threshold',  type=int,   default=10000)
    p.add_argument('--max_seq_len',       type=int,   default=2048)
    p.add_argument('--dropout_rate',      type=float, default=0.01)

    # ── StaticEncoder (array，需手动指定 max_dim) ────────────────────────
    # scalar static fields 由 schema 自动推导，无需手动指定。
    p.add_argument(
        '--static_array_fids', type=str, default='',
        help=(
            'Comma-separated "fid:group:vocab_size:max_dim" 4-tuples for array static fields. '
            'Example: "30:user:500:10,7:item:200:5". Leave empty to disable.'
        ),
    )

    # ── Training ──────────────────────────────
    p.add_argument('--batch_size',    type=int,   default=128)
    p.add_argument('--num_epochs',    type=int,   default=3)
    p.add_argument('--lr',            type=float, default=1e-4)
    p.add_argument('--sparse_lr',     type=float, default=1e-3)
    p.add_argument('--weight_decay',  type=float, default=1e-2)
    p.add_argument('--grad_clip',     type=float, default=1.0)
    p.add_argument('--grad_accum',    type=int,   default=1)
    p.add_argument('--log_interval',  type=int,   default=100)
    p.add_argument('--eval_interval', type=int,   default=5000)
    p.add_argument('--patience',      type=int,   default=3)
    p.add_argument(
        '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    p.add_argument('--label_key', default='label')
    p.add_argument('--loss_type', default='bce', choices=['bce', 'ce'])
    p.add_argument(
        '--amp_dtype', default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
    )

    # ── Resume ────────────────────────────────
    p.add_argument('--resume_from', default=None,
                   help='Path to a saved BehaviorEncoder directory to resume from.')

    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_static_array_fids(s: str) -> List[Tuple[int, str, int, int]]:
    result = []
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) != 4:
            raise ValueError(
                f"--static_array_fids: expected 'fid:group:vocab_size:max_dim', got '{token}'"
            )
        fid, group, vs, dim = int(parts[0]), parts[1].strip(), int(parts[2]), int(parts[3])
        if group not in ('user', 'item'):
            raise ValueError(f"--static_array_fids group must be 'user' or 'item', got '{group}'")
        result.append((fid, group, vs, dim))
    return result


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # ── static_array_fids 仍可手动指定（array 需要 max_dim，schema 里没有） ──
    static_array_fids    = _parse_static_array_fids(args.static_array_fids)

    logger.info("=" * 60)
    logger.info("BehaviorEncoder training")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # ── AMP dtype ─────────────────────────────
    amp_dtype = {"bfloat16": torch.bfloat16,
                 "float16":  torch.float16,
                 "float32":  torch.float32}[args.amp_dtype]

    # ── Dataset (第一次：不带 static，只为读 schema) ──────────────────────
    # 用单个 row group、batch_size=1 快速拿到 schema，避免完整扫描。
    logger.info("Loading schema ...")
    _schema_ds = PCVRParquetDataset(
        parquet_path=args.data_dir,
        schema_path=args.schema_path,
        batch_size=1,
        shuffle=False,
        buffer_batches=0,
    )

    # ── 自动构建 static_fids：所有 scalar (dim==1) user_int + item_int ──────
    static_fids: List[Tuple[int, str, int]] = []
    for fid, vs, dim in _schema_ds._user_int_cols:
        if dim == 1 and vs > 0:
            static_fids.append((fid, 'user', vs))
    for fid, vs, dim in _schema_ds._item_int_cols:
        if dim == 1 and vs > 0:
            static_fids.append((fid, 'item', vs))
    static_vocab_sizes = [vs for _, _, vs in static_fids]

    # ── 自动构建 static_dense_fids：所有 scalar (dim==1) user_dense ──────────
    static_dense_fids: List[Tuple[int, str]] = []
    for fid, dim in _schema_ds._user_dense_cols:
        if dim == 1:
            static_dense_fids.append((fid, 'user'))
    n_dense_scalar = len(static_dense_fids)

    # int array / dense array 仍需手动指定（需要 max_dim / dim 信息）
    static_array_configs = [
        (f"static_array_{fid}", vs, dim)
        for fid, _grp, vs, dim in static_array_fids
    ]
    dense_array_configs = [
        (f"static_dense_array_{fid}", dim)
        for fid, _grp, dim in []        # 当前无 dense array fids，留空占位
    ]

    logger.info(
        f"Auto static_int_scalar_fields:   {len(static_fids)} fids "
        f"(incl. timestamp-derived 1000-1003)"
    )
    logger.info(
        f"Auto static_dense_scalar_fields: {n_dense_scalar} fids"
    )
    logger.info(
        f"static_int_array_fields:   {len(static_array_fids)}"
    )

    # ── Dataset (正式) ───────────────────────────────────────────────────
    logger.info("Loading dataset ...")
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
        static_fids=static_fids or None,
        static_array_fids=static_array_fids or None,
        static_dense_fids=static_dense_fids or None,
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}"
    )

    # ── Model ─────────────────────────────────
    if args.resume_from is not None:
        logger.info(f"Resuming from {args.resume_from}")
        model = BehaviorEncoder.load(args.resume_from)
    else:
        model = BehaviorEncoder(
            seq_vocab_sizes=train_ds.seq_domain_vocab_sizes,
            static_vocab_sizes=static_vocab_sizes or None,
            static_array_configs=static_array_configs or None,
            n_dense_scalar=n_dense_scalar,
            dense_array_configs=dense_array_configs or None,
            d_model=args.d_model,
            emb_dim=args.emb_dim,
            num_heads=args.num_heads,
            num_domain_layers=args.num_domain_layers,
            hidden_mult=args.hidden_mult,
            dropout_rate=args.dropout_rate,
            num_time_buckets=args.num_time_buckets,
            action_num=args.action_num,
            use_rope=args.use_rope,
            rope_base=args.rope_base,
            emb_skip_threshold=args.emb_skip_threshold,
            seq_id_threshold=args.seq_id_threshold,
            max_seq_len=args.max_seq_len,
            num_cross_layers=args.num_cross_layers,
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {n_params:,}")

    # ── Save full config for reproducibility ──
    model_config = {
        "seq_vocab_sizes":       train_ds.seq_domain_vocab_sizes,
        "static_vocab_sizes":    static_vocab_sizes,
        "static_array_configs":  static_array_configs,
        "n_dense_scalar":        n_dense_scalar,
        "dense_array_configs":   dense_array_configs,
        "static_fids":           [[f, g, v] for f, g, v in static_fids],
        "static_array_fids":     [[f, g, v, d] for f, g, v, d in static_array_fids],
        "static_dense_fids":     [[f, g] for f, g in static_dense_fids],
        "d_model":               args.d_model,
        "emb_dim":               args.emb_dim,
        "num_heads":             args.num_heads,
        "num_domain_layers":     args.num_domain_layers,
        "num_cross_layers":      args.num_cross_layers,
        "hidden_mult":           args.hidden_mult,
        "dropout_rate":          args.dropout_rate,
        "num_time_buckets":      args.num_time_buckets,
        "action_num":            args.action_num,
        "use_rope":              args.use_rope,
        "rope_base":             args.rope_base,
        "emb_skip_threshold":    args.emb_skip_threshold,
        "seq_id_threshold":      args.seq_id_threshold,
        "max_seq_len":           args.max_seq_len,
    }

    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'train_config.json'), 'w') as f:
        json.dump(model_config, f, indent=2)

    # ── Trainer ───────────────────────────────
    trainer = BehaviorEncoderTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
        lr=args.lr,
        sparse_lr=args.sparse_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.grad_clip,
        grad_accum_steps=args.grad_accum,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        patience=args.patience,
        device=args.device,
        label_key=args.label_key,
        loss_type=args.loss_type,
        amp_dtype=amp_dtype,
    )
    trainer.train()


if __name__ == '__main__':
    main()