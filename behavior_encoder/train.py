"""
train.py — Entry point for BehaviorEncoder training.

Usage example:
    python train.py \
        --data_dir /data/pcvr \
        --schema_path /data/pcvr/schema.json \
        --save_dir checkpoints/run01 \
        --d_model 128 \
        --emb_dim 64 \
        --num_heads 4 \
        --num_domain_layers 2 \
        --num_hyformer_blocks 2 \
        --batch_size 256 \
        --num_epochs 10 \
        --lr 1e-3 \
        --device cuda:0

The dataset class (pcvr_parquet_dataset.py) must expose:
    train_ds.seq_domain_vocab_sizes  -> Dict[str, List[int]]

Batch dict keys expected per domain `d` (e.g. d = 'a', 'b', 'c', 'd'):
    seq_{d}             : (B, num_fids, L)  int tensor   e.g. seq_a
    seq_{d}_len         : (B,)              int tensor   e.g. seq_a_len
    seq_{d}_time_bucket : (B, L)            int tensor   e.g. seq_a_time_bucket
    label               : (B,) or (B, action_num) float tensor
"""

import argparse
import logging
import sys

import torch

from model import BehaviorEncoder
from trainer import BehaviorEncoderTrainer

# Import dataset utility (provided separately)
from pcvr_parquet_dataset import get_pcvr_data


# ─────────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train BehaviorEncoder")

    # Data
    p.add_argument("--data_dir", required=True, help="Path to parquet data directory")
    p.add_argument("--schema_path", required=True, help="Path to schema JSON file")
    p.add_argument("--valid_ratio", type=float, default=0.1)
    p.add_argument("--train_ratio", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--buffer_batches", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)

    # Model architecture
    p.add_argument("--d_model", type=int, default=64)
    p.add_argument("--emb_dim", type=int, default=64)
    p.add_argument("--num_heads", type=int, default=4)
    p.add_argument("--num_domain_layers", type=int, default=2,
                   help="Transformer layers per DomainEncoder")
    p.add_argument("--hidden_mult", type=int, default=4)
    p.add_argument("--num_time_buckets", type=int, default=65)
    p.add_argument("--action_num", type=int, default=1)
    p.add_argument("--use_rope", action="store_true", default=True)
    p.add_argument("--no_rope", dest="use_rope", action="store_false")
    p.add_argument("--rope_base", type=float, default=10000.0)
    p.add_argument("--emb_skip_threshold", type=int, default=0,
                   help="Skip Embedding for features with vocab > threshold (0=off)")
    p.add_argument("--seq_id_threshold", type=int, default=10000)
    p.add_argument("--max_seq_len", type=int, default=2048)
    p.add_argument("--dropout_rate", type=float, default=0.01)

    # Training
    p.add_argument("--save_dir", default="checkpoints")
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--sparse_lr", type=float, default=0.05)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--eval_interval", type=int, default=0,
                   help="Evaluate every N steps; 0 = once per epoch")
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--use_separate_optim", action="store_true", default=True,
                   help="Use Adagrad for embeddings + AdamW for dense params")
    p.add_argument("--label_key", default="label")
    p.add_argument("--loss_type", default="bce", choices=["bce", "ce"])
    p.add_argument("--amp_dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                   help="AMP dtype. bfloat16 recommended for A100/H100; "
                        "float16 for older GPUs; float32 to disable AMP.")

    # Checkpoint resume
    p.add_argument("--resume_from", default=None,
                   help="Directory containing model_config.json + model.pt to resume from")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    logger.info("=" * 60)
    logger.info("BehaviorEncoder training")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # ── AMP dtype ────────────────────────────────────────────────────────
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    amp_dtype = dtype_map[args.amp_dtype]

    # ── Dataset ──────────────────────────────────────────────────────────
    logger.info("Loading dataset …")
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
    logger.info(f"Train batches: {len(train_loader)}  Valid batches: {len(valid_loader)}")

    # seq_vocab_sizes is a field on train_ds exposed by pcvr_parquet_dataset
    seq_vocab_sizes = train_ds.seq_domain_vocab_sizes
    logger.info(f"Sequence domains: {sorted(seq_vocab_sizes.keys())}")
    for domain, vsizes in seq_vocab_sizes.items():
        logger.info(f"  {domain}: {len(vsizes)} fids, vocab sizes = {vsizes[:5]}{'…' if len(vsizes) > 5 else ''}")

    # ── Model ────────────────────────────────────────────────────────────
    if args.resume_from is not None:
        logger.info(f"Resuming from {args.resume_from}")
        model = BehaviorEncoder.load(args.resume_from, map_location="cpu")
    else:
        model = BehaviorEncoder(
            seq_vocab_sizes=seq_vocab_sizes,
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
        )

    n_params = sum(p.numel() for p in model.parameters())
    n_sparse = sum(p.numel() for p in model.get_sparse_params())
    logger.info(f"Total params: {n_params:,}  (sparse/embedding: {n_sparse:,})")

    # ── Trainer ──────────────────────────────────────────────────────────
    trainer = BehaviorEncoderTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=args.save_dir,
        lr=args.lr,
        sparse_lr=args.sparse_lr,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        num_epochs=args.num_epochs,
        warmup_ratio=args.warmup_ratio,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        patience=args.patience,
        device=args.device,
        use_separate_optim=args.use_separate_optim,
        label_key=args.label_key,
        loss_type=args.loss_type,
        amp_dtype=amp_dtype,
    )

    trainer.train()


if __name__ == "__main__":
    main()