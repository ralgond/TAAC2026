"""
train_cvr.py  —  Train CVRModel (multi-behavior, single CVR tower, no item_id).

Usage example
─────────────
python train_cvr.py \\
    --data_dir   /data/pcvr/train \\
    --schema_path /data/pcvr/schema.json \\
    --save_dir   checkpoints/cvr \\
    --emb_skip_threshold 1000000 \\
    --seq_max_lens "a:256,b:128,c:64,d:32"

Key differences vs. train.py (BehaviorEncoder)
────────────────────────────────────────────────
1. item_id feature is NEVER loaded (dropped from dataset batches).
   The parquet pipeline may still emit an item_id column; it is silently
   ignored by CVRTrainer._unpack_batch.

2. emb_skip_threshold defaults to 1_000_000.  Any feature whose vocab_size
   exceeds this is represented by a zero-vector (no Embedding allocated).
   Applies to BOTH static scalar features AND per-domain sequence features.

3. action_num is always 1 (single CVR tower, BCE loss).
   --loss_type is removed; BCE is always used.

4. time features (fid 1000-1003 in user_int space) are auto-detected by the
   dataset and flow through static_ids like any other user feature.
   No special handling needed in this script.

5. --seq_max_lens accepts "domain:len,domain:len" format where domain names
   match the keys in schema.json["seq"] (e.g. "a", "b", "c", "d").
   If a domain is not listed it defaults to 256.

Static feature discovery is identical to train.py:
  - scalar int: all dim==1 user_int + item_int fields (incl. fids 1000-1003)
  - array  int: all dim>1  user_int + item_int fields
  - scalar dense: all dim==1 user_dense fields
  - array  dense: all dim>1  user_dense fields
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import torch

from pcvr_parquet_dataset import PCVRParquetDataset, get_pcvr_data
from cvr_model import CVRModel
from cvr_trainer import CVRTrainer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('train_cvr')


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train CVRModel')

    # ── Data ──────────────────────────────────────────────────────────────
    p.add_argument('--data_dir',       required=True)
    p.add_argument('--schema_path',    required=True)
    p.add_argument('--save_dir',       default='checkpoints/cvr')
    p.add_argument('--valid_ratio',    type=float, default=0.1)
    p.add_argument('--train_ratio',    type=float, default=1.0)
    p.add_argument('--num_workers',    type=int,   default=8)
    p.add_argument('--buffer_batches', type=int,   default=20)
    p.add_argument('--seed',           type=int,   default=42)
    p.add_argument(
        '--seq_max_lens', type=str, default='',
        help=(
            'Per-domain max sequence length. '
            'Domain names come from schema.json["seq"] keys (e.g. "a", "b", "c", "d"). '
            'Format: "a:256,b:128,c:64,d:32". '
            'Unspecified domains default to 256.'
        ),
    )

    # ── CVRModel ──────────────────────────────────────────────────────────
    p.add_argument('--d_model',           type=int,   default=128)
    p.add_argument('--emb_dim',           type=int,   default=64)
    p.add_argument('--num_heads',         type=int,   default=4)
    p.add_argument('--num_domain_layers', type=int,   default=2)
    p.add_argument('--num_cross_layers',  type=int,   default=2)
    p.add_argument('--hidden_mult',       type=int,   default=4)
    p.add_argument('--num_time_buckets',  type=int,   default=65)
    p.add_argument('--use_rope',          action='store_true', default=True)
    p.add_argument('--no_rope',           dest='use_rope', action='store_false')
    p.add_argument('--rope_base',         type=float, default=10000.0)
    p.add_argument(
        '--emb_skip_threshold', type=int, default=1_000_000,
        help=(
            'Skip Embedding creation for features whose vocab_size > threshold. '
            'Zero-vector placeholder used instead. '
            'Applies to both static and per-domain sequence features. '
            'Default: 1_000_000 (per task requirement).'
        ),
    )
    p.add_argument('--seq_id_threshold',  type=int,   default=10000)
    p.add_argument('--max_seq_len',       type=int,   default=2048)
    p.add_argument('--dropout_rate',      type=float, default=0.01)

    # ── MoE ───────────────────────────────────────────────────────────────
    p.add_argument('--moe_num_experts',     type=int,   default=4)
    p.add_argument('--moe_top_k',           type=int,   default=2)
    p.add_argument('--moe_aux_loss_weight', type=float, default=0.01)

    # ── Static array fields (manual override; schema auto-fills the rest) ─
    p.add_argument(
        '--static_array_fids', type=str, default='',
        help='Comma-separated "fid:group:vocab_size:max_dim". E.g. "30:user:500:10".',
    )

    # ── Training ──────────────────────────────────────────────────────────
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
    p.add_argument(
        '--amp_dtype', default='bfloat16',
        choices=['bfloat16', 'float16', 'float32'],
    )

    # ── Resume ────────────────────────────────────────────────────────────
    p.add_argument('--resume_from', default=None,
                   help='Path to a saved CVRModel directory to resume from.')

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

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
            raise ValueError(f"group must be 'user' or 'item', got '{group}'")
        result.append((fid, group, vs, dim))
    return result


def _parse_seq_max_lens(s: str) -> Dict[str, int]:
    """Parse "click:256,cart:128,buy:64" → {"click": 256, "cart": 128, "buy": 64}."""
    result = {}
    for token in s.split(','):
        token = token.strip()
        if not token:
            continue
        parts = token.split(':')
        if len(parts) != 2:
            raise ValueError(
                f"--seq_max_lens: expected 'domain:length', got '{token}'"
            )
        result[parts[0].strip()] = int(parts[1])
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    amp_dtype = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }[args.amp_dtype]

    seq_max_lens = _parse_seq_max_lens(args.seq_max_lens)

    logger.info("=" * 60)
    logger.info("CVRModel training  (multi-behavior, single CVR tower, no item_id)")
    logger.info("=" * 60)
    logger.info(f"Args: {vars(args)}")

    # ── Static array fids: manual overrides ──────────────────────────────
    static_array_fids = _parse_static_array_fids(args.static_array_fids)

    # ── Schema probe (lightweight single-RG read) ─────────────────────────
    logger.info("Loading schema ...")
    _schema_ds = PCVRParquetDataset(
        parquet_path=args.data_dir,
        schema_path=args.schema_path,
        batch_size=1,
        shuffle=False,
        buffer_batches=0,
    )

    # ── Auto-discover static_array_fids from schema ───────────────────────
    # Manual entries (from CLI) take priority; schema fills the rest.
    _schema_array_fids: List[Tuple[int, str, int, int]] = []
    for fid, vs, dim in _schema_ds._user_int_cols:
        if dim > 1 and vs > 0:
            _schema_array_fids.append((fid, 'user', vs, dim))
    for fid, vs, dim in _schema_ds._item_int_cols:
        if dim > 1 and vs > 0:
            _schema_array_fids.append((fid, 'item', vs, dim))
    _manual_fids_set = {fid for fid, _, _, _ in static_array_fids}
    static_array_fids = static_array_fids + [
        t for t in _schema_array_fids if t[0] not in _manual_fids_set
    ]

    # ── Auto-discover static_fids (scalar int) ────────────────────────────
    # Includes time features: fid 1000 (hour), 1001 (dow), 1002 (is_weekend),
    # 1003 (time_slot) — these are appended to _user_int_cols by the dataset.
    static_fids: List[Tuple[int, str, int]] = []
    for fid, vs, dim in _schema_ds._user_int_cols:
        if dim == 1 and vs > 0:
            static_fids.append((fid, 'user', vs))
    for fid, vs, dim in _schema_ds._item_int_cols:
        if dim == 1 and vs > 0:
            static_fids.append((fid, 'item', vs))
    static_vocab_sizes = [vs for _, _, vs in static_fids]

    # ── Auto-discover static_dense_fids ───────────────────────────────────
    static_dense_fids: List[Tuple[int, str]] = []
    for fid, dim in _schema_ds._user_dense_cols:
        if dim == 1:
            static_dense_fids.append((fid, 'user'))
    n_dense_scalar = len(static_dense_fids)

    # ── Auto-discover dense_array_fids ────────────────────────────────────
    static_dense_array_fids: List[Tuple[int, str, int]] = []
    for fid, dim in _schema_ds._user_dense_cols:
        if dim > 1:
            static_dense_array_fids.append((fid, 'user', dim))

    static_array_configs = [
        (f"static_array_{fid}", vs, dim)
        for fid, _grp, vs, dim in static_array_fids
    ]
    dense_array_configs = [
        (f"static_dense_array_{fid}", dim)
        for fid, _grp, dim in static_dense_array_fids
    ]

    # ── Group labels ──────────────────────────────────────────────────────
    scalar_groups       = [grp for _fid, grp, _vs in static_fids]
    array_groups        = [grp for _fid, grp, _vs, _dim in static_array_fids]
    dense_scalar_groups = [grp for _fid, grp in static_dense_fids]
    dense_array_groups  = [grp for _fid, grp, _dim in static_dense_array_fids]

    logger.info(f"static_int_scalar_fields:  {len(static_fids)} "
                f"(user incl. time fids 1000-1003)")
    logger.info(f"static_int_array_fields:   {len(static_array_fids)}")
    logger.info(f"static_dense_scalar:       {n_dense_scalar}")
    logger.info(f"static_dense_array_fields: {len(static_dense_array_fids)}")
    logger.info(f"emb_skip_threshold:        {args.emb_skip_threshold:,}")
    logger.info(f"seq_max_lens:              {seq_max_lens}")

    # ── Dataset ───────────────────────────────────────────────────────────
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
        seq_max_lens=seq_max_lens or None,
        static_fids=static_fids or None,
        static_array_fids=static_array_fids or None,
        static_dense_fids=static_dense_fids or None,
        static_dense_array_fids=static_dense_array_fids or None,
    )
    logger.info(
        f"seq_domains: {train_ds.seq_domains}"
    )
    logger.info(
        f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}"
    )

    # ── Model ─────────────────────────────────────────────────────────────
    if args.resume_from is not None:
        logger.info(f"Resuming from {args.resume_from}")
        model = CVRModel.load(args.resume_from)
    else:
        model = CVRModel(
            seq_vocab_sizes=train_ds.seq_domain_vocab_sizes,
            static_vocab_sizes=static_vocab_sizes or None,
            static_array_configs=static_array_configs or None,
            n_dense_scalar=n_dense_scalar,
            dense_array_configs=dense_array_configs or None,
            scalar_groups=scalar_groups or None,
            array_groups=array_groups or None,
            dense_scalar_groups=dense_scalar_groups or None,
            dense_array_groups=dense_array_groups or None,
            d_model=args.d_model,
            emb_dim=args.emb_dim,
            num_heads=args.num_heads,
            num_domain_layers=args.num_domain_layers,
            num_cross_layers=args.num_cross_layers,
            hidden_mult=args.hidden_mult,
            dropout_rate=args.dropout_rate,
            num_time_buckets=args.num_time_buckets,
            use_rope=args.use_rope,
            rope_base=args.rope_base,
            emb_skip_threshold=args.emb_skip_threshold,
            seq_id_threshold=args.seq_id_threshold,
            max_seq_len=args.max_seq_len,
            moe_num_experts=args.moe_num_experts,
            moe_top_k=args.moe_top_k,
            moe_aux_loss_weight=args.moe_aux_loss_weight,
        )

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total params: {n_params:,}")

    if args.moe_num_experts > 0:
        logger.info(
            f"MoE: {args.num_cross_layers} CrossLayer(s) | "
            f"experts={args.moe_num_experts}  top_k={args.moe_top_k}  "
            f"aux_weight={args.moe_aux_loss_weight}"
        )
    else:
        logger.info("MoE disabled (moe_num_experts=0)")

    n_u = sum(1 for g in scalar_groups if g == 'user')
    n_i = sum(1 for g in scalar_groups if g == 'item')
    logger.info(
        f"StaticEncoder scalar groups: {n_u} user / {n_i} item  "
        f"| 3 output tokens (no item_id)"
    )

    # ── Save train config ─────────────────────────────────────────────────
    train_config = {
        "seq_vocab_sizes":          train_ds.seq_domain_vocab_sizes,
        "static_vocab_sizes":       static_vocab_sizes,
        "static_array_configs":     static_array_configs,
        "n_dense_scalar":           n_dense_scalar,
        "dense_array_configs":      dense_array_configs,
        "scalar_groups":            scalar_groups,
        "array_groups":             array_groups,
        "dense_scalar_groups":      dense_scalar_groups,
        "dense_array_groups":       dense_array_groups,
        # fid metadata
        "static_fids":              [[f, g, v] for f, g, v in static_fids],
        "static_array_fids":        [[f, g, v, d] for f, g, v, d in static_array_fids],
        "static_dense_fids":        [[f, g] for f, g in static_dense_fids],
        "static_dense_array_fids":  [[f, g, d] for f, g, d in static_dense_array_fids],
        # model hypers
        "d_model":                  args.d_model,
        "emb_dim":                  args.emb_dim,
        "num_heads":                args.num_heads,
        "num_domain_layers":        args.num_domain_layers,
        "num_cross_layers":         args.num_cross_layers,
        "hidden_mult":              args.hidden_mult,
        "dropout_rate":             args.dropout_rate,
        "num_time_buckets":         args.num_time_buckets,
        "use_rope":                 args.use_rope,
        "rope_base":                args.rope_base,
        "emb_skip_threshold":       args.emb_skip_threshold,
        "seq_id_threshold":         args.seq_id_threshold,
        "max_seq_len":              args.max_seq_len,
        "moe_num_experts":          args.moe_num_experts,
        "moe_top_k":                args.moe_top_k,
        "moe_aux_loss_weight":      args.moe_aux_loss_weight,
        # data
        "seq_max_lens":             seq_max_lens,
    }
    os.makedirs(args.save_dir, exist_ok=True)
    with open(os.path.join(args.save_dir, 'train_config.json'), 'w') as f:
        json.dump(train_config, f, indent=2)
    logger.info(f"train_config.json saved to {args.save_dir}")

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = CVRTrainer(
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
        amp_dtype=amp_dtype,
    )
    trainer.train()


if __name__ == '__main__':
    main()