from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import List, Tuple

import torch

from pcvr_parquet_dataset import (
    PCVRParquetDataset,
    get_pcvr_data,
)

from model import GatedFusion
from trainer import GatedFusionTrainer

from deepfm_model import DeepFM
from behavior_encoder_model import BehaviorEncoder


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

    p = argparse.ArgumentParser(
        description='Train GatedFusion'
    )

    # ─────────────────────────
    # Data
    # ─────────────────────────

    p.add_argument('--data_dir', required=True)
    p.add_argument('--schema_path', required=True)

    p.add_argument(
        '--save_dir',
        default='checkpoints',
    )

    p.add_argument(
        '--valid_ratio',
        type=float,
        default=0.1,
    )

    p.add_argument(
        '--train_ratio',
        type=float,
        default=1.0,
    )

    p.add_argument(
        '--num_workers',
        type=int,
        default=8,
    )

    p.add_argument(
        '--buffer_batches',
        type=int,
        default=20,
    )

    p.add_argument(
        '--seed',
        type=int,
        default=42,
    )

    # ─────────────────────────
    # Gate
    # ─────────────────────────
    p.add_argument(
        '--gate_in_dim',
        type=int,
        default=128,
    )

    # ─────────────────────────
    # DeepFM
    # ─────────────────────────

    p.add_argument(
        '--embed_dim',
        type=int,
        default=16,
    )

    p.add_argument(
        '--fusion_dim',
        type=int,
        default=64,
    )

    p.add_argument(
        '--dnn_hidden',
        type=str,
        default='256,128',
    )

    p.add_argument(
        '--dnn_dropout',
        type=float,
        default=0.1,
    )

    p.add_argument(
        '--num_buckets',
        type=int,
        default=None,
    )

    p.add_argument(
        '--num_items',
        type=int,
        default=None,
    )

    # ─────────────────────────
    # BehaviorEncoder
    # ─────────────────────────

    p.add_argument(
        '--d_model',
        type=int,
        default=64,
    )

    p.add_argument(
        '--emb_dim',
        type=int,
        default=64,
    )

    p.add_argument(
        '--num_heads',
        type=int,
        default=4,
    )

    p.add_argument(
        '--num_domain_layers',
        type=int,
        default=2,
    )

    p.add_argument(
        '--hidden_mult',
        type=int,
        default=4,
    )

    p.add_argument(
        '--num_time_buckets',
        type=int,
        default=65,
    )

    p.add_argument(
        '--action_num',
        type=int,
        default=1,
    )

    p.add_argument(
        '--use_rope',
        action='store_true',
        default=True,
    )

    p.add_argument(
        '--no_rope',
        dest='use_rope',
        action='store_false',
    )

    p.add_argument(
        '--rope_base',
        type=float,
        default=10000.0,
    )

    p.add_argument(
        '--emb_skip_threshold',
        type=int,
        default=0,
    )

    p.add_argument(
        '--seq_id_threshold',
        type=int,
        default=10000,
    )

    p.add_argument(
        '--max_seq_len',
        type=int,
        default=2048,
    )

    p.add_argument(
        '--dropout_rate',
        type=float,
        default=0.01,
    )

    # ─────────────────────────
    # Training
    # ─────────────────────────

    p.add_argument(
        '--batch_size',
        type=int,
        default=128,
    )

    p.add_argument(
        '--num_epochs',
        type=int,
        default=3,
    )

    p.add_argument(
        '--weight_decay',
        type=float,
        default=1e-2,
    )

    p.add_argument(
        '--grad_clip',
        type=float,
        default=1.0,
    )

    p.add_argument(
        '--grad_accum',
        type=int,
        default=1,
    )

    p.add_argument(
        '--log_interval',
        type=int,
        default=100,
    )

    p.add_argument(
        '--eval_interval',
        type=int,
        default=0,
    )

    p.add_argument(
        '--patience',
        type=int,
        default=3,
    )

    p.add_argument(
        '--device',
        default='cuda'
        if torch.cuda.is_available()
        else 'cpu',
    )

    p.add_argument(
        '--label_key',
        default='label',
    )

    p.add_argument(
        '--loss_type',
        default='bce',
        choices=['bce', 'ce'],
    )

    p.add_argument(
        '--amp_dtype',
        default='bfloat16',
        choices=[
            'bfloat16',
            'float16',
            'float32',
        ],
    )

    # ─────────────────────────
    # Resume
    # ─────────────────────────

    p.add_argument(
        '--resume_from',
        default=None,
    )

    return p.parse_args()


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _parse_hidden(s: str) -> List[int]:

    return [
        int(x)
        for x in s.split(',')
        if x.strip()
    ]


def _build_int_fields(
    schema: PCVRParquetDataset,
    group: str,
) -> List[Tuple[int, int]]:

    cols = (
        schema._user_int_cols
        if group == 'user'
        else schema._item_int_cols
    )

    fields = []

    for fid, vs, dim in cols:

        if dim == 1:

            fields.append(
                (fid, vs)
            )

    return fields


def _dense_dim_and_fids(
    schema: PCVRParquetDataset,
):

    fids = [
        fid
        for fid, dim
        in schema._user_dense_cols
        if dim == 1
    ]

    return len(fids), fids


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():

    args = parse_args()

    torch.manual_seed(args.seed)

    logger.info("=" * 60)
    logger.info("GatedFusion training")
    logger.info("=" * 60)

    logger.info(
        f"Args: {vars(args)}"
    )

    # ─────────────────────────
    # AMP
    # ─────────────────────────

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }

    amp_dtype = dtype_map[
        args.amp_dtype
    ]

    # ─────────────────────────
    # Dataset
    # ─────────────────────────

    logger.info(
        "Loading dataset ..."
    )

    train_loader, valid_loader, train_ds = (
        get_pcvr_data(
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
    )

    logger.info(
        f"Train batches: "
        f"{len(train_loader)} | "
        f"Valid batches: "
        f"{len(valid_loader)}"
    )

    # ─────────────────────────
    # DeepFM schema
    # ─────────────────────────

    user_int_fields = _build_int_fields(
        train_ds,
        'user',
    )

    item_int_fields = _build_int_fields(
        train_ds,
        'item',
    )

    dense_dim, dense_fids = (
        _dense_dim_and_fids(train_ds)
    )

    num_items = (
        args.num_items
        or 10_000_000
    )

    logger.info(
        f"user_int fields: "
        f"{len(user_int_fields)} | "
        f"item_int fields: "
        f"{len(item_int_fields)} | "
        f"dense_dim: {dense_dim}"
    )

    # ─────────────────────────
    # Resume
    # ─────────────────────────

    if args.resume_from is not None:

        logger.info(
            f"Loading from "
            f"{args.resume_from}"
        )

        model = GatedFusion.load(
            args.resume_from,
            deepfm_cls=DeepFM,
            behavior_cls=BehaviorEncoder,
            map_location='cpu',
        )

    else:

        # ─────────────────────
        # DeepFM
        # ─────────────────────

        deepfm = DeepFM(
            user_int_fields=user_int_fields,
            item_int_fields=item_int_fields,
            dense_dim=dense_dim,
            num_items=num_items,
            embed_dim=args.embed_dim,
            fusion_dim=args.fusion_dim,
            dnn_hidden_dims=_parse_hidden(
                args.dnn_hidden
            ),
            dnn_dropout=args.dnn_dropout,
            num_buckets=args.num_buckets,
        )

        deepfm.register_dense_fids(
            dense_fids
        )

        # ─────────────────────
        # BehaviorEncoder
        # ─────────────────────

        behavior_encoder = (
            BehaviorEncoder(
                seq_vocab_sizes=train_ds.seq_domain_vocab_sizes,
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
        )

        # ─────────────────────
        # Fusion model
        # ─────────────────────

        model = GatedFusion(
            deepfm=deepfm,
            behavior_encoder=behavior_encoder,
            gate_in_dim=args.gate_in_dim,
            action_num=args.action_num,
        )

    # ─────────────────────────
    # Params
    # ─────────────────────────

    n_params = sum(
        p.numel()
        for p
        in model.parameters()
    )

    logger.info(
        f"Total params: "
        f"{n_params:,}"
    )

    # ─────────────────────────
    # Config
    # ─────────────────────────

    model_config = {

        "deepfm": {

            "user_int_fields":
                user_int_fields,

            "item_int_fields":
                item_int_fields,

            "dense_dim":
                dense_dim,

            "num_items":
                num_items,

            "embed_dim":
                args.embed_dim,

            "fusion_dim":
                args.fusion_dim,

            "dnn_hidden_dims":
                _parse_hidden(
                    args.dnn_hidden
                ),

            "dnn_dropout":
                args.dnn_dropout,

            "num_buckets":
                args.num_buckets,
        },

        "behavior_encoder": {

            "seq_vocab_sizes":
                train_ds.seq_domain_vocab_sizes,

            "d_model":
                args.d_model,

            "emb_dim":
                args.emb_dim,

            "num_heads":
                args.num_heads,

            "num_domain_layers":
                args.num_domain_layers,

            "hidden_mult":
                args.hidden_mult,

            "dropout_rate":
                args.dropout_rate,

            "num_time_buckets":
                args.num_time_buckets,

            "action_num":
                args.action_num,

            "use_rope":
                args.use_rope,

            "rope_base":
                args.rope_base,

            "emb_skip_threshold":
                args.emb_skip_threshold,

            "seq_id_threshold":
                args.seq_id_threshold,

            "max_seq_len":
                args.max_seq_len,
        },

        "fusion": {

            "action_num":
                args.action_num,

            "gate_in_dim":
                args.gate_in_dim,
        }
    }

    # ─────────────────────────
    # Trainer
    # ─────────────────────────

    trainer = GatedFusionTrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        save_dir=args.save_dir,
        num_epochs=args.num_epochs,
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

    # save config into trainer
    trainer.model_config = model_config

    trainer.train()


if __name__ == '__main__':
    main()