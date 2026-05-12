"""infer.py – run inference with a trained GatedFusion model.

Environment variables
---------------------
MODEL_OUTPUT_PATH   Directory whose ``best/`` sub-directory contains
                    ``model.pt`` and ``model_config.json``.
EVAL_DATA_PATH      Test data directory (``*.parquet`` files).
EVAL_SCHEMA_PATH    Path to ``schema.json``.  Defaults to
                    ``<MODEL_OUTPUT_PATH>/schema.json`` when not set.
EVAL_RESULT_PATH    Directory for the generated ``predictions.json``.

Usage
-----
MODEL_OUTPUT_PATH=/checkpoints \\
EVAL_DATA_PATH=/data/pcvr/test \\
EVAL_RESULT_PATH=/results \\
python infer.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from pcvr_parquet_dataset import PCVRParquetDataset
from model import GatedFusion


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('infer')


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _unpack_seq(
    batch: Dict,
    seq_domains: List[str],
    device: torch.device,
):
    """Split the flat batch dict into the three dicts GatedFusion expects."""
    seq_data, seq_lens, seq_time_buckets = {}, {}, {}
    for domain in seq_domains:
        seq_data[domain]         = batch[domain].to(device, non_blocking=True)
        seq_lens[domain]         = batch[f"{domain}_len"].to(device, non_blocking=True)
        seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(
            device, non_blocking=True
        )
    return seq_data, seq_lens, seq_time_buckets


def _unpack_deepfm(
    batch: Dict,
    deepfm,
    device: torch.device,
) -> Dict:
    """
    Build the deepfm_batch dict that DeepFM.forward() expects.

    Keys required by DeepFM:
      - user_int_feats_{fid}   for each fid in user_int_fids
      - item_int_feats_{fid}   for each fid in item_int_fids
      - item_id
      - user_dense_feats_{fid} for each fid in dense_fids  (when present)
    """
    deepfm_batch: Dict = {}

    for fid in deepfm.user_int_fids:
        key = f"user_int_feats_{fid}"
        deepfm_batch[key] = batch[key].to(device, non_blocking=True)

    for fid in deepfm.item_int_fids:
        key = f"item_int_feats_{fid}"
        deepfm_batch[key] = batch[key].to(device, non_blocking=True)

    deepfm_batch["item_id"] = batch["item_id"].to(device, non_blocking=True)

    for fid in deepfm.dense_fids:
        key = f"user_dense_feats_{fid}"
        if key in batch:
            deepfm_batch[key] = batch[key].to(device, non_blocking=True)

    return deepfm_batch


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:

    # ── Read environment variables ─────────────────────────────────────────
    model_dir  = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir   = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')
    schema_path = os.environ.get(
        'EVAL_SCHEMA_PATH',
        os.path.join(model_dir or '', 'schema.json'),
    )

    missing = [
        k for k, v in [
            ('MODEL_OUTPUT_PATH', model_dir),
            ('EVAL_DATA_PATH',    data_dir),
            ('EVAL_RESULT_PATH',  result_dir),
        ]
        if not v
    ]
    if missing:
        raise EnvironmentError(
            f"Required environment variable(s) not set: {', '.join(missing)}"
        )

    # Weights and config live under the best/ sub-directory
    model_subdir = os.path.join(model_dir, 'best')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device            : {device}")
    logger.info(f"MODEL_OUTPUT_PATH : {model_dir}")
    logger.info(f"EVAL_DATA_PATH    : {data_dir}")
    logger.info(f"EVAL_RESULT_PATH  : {result_dir}")
    logger.info(f"schema_path       : {schema_path}")

    os.makedirs(result_dir, exist_ok=True)

    # ── Dataset (no shuffle, is_training=False) ───────────────────────────
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=256,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        pin_memory=device.type == 'cuda',
        prefetch_factor=2,
    )

    # ── Load GatedFusion model ─────────────────────────────────────────────
    if not os.path.isdir(model_subdir):
        raise FileNotFoundError(
            f"model sub-directory not found: {model_subdir}"
        )

    model = GatedFusion.load(model_subdir, map_location=str(device))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded GatedFusion from {model_subdir}  ({n_params:,} params)")

    # Sequence domains come from the embedded BehaviorEncoder
    seq_domains: List[str] = model.behavior_encoder.seq_domains

    # ── Inference ─────────────────────────────────────────────────────────
    all_user_ids: List[int]   = []
    all_probs:    List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):

            deepfm_batch = _unpack_deepfm(batch, model.deepfm, device)
            seq_data, seq_lens, seq_time_buckets = _unpack_seq(
                batch, seq_domains, device
            )

            # model.predict() handles eval mode, sigmoid/softmax, and no_grad
            with torch.amp.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
            ):
                probs = model.predict(
                    deepfm_batch,
                    seq_data,
                    seq_lens,
                    seq_time_buckets,
                )  # (B,)  – P(label=1)

            all_user_ids.extend(batch['user_id'].tolist())
            all_probs.extend(probs.cpu().float().tolist())

            if (batch_idx + 1) % 100 == 0:
                logger.info(
                    f"  Processed {len(all_probs):,} samples ..."
                )

    logger.info(
        f"Inference complete: {len(all_probs):,} samples total"
    )

    # ── Save predictions.json ─────────────────────────────────────────────
    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)

    logger.info(f"Saved {len(all_probs):,} predictions → {output_path}")


if __name__ == '__main__':
    main()