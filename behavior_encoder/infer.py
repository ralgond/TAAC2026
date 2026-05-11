"""infer.py – run inference with a trained BehaviorEncoder model.

Environment variables
---------------------
MODEL_OUTPUT_PATH   Directory containing ``best/model.pt``
                    and ``best/model_config.json``.
EVAL_DATA_PATH      Test data directory (``*.parquet`` + ``schema.json``).
EVAL_RESULT_PATH    Directory for the generated ``predictions.json``.

Usage
-----
MODEL_OUTPUT_PATH=/checkpoints \
EVAL_DATA_PATH=/data/pcvr/test \
EVAL_RESULT_PATH=/results \
python infer.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import List

import torch
from torch.utils.data import DataLoader

from pcvr_parquet_dataset import PCVRParquetDataset
from model import BehaviorEncoder


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
# Main
# ──────────────────────────────────────────────

def main() -> None:
    # ── Read environment variables ────────────────────────────────────────
    model_dir  = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir   = os.environ.get('EVAL_DATA_PATH')
    result_dir = os.environ.get('EVAL_RESULT_PATH')

    missing = [k for k, v in [
        ('MODEL_OUTPUT_PATH', model_dir),
        ('EVAL_DATA_PATH',    data_dir),
        ('EVAL_RESULT_PATH',  result_dir),
    ] if not v]
    if missing:
        raise EnvironmentError(
            f"Required environment variable(s) not set: {', '.join(missing)}")

    # weights and config live under the best/ subdirectory
    model_subdir = os.path.join(model_dir, 'best')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"MODEL_OUTPUT_PATH : {model_dir}")
    logger.info(f"EVAL_DATA_PATH    : {data_dir}")
    logger.info(f"EVAL_RESULT_PATH  : {result_dir}")

    os.makedirs(result_dir, exist_ok=True)

    schema_path = os.path.join(model_dir, 'schema.json')

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
        pin_memory=False,
        prefetch_factor=1,
    )

    # ── Load model (config + weights) ─────────────────────────────────────
    config_path  = os.path.join(model_subdir, 'model_config.json')
    weights_path = os.path.join(model_subdir, 'model.pt')

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"model_config.json not found at {config_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"model.pt not found at {weights_path}")

    model = BehaviorEncoder.load(model_subdir, map_location=str(device))
    model.to(device)
    model.eval()
    logger.info(f"Loaded model from {model_subdir}")

    # ── Inference ─────────────────────────────────────────────────────────
    all_user_ids: List[int]   = []
    all_probs:    List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            # Unpack sequence inputs
            # Key convention: seq_a / seq_a_len / seq_a_time_bucket
            seq_data, seq_lens, seq_time_buckets = {}, {}, {}
            for domain in model.seq_domains:
                seq_data[domain] = batch[domain].to(device, non_blocking=True)
                seq_lens[domain] = batch[f"{domain}_len"].to(device, non_blocking=True)
                seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(
                    device, non_blocking=True
                )

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(seq_data, seq_lens, seq_time_buckets)  # (B, action_num)

            # Convert logits to probabilities
            if logits.shape[-1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1))   # (B,)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1] # binary: prob of class 1

            all_user_ids.extend(batch['user_id'].tolist())
            all_probs.extend(probs.cpu().float().tolist())

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Processed {len(all_probs)} samples ...")

    logger.info(f"Inference complete: {len(all_probs)} samples total")

    # ── Save predictions.json ─────────────────────────────────────────────
    predictions = {
        "predictions": dict(zip(all_user_ids, all_probs)),
    }
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump(predictions, f)
    logger.info(f"Saved {len(all_probs)} predictions to {output_path}")


if __name__ == '__main__':
    main()