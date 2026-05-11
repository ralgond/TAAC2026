"""infer.py – run inference with a trained DeepFM model.

Environment variables
---------------------
MODEL_OUTPUT_PATH   Directory containing ``global_step100/best_model.pt``
                    and ``global_step100/model_config.json``.
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
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader

from pcvr_parquet_dataset import PCVRParquetDataset
from model import DeepFM


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
# Helpers (shared with train.py)
# ──────────────────────────────────────────────

def _build_int_fields(
    schema: PCVRParquetDataset,
    group: str,
) -> List[Tuple[int, int]]:
    cols = schema._user_int_cols if group == 'user' else schema._item_int_cols
    return [(fid, vs) for fid, vs, dim in cols if dim == 1]


def _dense_dim_and_fids(
    schema: PCVRParquetDataset,
) -> Tuple[int, List[int]]:
    fids = [fid for fid, dim in schema._user_dense_cols if dim == 1]
    return len(fids), fids


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

    # model files live under the global_step100 subdirectory
    model_subdir = os.path.join(model_dir, 'global_step100')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"MODEL_OUTPUT_PATH : {model_dir}")
    logger.info(f"EVAL_DATA_PATH    : {data_dir}")
    logger.info(f"EVAL_RESULT_PATH  : {result_dir}")

    os.makedirs(result_dir, exist_ok=True)

    schema_path = os.path.join(data_dir, 'schema.json')

    # ── Dataset (no shuffle, is_training=False) ───────────────────────────
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=512,
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

    # ── Load model config ─────────────────────────────────────────────────
    config_path = os.path.join(model_subdir, 'model_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"model_config.json not found at {config_path}")
    with open(config_path) as f:
        cfg = json.load(f)
    logger.info(f"Loaded model config from {config_path}")

    # ── Build model ───────────────────────────────────────────────────────
    user_int_fields = _build_int_fields(dataset, 'user')
    item_int_fields = _build_int_fields(dataset, 'item')
    dense_dim, dense_fids = _dense_dim_and_fids(dataset)

    model = DeepFM(
        user_int_fields=user_int_fields,
        item_int_fields=item_int_fields,
        dense_dim=dense_dim,
        num_items=cfg['num_items'],
        embed_dim=cfg['embed_dim'],
        dnn_hidden_dims=cfg['dnn_hidden_dims'],
        dnn_dropout=cfg['dnn_dropout'],
        num_buckets=cfg.get('num_buckets'),
    ).to(device)
    model.register_dense_fids(dense_fids)

    # ── Load weights ──────────────────────────────────────────────────────
    weights_path = os.path.join(model_subdir, 'best_model.pt')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"best_model.pt not found at {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    logger.info(f"Loaded weights from {weights_path}")

    # ── Inference ─────────────────────────────────────────────────────────
    all_user_ids: List[int] = []
    all_probs:    List[float] = []

    for batch_idx, batch in enumerate(loader):
        probs = model.predict(batch)   # [B], float32, in [0, 1]
        all_user_ids.extend(batch['user_id'].tolist())
        all_probs.extend(probs.cpu().tolist())

        if (batch_idx + 1) % 100 == 0:
            logger.info(f"  Processed {len(all_probs)} samples …")

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