"""
infer_cvr.py  –  Run inference with a trained CVRModel.

Differences from infer.py (BehaviorEncoder)
────────────────────────────────────────────
1. Imports CVRModel instead of BehaviorEncoder.
2. item_id is never unpacked or forwarded — CVRModel does not accept it.
3. loss_type is always "bce" (CVRModel has a single CVR tower); the config
   key is kept for logging parity but never changes the predict path.
4. _load_train_config() priority order unchanged: train_config.json first,
   best/model_config.json as fallback.
5. Config cross-validation now checks CVRModel-side keys
   (seq_vocab_sizes, static_vocab_sizes, etc.) instead of BehaviorEncoder keys.

Environment variables
─────────────────────
MODEL_OUTPUT_PATH   Directory whose ``best/`` sub-directory contains
                    ``model.pt`` and ``model_config.json``.
EVAL_DATA_PATH      Test data directory (``*.parquet`` files).
EVAL_SCHEMA_PATH    Path to ``schema.json``.  Defaults to
                    ``<MODEL_OUTPUT_PATH>/schema.json`` when not set.
EVAL_RESULT_PATH    Directory for the generated ``predictions.json``.

Usage
─────
MODEL_OUTPUT_PATH=/checkpoints \\
EVAL_DATA_PATH=/data/pcvr/test \\
EVAL_RESULT_PATH=/results \\
python infer_cvr.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from pcvr_parquet_dataset import PCVRParquetDataset
from cvr_model import CVRModel   # ← CVRModel instead of BehaviorEncoder


# ──────────────────────────────────────────────
# Logging
# ──────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s  %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('infer_cvr')


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _unpack_seq(
    batch: Dict,
    seq_domains: List[str],
    device: torch.device,
) -> Tuple[Dict, Dict, Dict]:
    seq_data, seq_lens, seq_time_buckets = {}, {}, {}
    for domain in seq_domains:
        seq_data[domain]         = batch[domain].to(device, non_blocking=True)
        seq_lens[domain]         = batch[f"{domain}_len"].to(device, non_blocking=True)
        seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(device, non_blocking=True)
    return seq_data, seq_lens, seq_time_buckets


def _unpack_static(
    batch: Dict,
    device: torch.device,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
    Optional[torch.Tensor],
    Optional[Dict[str, torch.Tensor]],
]:
    """Return (static_ids, static_array_ids, static_dense, static_dense_array_ids).

    NOTE: item_id is intentionally NOT returned — CVRModel does not accept it.
    """
    static_ids = (
        batch['static_ids'].to(device, non_blocking=True)
        if 'static_ids' in batch else None
    )

    array_keys = [k for k in batch if k.startswith('static_array_')]
    static_array_ids = (
        {k: batch[k].to(device, non_blocking=True) for k in array_keys}
        if array_keys else None
    )

    static_dense = (
        batch['static_dense'].to(device, non_blocking=True)
        if 'static_dense' in batch else None
    )

    dense_array_keys = [k for k in batch if k.startswith('static_dense_array_')]
    static_dense_array_ids = (
        {k: batch[k].to(device, non_blocking=True) for k in dense_array_keys}
        if dense_array_keys else None
    )

    # item_id: intentionally NOT unpacked — CVRModel has no item_id input

    return static_ids, static_array_ids, static_dense, static_dense_array_ids


def _predict(
    model: CVRModel,
    seq_data: Dict,
    seq_lens: Dict,
    seq_time_buckets: Dict,
    static_ids: Optional[torch.Tensor],
    static_array_ids: Optional[Dict[str, torch.Tensor]],
    static_dense: Optional[torch.Tensor],
    static_dense_array_ids: Optional[Dict[str, torch.Tensor]],
) -> torch.Tensor:
    """Run forward and return CVR probabilities (B,).

    CVRModel always uses BCE (single logit output); no loss_type branching
    needed.
    """
    out = model(
        seq_data, seq_lens, seq_time_buckets,
        static_ids=static_ids,
        static_array_ids=static_array_ids,
        static_dense=static_dense,
        static_dense_array_ids=static_dense_array_ids,
        # item_id: intentionally not passed
    )
    logits = out["logits"].float()       # (B, 1)
    return torch.sigmoid(logits.squeeze(-1))   # (B,)


# ──────────────────────────────────────────────
# Config loading
# ──────────────────────────────────────────────

def _load_train_config(model_subdir: str) -> Dict:
    """Load training config.

    Priority (matches original infer.py fix):
      1. <MODEL_OUTPUT_PATH>/train_config.json   — written by train_cvr.py,
         always contains static_fids.
      2. <MODEL_OUTPUT_PATH>/best/model_config.json — CVRModel.save() output;
         may lack static_fids.

    Only accepts a file if it contains 'static_fids'; otherwise tries the
    next candidate so features are never silently dropped.
    """
    candidates = [
        (os.path.join(model_subdir, '..', 'train_config.json'), 'train_config.json'),
        (os.path.join(model_subdir, 'model_config.json'),       'best/model_config.json'),
    ]
    for cfg_path, label in candidates:
        cfg_path = os.path.normpath(cfg_path)
        if not os.path.isfile(cfg_path):
            continue
        with open(cfg_path) as f:
            cfg = json.load(f)
        if cfg.get('static_fids'):
            logger.info(f"Loaded train config from: {cfg_path}  (via {label})")
            return cfg
        else:
            logger.warning(
                f"Config file {cfg_path} found but contains no 'static_fids'; "
                "trying next candidate."
            )

    # Fallback: return whatever we can find, warn loudly
    for cfg_path, label in candidates:
        cfg_path = os.path.normpath(cfg_path)
        if os.path.isfile(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            logger.warning(
                f"No config file contains 'static_fids'.  Using {cfg_path} as "
                "fallback.  Static features will be MISSING — check your "
                "MODEL_OUTPUT_PATH."
            )
            return cfg

    logger.warning(
        "Neither train_config.json nor best/model_config.json was found. "
        "Static features will be missing."
    )
    return {}


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main() -> None:

    # ── Env vars ──────────────────────────────
    model_dir   = os.environ.get('MODEL_OUTPUT_PATH')
    data_dir    = os.environ.get('EVAL_DATA_PATH')
    result_dir  = os.environ.get('EVAL_RESULT_PATH')
    schema_path = os.environ.get(
        'EVAL_SCHEMA_PATH',
        os.path.join(model_dir or '', 'schema.json'),
    )

    missing = [k for k, v in [
        ('MODEL_OUTPUT_PATH', model_dir),
        ('EVAL_DATA_PATH',    data_dir),
        ('EVAL_RESULT_PATH',  result_dir),
    ] if not v]
    if missing:
        raise EnvironmentError(
            f"Required env var(s) not set: {', '.join(missing)}"
        )

    model_subdir = os.path.join(model_dir, 'best')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device            : {device}")
    logger.info(f"MODEL_OUTPUT_PATH : {model_dir}")
    logger.info(f"EVAL_DATA_PATH    : {data_dir}")
    logger.info(f"EVAL_RESULT_PATH  : {result_dir}")
    logger.info(f"schema_path       : {schema_path}")

    os.makedirs(result_dir, exist_ok=True)

    # ── Load training config ──────────────────
    cfg = _load_train_config(model_subdir)

    static_fids:             List = [tuple(x) for x in cfg.get('static_fids',             [])]
    static_array_fids:       List = [tuple(x) for x in cfg.get('static_array_fids',       [])]
    static_dense_fids:       List = [tuple(x) for x in cfg.get('static_dense_fids',       [])]
    static_dense_array_fids: List = [tuple(x) for x in cfg.get('static_dense_array_fids', [])]
    # CVRModel is always BCE; log for transparency but do not branch on it
    loss_type: str = cfg.get('loss_type', 'bce')

    logger.info(f"static_int_scalar_fields:    {len(static_fids)}")
    logger.info(f"static_int_array_fields:     {len(static_array_fids)}")
    logger.info(f"static_dense_scalar_fields:  {len(static_dense_fids)}")
    logger.info(f"static_dense_array_fields:   {len(static_dense_array_fids)}")
    logger.info(f"loss_type (informational)  : {loss_type}")

    # ── Cross-validate config vs model_config.json ────────────────────────
    # Catches the common mistake of using a stale / mismatched train_config.
    # CVRModel stores these keys in model_config.json (written by CVRModel.save).
    _mcfg_path = os.path.normpath(os.path.join(model_subdir, 'model_config.json'))
    if os.path.isfile(_mcfg_path):
        with open(_mcfg_path) as _f:
            _mcfg = json.load(_f)
        _checks = [
            # (model_config key,        expected count from train_config,  train_config key)
            ('static_vocab_sizes',   len(static_fids),             'static_fids'),
            ('static_array_configs', len(static_array_fids),       'static_array_fids'),
            ('dense_array_configs',  len(static_dense_array_fids), 'static_dense_array_fids'),
        ]
        for model_key, train_len, train_key in _checks:
            model_len = len(_mcfg.get(model_key) or [])
            if model_len != train_len:
                logger.warning(
                    f"Field count mismatch: train_config['{train_key}']={train_len} "
                    f"vs model_config['{model_key}']={model_len}. "
                    "Check that the correct train_config.json is being used."
                )
        model_n_dense = int(_mcfg.get('n_dense_scalar', 0))
        if model_n_dense != len(static_dense_fids):
            logger.warning(
                f"Field count mismatch: train_config['static_dense_fids']="
                f"{len(static_dense_fids)} vs model_config['n_dense_scalar']="
                f"{model_n_dense}."
            )
        # Log seq domains from model_config for visibility
        seq_domains_from_cfg = sorted(((_mcfg.get('seq_vocab_sizes') or {}).keys()))
        if seq_domains_from_cfg:
            logger.info(f"seq_domains (from model_config): {seq_domains_from_cfg}")

    # ── Dataset ───────────────────────────────
    dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=256,
        shuffle=False,
        buffer_batches=0,
        is_training=False,
        static_fids=static_fids or None,
        static_array_fids=static_array_fids or None,
        static_dense_fids=static_dense_fids or None,
        static_dense_array_fids=static_dense_array_fids or None,
        # item_id: PCVRParquetDataset may still emit it; CVRTrainer ignores it
        # and CVRModel does not accept it — no need to suppress at dataset level
    )

    loader = DataLoader(
        dataset,
        batch_size=None,
        num_workers=4,
        pin_memory=(device.type == 'cuda'),
        prefetch_factor=2,
    )

    # ── Load model ────────────────────────────
    if not os.path.isdir(model_subdir):
        raise FileNotFoundError(f"model sub-directory not found: {model_subdir}")

    model = CVRModel.load(model_subdir, map_location=str(device))
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Loaded CVRModel from {model_subdir}  ({n_params:,} params)")

    seq_domains: List[str] = model.seq_domains
    logger.info(f"seq_domains: {seq_domains}")

    # ── Inference ─────────────────────────────
    all_user_ids: List[int]   = []
    all_probs:    List[float] = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            seq_data, seq_lens, seq_time_buckets = _unpack_seq(batch, seq_domains, device)

            # _unpack_static returns 4 values (no item_id) for CVRModel
            static_ids, static_array_ids, static_dense, static_dense_array_ids = \
                _unpack_static(batch, device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                probs = _predict(
                    model,
                    seq_data, seq_lens, seq_time_buckets,
                    static_ids=static_ids,
                    static_array_ids=static_array_ids,
                    static_dense=static_dense,
                    static_dense_array_ids=static_dense_array_ids,
                    # item_id: intentionally not passed to CVRModel
                )  # (B,)

            all_user_ids.extend(batch['user_id'].tolist())
            all_probs.extend(probs.cpu().float().tolist())

            if (batch_idx + 1) % 100 == 0:
                logger.info(f"  Processed {len(all_probs):,} samples ...")

    logger.info(f"Inference complete: {len(all_probs):,} samples total")

    # ── Save predictions.json ─────────────────
    output_path = os.path.join(result_dir, 'predictions.json')
    with open(output_path, 'w') as f:
        json.dump({"predictions": dict(zip(all_user_ids, all_probs))}, f)

    logger.info(f"Saved {len(all_probs):,} predictions → {output_path}")


if __name__ == '__main__':
    main()