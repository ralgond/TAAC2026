"""debug entry point (self-contained baseline).

Usage:
    python debug.py

Environment variables (take precedence over CLI flags):
    TRAIN_DATA_PATH  Training data directory (*.parquet + schema.json)
    TRAIN_CKPT_PATH  Checkpoint output directory
    TRAIN_LOG_PATH   Log directory
"""
import os
import torch
import logging
from dataset import get_pcvr_data

data_dir = os.environ.get('TRAIN_DATA_PATH')
log_dir = os.environ.get('TRAIN_LOG_PATH')
schema_path = os.path.join(data_dir, 'schema.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

def main():
    train_loader, valid_loader, pcvr_dataset = get_pcvr_data(data_dir,schema_path)
    logging.info(f"train_loader.len: {len(train_loader)}")

    for batch in train_loader:
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                logging.info(f"{k} is torch.Tensor, and shape={v.shape}")
            else:
                logging.info(f"{k} is not torch.Tensor")
        break
        
if __name__ == "__main__":
    main()