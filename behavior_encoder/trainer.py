"""
Trainer for BehaviorEncoder.

Features:
  - Mixed-precision training with torch.amp (bfloat16) to reduce VRAM usage
  - AdamW for dense parameters + optional Adagrad for sparse embedding params
  - Learning-rate warm-up + cosine decay
  - Gradient clipping
  - Periodic validation and model checkpointing
  - Early stopping
"""

import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from model import BehaviorEncoder

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Loss helpers
# ─────────────────────────────────────────────────────────────────────────────


def bce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy loss for multi-label or single-label tasks.

    Handles the common case where action_num=1 produces logits of shape (B, 1)
    but labels are (B,): squeeze the last dim before computing loss.
    For multi-label (action_num>1), both logits and labels should be (B, N).
    """
    if logits.shape[-1] == 1:
        logits = logits.squeeze(-1)   # (B, 1) -> (B,)
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def cross_entropy_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, labels.long())


import torch.nn.functional as F  # noqa: E402 (needed after bce_loss def)


# ─────────────────────────────────────────────────────────────────────────────
# LR schedule
# ─────────────────────────────────────────────────────────────────────────────


def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
) -> LambdaLR:
    def lr_lambda(current_step: int) -> float:
        if current_step < num_warmup_steps:
            return float(current_step) / max(1, num_warmup_steps)
        progress = float(current_step - num_warmup_steps) / max(
            1, num_training_steps - num_warmup_steps
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def compute_auc_approx(
    logits: torch.Tensor, labels: torch.Tensor
) -> float:
    """Approximate AUC-ROC via sklearn if available, else returns -1."""
    try:
        from sklearn.metrics import roc_auc_score
        scores = torch.sigmoid(logits).cpu().float().numpy()
        y = labels.cpu().float().numpy()
        if y.ndim > 1:
            # Multi-label: macro average
            aucs = []
            for i in range(y.shape[1]):
                if y[:, i].sum() > 0:
                    aucs.append(roc_auc_score(y[:, i], scores[:, i]))
            return float(sum(aucs) / len(aucs)) if aucs else -1.0
        return float(roc_auc_score(y, scores))
    except Exception:
        return -1.0


# ─────────────────────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────────────────────


class BehaviorEncoderTrainer:
    """
    Trainer for BehaviorEncoder with AMP bfloat16.

    Args:
        model:              BehaviorEncoder instance.
        train_loader:       DataLoader for training set.
        valid_loader:       DataLoader for validation set.
        save_dir:           Directory for checkpoints and config.
        lr:                 Peak learning rate for dense params.
        sparse_lr:          Learning rate for embedding params (Adagrad).
        weight_decay:       AdamW weight decay.
        max_grad_norm:      Gradient clipping norm.
        num_epochs:         Maximum training epochs.
        warmup_ratio:       Fraction of total steps used for LR warm-up.
        log_interval:       Log every N steps.
        eval_interval:      Evaluate every N steps (0 = once per epoch).
        patience:           Early-stopping patience (epochs with no improvement).
        device:             Training device string, e.g. 'cuda:0'.
        use_separate_optim: Use Adagrad for embeddings + AdamW for dense params.
        label_key:          Key in batch dict for labels.
        loss_type:          'bce' or 'ce'.
        amp_dtype:          torch.bfloat16 or torch.float16 (default bfloat16).
    """

    def __init__(
        self,
        model: BehaviorEncoder,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        save_dir: str = "checkpoints",
        lr: float = 1e-3,
        sparse_lr: float = 0.05,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        num_epochs: int = 10,
        warmup_ratio: float = 0.05,
        log_interval: int = 100,
        eval_interval: int = 0,
        patience: int = 3,
        device: str = "cuda",
        use_separate_optim: bool = True,
        label_key: str = "label",
        loss_type: str = "bce",
        amp_dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.lr = lr
        self.sparse_lr = sparse_lr
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.num_epochs = num_epochs
        self.warmup_ratio = warmup_ratio
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.patience = patience
        self.device = torch.device(device)
        self.label_key = label_key
        self.amp_dtype = amp_dtype

        self.loss_fn = bce_loss if loss_type == "bce" else cross_entropy_loss

        # Move model to device
        self.model.to(self.device)

        # ── Optimizers ──────────────────────────────────────────────────
        if use_separate_optim:
            dense_params = model.get_dense_params()
            sparse_params = model.get_sparse_params()
            self.dense_optimizer = AdamW(
                dense_params, lr=lr, weight_decay=weight_decay
            )
            self.sparse_optimizer = torch.optim.Adagrad(sparse_params, lr=sparse_lr)
            self.optimizers = [self.dense_optimizer, self.sparse_optimizer]
        else:
            self.dense_optimizer = AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )
            self.sparse_optimizer = None
            self.optimizers = [self.dense_optimizer]

        # ── LR scheduler (dense only; Adagrad has its own adaptive schedule) ──
        total_steps = num_epochs * len(train_loader)
        warmup_steps = max(1, int(total_steps * warmup_ratio))
        self.scheduler = get_cosine_schedule_with_warmup(
            self.dense_optimizer, warmup_steps, total_steps
        )

        # ── AMP scaler ──────────────────────────────────────────────────
        # GradScaler is only needed for float16; bfloat16 does not overflow
        self._use_scaler = amp_dtype == torch.float16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self._use_scaler)

        # ── State ────────────────────────────────────────────────────────
        self.global_step = 0
        self.best_valid_loss = float("inf")
        self.best_valid_auc = -1.0
        self.no_improve_epochs = 0

    # ── Batch unpacking ────────────────────────────────────────────────────

    def _unpack_batch(self, batch: dict) -> Tuple[Dict, Dict, Dict, torch.Tensor]:
        """
        Extracts seq_data, seq_lens, seq_time_buckets and labels from a batch dict.
        Moves tensors to device.
        """
        seq_data, seq_lens, seq_time_buckets = {}, {}, {}

        for domain in self.model.seq_domains:
            # Batch key convention (confirmed by dataset inspection):
            #   seq_a               → (B, num_fids, L)
            #   seq_a_len           → (B,)
            #   seq_a_time_bucket   → (B, L)
            # domain is already the full batch key, e.g. 'seq_a'
            seq_data[domain] = batch[domain].to(self.device, non_blocking=True)
            seq_lens[domain] = batch[f"{domain}_len"].to(self.device, non_blocking=True)
            seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(
                self.device, non_blocking=True
            )

        labels = batch[self.label_key].to(self.device, non_blocking=True)
        return seq_data, seq_lens, seq_time_buckets, labels

    # ── Training step ──────────────────────────────────────────────────────

    def _train_step(self, batch: dict) -> float:
        self.model.train()
        seq_data, seq_lens, seq_time_buckets, labels = self._unpack_batch(batch)

        for opt in self.optimizers:
            opt.zero_grad(set_to_none=True)

        # Forward pass under AMP context (bfloat16)
        with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
            logits = self.model(seq_data, seq_lens, seq_time_buckets)
            loss = self.loss_fn(logits, labels)

        # Backward
        if self._use_scaler:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.dense_optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            for opt in self.optimizers:
                self.scaler.step(opt)
            self.scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            for opt in self.optimizers:
                opt.step()

        self.scheduler.step()
        self.global_step += 1
        return loss.item()

    # ── Validation ─────────────────────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        """Returns (avg_loss, auc)."""
        self.model.eval()
        total_loss = 0.0
        all_logits, all_labels = [], []

        for batch in self.valid_loader:
            seq_data, seq_lens, seq_time_buckets, labels = self._unpack_batch(batch)

            with torch.amp.autocast(device_type=self.device.type, dtype=self.amp_dtype):
                logits = self.model(seq_data, seq_lens, seq_time_buckets)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            all_logits.append(logits.float().cpu())
            all_labels.append(labels.float().cpu())

        avg_loss = total_loss / max(1, len(self.valid_loader))
        all_logits = torch.cat(all_logits, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        auc = compute_auc_approx(all_logits, all_labels)
        return avg_loss, auc

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    def _save_checkpoint(self, tag: str) -> None:
        ckpt_dir = self.save_dir / tag
        self.model.save(str(ckpt_dir))
        logger.info(f"Checkpoint saved → {ckpt_dir}")

    # ── Main training loop ─────────────────────────────────────────────────

    def train(self) -> None:
        logger.info(
            f"Starting training: {self.num_epochs} epochs, "
            f"device={self.device}, amp_dtype={self.amp_dtype}"
        )

        # Save initial config so architecture is always recoverable
        self.model.save(str(self.save_dir / "init"))

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            running_loss = 0.0
            step_count = 0

            for step, batch in enumerate(self.train_loader):
                loss = self._train_step(batch)
                running_loss += loss
                step_count += 1

                # Logging
                if self.global_step % self.log_interval == 0:
                    avg = running_loss / step_count
                    lr_now = self.scheduler.get_last_lr()[0]
                    logger.info(
                        f"[E{epoch} S{self.global_step}] loss={avg:.4f}  lr={lr_now:.2e}"
                    )
                    running_loss = 0.0
                    step_count = 0

                # Mid-epoch validation
                if self.eval_interval > 0 and self.global_step % self.eval_interval == 0:
                    val_loss, val_auc = self.evaluate()
                    logger.info(
                        f"  [mid-epoch valid] loss={val_loss:.4f}  auc={val_auc:.4f}"
                    )
                    if val_loss < self.best_valid_loss:
                        self.best_valid_loss = val_loss
                        self._save_checkpoint("best")

            # End-of-epoch validation
            val_loss, val_auc = self.evaluate()
            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch}/{self.num_epochs} done in {elapsed:.1f}s  |  "
                f"valid loss={val_loss:.4f}  auc={val_auc:.4f}"
            )

            # Save latest checkpoint every epoch
            self._save_checkpoint(f"epoch_{epoch:03d}")

            # Best-model tracking
            improved = val_loss < self.best_valid_loss
            if improved:
                self.best_valid_loss = val_loss
                self.best_valid_auc = val_auc
                self.no_improve_epochs = 0
                self._save_checkpoint("best")
                logger.info(f"  ✓ New best: loss={val_loss:.4f}  auc={val_auc:.4f}")
            else:
                self.no_improve_epochs += 1
                logger.info(
                    f"  No improvement for {self.no_improve_epochs}/{self.patience} epochs."
                )

            # Early stopping
            if self.no_improve_epochs >= self.patience:
                logger.info("Early stopping triggered.")
                break

        logger.info(
            f"Training finished. Best valid loss={self.best_valid_loss:.4f}  "
            f"auc={self.best_valid_auc:.4f}"
        )