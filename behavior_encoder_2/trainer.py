"""
trainer.py

Trainer for BehaviorEncoder (standalone, no GatedFusion / DeepFM).

Features
--------
- AMP training (bfloat16 by default)
- Separate LR for sparse (embedding) vs dense params
- Cosine warmup scheduler
- Gradient accumulation
- Early stopping on validation AUC
- Save best checkpoint only
- BCE / CE loss support
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from behavior_encoder_model import BehaviorEncoder

import sys
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger('trainer')


# ─────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────

def get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.0,
):
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(min_lr_ratio, cosine)

    return LambdaLR(optimizer, lr_lambda)


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class BehaviorEncoderTrainer:

    def __init__(
        self,
        model: BehaviorEncoder,
        train_loader,
        valid_loader,
        save_dir: str,
        num_epochs: int = 3,
        lr: float = 1e-4,
        sparse_lr: float = 1e-3,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 100,
        eval_interval: int = 5000,
        patience: int = 3,
        device: str = "cuda",
        label_key: str = "label",
        loss_type: str = "bce",
        amp_dtype=torch.bfloat16,
    ) -> None:

        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.num_epochs = num_epochs
        self.max_grad_norm = max_grad_norm
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.patience = patience
        self.device = torch.device(device)
        self.label_key = label_key
        self.loss_type = loss_type
        self.amp_dtype = amp_dtype
        self.best_valid_auc = float("-inf")
        self.global_step = 0

        self.model.to(self.device)

        # ── Optimizer: separate LR for sparse (embedding) and dense params ──
        sparse_params = model.get_sparse_params()
        dense_params  = model.get_dense_params()
        self.optimizer = AdamW(
            [
                {"params": dense_params,  "lr": lr},
                {"params": sparse_params, "lr": sparse_lr},
            ],
            weight_decay=weight_decay,
        )

        total_steps = num_epochs * len(train_loader) // grad_accum_steps
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=2000,
            num_training_steps=total_steps,
        )

        self.use_grad_scaler = (amp_dtype == torch.float16)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_grad_scaler)

    # ─────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if self.loss_type == "bce":
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            return F.binary_cross_entropy_with_logits(logits, labels.float())
        elif self.loss_type == "ce":
            return F.cross_entropy(logits, labels.long())
        else:
            raise ValueError(f"Unknown loss_type={self.loss_type}")

    # ─────────────────────────────────────────
    # Batch unpack
    # ─────────────────────────────────────────

    def _unpack_batch(self, batch: dict):
        seq_data, seq_lens, seq_time_buckets = {}, {}, {}

        for domain in self.model.seq_domains:
            seq_data[domain]         = batch[domain].to(self.device, non_blocking=True)
            seq_lens[domain]         = batch[f"{domain}_len"].to(self.device, non_blocking=True)
            seq_time_buckets[domain] = batch[f"{domain}_time_bucket"].to(self.device, non_blocking=True)

        labels = batch[self.label_key].to(self.device, non_blocking=True)

        static_ids: Optional[torch.Tensor] = None
        if 'static_ids' in batch:
            static_ids = batch['static_ids'].to(self.device, non_blocking=True)

        static_array_ids: Optional[Dict[str, torch.Tensor]] = None
        _array_keys = [k for k in batch if k.startswith('static_array_')]
        if _array_keys:
            static_array_ids = {k: batch[k].to(self.device, non_blocking=True) for k in _array_keys}

        static_dense: Optional[torch.Tensor] = None
        if 'static_dense' in batch:
            static_dense = batch['static_dense'].to(self.device, non_blocking=True)

        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None
        _dense_array_keys = [k for k in batch if k.startswith('static_dense_array_')]
        if _dense_array_keys:
            static_dense_array_ids = {k: batch[k].to(self.device, non_blocking=True) for k in _dense_array_keys}

        return (
            seq_data, seq_lens, seq_time_buckets, labels,
            static_ids, static_array_ids,
            static_dense, static_dense_array_ids,
        )

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────

    def _forward(
        self,
        seq_data,
        seq_lens,
        seq_time_buckets,
        static_ids: Optional[torch.Tensor] = None,
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,
        static_dense: Optional[torch.Tensor] = None,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:

        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(self.amp_dtype != torch.float32),
        ):
            out = self.model(
                seq_data,
                seq_lens,
                seq_time_buckets,
                static_ids=static_ids,
                static_array_ids=static_array_ids,
                static_dense=static_dense,
                static_dense_array_ids=static_dense_array_ids,
            )

        return out["logits"].float()

    # ─────────────────────────────────────────
    # Train step
    # ─────────────────────────────────────────

    def _train_step(self, batch) -> float:
        self.model.train()

        if batch[self.label_key].shape[0] == 1:
            return 0.0

        (
            seq_data, seq_lens, seq_time_buckets, labels,
            static_ids, static_array_ids,
            static_dense, static_dense_array_ids,
        ) = self._unpack_batch(batch)

        logits = self._forward(
            seq_data, seq_lens, seq_time_buckets,
            static_ids=static_ids, static_array_ids=static_array_ids,
            static_dense=static_dense, static_dense_array_ids=static_dense_array_ids,
        )

        loss = self._compute_loss(logits, labels)
        loss_for_backward = loss / self.grad_accum_steps

        if self.use_grad_scaler:
            self.scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        do_step = (self.global_step + 1) % self.grad_accum_steps == 0
        if do_step:
            if self.use_grad_scaler:
                self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            if self.use_grad_scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        self.global_step += 1
        return loss.item()

    # ─────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss, total_batches = 0.0, 0
        all_probs, all_labels = [], []

        for batch in self.valid_loader:
            (
                seq_data, seq_lens, seq_time_buckets, labels,
                static_ids, static_array_ids,
                static_dense, static_dense_array_ids,
            ) = self._unpack_batch(batch)

            logits = self._forward(
                seq_data, seq_lens, seq_time_buckets,
                static_ids=static_ids, static_array_ids=static_array_ids,
                static_dense=static_dense, static_dense_array_ids=static_dense_array_ids,
            )

            loss = self._compute_loss(logits, labels)
            total_loss   += loss.item()
            total_batches += 1

            if self.loss_type == "bce":
                if logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

        avg_loss      = total_loss / max(1, total_batches)
        all_probs_np  = np.concatenate(all_probs)
        all_labels_np = np.concatenate(all_labels)

        if len(np.unique(all_labels_np)) < 2:
            logger.warning("Valid set has only one class; AUC undefined, returning 0.5")
            auc = 0.5
        else:
            auc = roc_auc_score(all_labels_np, all_probs_np)

        return avg_loss, auc

    # ─────────────────────────────────────────
    # Train loop
    # ─────────────────────────────────────────

    def train(self):
        init_path = self.save_dir / "init"
        init_path.mkdir(parents=True, exist_ok=True)
        self.model.save(str(init_path))

        logger.info(f"Training BehaviorEncoder (epochs={self.num_epochs})")
        self.optimizer.zero_grad(set_to_none=True)
        no_improve = 0

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()
            running_loss, running_steps = 0.0, 0

            for batch in self.train_loader:
                loss = self._train_step(batch)
                running_loss  += loss
                running_steps += 1

                if self.global_step % self.log_interval == 0:
                    avg = running_loss / max(1, running_steps)
                    lrs = [round(g["lr"], 8) for g in self.optimizer.param_groups]
                    logger.info(
                        f"[E{epoch}] [S{self.global_step}] loss={avg:.6f} lr={lrs}"
                    )
                    running_loss, running_steps = 0.0, 0

                if (
                    self.eval_interval > 0
                    and self.global_step > 0
                    and self.global_step % self.eval_interval == 0
                ):
                    valid_loss, valid_auc = self.evaluate()
                    logger.info(
                        f"[Eval@Step {self.global_step}] "
                        f"valid_loss={valid_loss:.6f} valid_auc={valid_auc:.6f}"
                    )

                    if valid_auc > self.best_valid_auc:
                        self.best_valid_auc = valid_auc
                        no_improve = 0
                        best_path = self.save_dir / "best"
                        self.model.save(str(best_path))
                        logger.info(
                            f"New best model → {best_path} (auc={valid_auc:.6f})"
                        )
                    else:
                        no_improve += 1
                        logger.info(f"No improvement ({no_improve}/{self.patience})")
                        if no_improve >= self.patience:
                            logger.info("Early stopping triggered")
                            return

            elapsed = time.time() - epoch_start
            logger.info(
                f"Epoch {epoch} done in {elapsed:.1f}s | "
                f"best_auc_so_far={self.best_valid_auc:.6f}"
            )

        logger.info("Training finished")


# Backward-compat alias
GatedFusionTrainer = BehaviorEncoderTrainer