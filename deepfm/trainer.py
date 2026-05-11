"""Trainer for DeepFM.

Responsibilities
----------------
- Training loop with gradient accumulation and optional AMP.
- Validation loop with AUC reporting.
- ``save_checkpoint`` / ``load_checkpoint`` for full training resumption.
- ``save_model`` / ``load_model`` for inference-only model weights.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from sklearn.metrics import roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False

from model import DeepFM


logger = logging.getLogger(__name__)


class Trainer:
    """Trains a :class:`DeepFM` model.

    Args:
        model:            DeepFM instance (already on the target device).
        optimizer:        PyTorch optimizer.
        scheduler:        optional LR scheduler (stepped once per optimizer step).
        device:           torch device.
        use_amp:          enable automatic mixed precision (CUDA only).
        grad_clip:        max gradient norm; ``None`` to disable.
        grad_accum_steps: accumulate gradients over N batches before stepping.
        log_interval:     log training loss every N batches.
    """

    def __init__(
        self,
        model: DeepFM,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: Optional[torch.device] = None,
        use_amp: bool = False,
        grad_clip: Optional[float] = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 100,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device or next(model.parameters()).device
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.grad_clip = grad_clip
        self.grad_accum_steps = max(1, grad_accum_steps)
        self.log_interval = log_interval

        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)
        self.criterion = nn.BCEWithLogitsLoss()

        self._global_step: int = 0
        self._epoch: int = 0

    # ──────────────────────────────────────────
    # Training
    # ──────────────────────────────────────────

    def train_epoch(self, loader: DataLoader) -> float:
        """Run one full pass over ``loader``.

        Returns:
            Mean training loss over the epoch.
        """
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        accum_loss = torch.tensor(0.0, device=self.device)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(loader):
            labels = batch['label'].to(self.device).float()
            B = labels.size(0)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(batch)
                loss = self.criterion(logits, labels) / self.grad_accum_steps

            self.scaler.scale(loss).backward()
            accum_loss += loss.detach() * self.grad_accum_steps

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.grad_clip is not None:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)  # free grad memory immediately
                if self.scheduler is not None:
                    self.scheduler.step()          # step-level LR update
                self._global_step += 1

            total_loss += accum_loss.item() * B
            total_samples += B
            accum_loss.fill_(0.0)

            if (batch_idx + 1) % self.log_interval == 0:
                avg = total_loss / max(total_samples, 1)
                lr = self.optimizer.param_groups[0]['lr']
                logger.info(
                    f"[epoch {self._epoch+1}] step {self._global_step} "
                    f"batch {batch_idx+1}  loss={avg:.5f}  lr={lr:.2e}")

        self._epoch += 1
        return total_loss / max(total_samples, 1)

    # ──────────────────────────────────────────
    # Validation
    # ──────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Dict[str, float]:
        """Evaluate on ``loader``.

        Returns:
            Dict with keys ``loss`` and (if sklearn is available) ``auc``.
        """
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_labels: list = []
        all_probs: list = []

        for batch in loader:
            labels = batch['label'].to(self.device).float()
            B = labels.size(0)

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(batch)
                loss = self.criterion(logits, labels)

            total_loss += loss.item() * B
            total_samples += B

            if _HAS_SKLEARN:
                all_labels.append(labels.cpu())
                all_probs.append(torch.sigmoid(logits).cpu())

        metrics: Dict[str, float] = {
            'loss': total_loss / max(total_samples, 1)
        }
        if _HAS_SKLEARN and all_labels:
            y_true = torch.cat(all_labels).numpy()
            y_score = torch.cat(all_probs).numpy()
            if y_true.sum() > 0 and (1 - y_true).sum() > 0:
                metrics['auc'] = float(roc_auc_score(y_true, y_score))
        return metrics

    # ──────────────────────────────────────────
    # Checkpoint (full training state)
    # ──────────────────────────────────────────

    def save_checkpoint(self, path: str) -> None:
        """Save full training state (model + optimizer + scheduler + counters).

        Use this during training to support resumption.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        state = {
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
            'epoch':                self._epoch,
            'global_step':          self._global_step,
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(state, path)
        logger.info(f"Checkpoint saved → {path}")

    def load_checkpoint(self, path: str) -> None:
        """Resume training from a checkpoint saved by :meth:`save_checkpoint`."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self._epoch       = ckpt.get('epoch', 0)
        self._global_step = ckpt.get('global_step', 0)
        if self.scheduler is not None and 'scheduler_state_dict' in ckpt:
            self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        logger.info(
            f"Checkpoint loaded ← {path}  "
            f"(epoch={self._epoch}, step={self._global_step})")

    # ──────────────────────────────────────────
    # Model-only save / load (for inference)
    # ──────────────────────────────────────────

    def save_model(self, path: str) -> None:
        """Save only the model weights.  Lighter than a full checkpoint;
        intended for serving / inference.
        """
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info(f"Model weights saved → {path}")

    def load_model(self, path: str) -> None:
        """Load model weights saved by :meth:`save_model`."""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        logger.info(f"Model weights loaded ← {path}")

    # ──────────────────────────────────────────
    # Convenience: run N epochs
    # ──────────────────────────────────────────

    def fit(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        num_epochs: int,
        save_dir: str = 'checkpoints',
        save_best: bool = True,
    ) -> None:
        """Train for ``num_epochs`` epochs, evaluating after each epoch.

        Args:
            train_loader: training DataLoader.
            valid_loader: validation DataLoader.
            num_epochs:   number of epochs to run.
            save_dir:     directory for checkpoints and best-model weights.
            save_best:    if True, save ``best_model.pt`` whenever validation
                          AUC (or loss when AUC is unavailable) improves.
        """
        best_metric: Optional[float] = None
        higher_is_better = _HAS_SKLEARN  # AUC: higher; loss: lower

        for epoch in range(num_epochs):
            t0 = time.time()
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(valid_loader)
            elapsed = time.time() - t0

            metric_str = '  '.join(f'{k}={v:.5f}' for k, v in val_metrics.items())
            logger.info(
                f"Epoch {self._epoch}/{num_epochs}  "
                f"train_loss={train_loss:.5f}  {metric_str}  "
                f"({elapsed:.1f}s)")

            # Save checkpoint under save_dir/global_step100/checkpoint-{N}.pt
            ckpt_path = os.path.join(
                save_dir, 'global_step100', f'checkpoint-{self._global_step}.pt')
            self.save_checkpoint(ckpt_path)

            # Save best model
            if save_best:
                current = val_metrics.get('auc', -val_metrics['loss'])
                if best_metric is None or current > best_metric:
                    best_metric = current
                    best_path = os.path.join(save_dir, 'global_step100', 'best_model.pt')
                    self.save_model(best_path)
                    logger.info(f"  ↑ New best ({current:.5f}) → {best_path}")