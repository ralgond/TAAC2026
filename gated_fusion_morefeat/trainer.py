"""
trainer.py

Trainer for:
    GatedFusion(
        DeepFM +
        BehaviorEncoder +
        Gate
    )

Features
--------
- AMP training
- BehaviorEncoder forced bf16
- Different LR per sub-model
- Cosine warmup scheduler
- Gradient accumulation
- Early stopping
- Save best only
- BCE / CE support
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import GatedFusion

# ++ 新增：用于计算 AUC
from sklearn.metrics import roc_auc_score
import numpy as np

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
    num_warmup_steps,
    num_training_steps,
    min_lr_ratio: float = 0.0,
):

    def lr_lambda(current_step: int):

        if current_step < num_warmup_steps:

            return (
                float(current_step)
                / float(max(1, num_warmup_steps))
            )

        progress = float(
            current_step - num_warmup_steps
        ) / float(
            max(
                1,
                num_training_steps
                - num_warmup_steps,
            )
        )

        cosine = 0.5 * (
            1.0
            + math.cos(math.pi * progress)
        )

        return max(
            min_lr_ratio,
            cosine,
        )

    return LambdaLR(
        optimizer,
        lr_lambda,
    )


# ─────────────────────────────────────────────────────────────
# Trainer
# ─────────────────────────────────────────────────────────────

class GatedFusionTrainer:

    def __init__(
        self,
        model: GatedFusion,
        train_loader,
        valid_loader,
        save_dir: str,
        num_epochs: int = 3,
        weight_decay: float = 1e-2,
        max_grad_norm: float = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 100,
        eval_interval: int = 0,
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
        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

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

        # ++ best model 判断指标改为 AUC（越大越好，初始为 -inf）
        self.best_valid_auc = float("-inf")

        self.global_step = 0

        self.model.to(self.device)

        # ─────────────────────────────────────
        # Optimizer
        # ─────────────────────────────────────

        self.optimizer = AdamW(
            [
                {
                    "params": model.deepfm.parameters(),
                    "lr": 1e-3,
                },
                {
                    "params": model.behavior_encoder.parameters(),
                    "lr": 1e-4,
                },
                {
                    "params": model.gate.parameters(),
                    "lr": 3e-4,
                },
            ],
            weight_decay=weight_decay,
        )

        total_steps = (
            num_epochs
            * len(train_loader)
            // grad_accum_steps
        )

        self.scheduler = (
            get_cosine_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=2000,
                num_training_steps=total_steps,
            )
        )

        # fp16 only
        self.use_grad_scaler = (
            amp_dtype == torch.float16
        )

        self.scaler = torch.cuda.amp.GradScaler(
            enabled=self.use_grad_scaler
        )

    # ─────────────────────────────────────────
    # Loss
    # ─────────────────────────────────────────

    def _compute_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:

        if self.loss_type == "bce":

            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)

            return F.binary_cross_entropy_with_logits(
                logits,
                labels.float(),
            )

        elif self.loss_type == "ce":

            return F.cross_entropy(
                logits,
                labels.long(),
            )

        else:
            raise ValueError(
                f"Unknown loss_type={self.loss_type}"
            )

    # ─────────────────────────────────────────
    # Batch unpack
    # ─────────────────────────────────────────

    def _unpack_batch(
        self,
        batch: dict,
    ):

        seq_data = {}
        seq_lens = {}
        seq_time_buckets = {}

        for domain in self.model.behavior_encoder.seq_domains:

            seq_data[domain] = batch[
                domain
            ].to(
                self.device,
                non_blocking=True,
            )

            seq_lens[domain] = batch[
                f"{domain}_len"
            ].to(
                self.device,
                non_blocking=True,
            )

            seq_time_buckets[domain] = batch[
                f"{domain}_time_bucket"
            ].to(
                self.device,
                non_blocking=True,
            )

        labels = batch[
            self.label_key
        ].to(
            self.device,
            non_blocking=True,
        )

        return (
            batch,
            seq_data,
            seq_lens,
            seq_time_buckets,
            labels,
        )

    # ─────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────

    def _forward(
        self,
        deepfm_batch,
        seq_data,
        seq_lens,
        seq_time_buckets,
    ):

        # Outer AMP
        with torch.amp.autocast(
            device_type=self.device.type,
            dtype=self.amp_dtype,
            enabled=(
                self.amp_dtype
                != torch.float32
            ),
        ):

            # BehaviorEncoder forced bf16
            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
            ):

                logits = self.model(
                    deepfm_batch,
                    seq_data,
                    seq_lens,
                    seq_time_buckets,
                )

        # Important:
        # unify dtype before BCE / gate
        logits = logits.float()

        return logits

    # ─────────────────────────────────────────
    # Train step
    # ─────────────────────────────────────────

    def _train_step(
        self,
        batch,
    ) -> float:

        self.model.train()

        # 跳过 batch size 为 1 的 batch（BN 层会报错）
        if batch[self.label_key].shape[0] == 1:
            return 0.0

        (
            deepfm_batch,
            seq_data,
            seq_lens,
            seq_time_buckets,
            labels,
        ) = self._unpack_batch(batch)

        logits = self._forward(
            deepfm_batch,
            seq_data,
            seq_lens,
            seq_time_buckets,
        )

        loss = self._compute_loss(
            logits,
            labels,
        )

        loss_for_backward = (
            loss
            / self.grad_accum_steps
        )

        # ─────────────────────────────────────
        # Backward
        # ─────────────────────────────────────

        if self.use_grad_scaler:

            self.scaler.scale(
                loss_for_backward
            ).backward()

        else:

            loss_for_backward.backward()

        # ─────────────────────────────────────
        # Optimizer step
        # ─────────────────────────────────────

        do_step = (
            (self.global_step + 1)
            % self.grad_accum_steps
            == 0
        )

        if do_step:

            if self.use_grad_scaler:

                self.scaler.unscale_(
                    self.optimizer
                )

            nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_grad_norm,
            )

            if self.use_grad_scaler:

                self.scaler.step(
                    self.optimizer
                )

                self.scaler.update()

            else:

                self.optimizer.step()

            self.scheduler.step()

            self.optimizer.zero_grad(
                set_to_none=True
            )

        self.global_step += 1

        return loss.item()

    # ─────────────────────────────────────────
    # Evaluate
    # ─────────────────────────────────────────

    @torch.no_grad()
    def evaluate(self):

        self.model.eval()

        total_loss = 0.0
        total_batches = 0

        # ++ 收集全量预测值和标签，用于计算 AUC
        all_probs = []
        all_labels = []

        for batch in self.valid_loader:

            (
                deepfm_batch,
                seq_data,
                seq_lens,
                seq_time_buckets,
                labels,
            ) = self._unpack_batch(batch)

            logits = self._forward(
                deepfm_batch,
                seq_data,
                seq_lens,
                seq_time_buckets,
            )

            loss = self._compute_loss(
                logits,
                labels,
            )

            total_loss += loss.item()
            total_batches += 1

            # ++ BCE 场景：sigmoid 转概率；CE 场景：softmax 取正类概率
            if self.loss_type == "bce":
                if logits.shape[-1] == 1:
                    logits = logits.squeeze(-1)
                probs = torch.sigmoid(logits)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]

            all_probs.append(
                probs.cpu().numpy()
            )
            all_labels.append(
                labels.cpu().numpy()
            )

        avg_loss = total_loss / max(1, total_batches)

        # ++ 拼接后计算 AUC；若验证集只有单一类别则回退到 0.5
        all_probs_np = np.concatenate(all_probs)
        all_labels_np = np.concatenate(all_labels)

        if len(np.unique(all_labels_np)) < 2:
            logger.warning(
                "Valid set contains only one class; "
                "AUC is undefined, returning 0.5"
            )
            auc = 0.5
        else:
            auc = roc_auc_score(
                all_labels_np,
                all_probs_np,
            )

        return avg_loss, auc

    # ─────────────────────────────────────────
    # Main train loop
    # ─────────────────────────────────────────

    def train(self):
        save_path = (
            self.save_dir / "init"
        )
        save_path.mkdir(parents=True, exist_ok=True)
        self.model.save(
            str(save_path)
        )

        logger.info(
            f"Training GatedFusion "
            f"(epochs={self.num_epochs})"
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        no_improve = 0

        for epoch in range(
            1,
            self.num_epochs + 1,
        ):

            epoch_start = time.time()

            running_loss = 0.0
            running_steps = 0

            for batch in self.train_loader:

                loss = self._train_step(
                    batch
                )

                running_loss += loss
                running_steps += 1

                # ─────────────────────────
                # Logging
                # ─────────────────────────

                if (
                    self.global_step
                    % self.log_interval
                    == 0
                ):

                    avg_loss = (
                        running_loss
                        / max(1, running_steps)
                    )

                    lrs = [
                        group["lr"]
                        for group
                        in self.optimizer.param_groups
                    ]

                    logger.info(
                        f"[E{epoch}] "
                        f"[S{self.global_step}] "
                        f"loss={avg_loss:.6f} "
                        f"lr="
                        f"{[round(x, 8) for x in lrs]}"
                    )

                    running_loss = 0.0
                    running_steps = 0

                # ─────────────────────────
                # Step eval
                # ─────────────────────────

                if (
                    self.eval_interval > 0
                    and self.global_step > 0
                    and (
                        self.global_step
                        % self.eval_interval
                        == 0
                    )
                ):

                    # ++ evaluate 返回 (loss, auc)
                    valid_loss, valid_auc = self.evaluate()

                    logger.info(
                        f"[Eval@Step "
                        f"{self.global_step}] "
                        f"valid_loss={valid_loss:.6f} "
                        f"valid_auc={valid_auc:.6f}"
                    )

            # ─────────────────────────────
            # Epoch eval
            # ─────────────────────────────

            # ++ evaluate 返回 (loss, auc)
            valid_loss, valid_auc = self.evaluate()

            elapsed = (
                time.time()
                - epoch_start
            )

            logger.info(
                f"Epoch {epoch} "
                f"done "
                f"in {elapsed:.1f}s | "
                f"valid_loss={valid_loss:.6f} "
                f"valid_auc={valid_auc:.6f}"
            )

            # ─────────────────────────────
            # Save best（以 AUC 为准，越大越好）
            # ─────────────────────────────

            if valid_auc > self.best_valid_auc:

                self.best_valid_auc = valid_auc

                no_improve = 0

                save_path = (
                    self.save_dir / "best"
                )

                self.model.save(
                    str(save_path)
                )

                logger.info(
                    f"New best model saved "
                    f"to {save_path} "
                    f"(auc={valid_auc:.6f})"
                )

            else:

                no_improve += 1

                logger.info(
                    f"No improvement "
                    f"({no_improve}/"
                    f"{self.patience})"
                )

                if (
                    no_improve
                    >= self.patience
                ):

                    logger.info(
                        "Early stopping triggered"
                    )

                    break

        logger.info(
            "Training finished"
        )