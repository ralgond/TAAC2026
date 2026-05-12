"""
GatedFusion model:
    output = gate * behavior_encoder + (1 - gate) * deepfm

Gate:
    gate = sigmoid(W[x1;x2])

Requirements:
  - DeepFM outputs logits:           (B, action_num)
  - BehaviorEncoder outputs logits:  (B, action_num)
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from deepfm_model import DeepFM
from behavior_encoder_model import BehaviorEncoder

logger = logging.getLogger(__name__)


class GateNetwork(nn.Module):
    """
    gate = sigmoid(W[x1;x2])

    x1 = deepfm hidden
    x2 = behavior hidden
    """

    def __init__(
        self,
        gate_in_dim: int = 128,
        action_num: int = 1,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()

        self.gate = nn.Sequential(
            nn.Linear(gate_in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_num),
        )

    def forward(
        self,
        deepfm_hidden: torch.Tensor,
        behavior_hidden: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([deepfm_hidden, behavior_hidden], dim=-1)
        ret = torch.sigmoid(self.gate(x))
        return ret


class GatedFusion(nn.Module):

    _CONFIG_KEYS = [
        "action_num",
    ]

    def __init__(
        self,
        deepfm: DeepFM,
        behavior_encoder: BehaviorEncoder,
        action_num: int = 1,
        gate_in_dim: int = 128
    ) -> None:
        super().__init__()

        self.deepfm = deepfm
        self.behavior_encoder = behavior_encoder

        self.action_num = action_num
        self.gate_in_dim = gate_in_dim

        self.gate = GateNetwork(
            gate_in_dim=gate_in_dim,
            action_num=action_num,
            hidden_dim=max(32, action_num * 4),
        )

    # ─────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────

    def forward(
        self,
        deepfm_batch: Dict,
        seq_data: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:

        res = self.deepfm(deepfm_batch)
        deepfm_hidden = res['hidden']
        deepfm_logits = res['logits']

        res = self.behavior_encoder(
            seq_data,
            seq_lens,
            seq_time_buckets,
        )
        behavior_hidden = res['hidden']
        behavior_logits = res['logits']

        gate = self.gate(
            deepfm_hidden,
            behavior_hidden,
        )

        behavior_logits = behavior_logits.squeeze(-1)
        deepfm_logits = deepfm_logits.squeeze(-1)
        gate = gate.squeeze(-1)
        logits = (
            gate * behavior_logits
            + (1.0 - gate) * deepfm_logits
        )

        return logits

    # ─────────────────────────────────────────────────────────────
    # Predict: return P(label=1) in [0, 1]
    # ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict(
        self,
        deepfm_batch: Dict,
        seq_data: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Run inference and return the probability of label=1.

        Signature mirrors forward(); no gradients are tracked.

        Returns
        -------
        probs : torch.Tensor, shape (B,) or (B, action_num)
            - action_num == 1  →  sigmoid(logits).squeeze(-1)  shape (B,)
            - action_num  > 1  →  softmax(logits)[:, 1]        shape (B,)
              (binary: probability of the positive class)
        """
        was_training = self.training
        self.eval()

        try:
            logits = self.forward(
                deepfm_batch,
                seq_data,
                seq_lens,
                seq_time_buckets,
            )  # (B,) or (B, action_num) after squeeze in forward

            # Re-add the trailing dim if forward squeezed it away
            if self.action_num == 1:
                probs = torch.sigmoid(logits)          # (B,)
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1]  # (B,)
        finally:
            self.train(was_training)

        return probs

    # ─────────────────────────────────────────────────────────────
    # Save / Load
    # ─────────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return {
            "action_num": self.action_num,
            "gate_in_dim": self.gate_in_dim
        }

    def save(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config = {
            "fusion_config": self.get_config(),
            "deepfm_config": self.deepfm.get_config(),
            "behavior_encoder_config": self.behavior_encoder.get_config(),
        }

        with open(save_path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        torch.save(self.state_dict(), save_path / "model.pt")

        logger.info(f"GatedFusion saved to {save_path}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        map_location: str = "cpu",
    ) -> "GatedFusion":

        save_path = Path(save_dir)

        with open(save_path / "model_config.json") as f:
            config = json.load(f)

        deepfm = DeepFM(
            **config["deepfm_config"]
        )

        behavior_encoder = BehaviorEncoder(
            **config["behavior_encoder_config"]
        )

        model = cls(
            deepfm=deepfm,
            behavior_encoder=behavior_encoder,
            **config["fusion_config"],
        )

        state = torch.load(
            save_path / "model.pt",
            map_location=map_location,
        )

        model.load_state_dict(state)

        logger.info(f"GatedFusion loaded from {save_path}")

        return model