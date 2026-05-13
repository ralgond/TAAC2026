"""
DeepFM model with Compositional Embedding for item_id.

UPDATED:
--------
Now returns:
    {
        "logits": [B]
        "hidden": [B, fusion_dim]
    }

for Gated Fusion.

Architecture
------------
FM:
    first-order
    second-order

DNN:
    embeddings + dense
        ↓
    hidden representation [B, fusion_dim]
        ↓
    scalar logit

Final:
    FM + DNN_logit
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Compositional Embedding
# ──────────────────────────────────────────────

class CompositionalEmbedding(nn.Module):

    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        num_buckets: Optional[int] = None,
    ) -> None:

        super().__init__()

        if num_buckets is None:
            num_buckets = math.ceil(
                math.sqrt(num_items)
            )

        self.num_buckets = num_buckets

        self.q_size = math.ceil(
            num_items / num_buckets
        )

        self.embed_dim = embed_dim

        self.table_q = nn.Embedding(
            self.q_size,
            embed_dim,
        )

        self.table_r = nn.Embedding(
            num_buckets,
            embed_dim,
        )

        nn.init.xavier_uniform_(
            self.table_q.weight
        )

        nn.init.xavier_uniform_(
            self.table_r.weight
        )

    def forward(
        self,
        item_id: torch.Tensor,
    ) -> torch.Tensor:

        num_items = (
            self.q_size
            * self.num_buckets
        )

        item_id = item_id.clamp(
            0,
            num_items - 1,
        )

        q = item_id // self.num_buckets
        r = item_id % self.num_buckets

        return (
            self.table_q(q)
            + self.table_r(r)
        )


# ──────────────────────────────────────────────
# DeepFM
# ──────────────────────────────────────────────

class DeepFM(nn.Module):

    # Keys saved to / loaded from model_config.json
    _CONFIG_KEYS = [
        "user_int_fields",
        "item_int_fields",
        "user_int_array_fields",
        "item_int_array_fields",
        "dense_dim",
        "num_items",
        "embed_dim",
        "dnn_hidden_dims",
        "fusion_dim",
        "dnn_dropout",
        "num_buckets"
    ]

    def __init__(
        self,
        user_int_fields: List[Tuple[int, int]],
        item_int_fields: List[Tuple[int, int]],
        user_int_array_fields: Optional[List[Tuple[int, int, int]]] = None,
        item_int_array_fields: Optional[List[Tuple[int, int, int]]] = None,
        dense_dim: int = 0,
        num_items: int = 10_000_000,
        embed_dim: int = 16,
        dnn_hidden_dims: List[int] = (512, 256, 128),
        fusion_dim: int = 128,
        dnn_dropout: float = 0.1,
        num_buckets: Optional[int] = None,
    ) -> None:

        super().__init__()

        self.user_int_fields = user_int_fields
        self.item_int_fields = item_int_fields
        self.user_int_array_fields = user_int_array_fields or []
        self.item_int_array_fields = item_int_array_fields or []
        self.dense_dim = dense_dim
        self.num_items = num_items
        self.embed_dim = embed_dim
        self.dnn_hidden_dims = dnn_hidden_dims
        self.fusion_dim = fusion_dim
        self.dnn_dropout = dnn_dropout
        self.num_buckets = num_buckets

        self.user_int_fids = [
            fid
            for fid, _
            in user_int_fields
        ]

        self.item_int_fids = [
            fid
            for fid, _
            in item_int_fields
        ]

        # (fid, vocab_size, dim) for array features
        self.user_int_array_fids = [
            fid
            for fid, _, _dim
            in self.user_int_array_fields
        ]

        self.item_int_array_fids = [
            fid
            for fid, _, _dim
            in self.item_int_array_fields
        ]

        self.dense_fids = []

        # ─────────────────────────
        # Embeddings
        # ─────────────────────────

        self.user_int_embs = nn.ModuleDict({
            str(fid): nn.Embedding(
                max(vs, 1) + 1,
                embed_dim,
                padding_idx=max(vs, 1),
            )
            for fid, vs
            in user_int_fields
        })

        self.item_int_embs = nn.ModuleDict({
            str(fid): nn.Embedding(
                max(vs, 1) + 1,
                embed_dim,
                padding_idx=max(vs, 1),
            )
            for fid, vs
            in item_int_fields
        })

        for emb in (
            list(self.user_int_embs.values())
            + list(self.item_int_embs.values())
        ):

            nn.init.xavier_uniform_(
                emb.weight[:-1]
            )

        # ─────────────────────────
        # EmbeddingBag for int array features
        # ─────────────────────────
        # Each array feature (dim>1) is pooled via mean EmbeddingBag so it
        # produces a single embed_dim vector regardless of array length.
        # num_embeddings = max(vs, 1) + 1 (last slot = padding, weight kept at 0).

        self.user_int_array_emb_bags = nn.ModuleDict({
            str(fid): nn.EmbeddingBag(
                max(vs, 1) + 1,
                embed_dim,
                mode='mean',
                padding_idx=max(vs, 1),
            )
            for fid, vs, _dim
            in self.user_int_array_fields
        })

        self.item_int_array_emb_bags = nn.ModuleDict({
            str(fid): nn.EmbeddingBag(
                max(vs, 1) + 1,
                embed_dim,
                mode='mean',
                padding_idx=max(vs, 1),
            )
            for fid, vs, _dim
            in self.item_int_array_fields
        })

        for bag in (
            list(self.user_int_array_emb_bags.values())
            + list(self.item_int_array_emb_bags.values())
        ):
            nn.init.xavier_uniform_(
                bag.weight[:-1]
            )

        # ─────────────────────────
        # item_id embedding
        # ─────────────────────────

        self.item_id_emb = (
            CompositionalEmbedding(
                num_items,
                embed_dim,
                num_buckets,
            )
        )

        self.n_emb_fields = (
            len(user_int_fields)
            + len(item_int_fields)
            + len(self.user_int_array_fields)
            + len(self.item_int_array_fields)
            + 1  # item_id
        )

        # ─────────────────────────
        # FM
        # ─────────────────────────

        self.fm_first_linears = (
            nn.ModuleList([
                nn.Linear(
                    embed_dim,
                    1,
                    bias=False,
                )
                for _ in range(
                    self.n_emb_fields
                )
            ])
        )

        self.fm_bias = nn.Parameter(
            torch.zeros(1)
        )

        # ─────────────────────────
        # DNN
        # ─────────────────────────

        dnn_input_dim = (
            self.n_emb_fields
            * embed_dim
            + dense_dim
        )

        layers = []

        in_dim = dnn_input_dim

        for h in dnn_hidden_dims:

            layers.append(
                nn.Linear(in_dim, h)
            )

            layers.append(
                nn.BatchNorm1d(h)
            )

            layers.append(
                nn.ReLU(inplace=True)
            )

            if dnn_dropout > 0:

                layers.append(
                    nn.Dropout(
                        dnn_dropout
                    )
                )

            in_dim = h

        # Final hidden representation
        layers.append(
            nn.Linear(
                in_dim,
                fusion_dim,
            )
        )

        layers.append(
            nn.BatchNorm1d(
                fusion_dim
            )
        )

        layers.append(
            nn.ReLU(inplace=True)
        )

        self.dnn = nn.Sequential(
            *layers
        )

        # scalar head
        self.dnn_head = nn.Linear(
            fusion_dim,
            1,
            bias=False,
        )

        # concat(dnn_hidden, fm_hidden)
        # fm_hidden = flat_emb: [B, n_emb_fields * embed_dim]
        self.hidden_proj = nn.Linear(
            fusion_dim + self.n_emb_fields * embed_dim,
            fusion_dim,
        )

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ):

        device = next(
            self.parameters()
        ).device

        # ─────────────────────────
        # Embeddings
        # ─────────────────────────

        field_embs = []

        for fid in self.user_int_fids:

            x = batch[
                f'user_int_feats_{fid}'
            ].to(device)

            field_embs.append(
                self.user_int_embs[
                    str(fid)
                ](x)
            )

        for fid in self.item_int_fids:

            x = batch[
                f'item_int_feats_{fid}'
            ].to(device)

            field_embs.append(
                self.item_int_embs[
                    str(fid)
                ](x)
            )

        item_id = batch[
            'item_id'
        ].to(device)

        field_embs.append(
            self.item_id_emb(item_id)
        )

        # ─────────────────────────
        # Array int embeddings (EmbeddingBag → mean pool)
        # ─────────────────────────

        for fid in self.user_int_array_fids:

            x = batch[
                f'user_int_array_feature_{fid}'
            ].to(device)  # [B, dim]

            field_embs.append(
                self.user_int_array_emb_bags[
                    str(fid)
                ](x)
            )  # [B, embed_dim]

        for fid in self.item_int_array_fids:

            x = batch[
                f'item_int_array_feature_{fid}'
            ].to(device)  # [B, dim]

            field_embs.append(
                self.item_int_array_emb_bags[
                    str(fid)
                ](x)
            )  # [B, embed_dim]

        stacked = torch.stack(
            field_embs,
            dim=1,
        )  # [B, F, D]

        # ─────────────────────────
        # FM First-order
        # ─────────────────────────

        first_order = sum(
            self.fm_first_linears[i](
                field_embs[i]
            )
            for i in range(
                self.n_emb_fields
            )
        )  # [B,1]

        # ─────────────────────────
        # FM Second-order
        # ─────────────────────────

        sum_emb = stacked.sum(
            dim=1
        )  # [B,D]

        sum_sq = (
            stacked ** 2
        ).sum(dim=1)

        second_order = 0.5 * (
            (
                sum_emb ** 2
            ) - sum_sq
        ).sum(
            dim=1,
            keepdim=True,
        )

        # ─────────────────────────
        # DNN input
        # ─────────────────────────

        flat_emb = stacked.view(
            stacked.size(0),
            -1,
        )

        dense_parts = []

        for fid in self.dense_fids:

            dense_parts.append(
                batch[
                    f'user_dense_feats_{fid}'
                ]
                .to(device)
                .float()
                .unsqueeze(1)
            )

        if dense_parts:

            dense_cat = torch.cat(
                dense_parts,
                dim=1,
            )

            dnn_input = torch.cat(
                [
                    flat_emb,
                    dense_cat,
                ],
                dim=1,
            )

        else:

            dnn_input = flat_emb

        # ─────────────────────────
        # DNN hidden
        # ─────────────────────────

        dnn_hidden = self.dnn(
            dnn_input
        )  # [B, fusion_dim]

        # FM interaction hidden: keep full field-level information
        # flat_emb is already [B, n_emb_fields * embed_dim]
        fm_hidden = flat_emb  # [B, n_emb_fields * embed_dim]

        # concat hidden
        fusion_hidden = torch.cat(
            [
                dnn_hidden,
                fm_hidden,
            ],
            dim=1,
        )

        hidden = self.hidden_proj(
            fusion_hidden
        )  # [B, fusion_dim]

        # ─────────────────────────
        # DNN scalar
        # ─────────────────────────

        dnn_logit = self.dnn_head(
            hidden
        )  # [B,1]

        # ─────────────────────────
        # Final logit
        # ─────────────────────────

        logit = (
            first_order
            + second_order
            + dnn_logit
        ).squeeze(1)

        logit = logit + self.fm_bias

        return {
            "logits": logit,      # [B]
            "hidden": hidden,     # [B, fusion_dim]
        }

    # ──────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────

    def register_dense_fids(
        self,
        fids: List[int],
    ) -> None:

        self.dense_fids = list(fids)

    @torch.no_grad()
    def predict(
        self,
        batch,
    ):

        training = self.training

        self.eval()

        out = self.forward(batch)

        probs = torch.sigmoid(
            out["logits"]
        )

        self.train(training)

        return probs

    def get_config(self) -> dict:
        return {k: getattr(self, k) for k in self._CONFIG_KEYS}