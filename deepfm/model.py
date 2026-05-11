"""DeepFM model with Compositional Embedding for item_id.

Architecture
------------
- item_id : quotient-remainder compositional embedding
    id // num_buckets  -> embedding table Q  (num_buckets  x embed_dim)
    id %  num_buckets  -> embedding table R  (num_buckets  x embed_dim)
    item_id_emb = Q[q] + R[r]          shape [B, embed_dim]

- int fields (user_int / item_int, each dim==1) :
    each field has its own nn.Embedding(vocab_size, embed_dim)
    field_emb                           shape [B, embed_dim]

- FM first-order :
    sum of Linear(embed_dim -> 1) over all fields   shape [B, 1]

- FM second-order :
    (sum_i v_i)^2 - sum_i v_i^2  / 2               shape [B, 1]
    computed over all field embeddings (including item_id)

- DNN :
    input = concat(all field embeddings, dense features)
                                        shape [B, n_fields*embed_dim + dense_dim]
    MLP → hidden layers → Linear → scalar

- Output :
    logit = FM_first + FM_second + DNN_out          shape [B]
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Compositional Embedding (quotient-remainder)
# ──────────────────────────────────────────────

class CompositionalEmbedding(nn.Module):
    """Quotient-remainder compositional embedding for item_id.

    Given an item id ``x``:
        q = x // num_buckets
        r = x %  num_buckets
        emb(x) = table_Q[q] + table_R[r]

    Both tables have shape ``[num_buckets, embed_dim]``, so total parameters
    are ``2 * num_buckets * embed_dim`` instead of ``vocab_size * embed_dim``.

    Args:
        num_items:   vocabulary size (max item_id + 1).
        embed_dim:   output embedding dimension.
        num_buckets: size of each sub-table.  Defaults to ``ceil(sqrt(num_items))``.
    """

    def __init__(
        self,
        num_items: int,
        embed_dim: int,
        num_buckets: Optional[int] = None,
    ) -> None:
        super().__init__()
        if num_buckets is None:
            num_buckets = math.ceil(math.sqrt(num_items))
        # r = item_id % num_buckets  →  range [0, num_buckets)
        # q = item_id // num_buckets →  range [0, ceil(num_items/num_buckets))
        # The two sub-tables have different row counts to prevent q from
        # indexing out of bounds when item_id approaches num_items-1.
        self.num_buckets = num_buckets
        self.q_size = math.ceil(num_items / num_buckets)
        self.embed_dim = embed_dim

        self.table_q = nn.Embedding(self.q_size, embed_dim)
        self.table_r = nn.Embedding(num_buckets, embed_dim)

        nn.init.xavier_uniform_(self.table_q.weight)
        nn.init.xavier_uniform_(self.table_r.weight)

    def forward(self, item_id: torch.Tensor) -> torch.Tensor:
        """
        Args:
            item_id: int64 tensor of shape ``[B]``, on the same device as the
                     embedding tables.
        Returns:
            Tensor of shape ``[B, embed_dim]``.
        """
        # Clamp to [0, num_items-1] so out-of-vocab ids never index OOB.
        num_items = self.q_size * self.num_buckets
        item_id = item_id.clamp(0, num_items - 1)
        q = item_id // self.num_buckets   # in [0, q_size)
        r = item_id % self.num_buckets    # in [0, num_buckets)
        return self.table_q(q) + self.table_r(r)


# ──────────────────────────────────────────────
# DeepFM
# ──────────────────────────────────────────────

class DeepFM(nn.Module):
    """DeepFM with quotient-remainder compositional embedding for item_id.

    Args:
        user_int_fields: list of ``(fid, vocab_size)`` for user int features.
        item_int_fields: list of ``(fid, vocab_size)`` for item int features
                         (excluding item_id itself).
        dense_dim:       total dimension of concatenated dense features fed
                         directly into the DNN (bypasses FM).
        num_items:       vocabulary size for item_id compositional embedding.
        embed_dim:       embedding dimension shared by all fields.
        dnn_hidden_dims: hidden layer sizes for the DNN tower.
        dnn_dropout:     dropout probability applied after each DNN hidden layer.
        num_buckets:     sub-table size for compositional embedding; defaults to
                         ``ceil(sqrt(num_items))``.
    """

    def __init__(
        self,
        user_int_fields: List[Tuple[int, int]],   # [(fid, vocab_size), ...]
        item_int_fields: List[Tuple[int, int]],   # [(fid, vocab_size), ...]
        dense_dim: int,
        num_items: int,
        embed_dim: int = 16,
        dnn_hidden_dims: List[int] = (256, 128, 64),
        dnn_dropout: float = 0.1,
        num_buckets: Optional[int] = None,
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.user_int_fids: List[int] = [fid for fid, _ in user_int_fields]
        self.item_int_fids: List[int] = [fid for fid, _ in item_int_fields]
        # dense_fids is fixed at construction time so forward always reads
        # exactly dense_dim scalars in a deterministic order, matching the
        # dnn_input_dim computed below.
        self.dense_fids: List[int] = []  # populated by caller via register_dense_fids

        # ── int field embeddings ──────────────────────────────────────────
        # Vocab layout:  indices [0, vs-1] are valid feature values (0 included).
        # Index vs is reserved as the padding token and is appended as an
        # extra row; its gradient is zeroed by setting padding_idx=vs.
        # For vs==0 (no vocab info) the table has 2 slots (0=value, 1=padding).
        self.user_int_embs = nn.ModuleDict({
            str(fid): nn.Embedding(max(vs, 1) + 1, embed_dim, padding_idx=max(vs, 1))
            for fid, vs in user_int_fields
        })
        self.item_int_embs = nn.ModuleDict({
            str(fid): nn.Embedding(max(vs, 1) + 1, embed_dim, padding_idx=max(vs, 1))
            for fid, vs in item_int_fields
        })
        for emb in list(self.user_int_embs.values()) + list(self.item_int_embs.values()):
            # Leave the padding row (last row) as zeros; init the rest.
            nn.init.xavier_uniform_(emb.weight[:-1])

        # ── item_id compositional embedding ──────────────────────────────
        self.item_id_emb = CompositionalEmbedding(num_items, embed_dim, num_buckets)

        # total number of embedding fields entering FM + DNN
        self.n_emb_fields = (
            len(user_int_fields) + len(item_int_fields) + 1  # +1 for item_id
        )

        # ── FM first-order weights (one linear per field) ─────────────────
        # Implemented as a Linear(embed_dim -> 1) applied to each field emb.
        self.fm_first_linears = nn.ModuleList([
            nn.Linear(embed_dim, 1, bias=False)
            for _ in range(self.n_emb_fields)
        ])

        # FM bias
        self.fm_bias = nn.Parameter(torch.zeros(1))

        # ── DNN ───────────────────────────────────────────────────────────
        dnn_input_dim = self.n_emb_fields * embed_dim + dense_dim
        layers: List[nn.Module] = []
        in_dim = dnn_input_dim
        for h in dnn_hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU(inplace=True))
            if dnn_dropout > 0:
                layers.append(nn.Dropout(dnn_dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, 1, bias=False))
        self.dnn = nn.Sequential(*layers)

    # ──────────────────────────────────────────
    # Forward
    # ──────────────────────────────────────────

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Args:
            batch: dict produced by ``PCVRParquetDataset._convert_batch``.
                   Required keys:
                     - ``item_id``                  [B], int64, on GPU
                     - ``user_int_feats_{fid}``     [B], int64  (for each fid)
                     - ``item_int_feats_{fid}``     [B], int64  (for each fid)
                     - ``user_dense_feats_{fid}``   [B], float32 (for each fid)

        Returns:
            logits tensor of shape ``[B]`` (raw, before sigmoid).
        """
        device = next(self.parameters()).device

        # ── collect field embeddings ──────────────────────────────────────
        # order: [user_int fields ..., item_int fields ..., item_id]
        field_embs: List[torch.Tensor] = []   # each [B, embed_dim]

        for fid in self.user_int_fids:
            x = batch[f'user_int_feats_{fid}'].to(device)
            field_embs.append(self.user_int_embs[str(fid)](x))

        for fid in self.item_int_fids:
            x = batch[f'item_int_feats_{fid}'].to(device)
            field_embs.append(self.item_int_embs[str(fid)](x))

        # item_id is already on GPU per dataset contract
        item_id = batch['item_id'].to(device)
        field_embs.append(self.item_id_emb(item_id))

        # stack → [B, n_fields, embed_dim]
        stacked = torch.stack(field_embs, dim=1)

        # ── FM first-order ─────────────────────────────────────────────
        # sum_i  w_i · v_i   where w_i is a Linear(embed_dim->1)
        first_order = sum(
            self.fm_first_linears[i](field_embs[i])   # [B, 1]
            for i in range(self.n_emb_fields)
        )  # [B, 1]

        # ── FM second-order ────────────────────────────────────────────
        # 0.5 * ( (Σ v_i)^2 - Σ v_i^2 )  summed over embed_dim → [B, 1]
        sum_emb = stacked.sum(dim=1)              # [B, embed_dim]
        sum_sq  = (stacked ** 2).sum(dim=1)       # [B, embed_dim]
        second_order = 0.5 * ((sum_emb ** 2) - sum_sq).sum(dim=1, keepdim=True)  # [B, 1]

        # ── DNN input ──────────────────────────────────────────────────
        # flat embeddings + dense features
        flat_emb = stacked.view(stacked.size(0), -1)   # [B, n_fields * embed_dim]

        dense_parts: List[torch.Tensor] = []
        for fid in self.dense_fids:
            dense_parts.append(
                batch[f'user_dense_feats_{fid}'].to(device).float().unsqueeze(1)
            )  # [B, 1]
        if dense_parts:
            dense_cat = torch.cat(dense_parts, dim=1)   # [B, dense_dim]
            dnn_input = torch.cat([flat_emb, dense_cat], dim=1)
        else:
            dnn_input = flat_emb

        dnn_out = self.dnn(dnn_input)   # [B, 1]

        # ── final logit ────────────────────────────────────────────────
        logit = (first_order + second_order + dnn_out).squeeze(1) + self.fm_bias
        return logit   # [B]

    # ──────────────────────────────────────────
    # Convenience
    # ──────────────────────────────────────────

    def register_dense_fids(self, fids: List[int]) -> None:
        """Register the ordered list of dense feature ids.

        Must be called once after construction (done automatically by
        ``train.py``).  The list must match the ``dense_dim`` passed to
        ``__init__`` (one id per scalar dense feature).
        """
        self.dense_fids = list(fids)

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Return label=1 probabilities, shape ``[B]``, in ``[0, 1]``.

        Temporarily switches to eval mode so BatchNorm uses running statistics
        rather than batch statistics, then restores the original training mode.
        """
        training = self.training
        self.eval()
        with torch.no_grad():
            probs = torch.sigmoid(self.forward(batch))
        self.train(training)
        return probs