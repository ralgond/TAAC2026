"""
cvr_model.py  —  Multi-behavior MoE ranking model (Pattern A, single CVR tower).

Architecture (matches the diagram discussed):
  Input streams
  ─────────────
  Static (user + item + context features):
    • user_int scalar/array (fid 1000-1003 are time features living in user space)
    • item_int scalar/array
    • user_dense scalar/array
    Processed by StaticEncoder → 3 tokens: user_token, item_token, cross_token
    NOTE: item_id is intentionally DROPPED.

  Behavior sequences (one per domain; domain names come from schema.json["seq"]
  keys, e.g. "a", "b", "c", "d"):
    • Each domain has its own DomainEncoder (independent weights, Pattern A)
    • High-cardinality features (vocab > emb_skip_threshold=1_000_000) are skipped
      inside both seq and static embeddings (zero-vector placeholder used instead)
    • time_bucket embedding is shared across all domains

  Processing pipeline
  ───────────────────
  1. StaticEncoder  → h_static_tokens  (B, 3, D)
  2. Per-domain:
       seq_emb  → DomainEncoder  → domain_enc  (B, L_d, D)
  3. CrossDomainAttention (CLS attends to all domain_encs + static_tokens)
       → behavior_emb  (B, D)
  4. CVR tower (single output head):
       Linear → LayerNorm → SiLU → Dropout → Linear(1)
       → cvr_logit  (B, 1)

Design notes
────────────
• item_id removed: StaticEncoder no longer builds CompositionalEmbedding;
  outputs exactly 3 tokens instead of 4.
• emb_skip_threshold=1_000_000 (default): any vocab_size > 1M is skipped for
  both static scalar embeddings AND per-domain sequence embeddings.
• time features (fid 1000-1003) are treated as ordinary user_int scalar
  features by the dataset and arrive in static_ids like any other user feature.
• The MoE FFN lives in CrossLayer (same as original BehaviorEncoder).
  DomainEncoder always uses dense FFN.
• Mixed-precision (bfloat16) safe: no fp16-unsafe ops.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# RoPE helpers (unchanged from BehaviorEncoder)
# ─────────────────────────────────────────────────────────────────────────────

class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0), persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:, :seq_len, :].to(device),
            self.sin_cached[:, :seq_len, :].to(device),
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    L = x.shape[2]
    return x * cos[:, :L, :].unsqueeze(1) + _rotate_half(x) * sin[:, :L, :].unsqueeze(1)


# ─────────────────────────────────────────────────────────────────────────────
# Gated Multi-Head Attention with optional RoPE (unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class RoPEMHA(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = d_model // num_heads
        self.dropout   = dropout
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.W_g = nn.Linear(d_model, d_model)
        nn.init.zeros_(self.W_g.weight)
        nn.init.constant_(self.W_g.bias, 1.0)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        apply_rope_to_q: bool = True,
    ) -> torch.Tensor:
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        Q = self.W_q(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        if rope_cos is not None and rope_sin is not None:
            K = _apply_rope(K, rope_cos, rope_sin)
            if apply_rope_to_q:
                Q = _apply_rope(Q, rope_cos, rope_sin)
        sdpa_mask = None
        if key_padding_mask is not None:
            sdpa_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2).expand(
                B, self.num_heads, Lq, Lk
            )
        dp  = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, dropout_p=dp)
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        G = torch.sigmoid(self.W_g(query))
        return self.W_o(out * G)


# ─────────────────────────────────────────────────────────────────────────────
# MoEFFN: Sparse Mixture-of-Experts FFN (vectorized, unchanged)
# ─────────────────────────────────────────────────────────────────────────────

class MoEFFN(nn.Module):
    """Vectorized sparse MoE FFN. aux_loss stored per-forward for collection."""

    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        self.num_experts = num_experts
        self.top_k       = top_k
        hid = d_model * hidden_mult
        self.router = nn.Linear(d_model, num_experts, bias=False)
        self.w1     = nn.Parameter(torch.empty(num_experts, d_model, hid))
        self.b1     = nn.Parameter(torch.zeros(num_experts, hid))
        self.w2     = nn.Parameter(torch.empty(num_experts, hid, d_model))
        self.b2     = nn.Parameter(torch.zeros(num_experts, d_model))
        self.dropout = nn.Dropout(dropout)
        self.act     = nn.GELU()
        for i in range(num_experts):
            nn.init.xavier_uniform_(self.w1[i])
            nn.init.xavier_uniform_(self.w2[i])
        nn.init.xavier_uniform_(self.router.weight)
        self.aux_loss: torch.Tensor = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        N = B * L
        x_flat = x.reshape(N, D)
        logits = self.router(x_flat)
        probs  = torch.softmax(logits, dim=-1)
        topk_vals, topk_ids = probs.topk(self.top_k, dim=-1)
        gate = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
        with torch.no_grad():
            dispatch = torch.zeros_like(probs)
            dispatch.scatter_(1, topk_ids, 1.0)
            f_e = dispatch.mean(dim=0)
        p_e = probs.mean(dim=0)
        self.aux_loss = self.num_experts * (f_e * p_e).sum()
        h          = torch.einsum('nd,edh->neh', x_flat, self.w1) + self.b1
        h          = self.dropout(self.act(h))
        expert_out = torch.einsum('neh,ehd->ned', h, self.w2) + self.b2
        idx      = topk_ids.unsqueeze(-1).expand(N, self.top_k, D)
        selected = expert_out.gather(1, idx)
        out      = (gate.unsqueeze(-1) * selected).sum(dim=1)
        return out.view(B, L, D)


# ─────────────────────────────────────────────────────────────────────────────
# DomainEncoder: per-behavior Transformer stack (dense FFN, Pattern A)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int = 4,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = RoPEMHA(d_model, num_heads, dropout)
        hid = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hid, d_model), nn.Dropout(dropout),
        )

    def forward(
        self, x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        res = x
        x   = self.norm1(x)
        x   = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                        rope_cos=rope_cos, rope_sin=rope_sin, apply_rope_to_q=True)
        x   = res + x
        res = x
        x   = self.norm2(x)
        return res + self.ffn(x)


class DomainEncoder(nn.Module):
    """Independent Transformer encoder for one behavior domain (Pattern A)."""

    def __init__(self, d_model: int, num_heads: int, num_layers: int,
                 hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, hidden_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask,
                      rope_cos=rope_cos, rope_sin=rope_sin)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# StaticEncoder: 3 tokens only (NO item_id / CompositionalEmbedding)
# ─────────────────────────────────────────────────────────────────────────────

def _make_group_mlp(n_fields: int, emb_dim: int, d_model: int,
                    hidden_mult: int, dropout: float) -> nn.Sequential:
    in_dim = max(n_fields, 1) * emb_dim
    hid    = d_model * hidden_mult
    return nn.Sequential(
        nn.Linear(in_dim, hid), nn.GELU(), nn.Dropout(dropout),
        nn.Linear(hid, d_model), nn.Dropout(dropout),
    )


def _intra_fm(embs: List[torch.Tensor]) -> torch.Tensor:
    if len(embs) < 2:
        t = embs[0] if embs else None
        B, dim = (t.shape[0], t.shape[-1]) if t is not None else (1, 1)
        dev = t.device if t is not None else torch.device("cpu")
        return torch.zeros(B, dim, device=dev)
    stack  = torch.stack(embs, dim=1)
    sum_sq = stack.sum(dim=1) ** 2
    sq_sum = (stack ** 2).sum(dim=1)
    return 0.5 * (sum_sq - sq_sum)


def _cross_fm(user_embs: List[torch.Tensor], item_embs: List[torch.Tensor]) -> torch.Tensor:
    u = torch.stack(user_embs, dim=1).sum(dim=1)
    v = torch.stack(item_embs, dim=1).sum(dim=1)
    return u * v


class StaticEncoder(nn.Module):
    """
    Static feature encoder — outputs exactly 3 tokens (NO item_id token).

      user_token   (B, 1, D) — intra-user FM + user MLP
      item_token   (B, 1, D) — intra-item FM + item MLP
      cross_token  (B, 1, D) — user×item FM  + cross MLP

    Output shape: (B, 3, D).

    Feature types handled
    ─────────────────────
    Int scalar   (static_ids,         (B, n) int64)   → nn.Embedding
    Int array    (static_array_ids,   dict name→(B, d) int64) → nn.EmbeddingBag(mean)
    Dense scalar (static_dense,       (B, n) float32) → Linear(1 → emb_dim)
    Dense array  (static_dense_array, dict name→(B, d) float32) → Linear(d → emb_dim)

    emb_skip_threshold: features whose vocab_size > threshold are skipped
    (zero-vector placeholder used).  Default=1_000_000 per task requirement.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        array_field_configs: Optional[List[Tuple[str, int, int]]] = None,
        n_dense_scalar: int = 0,
        dense_array_configs: Optional[List[Tuple[str, int]]] = None,
        scalar_groups: Optional[List[str]] = None,
        array_groups: Optional[List[str]] = None,
        dense_scalar_groups: Optional[List[str]] = None,
        dense_array_groups: Optional[List[str]] = None,
        emb_dim: int = 64,
        d_model: int = 64,
        hidden_mult: int = 4,
        dropout: float = 0.01,
        emb_skip_threshold: int = 1_000_000,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model

        # ── Int scalar embeddings (skip if vocab > threshold) ─────────────
        emb_modules: List[Optional[nn.Embedding]] = []
        for vs in vocab_sizes:
            skip = (int(vs) <= 0 or
                    (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold))
            if skip:
                emb_modules.append(None)
            else:
                emb = nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0.0
                emb_modules.append(emb)
        self._scalar_embs = nn.ModuleList([e for e in emb_modules if e is not None])
        _scalar_emb_index: List[int] = []
        real_i = 0
        for e in emb_modules:
            if e is not None:
                _scalar_emb_index.append(real_i); real_i += 1
            else:
                _scalar_emb_index.append(-1)
        self._scalar_emb_index = _scalar_emb_index

        self._scalar_groups: List[str] = []
        for i, idx in enumerate(_scalar_emb_index):
            if idx == -1:
                continue
            g = (scalar_groups[i] if scalar_groups and i < len(scalar_groups) else 'user')
            self._scalar_groups.append(g)

        # ── Int array EmbeddingBag ────────────────────────────────────────
        self._int_array_names:  List[str] = []
        self._int_array_bags    = nn.ModuleList()
        self._int_array_groups: List[str] = []
        if array_field_configs:
            for fi, (name, vs, _max_dim) in enumerate(array_field_configs):
                self._int_array_names.append(name)
                bag = nn.EmbeddingBag(int(vs) + 1, emb_dim, mode='mean', padding_idx=0)
                nn.init.xavier_normal_(bag.weight.data)
                bag.weight.data[0, :] = 0.0
                self._int_array_bags.append(bag)
                g = (array_groups[fi] if array_groups and fi < len(array_groups) else 'user')
                self._int_array_groups.append(g)

        # ── Dense scalar: Linear(1 → emb_dim) ────────────────────────────
        self._dense_scalar_projs = nn.ModuleList(
            [nn.Linear(1, emb_dim) for _ in range(n_dense_scalar)]
        )
        self._dense_scalar_groups: List[str] = []
        for i in range(n_dense_scalar):
            g = (dense_scalar_groups[i]
                 if dense_scalar_groups and i < len(dense_scalar_groups) else 'user')
            self._dense_scalar_groups.append(g)

        # ── Dense array: Linear(dim → emb_dim) ───────────────────────────
        self._dense_array_names:  List[str] = []
        self._dense_array_projs  = nn.ModuleList()
        self._dense_array_groups: List[str] = []
        if dense_array_configs:
            for fi, (name, dim) in enumerate(dense_array_configs):
                self._dense_array_names.append(name)
                self._dense_array_projs.append(nn.Linear(dim, emb_dim))
                g = (dense_array_groups[fi]
                     if dense_array_groups and fi < len(dense_array_groups) else 'user')
                self._dense_array_groups.append(g)

        # ── Count fields per group ────────────────────────────────────────
        all_groups = (
            self._scalar_groups + self._int_array_groups
            + self._dense_scalar_groups + self._dense_array_groups
        )
        n_user = sum(1 for g in all_groups if g == 'user')
        n_item = sum(1 for g in all_groups if g == 'item')

        self._user_placeholder: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(1, emb_dim)) if n_user == 0 else None
        )
        self._item_placeholder: Optional[nn.Parameter] = (
            nn.Parameter(torch.zeros(1, emb_dim)) if n_item == 0 else None
        )
        n_user_eff = max(n_user, 1)
        n_item_eff = max(n_item, 1)

        # ── MLPs ──────────────────────────────────────────────────────────
        self.user_mlp  = _make_group_mlp(n_user_eff, emb_dim, d_model, hidden_mult, dropout)
        self.item_mlp  = _make_group_mlp(n_item_eff, emb_dim, d_model, hidden_mult, dropout)
        self.cross_mlp = _make_group_mlp(2, emb_dim, d_model, hidden_mult, dropout)

        # ── FM projections ────────────────────────────────────────────────
        self.user_fm_proj  = nn.Linear(emb_dim, d_model)
        self.item_fm_proj  = nn.Linear(emb_dim, d_model)
        self.cross_fm_proj = nn.Linear(emb_dim, d_model)

        # ── Output norms ──────────────────────────────────────────────────
        self.user_norm  = nn.LayerNorm(d_model)
        self.item_norm  = nn.LayerNorm(d_model)
        self.cross_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_int_scalar(self, static_ids, ref_B, ref_device) -> List[torch.Tensor]:
        out = []
        for i, real_idx in enumerate(self._scalar_emb_index):
            if real_idx == -1:
                continue
            if static_ids is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                out.append(self._scalar_embs[real_idx](static_ids[:, i].long()))
        return out

    def _embed_int_array(self, static_array_ids, ref_B, ref_device) -> List[torch.Tensor]:
        out = []
        for bag, name in zip(self._int_array_bags, self._int_array_names):
            ids = static_array_ids.get(name) if static_array_ids else None
            if ids is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                out.append(bag(ids.long()))
        return out

    def _embed_dense_scalar(self, static_dense, ref_B, ref_device) -> List[torch.Tensor]:
        out = []
        for i, proj in enumerate(self._dense_scalar_projs):
            if static_dense is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                out.append(proj(static_dense[:, i].unsqueeze(-1).float()))
        return out

    def _embed_dense_array(self, static_dense_array_ids, ref_B, ref_device) -> List[torch.Tensor]:
        out = []
        for proj, name in zip(self._dense_array_projs, self._dense_array_names):
            arr = static_dense_array_ids.get(name) if static_dense_array_ids else None
            if arr is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                out.append(proj(arr.float()))
        return out

    def _group_embs(self, static_ids, static_array_ids, static_dense,
                    static_dense_array_ids, ref_B, ref_device):
        all_embs, all_groups = [], []
        se  = self._embed_int_scalar(static_ids, ref_B, ref_device)
        all_embs.extend(se);  all_groups.extend(self._scalar_groups)
        ae  = self._embed_int_array(static_array_ids, ref_B, ref_device)
        all_embs.extend(ae);  all_groups.extend(self._int_array_groups)
        de  = self._embed_dense_scalar(static_dense, ref_B, ref_device)
        all_embs.extend(de);  all_groups.extend(self._dense_scalar_groups)
        dae = self._embed_dense_array(static_dense_array_ids, ref_B, ref_device)
        all_embs.extend(dae); all_groups.extend(self._dense_array_groups)

        user_embs = [e for e, g in zip(all_embs, all_groups) if g == 'user']
        item_embs = [e for e, g in zip(all_embs, all_groups) if g == 'item']
        if not user_embs:
            user_embs = [self._user_placeholder.expand(ref_B, -1)]  # type: ignore[union-attr]
        if not item_embs:
            item_embs = [self._item_placeholder.expand(ref_B, -1)]  # type: ignore[union-attr]
        return user_embs, item_embs

    def forward(
        self,
        static_ids: Optional[torch.Tensor] = None,
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,
        static_dense: Optional[torch.Tensor] = None,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Returns (B, 3, D) — [user_token, item_token, cross_token].
        NOTE: item_id token intentionally omitted (3 tokens, not 4).
        """
        ref_B, ref_device = None, None
        for t in [static_ids, static_dense]:
            if t is not None:
                ref_B, ref_device = t.shape[0], t.device
                break
        if ref_B is None:
            for d in [static_array_ids, static_dense_array_ids]:
                if d:
                    _v = next(iter(d.values()))
                    ref_B, ref_device = _v.shape[0], _v.device
                    break
        if ref_B is None:
            return torch.zeros(1, 3, self.d_model)

        user_embs, item_embs = self._group_embs(
            static_ids, static_array_ids, static_dense, static_dense_array_ids,
            ref_B, ref_device,
        )

        # user token
        u_fm  = self.user_fm_proj(self.dropout_layer(_intra_fm(user_embs)))
        u_mlp = self.user_mlp(torch.cat(user_embs, dim=-1))
        u_tok = self.user_norm(u_fm + u_mlp).unsqueeze(1)    # (B, 1, D)

        # item token
        i_fm  = self.item_fm_proj(self.dropout_layer(_intra_fm(item_embs)))
        i_mlp = self.item_mlp(torch.cat(item_embs, dim=-1))
        i_tok = self.item_norm(i_fm + i_mlp).unsqueeze(1)    # (B, 1, D)

        # cross token
        c_fm   = self.cross_fm_proj(self.dropout_layer(_cross_fm(user_embs, item_embs)))
        u_mean = torch.stack(user_embs, dim=1).mean(dim=1)
        v_mean = torch.stack(item_embs, dim=1).mean(dim=1)
        c_mlp  = self.cross_mlp(torch.cat([u_mean, v_mean], dim=-1))
        c_tok  = self.cross_norm(c_fm + c_mlp).unsqueeze(1)  # (B, 1, D)

        return torch.cat([u_tok, i_tok, c_tok], dim=1)        # (B, 3, D)


# ─────────────────────────────────────────────────────────────────────────────
# CrossLayer + CrossDomainAttention (MoE in CrossLayer FFN)
# ─────────────────────────────────────────────────────────────────────────────

class CrossLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, hidden_mult: int,
                 dropout: float, moe_num_experts: int = 0, moe_top_k: int = 2) -> None:
        super().__init__()
        self.norm_q   = nn.LayerNorm(d_model)
        self.norm_kv  = nn.LayerNorm(d_model)
        self.attn     = RoPEMHA(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        if moe_num_experts > 0:
            self.ffn: nn.Module = MoEFFN(
                d_model=d_model, num_experts=moe_num_experts,
                top_k=moe_top_k, hidden_mult=hidden_mult, dropout=dropout,
            )
        else:
            hid = d_model * hidden_mult
            self.ffn = nn.Sequential(
                nn.Linear(d_model, hid), nn.GELU(), nn.Dropout(dropout),
                nn.Linear(hid, d_model),
            )

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(
            self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv),
            key_padding_mask=kv_mask, apply_rope_to_q=False,
        )
        q = q + attn_out
        q = q + self.ffn(self.ffn_norm(q))
        return q


class CrossDomainAttention(nn.Module):
    """CLS token attends to all domain outputs + static tokens (3)."""

    def __init__(self, d_model: int, num_heads: int, num_domains: int,
                 dropout: float = 0.0, hidden_mult: int = 4,
                 num_cross_layers: int = 2,
                 moe_num_experts: int = 0, moe_top_k: int = 2) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.layers = nn.ModuleList([
            CrossLayer(d_model, num_heads, hidden_mult, dropout, moe_num_experts, moe_top_k)
            for _ in range(num_cross_layers)
        ])
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        domain_outputs: List[torch.Tensor],
        domain_masks:   List[torch.Tensor],
        static_tokens:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = domain_outputs[0].shape[0]
        kv_parts   = list(domain_outputs)
        mask_parts = list(domain_masks)
        if static_tokens is not None:
            kv_parts.append(static_tokens)
            n_s = static_tokens.shape[1]
            mask_parts.append(
                torch.zeros(B, n_s, dtype=torch.bool, device=static_tokens.device)
            )
        kv   = torch.cat(kv_parts,   dim=1)
        mask = torch.cat(mask_parts, dim=1)
        q = self.cls_token.expand(B, -1, -1)
        for layer in self.layers:
            q = layer(q, kv, mask)
        return self.out_norm(q).squeeze(1)   # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# CVRModel: the top-level model
# ─────────────────────────────────────────────────────────────────────────────

class CVRModel(nn.Module):
    """
    Multi-behavior CVR ranking model (Pattern A, single CVR tower).

    Key differences from BehaviorEncoder
    ─────────────────────────────────────
    • item_id feature and CompositionalEmbedding completely removed.
    • StaticEncoder outputs 3 tokens (not 4).
    • action_num is always 1 (CVR-only, binary cross-entropy loss).
    • emb_skip_threshold defaults to 1_000_000.
    • seq_vocab_sizes supports any number of behavior domains whose names
      come from schema.json["seq"] keys (e.g. "a", "b", "c", "d").
      Each domain has INDEPENDENT DomainEncoder weights (Pattern A).
    • time features (fid 1000-1003) arrive as ordinary user_int scalars
      via static_ids — no special handling needed here.
    """

    _CONFIG_KEYS = [
        "seq_vocab_sizes",
        "static_vocab_sizes",
        "static_array_configs",
        "n_dense_scalar",
        "dense_array_configs",
        "scalar_groups",
        "array_groups",
        "dense_scalar_groups",
        "dense_array_groups",
        "d_model",
        "emb_dim",
        "num_heads",
        "num_domain_layers",
        "hidden_mult",
        "dropout_rate",
        "num_time_buckets",
        "use_rope",
        "rope_base",
        "emb_skip_threshold",
        "seq_id_threshold",
        "max_seq_len",
        "num_cross_layers",
        "moe_num_experts",
        "moe_top_k",
        "moe_aux_loss_weight",
    ]

    def __init__(
        self,
        seq_vocab_sizes: Dict[str, List[int]],
        static_vocab_sizes: Optional[List[int]] = None,
        static_array_configs: Optional[List[Tuple[str, int, int]]] = None,
        n_dense_scalar: int = 0,
        dense_array_configs: Optional[List[Tuple[str, int]]] = None,
        scalar_groups: Optional[List[str]] = None,
        array_groups: Optional[List[str]] = None,
        dense_scalar_groups: Optional[List[str]] = None,
        dense_array_groups: Optional[List[str]] = None,
        d_model: int = 128,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_domain_layers: int = 2,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        num_time_buckets: int = 65,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 1_000_000,
        seq_id_threshold: int = 10000,
        max_seq_len: int = 2048,
        num_cross_layers: int = 2,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_weight: float = 0.01,
    ) -> None:
        super().__init__()

        # ── Store config for save/load ─────────────────────────────────────
        self.seq_vocab_sizes        = {k: list(v) for k, v in seq_vocab_sizes.items()}
        self.static_vocab_sizes     = list(static_vocab_sizes) if static_vocab_sizes else []
        self.static_array_configs   = [tuple(x) for x in static_array_configs] if static_array_configs else []  # type: ignore
        self.n_dense_scalar         = int(n_dense_scalar)
        self.dense_array_configs    = [tuple(x) for x in dense_array_configs] if dense_array_configs else []  # type: ignore
        self.scalar_groups          = list(scalar_groups)       if scalar_groups       else []
        self.array_groups           = list(array_groups)        if array_groups        else []
        self.dense_scalar_groups    = list(dense_scalar_groups) if dense_scalar_groups else []
        self.dense_array_groups     = list(dense_array_groups)  if dense_array_groups  else []
        self.d_model                = d_model
        self.emb_dim                = emb_dim
        self.num_heads              = num_heads
        self.num_domain_layers      = num_domain_layers
        self.hidden_mult            = hidden_mult
        self.dropout_rate           = dropout_rate
        self.num_time_buckets       = num_time_buckets
        self.use_rope               = use_rope
        self.rope_base              = rope_base
        self.emb_skip_threshold     = emb_skip_threshold
        self.seq_id_threshold       = seq_id_threshold
        self.max_seq_len            = max_seq_len
        self.num_cross_layers       = num_cross_layers
        self.moe_num_experts        = int(moe_num_experts)
        self.moe_top_k              = int(moe_top_k)
        self.moe_aux_loss_weight    = float(moe_aux_loss_weight)

        self.seq_domains = sorted(seq_vocab_sizes.keys())

        # ── Shared time-bucket embedding ──────────────────────────────────
        if num_time_buckets > 0:
            self.time_embedding: Optional[nn.Embedding] = nn.Embedding(
                num_time_buckets, emb_dim, padding_idx=0
            )
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0.0
        else:
            self.time_embedding = None

        # ── Per-domain token embeddings + projection (Pattern A: independent) ──
        self._seq_embs:      nn.ModuleDict         = nn.ModuleDict()
        self._seq_emb_index: Dict[str, List[int]]  = {}
        self._seq_is_id:     Dict[str, List[bool]] = {}
        self._seq_proj:      nn.ModuleDict         = nn.ModuleDict()
        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        for domain in self.seq_domains:
            voc = self.seq_vocab_sizes[domain]
            embs_raw, idx_map, is_id = self._make_seq_embs(voc)
            self._seq_embs[domain]      = nn.ModuleList([e for e in embs_raw if e is not None])
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain]     = is_id
            # input dim: n_fids * emb_dim + time_bucket_emb (if enabled)
            in_dim = len(voc) * emb_dim + (emb_dim if num_time_buckets > 0 else 0)
            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(in_dim, d_model), nn.LayerNorm(d_model),
            )

        # ── Per-domain encoders (INDEPENDENT weights — Pattern A) ─────────
        self.domain_encoders: nn.ModuleDict = nn.ModuleDict({
            domain: DomainEncoder(d_model, num_heads, num_domain_layers,
                                  hidden_mult, dropout_rate)
            for domain in self.seq_domains
        })

        # ── Static encoder (3 tokens, NO item_id) ────────────────────────
        _has_static = (
            bool(self.static_vocab_sizes)
            or bool(self.static_array_configs)
            or self.n_dense_scalar > 0
            or bool(self.dense_array_configs)
        )
        if _has_static:
            self.static_encoder: Optional[StaticEncoder] = StaticEncoder(
                vocab_sizes=self.static_vocab_sizes,
                array_field_configs=self.static_array_configs or None,
                n_dense_scalar=self.n_dense_scalar,
                dense_array_configs=self.dense_array_configs or None,
                scalar_groups=self.scalar_groups or None,
                array_groups=self.array_groups or None,
                dense_scalar_groups=self.dense_scalar_groups or None,
                dense_array_groups=self.dense_array_groups or None,
                emb_dim=emb_dim,
                d_model=d_model,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                emb_skip_threshold=emb_skip_threshold,
            )
        else:
            self.static_encoder = None

        # ── Cross-domain attention (MoE in CrossLayer when enabled) ───────
        self.cross_attn = CrossDomainAttention(
            d_model=d_model, num_heads=num_heads,
            num_domains=len(self.seq_domains),
            dropout=dropout_rate, hidden_mult=hidden_mult,
            num_cross_layers=num_cross_layers,
            moe_num_experts=moe_num_experts, moe_top_k=moe_top_k,
        )

        # ── RoPE ──────────────────────────────────────────────────────────
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb: Optional[RotaryEmbedding] = RotaryEmbedding(
                dim=head_dim, max_seq_len=max_seq_len, base=rope_base
            )
        else:
            self.rotary_emb = None

        self.emb_dropout = nn.Dropout(dropout_rate)

        # ── Single CVR tower ──────────────────────────────────────────────
        # action_num=1, binary cross-entropy loss in trainer
        self.cvr_tower = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, 1),   # single CVR logit
        )

        self._init_linear()

    # ── Internal helpers ───────────────────────────────────────────────────

    def _make_seq_embs(self, vocab_sizes: List[int]) -> Tuple[list, List[int], List[bool]]:
        """Build per-feature embeddings for one domain, respecting emb_skip_threshold."""
        embs_raw = []
        for vs in vocab_sizes:
            skip = int(vs) <= 0 or (
                self.emb_skip_threshold > 0 and int(vs) > self.emb_skip_threshold
            )
            if skip:
                embs_raw.append(None)
            else:
                emb = nn.Embedding(int(vs) + 1, self.emb_dim, padding_idx=0)
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0.0
                embs_raw.append(emb)
        idx_map, real_idx = [], 0
        for e in embs_raw:
            if e is not None:
                idx_map.append(real_idx); real_idx += 1
            else:
                idx_map.append(-1)
        is_id = [int(vs) > self.seq_id_threshold for vs in vocab_sizes]
        return embs_raw, idx_map, is_id

    def _embed_domain(
        self, seq: torch.Tensor, domain: str, time_buckets: torch.Tensor,
    ) -> torch.Tensor:
        """Embed one domain sequence: concat(embs, time_bucket_emb) → Linear → d_model."""
        B, n_fid, L = seq.shape
        if L > self.max_seq_len:
            seq          = seq[:, :, :self.max_seq_len]
            time_buckets = time_buckets[:, :self.max_seq_len]
            L            = self.max_seq_len

        parts: List[torch.Tensor] = []
        idx_map = self._seq_emb_index[domain]
        is_id   = self._seq_is_id[domain]
        embs    = self._seq_embs[domain]

        for fid_i in range(n_fid):
            real_idx = idx_map[fid_i]
            if real_idx == -1:
                # skipped (vocab > emb_skip_threshold): zero-vector placeholder
                parts.append(torch.zeros(B, L, self.emb_dim, device=seq.device))
            else:
                ids = seq[:, fid_i, :]
                e   = embs[real_idx](ids)
                # extra dropout for high-cardinality ID features
                if is_id[fid_i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                parts.append(e)

        if self.time_embedding is not None:
            parts.append(self.time_embedding(time_buckets))

        x = torch.cat(parts, dim=-1)          # (B, L, n_fid*emb_dim + emb_dim)
        return self._seq_proj[domain](x)       # (B, L, d_model)

    @staticmethod
    def _make_padding_mask(seq_lens: torch.Tensor, max_len: int) -> torch.Tensor:
        """True where position is padding (i.e. >= seq_len)."""
        idx = torch.arange(max_len, device=seq_lens.device).unsqueeze(0)
        return idx >= seq_lens.unsqueeze(1)

    def _init_linear(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── MoE aux loss ──────────────────────────────────────────────────────

    def get_moe_aux_loss(self) -> torch.Tensor:
        """Collect weighted MoE aux load-balancing loss from all CrossLayer FFNs."""
        total = torch.tensor(0.0)
        for m in self.modules():
            if isinstance(m, MoEFFN):
                total = total + m.aux_loss
        return self.moe_aux_loss_weight * total

    # ── Sparse / Dense param split (for separate LR in optimizer) ─────────

    def get_sparse_params(self) -> List[nn.Parameter]:
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        seq_data: Dict[str, torch.Tensor],        # {domain: (B, n_feats, L)}
        seq_lens: Dict[str, torch.Tensor],         # {domain: (B,)}
        seq_time_buckets: Dict[str, torch.Tensor], # {domain: (B, L)}
        static_ids: Optional[torch.Tensor] = None,                          # (B, n_static)
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,         # {name: (B, d)}
        static_dense: Optional[torch.Tensor] = None,                        # (B, n_dense)
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,   # {name: (B, d)}
    ) -> Dict[str, torch.Tensor]:
        """
        Returns dict with keys:
          'logits'  (B, 1)  — CVR logit (before sigmoid)
          'hidden'  (B, D)  — behavior embedding (for analysis / distillation)

        NOTE: item_id is NOT accepted. Remove it from call sites.
        """
        # ── Per-domain encoding ────────────────────────────────────────────
        domain_outputs: List[torch.Tensor] = []
        domain_masks:   List[torch.Tensor] = []

        for domain in self.seq_domains:
            seq  = seq_data[domain]
            L    = min(seq.shape[2], self.max_seq_len)
            mask = self._make_padding_mask(seq_lens[domain].clamp(max=L), L)
            tok  = self.emb_dropout(
                self._embed_domain(seq, domain, seq_time_buckets[domain])
            )
            rope_cos, rope_sin = None, None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(L, seq.device)
            enc = self.domain_encoders[domain](
                tok, key_padding_mask=mask, rope_cos=rope_cos, rope_sin=rope_sin
            )
            domain_outputs.append(enc)
            domain_masks.append(mask)

        # ── Static encoding → 3 tokens ────────────────────────────────────
        static_tokens: Optional[torch.Tensor] = None
        if self.static_encoder is not None:
            _has = any(x is not None for x in [
                static_ids, static_array_ids, static_dense, static_dense_array_ids
            ])
            if _has:
                static_tokens = self.static_encoder(
                    static_ids=static_ids,
                    static_array_ids=static_array_ids,
                    static_dense=static_dense,
                    static_dense_array_ids=static_dense_array_ids,
                )  # (B, 3, D)

        # ── Cross-domain attention → behavior embedding ───────────────────
        behavior_emb = self.cross_attn(
            domain_outputs, domain_masks, static_tokens=static_tokens
        )  # (B, D)

        # ── Single CVR tower ──────────────────────────────────────────────
        cvr_logit = self.cvr_tower(behavior_emb)   # (B, 1)

        return {"logits": cvr_logit, "hidden": behavior_emb}

    def encode(
        self,
        seq_data: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
        static_ids: Optional[torch.Tensor] = None,
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,
        static_dense: Optional[torch.Tensor] = None,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Returns behavior embedding (B, D) without CVR head."""
        return self.forward(
            seq_data, seq_lens, seq_time_buckets,
            static_ids=static_ids, static_array_ids=static_array_ids,
            static_dense=static_dense, static_dense_array_ids=static_dense_array_ids,
        )["hidden"]

    # ── Save / Load ────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return {k: getattr(self, k) for k in self._CONFIG_KEYS}

    def save(self, save_dir: str) -> None:
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "model_config.json", "w") as f:
            json.dump(self.get_config(), f, indent=2)
        torch.save(self.state_dict(), p / "model.pt")
        logger.info(f"CVRModel saved to {p}")

    @classmethod
    def load(cls, save_dir: str, map_location: str = "cpu") -> "CVRModel":
        p = Path(save_dir)
        with open(p / "model_config.json") as f:
            config = json.load(f)
        model = cls(**config)
        model.load_state_dict(torch.load(p / "model.pt", map_location=map_location))
        logger.info(f"CVRModel loaded from {p}")
        return model


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, L = 4, 20

    # Domain names come from schema.json["seq"] keys — typically short labels
    # like "a", "b", "c", "d" (matching behavior_encoder_model.py convention).
    # The first fid in each domain has vocab > emb_skip_threshold (1_000_000)
    # so its embedding is skipped and a zero-vector placeholder is used.
    seq_vocab_sizes = {
        "a": [2_000_000, 500, 200],   # fid-0 skipped (vocab > threshold)
        "b": [2_000_000, 300],         # fid-0 skipped
        "c": [2_000_000, 800],         # fid-0 skipped
        "d": [2_000_000, 150],         # fid-0 skipped
    }

    model = CVRModel(
        seq_vocab_sizes=seq_vocab_sizes,
        static_vocab_sizes=[50, 24, 7, 2, 4, 200],  # last two are time fids 1000-1003
        static_array_configs=[("static_array_30", 500, 10)],
        n_dense_scalar=3,
        dense_array_configs=[("static_dense_array_7", 8)],
        scalar_groups=['user', 'user', 'user', 'user', 'user', 'item'],
        array_groups=['user'],
        dense_scalar_groups=['user', 'user', 'item'],
        dense_array_groups=['user'],
        d_model=64, emb_dim=32, num_heads=4,
        num_domain_layers=2, num_cross_layers=2,
        moe_num_experts=4, moe_top_k=2,
        emb_skip_threshold=1_000_000,
    )

    # Build seq tensors: values for skipped fids (vocab > threshold) are
    # arbitrary (they get zeroed out); kept fids must stay within vocab range.
    def _make_seq(vocs):
        return torch.stack([torch.randint(0, max(vs, 1), (B, L)) for vs in vocs], dim=1)

    seq_data  = {d: _make_seq(v) for d, v in seq_vocab_sizes.items()}
    seq_lens  = {d: torch.randint(5, L + 1, (B,)) for d in seq_vocab_sizes}
    seq_tbuck = {d: torch.randint(0, 65, (B, L))  for d in seq_vocab_sizes}

    s_ids = torch.stack([
        torch.randint(0, 50,  (B,)),
        torch.randint(0, 24,  (B,)),   # hour        (fid 1000)
        torch.randint(0, 7,   (B,)),   # day_of_week (fid 1001)
        torch.randint(0, 2,   (B,)),   # is_weekend  (fid 1002)
        torch.randint(0, 4,   (B,)),   # time_slot   (fid 1003)
        torch.randint(0, 200, (B,)),
    ], dim=1)
    s_arr   = {"static_array_30":      torch.randint(0, 500, (B, 10))}
    s_dense = torch.randn(B, 3)
    s_darr  = {"static_dense_array_7": torch.randn(B, 8)}

    out = model(
        seq_data, seq_lens, seq_tbuck,
        static_ids=s_ids, static_array_ids=s_arr,
        static_dense=s_dense, static_dense_array_ids=s_darr,
    )
    print(f"logits={out['logits'].shape}  hidden={out['hidden'].shape}")
    assert out['logits'].shape == (B, 1), f"Expected (B,1), got {out['logits'].shape}"

    aux = model.get_moe_aux_loss()
    (out['logits'].mean() + aux).backward()
    print(f"aux_loss={aux.item():.6f}")

    # Verify StaticEncoder outputs exactly 3 tokens (no item_id token)
    st = model.static_encoder(
        static_ids=s_ids, static_array_ids=s_arr,
        static_dense=s_dense, static_dense_array_ids=s_darr,
    )
    assert st.shape == (B, 3, 64), f"Expected (B,3,64), got {st.shape}"
    print(f"static_tokens={st.shape}  ✓  (3 tokens, no item_id)")

    # Verify no-static path still works
    out2 = model(seq_data, seq_lens, seq_tbuck)
    assert out2['logits'].shape == (B, 1)
    print(f"no-static  logits={out2['logits'].shape}  ✓")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
    print("Smoke test passed ✓")