"""
BehaviorEncoder: A standalone behavior sequence encoder.

Architecture:
  1. Per-domain token embedding: concat(sideinfo_embs, time_bucket_emb) -> MLP -> d_model
  2. Per-domain DomainEncoder (Transformer self-attention layers)
  3. [NEW] StaticEncoder: static features -> FM cross + MLP -> (B, 1, D) token
  4. Cross-attention across all domains + static token: a learnable CLS token attends to
     all concatenated domain outputs + static token, producing the final behavior embedding.

Key design decisions (derived from PCVRHyFormer):
  - seq_vocab_sizes: {domain: [vocab_size_per_fid, ...]}
  - static_vocab_sizes: [vocab_size_fid0, vocab_size_fid1, ...]  (NEW)
  - time_bucket embedding: nn.Embedding(num_time_buckets=65, emb_dim=64)
  - Sequences are in reverse-chronological order (most recent first)
  - RoPE is supported and applied to sequence positions
  - emb_skip_threshold: skip Embedding creation for high-cardinality features
  - Mixed precision (bfloat16) compatible design (no fp16-unsafe ops)

StaticEncoder design (DeepFM-style):
  - Embedding layer per static feature
  - Second-order FM interaction: sum_i<j <e_i, e_j>  ->  (B, D)
  - MLP over concatenated embeddings                  ->  (B, D)
  - Fusion: FM output + MLP output -> proj -> (B, 1, D) token
  - No RoPE applied (static features have no sequence position)
"""

import json
import logging
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared with PCVRHyFormer (self-contained copies)
# ─────────────────────────────────────────────────────────────────────────────


class RotaryEmbedding(nn.Module):
    """Precomputes and caches RoPE cos/sin for up to max_seq_len positions."""

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0) -> None:
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
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
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, H, L, head_dim), cos/sin: (1, L, head_dim)."""
    L = x.shape[2]
    cos_ = cos[:, :L, :].unsqueeze(1)
    sin_ = sin[:, :L, :].unsqueeze(1)
    return x * cos_ + _rotate_half(x) * sin_


# ─────────────────────────────────────────────────────────────────────────────
# Multi-head self-attention with optional RoPE
# ─────────────────────────────────────────────────────────────────────────────


class RoPEMHA(nn.Module):
    """Multi-head attention with optional Rotary Position Embedding."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        # Gating (same as PCVRHyFormer)
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
            # True = padding → False = attend; SDPA expects True = attend
            sdpa_mask = (~key_padding_mask).unsqueeze(1).unsqueeze(2)  # (B,1,1,Lk)
            sdpa_mask = sdpa_mask.expand(B, self.num_heads, Lq, Lk)

        dp = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=sdpa_mask, dropout_p=dp)
        out = torch.nan_to_num(out, nan=0.0)
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)

        G = torch.sigmoid(self.W_g(query))
        out = out * G
        return self.W_o(out)


# ─────────────────────────────────────────────────────────────────────────────
# DomainEncoder: Transformer encoder stack for one behavior domain
# ─────────────────────────────────────────────────────────────────────────────


class TransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with RoPE support."""

    def __init__(self, d_model: int, num_heads: int, hidden_mult: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = RoPEMHA(d_model, num_heads, dropout)
        hid = d_model * hidden_mult
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = self.attn(x, x, x, key_padding_mask=key_padding_mask,
                      rope_cos=rope_cos, rope_sin=rope_sin, apply_rope_to_q=True)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        return residual + x


class DomainEncoder(nn.Module):
    """Stack of Transformer encoder layers for one sequence domain."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_layers: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, hidden_mult, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask,
                      rope_cos=rope_cos, rope_sin=rope_sin)
        return self.norm(x)


# ─────────────────────────────────────────────────────────────────────────────
# [NEW] StaticEncoder: DeepFM-style static feature cross encoder
# ─────────────────────────────────────────────────────────────────────────────


class StaticEncoder(nn.Module):
    """
    DeepFM-style encoder for static features, supporting four field types:

    **Int scalar**  (``static_ids``, shape ``(B, n_int_scalar)`` int64):
        Each field is a single integer id → Embedding → ``(B, emb_dim)``.

    **Int array**   (``static_array_ids``, ``Dict[name, (B, dim)]`` int64):
        Multi-value integer ids padded to ``dim``.  Zero = padding.
        Embedding + mask mean-pool → ``(B, emb_dim)``.

    **Dense scalar** (``static_dense``, shape ``(B, n_dense)`` float32):
        Each field is a single float value → Linear(1, emb_dim) per field
        → ``(B, emb_dim)``.

    **Dense array**  (``static_dense_array_ids``, ``Dict[name, (B, dim)]`` float32):
        Multi-value float vector padded to ``dim``.  Zero = padding (mask by
        any nonzero position in the row).
        Linear(dim, emb_dim) → ``(B, emb_dim)``.

    All four branch outputs are concatenated into ``field_embs`` and fed
    jointly into FM + MLP:
      FM:  0.5*(‖Σe_i‖²−Σ‖e_i‖²) → proj → (B, d_model)
      MLP: concat(all field_embs)  → MLP  → (B, d_model)
      Fusion: LayerNorm(fm_out + mlp_out) → (B, 1, d_model)

    The output token is appended to CrossDomainAttention KV (no RoPE).

    Args:
        vocab_sizes:              ``[vocab_size, ...]`` for int scalar fields.
                                  Use 0/negative → field skipped (zero emb).
        array_field_configs:      ``[(name, vocab_size, max_dim), ...]`` for
                                  int array fields.
        n_dense_scalar:           Number of dense scalar fields (int ≥ 0).
        dense_array_configs:      ``[(name, dim), ...]`` for dense array fields.
                                  ``name`` matches keys in ``static_dense_array_ids``.
        emb_dim:                  Embedding/projection dimension per field.
        d_model:                  Output token dimension.
        hidden_mult:              MLP hidden expansion factor.
        dropout:                  Dropout probability.
        emb_skip_threshold:       Skip int scalar Embedding for vocab > threshold.
    """

    def __init__(
        self,
        vocab_sizes: List[int],
        array_field_configs: Optional[List[Tuple[str, int, int]]] = None,
        n_dense_scalar: int = 0,
        dense_array_configs: Optional[List[Tuple[str, int]]] = None,
        emb_dim: int = 64,
        d_model: int = 64,
        hidden_mult: int = 4,
        dropout: float = 0.01,
        emb_skip_threshold: int = 0,
    ) -> None:
        super().__init__()
        self.emb_dim = emb_dim
        self.d_model = d_model

        # ── Int scalar: Embedding per field ───────────────────────────────
        emb_modules: List[Optional[nn.Embedding]] = []
        for vs in vocab_sizes:
            skip = int(vs) <= 0 or (emb_skip_threshold > 0 and int(vs) > emb_skip_threshold)
            if skip:
                emb_modules.append(None)
            else:
                emb = nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0.0
                emb_modules.append(emb)

        self._scalar_embs = nn.ModuleList([e for e in emb_modules if e is not None])
        self._scalar_emb_index: List[int] = []
        real_idx = 0
        for e in emb_modules:
            if e is not None:
                self._scalar_emb_index.append(real_idx)
                real_idx += 1
            else:
                self._scalar_emb_index.append(-1)
        num_valid_int_scalar = sum(1 for e in emb_modules if e is not None)

        # ── Int array: Embedding + mask mean-pool per field ───────────────
        self._int_array_names: List[str] = []
        self._int_array_embs = nn.ModuleList()
        if array_field_configs:
            for name, vs, _max_dim in array_field_configs:
                self._int_array_names.append(name)
                emb = nn.Embedding(int(vs) + 1, emb_dim, padding_idx=0)
                nn.init.xavier_normal_(emb.weight.data)
                emb.weight.data[0, :] = 0.0
                self._int_array_embs.append(emb)
        num_int_array = len(self._int_array_names)

        # ── Dense scalar: Linear(1 → emb_dim) per field ───────────────────
        self.n_dense_scalar = n_dense_scalar
        # One Linear per dense scalar field (shared weight doesn't make sense
        # across semantically different features).
        self._dense_scalar_projs = nn.ModuleList(
            [nn.Linear(1, emb_dim) for _ in range(n_dense_scalar)]
        )

        # ── Dense array: Linear(dim → emb_dim) per field ─────────────────
        self._dense_array_names: List[str] = []
        self._dense_array_projs = nn.ModuleList()
        if dense_array_configs:
            for name, dim in dense_array_configs:
                self._dense_array_names.append(name)
                self._dense_array_projs.append(nn.Linear(dim, emb_dim))
        num_dense_array = len(self._dense_array_names)

        # ── Total field count (all four branches) ─────────────────────────
        num_total = num_valid_int_scalar + num_int_array + n_dense_scalar + num_dense_array

        # ── MLP branch ────────────────────────────────────────────────────
        mlp_in_dim = num_total * emb_dim
        mlp_hidden  = d_model * hidden_mult
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, d_model),
            nn.Dropout(dropout),
        )

        # ── FM proj: emb_dim → d_model ────────────────────────────────────
        self.fm_proj     = nn.Linear(emb_dim, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        self._num_total = num_total
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Branch embed helpers ───────────────────────────────────────────────

    def _embed_int_scalar(
        self,
        static_ids: Optional[torch.Tensor],
        ref_B: int,
        ref_device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Returns exactly ``num_valid_int_scalar`` tensors of shape ``(B, emb_dim)``.
        Missing ``static_ids`` → zero vectors (concat dim stays fixed).
        """
        out: List[torch.Tensor] = []
        for i, real_idx in enumerate(self._scalar_emb_index):
            if real_idx == -1:
                continue
            if static_ids is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                out.append(self._scalar_embs[real_idx](static_ids[:, i].long()))
        return out

    def _embed_int_array(
        self,
        static_array_ids: Optional[Dict[str, torch.Tensor]],
        ref_B: int,
        ref_device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Returns exactly ``num_int_array`` tensors of shape ``(B, emb_dim)``.
        Missing field → zero vector.
        """
        out: List[torch.Tensor] = []
        for emb_mod, name in zip(self._int_array_embs, self._int_array_names):
            ids = static_array_ids.get(name) if static_array_ids else None
            if ids is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
                continue
            ids = ids.long()                              # (B, dim)
            emb = emb_mod(ids)                            # (B, dim, emb_dim)
            mask = (ids > 0).float().unsqueeze(-1)        # (B, dim, 1)
            valid_cnt = mask.sum(dim=1).clamp(min=1.0)   # (B, 1)
            out.append((emb * mask).sum(dim=1) / valid_cnt)  # (B, emb_dim)
        return out

    def _embed_dense_scalar(
        self,
        static_dense: Optional[torch.Tensor],
        ref_B: int,
        ref_device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Returns exactly ``n_dense_scalar`` tensors of shape ``(B, emb_dim)``.
        Each field: Linear(1 → emb_dim) applied to the scalar value.
        Missing ``static_dense`` → zero vectors.
        """
        out: List[torch.Tensor] = []
        for i, proj in enumerate(self._dense_scalar_projs):
            if static_dense is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
            else:
                val = static_dense[:, i].unsqueeze(-1).float()  # (B, 1)
                out.append(proj(val))                            # (B, emb_dim)
        return out

    def _embed_dense_array(
        self,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]],
        ref_B: int,
        ref_device: torch.device,
    ) -> List[torch.Tensor]:
        """
        Returns exactly ``num_dense_array`` tensors of shape ``(B, emb_dim)``.
        Each field: Linear(dim → emb_dim) applied to the float vector.
        Missing field → zero vector.
        """
        out: List[torch.Tensor] = []
        for proj, name in zip(self._dense_array_projs, self._dense_array_names):
            arr = static_dense_array_ids.get(name) if static_dense_array_ids else None
            if arr is None:
                out.append(torch.zeros(ref_B, self.emb_dim, device=ref_device))
                continue
            out.append(proj(arr.float()))   # (B, emb_dim)
        return out

    # ── FM interaction ─────────────────────────────────────────────────────

    def _fm_interaction(self, field_embs: List[torch.Tensor]) -> torch.Tensor:
        """Squared-sum FM trick → (B, emb_dim). Returns zeros if < 2 fields."""
        if len(field_embs) < 2:
            B   = field_embs[0].shape[0] if field_embs else 1
            dev = field_embs[0].device   if field_embs else torch.device("cpu")
            return torch.zeros(B, self.emb_dim, device=dev)
        stack   = torch.stack(field_embs, dim=1)  # (B, n, emb_dim)
        sum_emb = stack.sum(dim=1)                # (B, emb_dim)
        sum_sq  = (stack ** 2).sum(dim=1)         # (B, emb_dim)
        return 0.5 * (sum_emb ** 2 - sum_sq)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        static_ids: Optional[torch.Tensor] = None,
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,
        static_dense: Optional[torch.Tensor] = None,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Args:
            static_ids:              (B, n_int_scalar) int64.
            static_array_ids:        {name: (B, max_dim) int64}.
            static_dense:            (B, n_dense_scalar) float32.
            static_dense_array_ids:  {name: (B, dim) float32}.

        Returns:
            token: (B, 1, d_model)
        """
        # Resolve ref_B / ref_device from whichever input is available
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
            return torch.zeros(1, 1, self.d_model)

        field_embs: List[torch.Tensor] = []

        # Each branch is always called so that concat dim = num_total * emb_dim
        # (missing inputs → zero vectors inside each helper).
        if self._scalar_emb_index:
            field_embs.extend(self._embed_int_scalar(static_ids, ref_B, ref_device))

        if self._int_array_names:
            field_embs.extend(self._embed_int_array(static_array_ids, ref_B, ref_device))

        if self.n_dense_scalar > 0:
            field_embs.extend(self._embed_dense_scalar(static_dense, ref_B, ref_device))

        if self._dense_array_names:
            field_embs.extend(self._embed_dense_array(static_dense_array_ids, ref_B, ref_device))

        if not field_embs:
            return torch.zeros(ref_B, 1, self.d_model, device=ref_device)

        # FM branch
        fm_out = self._fm_interaction(field_embs)          # (B, emb_dim)
        fm_out = self.dropout_layer(fm_out)
        fm_out = self.fm_proj(fm_out)                      # (B, d_model)

        # MLP branch
        concat  = torch.cat(field_embs, dim=-1)            # (B, num_total * emb_dim)
        mlp_out = self.mlp(concat)                         # (B, d_model)

        # Fusion
        fused = self.fusion_norm(fm_out + mlp_out)         # (B, d_model)
        return fused.unsqueeze(1)                          # (B, 1, d_model)


# ─────────────────────────────────────────────────────────────────────────────
# Cross-domain attention: CLS token attends to all domain outputs
# ─────────────────────────────────────────────────────────────────────────────


class CrossLayer(nn.Module):
    """
    One cross-attention + FFN layer (Pre-LN).

    The CLS query attends to the concatenated domain KV tokens.
    RoPE is intentionally skipped on the query side since DomainEncoder
    has already baked positional information into the KV tokens.
    The static token has no position either, so no RoPE is needed there.
    """

    def __init__(self, d_model: int, num_heads: int, hidden_mult: int, dropout: float) -> None:
        super().__init__()
        self.norm_q  = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)
        self.attn    = RoPEMHA(d_model, num_heads, dropout)

        hid = d_model * hidden_mult
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, d_model),
        )

    def forward(
        self,
        q: torch.Tensor,           # (B, 1, D)  CLS token
        kv: torch.Tensor,          # (B, S, D)  concatenated domain tokens (+ static token)
        kv_mask: torch.Tensor,     # (B, S)     True = padding
    ) -> torch.Tensor:             # (B, 1, D)
        # Cross-attention (Pre-LN, no RoPE on query)
        attn_out = self.attn(
            self.norm_q(q),
            self.norm_kv(kv),
            self.norm_kv(kv),
            key_padding_mask=kv_mask,
            apply_rope_to_q=False,
        )
        q = q + attn_out

        # FFN (Pre-LN)
        q = q + self.ffn(self.ffn_norm(q))
        return q


class CrossDomainAttention(nn.Module):
    """
    A learnable CLS token attends to all concatenated domain outputs (+ static token)
    through ``num_cross_layers`` stacked CrossLayer blocks.

    Output: final CLS token representation -> (B, D) behavior embedding.

    Args:
        d_model:          Token dimension.
        num_heads:        Attention heads.
        num_domains:      Number of behavior domains (kept for API compatibility).
        dropout:          Dropout probability.
        hidden_mult:      FFN expansion factor (hidden = d_model * hidden_mult).
        num_cross_layers: Number of stacked cross-attention layers (default 2).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_domains: int,
        dropout: float = 0.0,
        hidden_mult: int = 4,
        num_cross_layers: int = 2,
    ) -> None:
        super().__init__()

        self.num_cross_layers = num_cross_layers

        # One global CLS token as the persistent query seed
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Stack of cross-attention layers
        self.layers = nn.ModuleList([
            CrossLayer(d_model, num_heads, hidden_mult, dropout)
            for _ in range(num_cross_layers)
        ])

        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        domain_outputs: List[torch.Tensor],    # list of (B, L_i, D) -- behavior domains
        domain_masks: List[torch.Tensor],      # list of (B, L_i) bool -- True = padding
        static_token: Optional[torch.Tensor] = None,  # [NEW] (B, 1, D) or None
    ) -> torch.Tensor:
        """
        Args:
            domain_outputs: list of (B, L_i, D) -- per-domain encoded tokens
            domain_masks:   list of (B, L_i) bool -- True = padding position
            static_token:   (B, 1, D) from StaticEncoder; None if not used
        Returns:
            (B, D) global behavior embedding
        """
        B = domain_outputs[0].shape[0]

        kv_parts   = list(domain_outputs)
        mask_parts = list(domain_masks)

        # [NEW] Append static token as an extra KV token (never masked)
        if static_token is not None:
            kv_parts.append(static_token)                         # (B, 1, D)
            static_mask = torch.zeros(B, 1, dtype=torch.bool,
                                      device=static_token.device) # never padded
            mask_parts.append(static_mask)

        # Concatenate all KV tokens -> (B, S, D)  where S = sum(L_i) [+ 1 for static]
        kv            = torch.cat(kv_parts,   dim=1)
        combined_mask = torch.cat(mask_parts, dim=1)

        # Expand CLS token for the batch
        q = self.cls_token.expand(B, -1, -1)               # (B, 1, D)

        # Pass through N cross-attention layers
        for layer in self.layers:
            q = layer(q, kv, combined_mask)                 # (B, 1, D)

        return self.out_norm(q).squeeze(1)                  # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Main BehaviorEncoder
# ─────────────────────────────────────────────────────────────────────────────


class BehaviorEncoder(nn.Module):
    """
    Standalone behavior sequence encoder with optional StaticEncoder.

    For each behavior domain:
      tokens = concat(emb_fid0, ..., emb_time_bucket) -> proj MLP -> (B, L, D)
      domain_out = DomainEncoder(tokens)                             -> (B, L, D)

    For static features (four field types):
      static_token = StaticEncoder(
          static_ids, static_array_ids,
          static_dense, static_dense_array_ids,
      )  -> (B, 1, D)

    Final:
      behavior_emb = CrossDomainAttention(domain_outs [+ static_token]) -> (B, D)
      logits       = classifier(behavior_emb)                           -> (B, action_num)

    Args:
        seq_vocab_sizes:        {domain: [vocab_size_fid0, ...]}
        static_vocab_sizes:     [vocab_size, ...] for int scalar static fields.
        static_array_configs:   [(name, vocab_size, max_dim), ...] for int array fields.
        n_dense_scalar:         Number of dense scalar static fields (float32).
        dense_array_configs:    [(name, dim), ...] for dense array static fields.
        d_model / emb_dim / num_heads / ...: standard transformer hyper-params.
    """

    _CONFIG_KEYS = [
        "seq_vocab_sizes",
        "static_vocab_sizes",     # int scalar static fields
        "static_array_configs",   # int array static fields
        "n_dense_scalar",         # dense scalar static fields count
        "dense_array_configs",    # dense array static fields
        "d_model",
        "emb_dim",
        "num_heads",
        "num_domain_layers",
        "hidden_mult",
        "dropout_rate",
        "num_time_buckets",
        "action_num",
        "use_rope",
        "rope_base",
        "emb_skip_threshold",
        "seq_id_threshold",
        "max_seq_len",
        "num_cross_layers",
    ]

    def __init__(
        self,
        seq_vocab_sizes: Dict[str, List[int]],
        static_vocab_sizes: Optional[List[int]] = None,
        static_array_configs: Optional[List[Tuple[str, int, int]]] = None,
        n_dense_scalar: int = 0,
        dense_array_configs: Optional[List[Tuple[str, int]]] = None,
        d_model: int = 64,
        emb_dim: int = 64,
        num_heads: int = 4,
        num_domain_layers: int = 2,
        hidden_mult: int = 4,
        dropout_rate: float = 0.01,
        num_time_buckets: int = 65,
        action_num: int = 1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        max_seq_len: int = 2048,
        num_cross_layers: int = 2,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.emb_dim = emb_dim
        self.action_num = action_num
        self.num_time_buckets = num_time_buckets
        self.use_rope = use_rope
        self.emb_skip_threshold = emb_skip_threshold
        self.seq_id_threshold = seq_id_threshold
        self.seq_domains = sorted(seq_vocab_sizes.keys())
        self.seq_vocab_sizes = {k: list(v) for k, v in seq_vocab_sizes.items()}
        self.static_vocab_sizes = list(static_vocab_sizes) if static_vocab_sizes else []
        self.static_array_configs: List[Tuple[str, int, int]] = (
            [tuple(x) for x in static_array_configs] if static_array_configs else []
        )
        self.n_dense_scalar = int(n_dense_scalar)
        self.dense_array_configs: List[Tuple[str, int]] = (
            [tuple(x) for x in dense_array_configs] if dense_array_configs else []
        )
        self.num_domain_layers = num_domain_layers
        self.num_heads = num_heads
        self.hidden_mult = hidden_mult
        self.dropout_rate = dropout_rate
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
        self.num_cross_layers = num_cross_layers

        # ── Time bucket embedding (shared across all domains) ──────────────
        if num_time_buckets > 0:
            self.time_embedding = nn.Embedding(num_time_buckets, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0.0
        else:
            self.time_embedding = None

        # ── Per-domain feature embeddings + projection ─────────────────────
        self._seq_embs: nn.ModuleDict = nn.ModuleDict()
        self._seq_emb_index: Dict[str, List[int]] = {}
        self._seq_is_id: Dict[str, List[bool]] = {}
        self._seq_proj: nn.ModuleDict = nn.ModuleDict()

        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        for domain in self.seq_domains:
            vocab_sizes = self.seq_vocab_sizes[domain]
            embs_raw, idx_map, is_id = self._make_seq_embs(vocab_sizes)
            self._seq_embs[domain] = nn.ModuleList([e for e in embs_raw if e is not None])
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id

            in_dim = len(vocab_sizes) * emb_dim
            if num_time_buckets > 0:
                in_dim += emb_dim

            self._seq_proj[domain] = nn.Sequential(
                nn.Linear(in_dim, d_model),
                nn.LayerNorm(d_model),
            )

        # ── Per-domain Transformer encoders ───────────────────────────────
        self.domain_encoders: nn.ModuleDict = nn.ModuleDict({
            domain: DomainEncoder(
                d_model=d_model,
                num_heads=num_heads,
                num_layers=num_domain_layers,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
            )
            for domain in self.seq_domains
        })

        # ── Static encoder (all four field types) ─────────────────────────
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
                emb_dim=emb_dim,
                d_model=d_model,
                hidden_mult=hidden_mult,
                dropout=dropout_rate,
                emb_skip_threshold=emb_skip_threshold,
            )
        else:
            self.static_encoder = None

        # ── Cross-domain attention ─────────────────────────────────────────
        self.cross_attn = CrossDomainAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_domains=len(self.seq_domains),
            dropout=dropout_rate,
            hidden_mult=hidden_mult,
            num_cross_layers=num_cross_layers,
        )

        # ── RoPE ──────────────────────────────────────────────────────────
        if use_rope:
            head_dim = d_model // num_heads
            self.rotary_emb = RotaryEmbedding(dim=head_dim, max_seq_len=max_seq_len, base=rope_base)
        else:
            self.rotary_emb = None

        # ── Dropout ───────────────────────────────────────────────────────
        self.emb_dropout = nn.Dropout(dropout_rate)

        # ── Classifier ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(d_model, action_num),
        )

        self._init_linear()

    # ── Internals ──────────────────────────────────────────────────────────

    def _make_seq_embs(
        self, vocab_sizes: List[int]
    ) -> Tuple[list, List[int], List[bool]]:
        """Create embedding tables for a domain, respecting emb_skip_threshold."""
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
                idx_map.append(real_idx)
                real_idx += 1
            else:
                idx_map.append(-1)

        is_id = [int(vs) > self.seq_id_threshold for vs in vocab_sizes]
        return embs_raw, idx_map, is_id

    def _init_linear(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _embed_domain(
        self,
        seq: torch.Tensor,
        domain: str,
        time_bucket_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Embeds one domain.

        seq layout: (B, num_fids, L)
          - axis 1 indexes feature-id (fid)
          - axis 2 indexes sequence position (most-recent first)

        Returns: (B, L, D)
        """
        B, num_fids, L = seq.shape
        emb_list: List[torch.Tensor] = []

        embs = self._seq_embs[domain]
        idx_map = self._seq_emb_index[domain]
        is_id = self._seq_is_id[domain]

        for i in range(num_fids):
            real_idx = idx_map[i] if i < len(idx_map) else -1
            if real_idx == -1:
                emb_list.append(seq.new_zeros(B, L, self.emb_dim, dtype=torch.float))
            else:
                e = embs[real_idx](seq[:, i, :].long())  # (B, L, emb_dim)
                if is_id[i] and self.training:
                    e = self.seq_id_emb_dropout(e)
                emb_list.append(e)

        if self.time_embedding is not None:
            emb_list.append(self.time_embedding(time_bucket_ids.long()))

        cat_emb = torch.cat(emb_list, dim=-1)
        proj = self._seq_proj[domain]
        token_emb = F.gelu(proj(cat_emb))
        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        """Returns (B, max_len) bool mask; True = padding position."""
        idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        seq_data: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
        static_ids: Optional[torch.Tensor] = None,
        static_array_ids: Optional[Dict[str, torch.Tensor]] = None,
        static_dense: Optional[torch.Tensor] = None,
        static_dense_array_ids: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seq_data / seq_lens / seq_time_buckets: behavior sequences.
            static_ids:              (B, n_int_scalar) int64.
            static_array_ids:        {name: (B, max_dim) int64}.
            static_dense:            (B, n_dense_scalar) float32.
            static_dense_array_ids:  {name: (B, dim) float32}.
        Returns:
            {'logits': (B, action_num), 'hidden': (B, D)}
        """
        domain_outputs: List[torch.Tensor] = []
        domain_masks: List[torch.Tensor] = []

        for domain in self.seq_domains:
            seq = seq_data[domain]
            L = seq.shape[2]
            mask = self._make_padding_mask(seq_lens[domain], L)
            tokens = self._embed_domain(seq, domain, seq_time_buckets[domain])
            tokens = self.emb_dropout(tokens)
            rope_cos, rope_sin = None, None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(L, seq.device)
            encoded = self.domain_encoders[domain](
                tokens, key_padding_mask=mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )
            domain_outputs.append(encoded)
            domain_masks.append(mask)

        static_token: Optional[torch.Tensor] = None
        if self.static_encoder is not None:
            _has_input = any(x is not None for x in [
                static_ids, static_array_ids, static_dense, static_dense_array_ids
            ])
            if _has_input:
                static_token = self.static_encoder(
                    static_ids=static_ids,
                    static_array_ids=static_array_ids,
                    static_dense=static_dense,
                    static_dense_array_ids=static_dense_array_ids,
                )

        behavior_emb = self.cross_attn(domain_outputs, domain_masks, static_token=static_token)
        logits = self.classifier(behavior_emb)
        return {"logits": logits, "hidden": behavior_emb}

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
        """Returns the behavior embedding (B, D) without the classifier head."""
        domain_outputs, domain_masks = [], []
        for domain in self.seq_domains:
            seq = seq_data[domain]
            L = seq.shape[2]
            mask = self._make_padding_mask(seq_lens[domain], L)
            tokens = self._embed_domain(seq, domain, seq_time_buckets[domain])
            rope_cos, rope_sin = None, None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(L, seq.device)
            encoded = self.domain_encoders[domain](
                tokens, key_padding_mask=mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )
            domain_outputs.append(encoded)
            domain_masks.append(mask)

        static_token = None
        if self.static_encoder is not None:
            _has_input = any(x is not None for x in [
                static_ids, static_array_ids, static_dense, static_dense_array_ids
            ])
            if _has_input:
                static_token = self.static_encoder(
                    static_ids=static_ids,
                    static_array_ids=static_array_ids,
                    static_dense=static_dense,
                    static_dense_array_ids=static_dense_array_ids,
                )

        return self.cross_attn(domain_outputs, domain_masks, static_token=static_token)

    # ── Save / Load ────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return {k: getattr(self, k) for k in self._CONFIG_KEYS}

    def save(self, save_dir: str) -> None:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        config = self.get_config()
        with open(save_path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)
        torch.save(self.state_dict(), save_path / "model.pt")
        logger.info(f"Model saved to {save_path}  (config + weights)")

    @classmethod
    def load(cls, save_dir: str, map_location: str = "cpu") -> "BehaviorEncoder":
        save_path = Path(save_dir)
        with open(save_path / "model_config.json") as f:
            config = json.load(f)
        model = cls(**config)
        state = torch.load(save_path / "model.pt", map_location=map_location)
        model.load_state_dict(state)
        logger.info(f"Model loaded from {save_path}")
        return model

    # ── Sparse / Dense param split ─────────────────────────────────────────

    def get_sparse_params(self) -> List[nn.Parameter]:
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, L = 4, 20

    seq_vocab_sizes = {"a": [1000, 500, 200], "b": [2000, 300], "c": [800]}

    model = BehaviorEncoder(
        seq_vocab_sizes=seq_vocab_sizes,
        static_vocab_sizes=[50, 3, 100],              # int scalar
        static_array_configs=[("user_tags", 200, 10), ("item_tags", 500, 5)],  # int array
        n_dense_scalar=4,                              # dense scalar
        dense_array_configs=[("user_profile", 8)],    # dense array
        d_model=64, emb_dim=32, num_heads=4,
        num_domain_layers=2, num_cross_layers=2, dropout_rate=0.0,
    )

    seq_data        = {d: torch.randint(1, 100, (B, len(v), L)) for d, v in seq_vocab_sizes.items()}
    seq_lens        = {d: torch.randint(5, L+1, (B,)) for d in seq_vocab_sizes}
    seq_time_buckets = {d: torch.randint(0, 65, (B, L)) for d in seq_vocab_sizes}

    static_ids      = torch.stack([torch.randint(1, 50, (B,)),
                                   torch.randint(1, 3, (B,)),
                                   torch.randint(1, 100, (B,))], dim=1)
    static_array_ids = {"user_tags": torch.randint(0, 200, (B, 10)),
                        "item_tags": torch.randint(0, 500, (B,  5))}
    static_dense    = torch.randn(B, 4)
    static_dense_array_ids = {"user_profile": torch.randn(B, 8)}

    # All four types
    out = model(seq_data, seq_lens, seq_time_buckets,
                static_ids=static_ids, static_array_ids=static_array_ids,
                static_dense=static_dense, static_dense_array_ids=static_dense_array_ids)
    print(f"all-types   logits: {out['logits'].shape}")

    # Only dense
    out2 = model(seq_data, seq_lens, seq_time_buckets,
                 static_dense=static_dense, static_dense_array_ids=static_dense_array_ids)
    print(f"dense-only  logits: {out2['logits'].shape}")

    # Only int
    out3 = model(seq_data, seq_lens, seq_time_buckets,
                 static_ids=static_ids, static_array_ids=static_array_ids)
    print(f"int-only    logits: {out3['logits'].shape}")

    # No static
    out4 = model(seq_data, seq_lens, seq_time_buckets)
    print(f"no-static   logits: {out4['logits'].shape}")

    print("Smoke test passed ✓")