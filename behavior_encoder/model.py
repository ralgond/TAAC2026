"""
BehaviorEncoder: A standalone behavior sequence encoder.

Architecture:
  1. Per-domain token embedding: concat(sideinfo_embs, time_bucket_emb) -> MLP -> d_model
  2. Per-domain DomainEncoder (Transformer self-attention layers)
  3. Cross-attention across all domains: a learnable CLS token attends to all
     concatenated domain outputs, producing the final behavior embedding.

Key design decisions (derived from PCVRHyFormer):
  - seq_vocab_sizes: {domain: [vocab_size_per_fid, ...]}
  - time_bucket embedding: nn.Embedding(num_time_buckets=65, emb_dim=64)
  - Sequences are in reverse-chronological order (most recent first)
  - RoPE is supported and applied to sequence positions
  - emb_skip_threshold: skip Embedding creation for high-cardinality features
  - Mixed precision (bfloat16) compatible design (no fp16-unsafe ops)
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
# Cross-domain attention: CLS token attends to all domain outputs
# ─────────────────────────────────────────────────────────────────────────────


class CrossDomainAttention(nn.Module):
    """
    A set of learnable CLS tokens (one per domain + one global) attend to
    all concatenated domain sequence outputs via cross-attention.

    Output: global CLS token representation → behavior embedding.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_domains: int,
        dropout: float = 0.0,
        hidden_mult: int = 4,        # FFN hidden_dim = d_model * hidden_mult; default 4 → 256
    ) -> None:
        super().__init__()
        # One global CLS token as the query
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.norm_q = nn.LayerNorm(d_model)
        self.norm_kv = nn.LayerNorm(d_model)

        self.attn = RoPEMHA(d_model, num_heads, dropout)
        self.ffn_norm = nn.LayerNorm(d_model)
        hid = d_model * hidden_mult  # e.g. 64*4=256, consistent with DomainEncoder
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hid, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        domain_outputs: List[torch.Tensor],
        domain_masks: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            domain_outputs: list of (B, L_i, D) tensors (already encoded)
            domain_masks:   list of (B, L_i) bool tensors, True = padding
        Returns:
            (B, D) global behavior embedding
        """
        B = domain_outputs[0].shape[0]

        # Concatenate all domain tokens → (B, sum(L_i), D)
        kv = torch.cat(domain_outputs, dim=1)
        combined_mask = torch.cat(domain_masks, dim=1)  # (B, sum(L_i))

        # CLS query
        q = self.cls_token.expand(B, -1, -1)  # (B, 1, D)

        # Pre-LN cross-attention (no RoPE on query; KV has no positional bias here
        # since domains are concatenated and RoPE was already applied inside DomainEncoder)
        q_normed = self.norm_q(q)
        kv_normed = self.norm_kv(kv)
        attn_out = self.attn(
            q_normed, kv_normed, kv_normed,
            key_padding_mask=combined_mask,
            apply_rope_to_q=False,
        )
        q = q + attn_out  # residual

        # FFN
        residual = q
        q = self.ffn_norm(q)
        q = self.ffn(q)
        q = residual + q

        return self.out_norm(q).squeeze(1)  # (B, D)


# ─────────────────────────────────────────────────────────────────────────────
# Main BehaviorEncoder
# ─────────────────────────────────────────────────────────────────────────────


class BehaviorEncoder(nn.Module):
    """
    Standalone behavior sequence encoder.

    For each domain:
      tokens = concat(emb_fid0, emb_fid1, ..., emb_time_bucket) -> proj MLP -> (B, L, D)
      domain_out = DomainEncoder(tokens)                                       -> (B, L, D)

    Final:
      behavior_emb = CrossDomainAttention(all domain_outs)                     -> (B, D)
      logits       = classifier(behavior_emb)                                  -> (B, action_num)

    Args:
        seq_vocab_sizes:      {domain: [vocab_size_fid0, vocab_size_fid1, ...]}
        d_model:              Model (token) dimension.
        emb_dim:              Dimension of each feature embedding.
        num_heads:            Attention heads in DomainEncoder & CrossDomainAttention.
        num_domain_layers:    Transformer layers per DomainEncoder.
        hidden_mult:          FFN expansion multiplier (hidden_dim = d_model * hidden_mult).
                              Default 4 → hidden_dim=256, standard Transformer config.
        dropout_rate:         Dropout probability.
        num_time_buckets:     Size of time-bucket embedding table (0 = disabled).
        action_num:           Number of output classes.
        use_rope:             Whether to inject Rotary Position Embeddings.
        rope_base:            Base frequency for RoPE.
        emb_skip_threshold:   Skip Embedding for features with vocab_size > threshold (0 = off).
        seq_id_threshold:     Features with vocab_size > threshold get extra id-dropout.
        max_seq_len:          Maximum sequence length (for RoPE cache).
    """

    # Keys saved to / loaded from model_config.json
    _CONFIG_KEYS = [
        "seq_vocab_sizes",
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
    ]

    def __init__(
        self,
        seq_vocab_sizes: Dict[str, List[int]],
        d_model: int = 64,           # token dim; FFN hidden_dim = d_model * hidden_mult = 64*4 = 256
        emb_dim: int = 64,
        num_heads: int = 4,          # DomainEncoder: num_heads = 4
        num_domain_layers: int = 2,  # DomainEncoder: num_layers = 2
        hidden_mult: int = 4,        # DomainEncoder FFN: hidden_dim = 64*4 = 256
        dropout_rate: float = 0.01,
        num_time_buckets: int = 65,
        action_num: int = 1,
        use_rope: bool = True,
        rope_base: float = 10000.0,
        emb_skip_threshold: int = 0,
        seq_id_threshold: int = 10000,
        max_seq_len: int = 2048,
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
        self.num_domain_layers = num_domain_layers
        self.num_heads = num_heads
        self.hidden_mult = hidden_mult
        self.dropout_rate = dropout_rate
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len

        # ── Time bucket embedding (shared across all domains) ──────────────
        if num_time_buckets > 0:
            # Matches PCVRHyFormer: Embedding(65, emb_dim), padding_idx=0
            self.time_embedding = nn.Embedding(num_time_buckets, emb_dim, padding_idx=0)
            nn.init.xavier_normal_(self.time_embedding.weight.data)
            self.time_embedding.weight.data[0, :] = 0.0
        else:
            self.time_embedding = None

        # ── Per-domain feature embeddings + projection ─────────────────────
        self._seq_embs: nn.ModuleDict = nn.ModuleDict()       # domain -> ModuleList
        self._seq_emb_index: Dict[str, List[int]] = {}        # domain -> index_map
        self._seq_is_id: Dict[str, List[bool]] = {}           # domain -> is_id
        self._seq_proj: nn.ModuleDict = nn.ModuleDict()       # domain -> Sequential

        self.seq_id_emb_dropout = nn.Dropout(dropout_rate * 2)

        for domain in self.seq_domains:
            vocab_sizes = self.seq_vocab_sizes[domain]
            embs_raw, idx_map, is_id = self._make_seq_embs(vocab_sizes)
            self._seq_embs[domain] = nn.ModuleList([e for e in embs_raw if e is not None])
            self._seq_emb_index[domain] = idx_map
            self._seq_is_id[domain] = is_id

            # Input to projection = num_fids * emb_dim + (emb_dim if time_buckets)
            in_dim = len(vocab_sizes) * emb_dim
            if num_time_buckets > 0:
                in_dim += emb_dim          # time bucket concat'd alongside fid embs

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

        # ── Cross-domain attention ─────────────────────────────────────────
        self.cross_attn = CrossDomainAttention(
            d_model=d_model,
            num_heads=num_heads,
            num_domains=len(self.seq_domains),
            dropout=dropout_rate,
            hidden_mult=hidden_mult,   # keep FFN width consistent with DomainEncoder
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

        # Xavier init for all linear layers not handled above
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
        seq: torch.Tensor,           # (B, num_fids, L) — fid-major layout
        domain: str,
        time_bucket_ids: torch.Tensor,  # (B, L) int
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

        # Append time bucket embedding
        if self.time_embedding is not None:
            emb_list.append(self.time_embedding(time_bucket_ids.long()))  # (B, L, emb_dim)

        cat_emb = torch.cat(emb_list, dim=-1)  # (B, L, num_fids*emb_dim [+ emb_dim])
        proj = self._seq_proj[domain]
        token_emb = F.gelu(proj(cat_emb))      # (B, L, D)
        return token_emb

    def _make_padding_mask(self, seq_len: torch.Tensor, max_len: int) -> torch.Tensor:
        """Returns (B, max_len) bool mask; True = padding position."""
        idx = torch.arange(max_len, device=seq_len.device).unsqueeze(0)
        return idx >= seq_len.unsqueeze(1)

    # ── Forward ────────────────────────────────────────────────────────────

    def forward(
        self,
        seq_data: Dict[str, torch.Tensor],        # {domain: (B, num_fids, L)}
        seq_lens: Dict[str, torch.Tensor],         # {domain: (B,) int}
        seq_time_buckets: Dict[str, torch.Tensor], # {domain: (B, L) int}
    ) -> torch.Tensor:
        """
        Args:
            seq_data:         Per-domain behavior sequences, keyed by domain name
                              (e.g. 'a', 'b', 'c', 'd').
                              Each tensor shape: (B, num_fids, L), most-recent first.
                              Batch key: ``seq_{domain}``  e.g. ``seq_a``.
            seq_lens:         Valid sequence length per sample per domain. (B,)
                              Batch key: ``seq_{domain}_len``  e.g. ``seq_a_len``.
            seq_time_buckets: Time-bucket ids per position per domain. (B, L)
                              Batch key: ``seq_{domain}_time_bucket``
                              e.g. ``seq_a_time_bucket``.
        Returns:
            logits: (B, action_num)
        """
        domain_outputs: List[torch.Tensor] = []
        domain_masks: List[torch.Tensor] = []

        for domain in self.seq_domains:
            seq = seq_data[domain]           # (B, num_fids, L)
            L = seq.shape[2]

            # Padding mask
            mask = self._make_padding_mask(seq_lens[domain], L)  # (B, L)

            # Token embedding
            tokens = self._embed_domain(seq, domain, seq_time_buckets[domain])  # (B, L, D)
            tokens = self.emb_dropout(tokens)

            # RoPE
            rope_cos, rope_sin = None, None
            if self.rotary_emb is not None:
                rope_cos, rope_sin = self.rotary_emb(L, seq.device)

            # Domain encoder
            encoded = self.domain_encoders[domain](
                tokens, key_padding_mask=mask,
                rope_cos=rope_cos, rope_sin=rope_sin,
            )  # (B, L, D)

            domain_outputs.append(encoded)
            domain_masks.append(mask)

        # Cross-domain attention → (B, D)
        behavior_emb = self.cross_attn(domain_outputs, domain_masks)

        # Classifier
        logits = self.classifier(behavior_emb)  # (B, action_num)
        return logits

    def encode(
        self,
        seq_data: Dict[str, torch.Tensor],
        seq_lens: Dict[str, torch.Tensor],
        seq_time_buckets: Dict[str, torch.Tensor],
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
        return self.cross_attn(domain_outputs, domain_masks)

    # ── Save / Load ────────────────────────────────────────────────────────

    def get_config(self) -> dict:
        return {k: getattr(self, k) for k in self._CONFIG_KEYS}

    def save(self, save_dir: str) -> None:
        """Save model weights and config to save_dir."""
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        config = self.get_config()
        config_path = save_path / "model_config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        ckpt_path = save_path / "model.pt"
        torch.save(self.state_dict(), ckpt_path)
        logger.info(f"Model saved to {save_path}  (config + weights)")

    @classmethod
    def load(cls, save_dir: str, map_location: str = "cpu") -> "BehaviorEncoder":
        """Load model from save_dir (reads model_config.json + model.pt)."""
        save_path = Path(save_dir)
        config_path = save_path / "model_config.json"
        ckpt_path = save_path / "model.pt"

        with open(config_path) as f:
            config = json.load(f)

        model = cls(**config)
        state = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(state)
        logger.info(f"Model loaded from {save_path}")
        return model

    # ── Sparse / Dense param split (for dual-optimizer setups) ────────────

    def get_sparse_params(self) -> List[nn.Parameter]:
        ptrs = {m.weight.data_ptr() for m in self.modules() if isinstance(m, nn.Embedding)}
        return [p for p in self.parameters() if p.data_ptr() in ptrs]

    def get_dense_params(self) -> List[nn.Parameter]:
        sparse_ptrs = {p.data_ptr() for p in self.get_sparse_params()}
        return [p for p in self.parameters() if p.data_ptr() not in sparse_ptrs]