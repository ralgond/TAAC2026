"""PCVR Parquet dataset module (performance-tuned).

Reads raw multi-column Parquet directly and obtains feature metadata from
``schema.json``.

Optimizations:
- Pre-allocated numpy buffers to eliminate ``np.zeros`` + ``np.stack`` overhead.
- Fused padding loop over sequence domains that writes directly into a 3D buffer.
- Pre-computed column-index lookup to avoid per-row string lookups.
- ``file_system`` tensor-sharing strategy to work around ``/dev/shm`` exhaustion
  when using many DataLoader workers.

Changes vs. original:
- Each scalar (dim==1) int/float feature is output as its own key in the batch
  dict, e.g. ``user_int_feats_1``, ``item_int_feats_3``, ``user_dense_feats_7``.
  Multi-dim features are skipped entirely.
- ``item_id`` is read from the dedicated ``item_id`` parquet column and placed
  in GPU memory (same as ``user_id``).
- ``user_id`` and ``item_id`` are both tensors on the CUDA device when available.
"""

import os
import logging
import random
import json
import gc

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.multiprocessing
from torch.utils.data import IterableDataset, DataLoader
from typing import Any, Dict, Iterator, List, Optional, Tuple

try:
    import numpy.typing as npt  # noqa: F401
except ImportError:  # pragma: no cover
    class _NptFallback:  # type: ignore[no-redef]
        NDArray = Any

    npt = _NptFallback()  # type: ignore[assignment]


# ─────────────────────────── Feature Schema ──────────────────────────────────


class FeatureSchema:
    """Records ``(feature_id, offset, length)`` for each feature so downstream
    code can locate the segment of the flattened tensor that belongs to a
    specific feature id.

    For int features:
      - int_value: length = 1
      - int_array: length = array length
      - int_array_and_float_array: int part length
    For dense features:
      - float_value: length = 1
      - float_array: length = array length
      - int_array_and_float_array: float part length
    """

    def __init__(self) -> None:
        self.entries: List[Tuple[int, int, int]] = []
        self.total_dim: int = 0
        self._fid_to_entry: Dict[int, Tuple[int, int]] = {}

    def add(self, feature_id: int, length: int) -> None:
        offset = self.total_dim
        self.entries.append((feature_id, offset, length))
        self._fid_to_entry[feature_id] = (offset, length)
        self.total_dim += length

    def get_offset_length(self, feature_id: int) -> Tuple[int, int]:
        return self._fid_to_entry[feature_id]

    @property
    def feature_ids(self) -> List[int]:
        return [fid for fid, _, _ in self.entries]

    def to_dict(self) -> Dict[str, Any]:
        return {'entries': self.entries, 'total_dim': self.total_dim}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'FeatureSchema':
        schema = cls()
        for fid, offset, length in d['entries']:
            schema.entries.append((fid, offset, length))
            schema._fid_to_entry[fid] = (offset, length)
        schema.total_dim = d['total_dim']
        return schema

    def __repr__(self) -> str:
        lines = [f"FeatureSchema(total_dim={self.total_dim}, features=["]
        for fid, offset, length in self.entries:
            lines.append(f"  fid={fid}: offset={offset}, length={length}")
        lines.append("])")
        return "\n".join(lines)


torch.multiprocessing.set_sharing_strategy('file_system')

BUCKET_BOUNDARIES = np.array([
    5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,
    120, 180, 240, 300, 360, 420, 480, 540, 600,
    900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3300, 3600,
    5400, 7200, 9000, 10800, 12600, 14400, 16200, 18000, 19800, 21600,
    32400, 43200, 54000, 64800, 75600, 86400,
    172800, 259200, 345600, 432000, 518400, 604800,
    1123200, 1641600, 2160000, 2592000,
    4320000, 6048000, 7776000,
    11664000, 15552000,
    31536000,
], dtype=np.int64)

NUM_TIME_BUCKETS = len(BUCKET_BOUNDARIES) + 1




class PCVRParquetDataset(IterableDataset):
    """PCVR dataset that reads raw multi-column Parquet directly.

    Output batch dict keys
    ----------------------
    Per scalar (dim==1) int feature:
        ``user_int_feats_{fid}``         shape ``[B]``,    dtype int64
        ``item_int_feats_{fid}``         shape ``[B]``,    dtype int64
    Per array (dim>1) int feature:
        ``user_int_array_feature_{fid}`` shape ``[B, dim]``, dtype int64
        ``item_int_array_feature_{fid}`` shape ``[B, dim]``, dtype int64
    Per scalar (dim==1) dense feature:
        ``user_dense_feats_{fid}``       shape ``[B]``,    dtype float32
    Multi-dim dense features are **not** included in the output.

    Special keys:
        ``item_id``     shape ``[B]``, int64, on GPU (if available)
        ``user_id``     shape ``[B]``, int64, on GPU (if available)
        ``label``       shape ``[B]``, int64
        ``timestamp``   shape ``[B]``, int64
        ``_seq_domains`` list of domain name strings (non-tensor metadata)
    Sequence keys (unchanged):
        ``{domain}``               shape ``[B, n_feats, max_len]``, int64
        ``{domain}_len``           shape ``[B]``, int64
        ``{domain}_time_bucket``   shape ``[B, max_len]``, int64
    """

    def __init__(
        self,
        parquet_path: str,
        schema_path: str,
        batch_size: int = 256,
        seq_max_lens: Optional[Dict[str, int]] = None,
        shuffle: bool = True,
        buffer_batches: int = 20,
        row_group_range: Optional[Tuple[int, int]] = None,
        clip_vocab: bool = True,
        is_training: bool = True,
    ) -> None:
        super().__init__()

        if os.path.isdir(parquet_path):
            import glob
            files = sorted(glob.glob(os.path.join(parquet_path, '*.parquet')))
            if not files:
                raise FileNotFoundError(f"No .parquet files in {parquet_path}")
            self._parquet_files = files
        else:
            self._parquet_files = [parquet_path]

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.buffer_batches = buffer_batches
        self.clip_vocab = clip_vocab
        self.is_training = is_training
        self._oob_stats: Dict[Tuple[str, int], Dict[str, int]] = {}

        self._rg_list = []
        for f in self._parquet_files:
            pf = pq.ParquetFile(f)
            for i in range(pf.metadata.num_row_groups):
                self._rg_list.append((f, i, pf.metadata.row_group(i).num_rows))

        if row_group_range is not None:
            start, end = row_group_range
            self._rg_list = self._rg_list[start:end]

        self.num_rows = sum(r[2] for r in self._rg_list)

        self._load_schema(schema_path, seq_max_lens or {})

        # ---- Pre-compute column index lookup ----
        pf = pq.ParquetFile(self._parquet_files[0])
        schema_names = pf.schema_arrow.names
        self._col_idx = {name: i for i, name in enumerate(schema_names)}

        # ---- item_id column index ----
        self._item_id_ci = self._col_idx.get('item_id')
        if self._item_id_ci is None:
            logging.warning("'item_id' column not found in parquet schema; "
                            "item_id will be all-zeros.")

        # ---- Build scalar-only (dim==1) plans ----
        # Each plan entry: (col_idx, fid, vocab_size, padding_val)
        # padding_val = max(vs, 1): one slot beyond the valid vocab range,
        # matching padding_idx in model.py (num_embeddings = max(vs,1)+1).
        # Null/missing values are filled with padding_val so they land on the
        # zero-grad padding row and are never confused with valid feature value 0.
        self._user_int_plan: List[Tuple[int, int, int, int]] = []
        for fid, vs, dim in self._user_int_cols:
            if dim != 1:
                continue
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_plan.append((ci, fid, vs, max(vs, 1)))

        self._item_int_plan: List[Tuple[int, int, int, int]] = []
        for fid, vs, dim in self._item_int_cols:
            if dim != 1:
                continue
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_plan.append((ci, fid, vs, max(vs, 1)))

        # ---- Build array (dim>1) int plans ----
        # Each plan entry: (col_idx, fid, vocab_size, padding_val, dim)
        self._user_int_array_plan: List[Tuple[int, int, int, int, int]] = []
        for fid, vs, dim in self._user_int_cols:
            if dim <= 1:
                continue
            ci = self._col_idx.get(f'user_int_feats_{fid}')
            self._user_int_array_plan.append((ci, fid, vs, max(vs, 1), dim))

        self._item_int_array_plan: List[Tuple[int, int, int, int, int]] = []
        for fid, vs, dim in self._item_int_cols:
            if dim <= 1:
                continue
            ci = self._col_idx.get(f'item_int_feats_{fid}')
            self._item_int_array_plan.append((ci, fid, vs, max(vs, 1), dim))

        # Dense scalar plan: (col_idx, fid)
        self._user_dense_plan: List[Tuple[int, int]] = []
        for fid, dim in self._user_dense_cols:
            if dim != 1:
                continue
            ci = self._col_idx.get(f'user_dense_feats_{fid}')
            self._user_dense_plan.append((ci, fid))

        # ---- Pre-allocate sequence buffers (unchanged) ----
        B = batch_size
        self._buf_seq: Dict[str, np.ndarray] = {}
        self._buf_seq_tb: Dict[str, np.ndarray] = {}
        self._buf_seq_lens: Dict[str, np.ndarray] = {}
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            n_feats = len(self.sideinfo_fids[domain])
            self._buf_seq[domain] = np.zeros((B, n_feats, max_len), dtype=np.int64)
            self._buf_seq_tb[domain] = np.zeros((B, max_len), dtype=np.int64)
            self._buf_seq_lens[domain] = np.zeros(B, dtype=np.int64)

        # ---- Sequence column plan (unchanged) ----
        self._seq_plan: Dict[str, Tuple[List[Tuple[int, int, int]], Optional[int]]] = {}
        for domain in self.seq_domains:
            prefix = self._seq_prefix[domain]
            sideinfo_fids = self.sideinfo_fids[domain]
            ts_fid = self.ts_fids[domain]
            side_plan = []
            for slot, fid in enumerate(sideinfo_fids):
                ci = self._col_idx.get(f'{prefix}_{fid}')
                vs = self.seq_vocab_sizes[domain][fid]
                side_plan.append((ci, slot, vs))
            ts_ci = self._col_idx.get(f'{prefix}_{ts_fid}') if ts_fid is not None else None
            self._seq_plan[domain] = (side_plan, ts_ci)

        logging.info(
            f"PCVRParquetDataset: {self.num_rows} rows from "
            f"{len(self._parquet_files)} file(s), batch_size={batch_size}, "
            f"buffer_batches={buffer_batches}, shuffle={shuffle}, "
            f"scalar user_int={len(self._user_int_plan)}, "
            f"scalar item_int={len(self._item_int_plan)}, "
            f"scalar user_dense={len(self._user_dense_plan)}, "
            f"array user_int={len(self._user_int_array_plan)}, "
            f"array item_int={len(self._item_int_array_plan)}")

    # ------------------------------------------------------------------
    # Schema loading (unchanged)
    # ------------------------------------------------------------------

    def _load_schema(self, schema_path: str, seq_max_lens: Dict[str, int]) -> None:
        with open(schema_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)

        self._user_int_cols: List[List[int]] = raw['user_int']
        self.user_int_schema: FeatureSchema = FeatureSchema()
        self.user_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._user_int_cols:
            self.user_int_schema.add(fid, dim)
            self.user_int_vocab_sizes.extend([vs] * dim)

        self._item_int_cols: List[List[int]] = raw['item_int']
        self.item_int_schema: FeatureSchema = FeatureSchema()
        self.item_int_vocab_sizes: List[int] = []
        for fid, vs, dim in self._item_int_cols:
            self.item_int_schema.add(fid, dim)
            self.item_int_vocab_sizes.extend([vs] * dim)

        self._user_dense_cols: List[List[int]] = raw['user_dense']
        self.user_dense_schema: FeatureSchema = FeatureSchema()
        for fid, dim in self._user_dense_cols:
            self.user_dense_schema.add(fid, dim)

        self.item_dense_schema: FeatureSchema = FeatureSchema()

        self._seq_cfg: Dict[str, Dict[str, Any]] = raw['seq']
        self.seq_domains: List[str] = sorted(self._seq_cfg.keys())
        self.seq_feature_ids: Dict[str, List[int]] = {}
        self.seq_vocab_sizes: Dict[str, Dict[int, int]] = {}
        self.seq_domain_vocab_sizes: Dict[str, List[int]] = {}
        self.ts_fids: Dict[str, Optional[int]] = {}
        self.sideinfo_fids: Dict[str, List[int]] = {}
        self._seq_prefix: Dict[str, str] = {}
        self._seq_maxlen: Dict[str, int] = {}

        for domain in self.seq_domains:
            cfg = self._seq_cfg[domain]
            self._seq_prefix[domain] = cfg['prefix']
            ts_fid = cfg['ts_fid']
            self.ts_fids[domain] = ts_fid

            all_fids = [fid for fid, vs in cfg['features']]
            self.seq_feature_ids[domain] = all_fids
            self.seq_vocab_sizes[domain] = {fid: vs for fid, vs in cfg['features']}

            sideinfo = [fid for fid in all_fids if fid != ts_fid]
            self.sideinfo_fids[domain] = sideinfo
            self.seq_domain_vocab_sizes[domain] = [
                self.seq_vocab_sizes[domain][fid] for fid in sideinfo
            ]
            self._seq_maxlen[domain] = seq_max_lens.get(domain, 256)

    # ------------------------------------------------------------------
    # Iterator
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return sum((n + self.batch_size - 1) // self.batch_size
                   for _, _, n in self._rg_list)

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        rg_list = self._rg_list
        if worker_info is not None and worker_info.num_workers > 1:
            rg_list = [rg for i, rg in enumerate(rg_list)
                       if i % worker_info.num_workers == worker_info.id]

        buffer: List[Dict[str, Any]] = []
        for file_path, rg_idx, _ in rg_list:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches(batch_size=self.batch_size, row_groups=[rg_idx]):
                batch_dict = self._convert_batch(batch)
                if self.shuffle and self.buffer_batches > 1:
                    buffer.append(batch_dict)
                    if len(buffer) >= self.buffer_batches:
                        yield from self._flush_buffer(buffer)
                        buffer = []
                else:
                    yield batch_dict

        if buffer:
            yield from self._flush_buffer(buffer)

        del buffer
        gc.collect()

    def _flush_buffer(
        self, buffer: List[Dict[str, Any]]
    ) -> Iterator[Dict[str, Any]]:
        """Shuffle and re-slice the buffered batches without full concatenation.

        Memory strategy
        ---------------
        Instead of cat-ing every tensor into one giant buffer (O(buffer_batches)
        peak memory per worker), we build a row-level index that maps each
        output row to ``(batch_slot, within_batch_row)``.  Each output mini-batch
        is then assembled by gathering one slice per source batch, which keeps
        peak extra memory at O(batch_size) rather than O(buffer_batches *
        batch_size).
        """
        non_tensor_keys: Dict[str, Any] = {
            k: v for k, v in buffer[0].items()
            if not isinstance(v, torch.Tensor)
        }
        tensor_keys = [k for k in buffer[0].keys() if isinstance(buffer[0][k], torch.Tensor)]

        # Build a flat row index: each entry is (buf_slot, row_within_slot).
        sizes = [buffer[b]['label'].shape[0] for b in range(len(buffer))]
        total_rows = sum(sizes)
        # flat_index[i] = (buf_slot, local_row)
        flat_index: List[Tuple[int, int]] = []
        for slot, sz in enumerate(sizes):
            for r in range(sz):
                flat_index.append((slot, r))

        if self.shuffle:
            perm = torch.randperm(total_rows).tolist()
            flat_index = [flat_index[i] for i in perm]

        for start in range(0, total_rows, self.batch_size):
            chunk = flat_index[start:start + self.batch_size]
            # Group rows by source slot to minimise indexing calls.
            slot_rows: Dict[int, List[int]] = {}
            chunk_order: List[Tuple[int, int]] = []  # (slot, local_pos_in_slot_rows)
            for slot, row in chunk:
                pos = len(slot_rows.get(slot, []))
                slot_rows.setdefault(slot, []).append(row)
                chunk_order.append((slot, pos))

            # For each tensor key, gather rows from the relevant source batches.
            out: Dict[str, Any] = {}
            for k in tensor_keys:
                parts = []
                for slot, rows in slot_rows.items():
                    idx = torch.tensor(rows, dtype=torch.long)
                    parts.append(buffer[slot][k][idx])
                # Re-order to match the shuffled chunk_order.
                cat = torch.cat(parts, dim=0)  # ordered by slot, not by chunk
                # Build a mapping from (slot, pos) → position in cat.
                offsets: Dict[int, int] = {}
                off = 0
                for slot in slot_rows:
                    offsets[slot] = off
                    off += len(slot_rows[slot])
                final_idx = [offsets[slot] + pos for slot, pos in chunk_order]
                out[k] = cat[torch.tensor(final_idx, dtype=torch.long)]
            out.update(non_tensor_keys)
            yield out

        buffer.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _record_oob(
        self,
        group: str,
        col_idx: int,
        arr: "npt.NDArray[np.int64]",
        vocab_size: int,
        padding_val: int,
    ) -> None:
        """Record and clip values >= vocab_size.

        Callers are responsible for clipping negatives to ``padding_val``
        before calling this method (``arr[arr < 0] = pv``).  By the time
        this method runs, ``arr`` contains only non-negative integers, so
        the only OOB case to handle here is the upper-bound violation
        (``arr >= vocab_size``).
        """
        hi_mask = arr >= vocab_size
        if not hi_mask.any():
            return
        key = (group, col_idx)
        oob_vals = arr[hi_mask]
        n  = int(hi_mask.sum())
        mx = int(oob_vals.max())
        mn = int(oob_vals.min())
        if key in self._oob_stats:
            s = self._oob_stats[key]
            s['count'] += n
            s['max']     = max(s['max'], mx)
            s['min_oob'] = min(s['min_oob'], mn)
        else:
            self._oob_stats[key] = {
                'count': n, 'max': mx, 'min_oob': mn, 'vocab': vocab_size,
            }
        if self.clip_vocab:
            arr[hi_mask] = padding_val
        else:
            raise ValueError(
                f"{group} col_idx={col_idx}: {n} values >= vocab_size={vocab_size}, "
                f"actual range=[{mn}, {mx}]. "
                f"Use clip_vocab=True to clip or fix schema.json")

    def dump_oob_stats(self, path: Optional[str] = None) -> None:
        if not self._oob_stats:
            logging.info("No out-of-bound values detected.")
            return
        lines = ["=== Out-of-Bound Stats ==="]
        for (group, ci), s in sorted(self._oob_stats.items()):
            direction = "TOO_HIGH" if s['min_oob'] >= s['vocab'] else "TOO_LOW"
            lines.append(
                f"  {group} col_idx={ci}: vocab={s['vocab']}, "
                f"oob_count={s['count']}, range=[{s['min_oob']}, {s['max']}], "
                f"{direction}")
        msg = "\n".join(lines)
        if path:
            with open(path, 'w') as f:
                f.write(msg + "\n")
            logging.info(f"OOB stats written to {path}")
        else:
            logging.info(msg)

    def _pad_varlen_int_column(
        self,
        arrow_col: "pa.ListArray",
        max_len: int,
        B: int,
    ) -> Tuple["npt.NDArray[np.int64]", "npt.NDArray[np.int64]"]:
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((B, max_len), dtype=np.int64)
        lengths = np.zeros(B, dtype=np.int64)
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_len)
            padded[i, :use_len] = values[start:start + use_len]
            lengths[i] = use_len
        padded[padded <= 0] = 0
        return padded, lengths

    _pad_varlen_column = _pad_varlen_int_column

    def _pad_varlen_float_column(
        self,
        arrow_col: "pa.ListArray",
        max_dim: int,
        B: int,
    ) -> "npt.NDArray[np.float32]":
        offsets = arrow_col.offsets.to_numpy()
        values = arrow_col.values.to_numpy()
        padded = np.zeros((B, max_dim), dtype=np.float32)
        for i in range(B):
            start, end = int(offsets[i]), int(offsets[i + 1])
            raw_len = end - start
            if raw_len <= 0:
                continue
            use_len = min(raw_len, max_dim)
            padded[i, :use_len] = values[start:start + use_len]
        return padded

    # ------------------------------------------------------------------
    # Core batch conversion
    # ------------------------------------------------------------------

    def _convert_batch(self, batch: "pa.RecordBatch") -> Dict[str, Any]:
        """Convert an Arrow RecordBatch into a training-ready dict of tensors.

        Each scalar (dim==1) int/float feature is stored under its own key:
            ``user_int_feats_{fid}``   shape [B], int64
            ``item_int_feats_{fid}``   shape [B], int64
            ``user_dense_feats_{fid}`` shape [B], float32
        Multi-dim features are skipped.

        ``user_id`` and ``item_id`` are int64 tensors placed on the GPU device
        (falls back to CPU when CUDA is unavailable).
        """
        B = batch.num_rows

        # ---- meta ----
        timestamps = batch.column(self._col_idx['timestamp']).to_numpy().astype(np.int64)
        if self.is_training:
            labels = (batch.column(self._col_idx['label_type']).fill_null(0)
                      .to_numpy(zero_copy_only=False).astype(np.int64) == 2).astype(np.int64)
        else:
            labels = np.zeros(B, dtype=np.int64)

        # user_id — CPU tensor
        user_id_raw = np.array(batch.column(self._col_idx['user_id']).to_pylist(),
                               dtype=np.int64)

        # item_id — CPU tensor
        if self._item_id_ci is not None:
            item_id_raw = (batch.column(self._item_id_ci)
                           .fill_null(0)
                           .to_numpy(zero_copy_only=False)
                           .astype(np.int64))
        else:
            item_id_raw = np.zeros(B, dtype=np.int64)

        result: Dict[str, Any] = {
            'label':        torch.from_numpy(labels),
            'timestamp':    torch.from_numpy(timestamps),
            'user_id':      torch.from_numpy(user_id_raw),
            'item_id':      torch.from_numpy(item_id_raw),
            '_seq_domains': self.seq_domains,
        }

        # ---- user_int: one key per scalar feature ----
        # fill_null maps Arrow nulls to pv (= padding_idx in the embedding).
        # Negative values (-1 sentinel for missing) are clipped to pv
        # unconditionally BEFORE the OOB check, so _record_oob only sees
        # values in [0, ...) and the embedding lookup is always in range.
        for ci, fid, vs, pv in self._user_int_plan:
            arr = (batch.column(ci)
                   .fill_null(pv)
                   .to_numpy(zero_copy_only=False)
                   .astype(np.int64))
            arr[arr < 0] = pv          # clip negatives → padding unconditionally
            if vs > 0:
                self._record_oob('user_int', ci, arr, vs, pv)
            else:
                arr[:] = pv
            result[f'user_int_feats_{fid}'] = torch.from_numpy(arr.copy())

        # ---- item_int: one key per scalar feature ----
        for ci, fid, vs, pv in self._item_int_plan:
            arr = (batch.column(ci)
                   .fill_null(pv)
                   .to_numpy(zero_copy_only=False)
                   .astype(np.int64))
            arr[arr < 0] = pv          # clip negatives → padding unconditionally
            if vs > 0:
                self._record_oob('item_int', ci, arr, vs, pv)
            else:
                arr[:] = pv
            result[f'item_int_feats_{fid}'] = torch.from_numpy(arr.copy())

        # ---- user_int_array: one key per array (dim>1) feature ----
        # Output shape: [B, dim], dtype int64.
        # Arrow list columns are padded/truncated to exactly `dim` slots.
        # Out-of-range values and nulls are mapped to the padding value.
        for ci, fid, vs, pv, dim in self._user_int_array_plan:
            col = batch.column(ci)
            padded, _ = self._pad_varlen_int_column(col, dim, B)
            padded[padded < 0] = pv
            if vs > 0:
                self._record_oob('user_int_array', ci, padded, vs, pv)
            else:
                padded[:] = pv
            result[f'user_int_array_feature_{fid}'] = torch.from_numpy(padded.copy())

        # ---- item_int_array: one key per array (dim>1) feature ----
        for ci, fid, vs, pv, dim in self._item_int_array_plan:
            col = batch.column(ci)
            padded, _ = self._pad_varlen_int_column(col, dim, B)
            padded[padded < 0] = pv
            if vs > 0:
                self._record_oob('item_int_array', ci, padded, vs, pv)
            else:
                padded[:] = pv
            result[f'item_int_array_feature_{fid}'] = torch.from_numpy(padded.copy())

        # ---- user_dense: one key per scalar feature ----
        for ci, fid in self._user_dense_plan:
            col = batch.column(ci)
            # Dense scalar columns may be stored as list<float> with length 1
            # or as a plain float column; handle both cases.
            if hasattr(col, 'offsets'):
                padded = self._pad_varlen_float_column(col, 1, B)
                arr_f = padded[:, 0].copy()
            else:
                arr_f = (col.fill_null(0.0)
                         .to_numpy(zero_copy_only=False)
                         .astype(np.float32))
            result[f'user_dense_feats_{fid}'] = torch.from_numpy(arr_f)

        # ---- Sequence features: fused padding directly into the 3D buffer ----
        for domain in self.seq_domains:
            max_len = self._seq_maxlen[domain]
            side_plan, ts_ci = self._seq_plan[domain]

            out = self._buf_seq[domain][:B]
            out[:] = 0
            lengths = self._buf_seq_lens[domain][:B]
            lengths[:] = 0

            col_data = []
            for ci, slot, vs in side_plan:
                col = batch.column(ci)
                col_data.append((col.offsets.to_numpy(), col.values.to_numpy(), vs, ci))

            for c, (offs, vals, vs, ci) in enumerate(col_data):
                for i in range(B):
                    s = int(offs[i])
                    e = int(offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    out[i, c, :ul] = vals[s:s + ul]
                    if ul > lengths[i]:
                        lengths[i] = ul

            out[out <= 0] = 0

            for c, (_, _, vs, ci) in enumerate(col_data):
                slice_c = out[:, c, :]
                if vs > 0:
                    self._record_oob(f'seq_{domain}', ci, slice_c, vs, 0)
                else:
                    slice_c[:] = 0

            result[domain] = torch.from_numpy(out.copy())
            result[f'{domain}_len'] = torch.from_numpy(lengths.copy())

            time_bucket = self._buf_seq_tb[domain][:B]
            time_bucket[:] = 0
            if ts_ci is not None:
                ts_col = batch.column(ts_ci)
                ts_offs = ts_col.offsets.to_numpy()
                ts_vals = ts_col.values.to_numpy()
                ts_padded = np.zeros((B, max_len), dtype=np.int64)
                for i in range(B):
                    s = int(ts_offs[i])
                    e = int(ts_offs[i + 1])
                    rl = e - s
                    if rl <= 0:
                        continue
                    ul = min(rl, max_len)
                    ts_padded[i, :ul] = ts_vals[s:s + ul]

                ts_expanded = timestamps.reshape(-1, 1)
                time_diff = np.maximum(ts_expanded - ts_padded, 0)
                raw_buckets = np.clip(
                    np.searchsorted(BUCKET_BOUNDARIES, time_diff.ravel()),
                    0, len(BUCKET_BOUNDARIES) - 1,
                )
                buckets = raw_buckets.reshape(B, max_len) + 1
                buckets[ts_padded == 0] = 0
                time_bucket[:] = buckets

            result[f'{domain}_time_bucket'] = torch.from_numpy(time_bucket.copy())

        return result


def get_pcvr_data(
    data_dir: str,
    schema_path: str,
    batch_size: int = 256,
    valid_ratio: float = 0.1,
    train_ratio: float = 1.0,
    num_workers: int = 16,
    buffer_batches: int = 20,
    shuffle_train: bool = True,
    seed: int = 42,
    clip_vocab: bool = True,
    seq_max_lens: Optional[Dict[str, int]] = None,
    **kwargs: Any,
) -> Tuple[DataLoader, DataLoader, PCVRParquetDataset]:
    """Create train / valid DataLoaders from raw multi-column Parquet files.

    Returns:
        ``(train_loader, valid_loader, train_dataset)``
    """
    random.seed(seed)

    import glob as _glob
    pq_files = sorted(_glob.glob(os.path.join(data_dir, '*.parquet')))

    rg_info = []
    for f in pq_files:
        pf = pq.ParquetFile(f)
        for i in range(pf.metadata.num_row_groups):
            rg_info.append((f, i, pf.metadata.row_group(i).num_rows))
    total_rgs = len(rg_info)

    n_valid_rgs = max(1, int(total_rgs * valid_ratio))
    n_train_rgs = total_rgs - n_valid_rgs

    if train_ratio < 1.0:
        n_train_rgs = max(1, int(n_train_rgs * train_ratio))
        logging.info(f"train_ratio={train_ratio}: using {n_train_rgs} train Row Groups")

    train_rows = sum(r[2] for r in rg_info[:n_train_rgs])
    valid_rows = sum(r[2] for r in rg_info[n_train_rgs:])

    logging.info(f"Row Group split: {n_train_rgs} train ({train_rows} rows), "
                 f"{n_valid_rgs} valid ({valid_rows} rows)")

    train_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=shuffle_train,
        buffer_batches=buffer_batches,
        row_group_range=(0, n_train_rgs),
        clip_vocab=clip_vocab,
    )

    use_cuda = torch.cuda.is_available()
    # pin_memory=False: pinning locks system RAM which competes with GPU memory
    # under pressure and makes the OOM killer more likely to fire.
    # persistent_workers=False: workers holding pre-allocated seq buffers
    # consume significant system RAM even between epochs.
    _train_kw = {}
    if num_workers > 0:
        _train_kw['prefetch_factor'] = 1

    train_loader = DataLoader(
        train_dataset, batch_size=None,
        num_workers=num_workers, pin_memory=False,
        **_train_kw,
    )

    valid_dataset = PCVRParquetDataset(
        parquet_path=data_dir,
        schema_path=schema_path,
        batch_size=batch_size,
        seq_max_lens=seq_max_lens,
        shuffle=False,
        buffer_batches=0,
        row_group_range=(n_train_rgs, total_rgs),
        clip_vocab=clip_vocab,
    )
    valid_loader = DataLoader(
        valid_dataset, batch_size=None,
        num_workers=0, pin_memory=False,
    )

    logging.info(f"Parquet train: {train_rows} rows, valid: {valid_rows} rows, "
                 f"batch_size={batch_size}, buffer_batches={buffer_batches}")

    return train_loader, valid_loader, train_dataset