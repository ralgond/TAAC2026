"""Diagnose user_id distribution in parquet files.

Usage:
    python diagnose_data.py --data_dir /path/to/parquet [--max_rgs 5]
"""
import argparse
import collections
import glob
import os
import logging
import pyarrow.parquet as pq

data_dir = os.environ.get('TRAIN_DATA_PATH')
log_dir = os.environ.get('TRAIN_LOG_PATH')
schema_path = os.path.join(data_dir, 'schema.json')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    max_rgs = 5

    files = sorted(glob.glob(os.path.join(data_dir, '*.parquet')))
    print(f"Files: {len(files)}")

    # Collect all RGs
    all_rgs = []
    for f in files:
        pf = pq.ParquetFile(f)
        print(f"  {os.path.basename(f)}: {pf.metadata.num_row_groups} RGs, "
              f"{pf.metadata.num_rows} rows")
        for i in range(pf.metadata.num_row_groups):
            all_rgs.append((f, i))

    print(f"\nTotal RGs: {len(all_rgs)}, sampling first {max_rgs}\n")

    # --- Per-RG stats ---
    uid_sets = []
    for f, rg_idx in all_rgs[:max_rgs]:
        pf = pq.ParquetFile(f)
        tbl = pf.read_row_group(rg_idx, columns=['user_id', 'timestamp'])
        uids = tbl.column('user_id').to_pylist()
        ts   = tbl.column('timestamp').to_numpy()
        cnt  = collections.Counter(uids)
        uid_sets.append(set(uids))
        print(f"RG {rg_idx} ({os.path.basename(f)}): "
              f"{tbl.num_rows} rows, "
              f"{len(cnt)} unique users, "
              f"max_appearances={max(cnt.values())}, "
              f"ts=[{ts.min()}, {ts.max()}]")

    # --- Cross-RG overlap ---
    print(f"\nCross-RG user_id overlap:")
    for i in range(len(uid_sets)):
        for j in range(i + 1, len(uid_sets)):
            overlap = uid_sets[i] & uid_sets[j]
            print(f"  RG{i} ∩ RG{j}: {len(overlap)} shared users")

    # --- Global unique users across sampled RGs ---
    all_uids = set()
    for s in uid_sets:
        all_uids |= s
    total_rows_sampled = sum(
        pq.ParquetFile(f).metadata.row_group(rg).num_rows
        for f, rg in all_rgs[:max_rgs]
    )
    print(f"\nAcross {max_rgs} RGs: "
          f"{total_rows_sampled} rows, {len(all_uids)} unique users, "
          f"avg rows/user = {total_rows_sampled / max(len(all_uids), 1):.2f}")


if __name__ == '__main__':
    main()