"""Remove observation.bboxes from an existing bbox dataset.

Usage:
    python fix_bbox_dataset.py ./data/DS1_1B1O_FF_rand40_bbox
"""
import sys
import json
from pathlib import Path

import pyarrow.parquet as pq
import pyarrow as pa

ds_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./data/DS1_1B1O_FF_rand40_bbox")

if not ds_dir.exists():
    print(f"Dataset not found: {ds_dir}")
    sys.exit(1)

# 1. Remove observation.bboxes from parquet files
data_dir = ds_dir / "data"
fixed = 0
for pf in sorted(data_dir.rglob("*.parquet")):
    tbl = pq.read_table(str(pf))
    if "observation.bboxes" in tbl.column_names:
        tbl = tbl.drop("observation.bboxes")
        pq.write_table(tbl, str(pf))
        fixed += 1
        print(f"  Fixed: {pf}")

# 2. Remove observation.bboxes from info.json
info_path = ds_dir / "meta" / "info.json"
if info_path.exists():
    info = json.loads(info_path.read_text())
    if "observation.bboxes" in info.get("features", {}):
        del info["features"]["observation.bboxes"]
        info_path.write_text(json.dumps(info, indent=4))
        print(f"  Fixed: {info_path}")

print(f"Done — fixed {fixed} parquet file(s)")
