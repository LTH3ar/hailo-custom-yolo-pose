# verify_label.py
import random, os
from pathlib import Path
import sys

DATASET_DIR = Path(sys.argv[1])

label_dir = DATASET_DIR / "labels" / "train2017"
sample = random.choice(os.listdir(label_dir))
with open(os.path.join(label_dir, sample)) as f:
    for line in f:
        parts = line.strip().split()
        expected = 5 + 13 * 3  # = 44 values per row
        print(f"File: {sample}")
        print(f"  Values per row: {len(parts)}  (expected {expected})")
        assert len(parts) == expected, "❌ Mismatch! Check filter script."
        print("  ✅ Label looks correct")
        break