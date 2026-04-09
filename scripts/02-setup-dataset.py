from ultralytics.data.converter import convert_coco
from pathlib import Path
import shutil
import sys

DATASET_DIR = Path(sys.argv[1])

ANNOTATIONS_DIR = DATASET_DIR / "annotations"

for pattern in ["captions_*", "instances_*"]:
    for file_path in ANNOTATIONS_DIR.glob(pattern):
        if file_path.is_file():
            file_path.unlink()
            print(f"Deleted: {file_path}")

temp_out = DATASET_DIR.parent / (DATASET_DIR.name + "_temp")

convert_coco(
    labels_dir=ANNOTATIONS_DIR,
    save_dir=temp_out,
    use_keypoints=True,
    use_segments=False
)

# 2. Move the generated 'labels' folder back to your main dir
# Ultralytics creates a 'labels' folder inside the save_dir
generated_labels = temp_out / "labels"
if generated_labels.exists():
    dest = DATASET_DIR / "labels"
    if dest.exists():
        shutil.rmtree(dest) # Remove old YOLO labels if they exist
    shutil.move(str(generated_labels), str(dest))
    shutil.rmtree(temp_out) # Clean up temp


import os

# ─────────────────────────────────────────────
# CONFIG — adjust these if needed
# ─────────────────────────────────────────────
INPUT_DIRS = {
    "train": f"{str(DATASET_DIR)}/labels/person_keypoints_train2017",
    "val":   f"{str(DATASET_DIR)}/labels/person_keypoints_val2017",
}
OUTPUT_DIRS = {
    "train": f"{str(DATASET_DIR)}/labels/train2017",
    "val":   f"{str(DATASET_DIR)}/labels/val2017",
}

KEEP_KP_INDICES = list(range(13))  # 0,1,2,3,4,5,6,7,8,9,10,11,12
KP_DIMS = 3                        # x, y, visibility
TOTAL_KPS = 17                     # original COCO keypoints
# ─────────────────────────────────────────────


def filter_label_file(src_path, dst_path, keep_indices, kp_dims, total_kps):
    with open(src_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5 + total_kps * kp_dims:
            # skip malformed lines silently
            continue

        header = parts[:5]  # class cx cy w h
        kp_flat = parts[5:]

        # Reshape into list of [x, y, vis] per keypoint
        all_kps = [
            kp_flat[i * kp_dims : (i + 1) * kp_dims]
            for i in range(total_kps)
        ]

        # Keep only wanted keypoints
        filtered_kps = [all_kps[i] for i in keep_indices]
        flat_filtered = [v for kp in filtered_kps for v in kp]

        new_lines.append(" ".join(header + flat_filtered) + "\n")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    with open(dst_path, "w") as f:
        f.writelines(new_lines)

for split in ("train", "val"):
    src_dir = INPUT_DIRS[split]
    dst_dir = OUTPUT_DIRS[split]
    files = [f for f in os.listdir(src_dir) if f.endswith(".txt")]
    print(f"[{split}] Processing {len(files)} label files → {dst_dir}")

    for i, fname in enumerate(files):
        filter_label_file(
            src_path=os.path.join(src_dir, fname),
            dst_path=os.path.join(dst_dir, fname),
            keep_indices=KEEP_KP_INDICES,
            kp_dims=KP_DIMS,
            total_kps=TOTAL_KPS,
        )
        if (i + 1) % 10000 == 0:
            print(f"  {i+1}/{len(files)} done...")

    print(f"[{split}] ✅ Done. Output: {dst_dir}")