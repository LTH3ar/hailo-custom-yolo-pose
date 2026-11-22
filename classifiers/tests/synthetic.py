"""Synthetic test data in the real formats: an SCB-style pose cache, and a ClaBehavior-style COCO
dataset (with a matching pose cache once it has been laid out by prepare_clabehavior).

Distracted students have their nose pushed towards one shoulder. Images are smooth random
pictures, so the perceptual-hash checks behave as on photos:
  - SCB: the first image of each lesson also appears byte-identical in a second subset, and
    lesson 1's first image appears resized and re-encoded in the last lesson (a near duplicate)
  - ClaBehavior: frames come from three fixed cameras (the same picture plus small noise)
"""
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

SCB_F = ["read", "write", "hand-raising"]
SCB_D = ["TurnHead", "BowHead", "discuss"]
CLA_F = ["read", "write", "raise_hand", "lookup"]
CLA_D = ["turn_head", "discuss", "stand"]
CLA_CATS = ["lookup", "read", "turn_head", "raise_hand", "write", "discuss", "stand"]


def picture(rng, w=640, h=360, grid=(5, 6)):
    small = (rng.random((*grid, 3)) * 255).astype(np.uint8)
    return Image.fromarray(small).resize((w, h), Image.BILINEAR)


def person(rng, x, y, distracted):
    w, h = 60.0, 80.0
    cx = x + w / 2
    dx = rng.choice([-1, 1]) * rng.uniform(9, 15) if distracted else rng.gauss(0, 2)
    drop = rng.uniform(4, 10) if distracted and rng.random() < 0.3 else rng.gauss(0, 1.5)
    nose = (cx + dx, y + 15 + drop)
    k = [
        nose,
        (nose[0] - 5, nose[1] - 3), (nose[0] + 5, nose[1] - 3),   # eyes
        (cx - 12, y + 15), (cx + 12, y + 15),                      # ears
        (cx - 18, y + 35), (cx + 18, y + 35),                      # shoulders
        (cx - 22, y + 55), (cx + 22, y + 55),                      # elbows
        (cx - 15, y + 70), (cx + 15, y + 70),                      # wrists
        (cx - 12, y + 78), (cx + 12, y + 78),                      # hips
    ]
    kpts = []
    for i, (a, b) in enumerate(k):
        c = 0.9
        if distracted and i in (3, 4) and abs(dx) > 12 and (i == 3) == (dx > 0):
            c = 0.05  # far ear hidden when the head is turned
        kpts.append([round(a + rng.gauss(0, 0.8), 1), round(b + rng.gauss(0, 0.8), 1), c])
    return [x, y, x + w, y + h], kpts


def students(rng, per_image, focused, distracted, p_distracted=0.35):
    gt, dets = [], []
    for s in range(per_image):
        dis = rng.random() < p_distracted
        label = rng.choice(distracted if dis else focused)
        box, kpts = person(rng, 20 + 80 * s, 40 + 10 * (s % 2), dis)
        gt.append([label] + [b + rng.uniform(-2, 2) for b in box])
        dets.append({"box": box, "score": round(rng.uniform(0.4, 0.95), 3), "kpts": kpts})
    return gt, dets


def write_cache(root, n_lessons=12, per_lesson=5, per_image=6, seed=0):
    """SCB-style pose cache with real image files. Returns the cache path."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    root = Path(root)
    (root / "img").mkdir(parents=True, exist_ok=True)
    cache = root / "poses.jsonl"
    first_pic = None
    with open(cache, "w") as out:
        for lesson in range(1, n_lessons + 1):
            for j in range(per_lesson):
                gt, dets = students(rng, per_image, SCB_F, SCB_D)
                img = root / "img" / f"{lesson}_{j:06d}.jpg"
                pic = picture(nrng)
                pic.save(img, quality=95)
                rec = {"img": str(img), "subset": "SCB5-Handrise-Read-write", "w": 640, "h": 360, "gt": gt, "dets": dets}
                out.write(json.dumps(rec) + "\n")
                if j == 0:  # byte-identical copy of each lesson's first image in a second subset
                    copy = root / "img" / f"{lesson}_{j:06d}_copy.jpg"
                    copy.write_bytes(img.read_bytes())
                    out.write(json.dumps(dict(rec, img=str(copy), subset="SCB5-Turn-Bow-Head",
                                              gt=[g for g in gt if g[0] in SCB_D])) + "\n")
                    if lesson == 1:
                        first_pic = pic
            if lesson == n_lessons:  # lesson 1's first picture, resized and re-encoded, filed under the last lesson
                near = root / "img" / f"{lesson}_{per_lesson:06d}.jpg"
                first_pic.resize((600, 338), Image.BILINEAR).save(near, quality=70)
                gt, dets = students(rng, per_image, SCB_F, SCB_D)
                out.write(json.dumps({"img": str(near), "subset": "SCB5-Handrise-Read-write", "w": 600, "h": 338,
                                      "gt": gt, "dets": dets}) + "\n")
    return cache


def write_clabehavior(root, cameras=3, frames=12, per_image=6, seed=1):
    """A ClaBehavior-style folder: COCO json (train and val) and images from fixed cameras."""
    rng, nrng = random.Random(seed), np.random.default_rng(seed)
    root = Path(root)
    (root / "coco" / "images").mkdir(parents=True, exist_ok=True)
    bases = [np.asarray(picture(nrng), dtype=np.int16) for _ in range(cameras)]
    splits = {"train": {"images": [], "annotations": []}, "val": {"images": [], "annotations": []}}
    truth = {}
    ann_id = 1
    for c in range(cameras):
        for f in range(frames):
            fn = f"cam{c}_{f:04d}.jpg"
            arr = np.clip(bases[c] + nrng.integers(-3, 4, bases[c].shape), 0, 255)
            for _ in range(4):  # students who move between frames: they break a fine hash, not the background
                x, y = int(nrng.integers(0, 580)), int(nrng.integers(60, 300))
                arr[y:y + 60, x:x + 45] = nrng.integers(0, 256, 3)
            arr = arr.astype(np.uint8)
            Image.fromarray(arr).save(root / "coco" / "images" / fn, quality=92)
            split = "val" if f % 6 == 0 else "train"
            iid = len(splits["train"]["images"]) + len(splits["val"]["images"]) + 1
            splits[split]["images"].append({"id": iid, "file_name": fn, "width": 640, "height": 360})
            gt, dets = students(rng, per_image, CLA_F, CLA_D)
            truth[fn] = (gt, dets)
            for g in gt:
                x1, y1, x2, y2 = g[1:]
                splits[split]["annotations"].append({"id": ann_id, "image_id": iid, "bbox": [x1, y1, x2 - x1, y2 - y1],
                                                     "category_id": CLA_CATS.index(g[0]) + 1, "area": (x2 - x1) * (y2 - y1),
                                                     "iscrowd": 0})
                ann_id += 1
    cats = [{"id": i + 1, "name": n} for i, n in enumerate(CLA_CATS)]
    for split, d in splits.items():
        json.dump(dict(d, categories=cats), open(root / "coco" / f"instances_{split}.json", "w"))
    return root, truth


def write_clabehavior_cache(prepared, truth, out_path):
    """A pose cache over a prepared (SCB-layout) ClaBehavior folder, as the extractor would write it."""
    prepared = Path(prepared)
    with open(out_path, "w") as out:
        for img in sorted(prepared.glob("images/*/*.jpg")):
            gt, dets = truth[img.name]
            out.write(json.dumps({"img": str(img), "subset": prepared.name, "w": 640, "h": 360,
                                  "gt": gt, "dets": dets, "tiles": 0}) + "\n")
    return out_path
