"""Write a small synthetic SCB-style pose cache (and its image files) for tests.

Same record format as the runtime's extract_hef.py: one JSON line per image with the subset,
the ground-truth boxes and the detections with 13 keypoints. Distracted students have their
nose pushed towards one shoulder; some images are byte-identical copies in a second subset.
"""
import json
import random
from pathlib import Path

LABELS_F = ["read", "write", "hand-raising"]
LABELS_D = ["TurnHead", "BowHead", "discuss"]


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
    box = [x, y, x + w, y + h]
    return box, kpts


def write_cache(root, n_lessons=12, per_lesson=5, per_image=6, seed=0):
    rng = random.Random(seed)
    root = Path(root)
    (root / "img").mkdir(parents=True, exist_ok=True)
    cache = root / "poses.jsonl"
    with open(cache, "w") as out:
        for lesson in range(1, n_lessons + 1):
            for j in range(per_lesson):
                name = f"{lesson}_{j:06d}.jpg"
                gt, dets = [], []
                for s in range(per_image):
                    distracted = rng.random() < 0.35
                    label = rng.choice(LABELS_D if distracted else LABELS_F)
                    box, kpts = person(rng, 20 + 80 * s, 40 + 10 * (s % 2), distracted)
                    gt.append([label] + [b + rng.uniform(-2, 2) for b in box])
                    dets.append({"box": box, "score": round(rng.uniform(0.4, 0.95), 3), "kpts": kpts})
                img = root / "img" / name
                img.write_bytes(f"synthetic image {lesson} {j}".encode())
                rec = {"img": str(img), "subset": "SCB5-Handrise-Read-write", "w": 640, "h": 360, "gt": gt, "dets": dets}
                out.write(json.dumps(rec) + "\n")
                if j == 0:  # a byte-identical copy of the first image of each lesson in another subset
                    copy = root / "img" / f"{lesson}_{j:06d}_copy.jpg"
                    copy.write_bytes(img.read_bytes())
                    rec2 = dict(rec, img=str(copy), subset="SCB5-Turn-Bow-Head",
                                gt=[g for g in gt if g[0] in LABELS_D])
                    out.write(json.dumps(rec2) + "\n")
    return cache
