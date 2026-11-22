"""Cut the student crops for the crop baseline and lay out one Ultralytics classification dataset
per training run. Run where the images are and the training happens (ubuntu3060); no model.

Crops are minors' images (SCB and ClaBehavior are handled like that): they stay on this machine,
never in a repo, a paper or a slide.

  OUT/crops/gt/<id>.jpg    the labelled box, padded by --pad on every side: training crops
  OUT/crops/det/<id>.jpg   the pose box hcpose found for the student: test crops (eval students)
  OUT/ds/<run>/{train,val}/<action>/<id>.jpg   symlinks to the gt crops, one folder per run:
      fold0..fold4   train on the other four folds (test = that fold, scored by score.py)
      all            every student: the model used on the classroom clips
      scb_only, cla_only   one dataset each, for the cross-dataset test

Ultralytics needs a val folder; it holds a 5% sample of the training students (every action
represented), so it overlaps training. Nothing is scored on it and predict.py uses the last
epoch's weights, never "best", so it cannot leak into a result.

The image paths in split.json are the ones on the machine that wrote it (pi5). --path-map OLD=NEW
rewrites a path prefix (repeatable); the run stops if any image is missing.

usage (ubuntu3060):
  python -m baselines.crop_cls.make_crops split.json ~/hcpose/crop_cls --path-map /home/pi=/home/me/hcpose/pi_data
"""
import argparse
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

RUNS_FOLDS = 5


def remap(path, maps):
    for old, new in maps:
        if path.startswith(old):
            return new + path[len(old):]
    return path


def crop(im, box, pad):
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    x1, y1 = max(0, x1 - pad * w), max(0, y1 - pad * h)
    x2, y2 = min(im.width, x2 + pad * w), min(im.height, y2 + pad * h)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return im.crop((int(x1), int(y1), int(round(x2)), int(round(y2))))


def runs_for(students):
    """{run name: the students it trains on}"""
    runs = {}
    for k in range(RUNS_FOLDS):
        runs[f"fold{k}"] = [s for s in students if s["fold"] != k]
    runs["all"] = list(students)
    runs["scb_only"] = [s for s in students if s["dataset"] == "scb"]
    runs["cla_only"] = [s for s in students if s["dataset"] == "clabehavior"]
    return {k: v for k, v in runs.items() if v}


def val_sample(train, frac=0.05, seed=0):
    """A small per-action sample, at least one per action, for Ultralytics' val folder."""
    rng = random.Random(seed)
    by = defaultdict(list)
    for s in train:
        by[s["action"]].append(s)
    out = []
    for act, ss in sorted(by.items()):
        out += rng.sample(ss, max(1, int(len(ss) * frac)))
    return out


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    os.symlink(src, dst)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("out", type=Path)
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--pad", type=float, default=0.1, help="padding around each box, as a share of its size (0.1)")
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args(argv)
    maps = [tuple(m.split("=", 1)) for m in args.path_map]

    S = json.load(open(args.split))
    students = S["students"]
    gt_dir, det_dir = args.out / "crops" / "gt", args.out / "crops" / "det"
    gt_dir.mkdir(parents=True, exist_ok=True)
    det_dir.mkdir(parents=True, exist_ok=True)

    by_img = defaultdict(list)
    for s in students:
        by_img[s["img"]].append(s)
    missing, small, kept = [], 0, []
    for i, (img, ss) in enumerate(sorted(by_img.items())):
        p = remap(img, maps)
        if not os.path.isfile(p):
            missing.append(p)
            continue
        with Image.open(p) as im:
            im = im.convert("RGB")
            for s in ss:
                g = gt_dir / f"{s['id']}.jpg"
                if not g.exists():
                    c = crop(im, s["box"], args.pad)
                    if c is None:
                        small += 1
                        continue
                    c.save(g, quality=args.quality)
                kept.append(s)
                if s.get("eval") and s.get("det_box"):
                    d = det_dir / f"{s['id']}.jpg"
                    if not d.exists():
                        c = crop(im, s["det_box"], args.pad)
                        if c is not None:
                            c.save(d, quality=args.quality)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(by_img)} images")
    if missing:
        raise SystemExit(f"{len(missing)} images not found, e.g. {missing[:3]}: check --path-map")

    for name, train in runs_for(kept).items():
        root = args.out / "ds" / name
        val = val_sample(train)
        for part, ss in (("train", train), ("val", val)):
            for s in ss:
                link((gt_dir / f"{s['id']}.jpg").resolve(), root / part / s["action"] / f"{s['id']}.jpg")
        print(f"{name}: {len(train)} training crops {dict(Counter(s['action'] for s in train).most_common())}")
    n_det = sum(1 for _ in det_dir.glob("*.jpg"))
    print(f"{len(kept)} gt crops ({small} boxes too small), {n_det} pose-box crops for testing -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
