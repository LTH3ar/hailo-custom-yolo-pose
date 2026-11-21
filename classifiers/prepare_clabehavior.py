"""Lay out ClaBehavior like SCB, so the runtime's pose extractor reads it unchanged.

ClaBehavior ships COCO json annotations. The extractor (the runtime's research/scb/extract_hef.py)
reads DIR/images/<split>/<img> with YOLO labels in DIR/labels/<split>/<stem>.txt and a class
list. This script writes that layout (images as symlinks, nothing copied) and:
  - classes.txt   the class names in label-index order
  - groups.json   a camera-view group per image: ClaBehavior's frames come from fixed cameras
                  and its file names carry no lesson id, so frames that look alike (64-bit
                  difference hash within 4 bits, linked transitively) share a group, and a
                  camera's frames stay on one side of every split
It prints the extractor's DIR:classes argument.

Run where the images are (no model involved):
  python -m classifiers.prepare_clabehavior ~/datasets/clabehavior ~/datasets/clabehavior_yolo
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from classifiers.common.neardup import camera_groups

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def find_coco_jsons(src):
    out = []
    for p in sorted(src.rglob("*.json")):
        try:
            d = json.load(open(p))
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and {"images", "annotations", "categories"} <= set(d):
            out.append((p, d))
    return out


def split_name(json_path):
    n = json_path.stem.lower()
    for s in ("train", "val", "test"):
        if s in n:
            return s
    return json_path.stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="the downloaded ClaBehavior folder (with the COCO json files)")
    ap.add_argument("out", type=Path, help="output folder in SCB layout")
    ap.add_argument("--group-bits", type=int, default=4,
                    help="frames whose hashes differ in at most this many bits share a camera group (default 4)")
    args = ap.parse_args()
    src, out = args.src.expanduser().resolve(), args.out.expanduser().resolve()

    jsons = find_coco_jsons(src)
    if not jsons:
        raise SystemExit(f"no COCO json (images/annotations/categories) under {src}")
    by_name = {}
    for p in src.rglob("*"):
        if p.suffix.lower() in IMG_EXT:
            by_name.setdefault(p.name, p)

    # one class list across all json files, ordered by category id
    cats = {}
    for _, d in jsons:
        for c in d["categories"]:
            cats.setdefault(c["id"], c["name"])
    ids = sorted(cats)
    index = {cid: i for i, cid in enumerate(ids)}
    names = [cats[c] for c in ids]

    written, missing, boxes = [], 0, Counter()
    for jp, d in jsons:
        split = split_name(jp)
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        anns = {}
        for a in d["annotations"]:
            anns.setdefault(a["image_id"], []).append(a)
        for im in d["images"]:
            fn = Path(im["file_name"]).name
            cand = [jp.parent / im["file_name"], jp.parent / "images" / fn, jp.parent.parent / "images" / split / fn]
            path = next((c for c in cand if c.is_file()), by_name.get(fn))
            if path is None:
                missing += 1
                continue
            w, h = im.get("width"), im.get("height")
            if not w or not h:
                with Image.open(path) as pil:
                    w, h = pil.size
            link = out / "images" / split / fn
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(path)
            lines = []
            for a in anns.get(im["id"], []):
                x, y, bw, bh = a["bbox"]
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{index[a['category_id']]} {(x + bw / 2) / w:.6f} {(y + bh / 2) / h:.6f} "
                             f"{bw / w:.6f} {bh / h:.6f}")
                boxes[names[index[a['category_id']]]] += 1
            (out / "labels" / split / (Path(fn).stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
            written.append((split, fn, path))

    (out / "classes.txt").write_text("\n".join(names) + "\n")
    paths = [str(p) for _, _, p in written]
    groups = camera_groups(paths, max_bits=args.group_bits, cache_path=str(out / "dhash.json"))
    group_of = {fn: g for (_, fn, _), g in zip(written, groups)}
    (out / "groups.json").write_text(json.dumps(group_of, indent=0, sort_keys=True))

    sizes = sorted(Counter(groups).values(), reverse=True)
    print(f"{len(written)} images ({missing} listed in the json but not found) in {sorted({s for s, _, _ in written})}")
    print("boxes per class:", ", ".join(f"{k} {v}" for k, v in boxes.most_common()))
    print(f"{len(sizes)} camera groups (frames per group: {sizes})")
    print(f"\nextractor argument:\n  {out}:{','.join(names)}")


if __name__ == "__main__":
    main()
