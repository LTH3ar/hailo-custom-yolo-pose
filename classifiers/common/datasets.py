"""Load pose caches of any supported dataset into one record format.

Every record is a pose-cache line (img, gt, dets, ...) plus:
  dataset  "scb" or "clabehavior"
  group    the split group, prefixed with the dataset ("scb:12", "clabehavior:3"). Records in
           one group always land on the same side of a train/test split.

SCB: byte-identical copies across subsets are merged (scb.py, as in the thesis). Pictures saved
twice but not byte-identical (re-encoded or resized) are not merged, but put in the same group,
so they can't leak across a split. Groups are lessons, from the file names.

ClaBehavior: groups are fixed-camera views from groups.json, written by prepare_clabehavior.py
next to the images.
"""
import json
import os
from pathlib import Path

from . import scb
from .neardup import UnionFind, same_picture_pairs

DATASETS = ("scb", "clabehavior")


def load_scb(path, near_dups=True, verbose=True):
    recs = scb.load(path, verbose=verbose)
    if near_dups:
        imgs = [r["img"] for r in recs]
        cache = os.path.join(os.path.dirname(os.path.abspath(path)), "scb.dhash.json")  # shared by all SCB caches
        pairs = same_picture_pairs(imgs, cache_path=cache)
        index = {im: i for i, im in enumerate(imgs)}
        ids = sorted({r["group"] for r in recs})
        pos = {g: i for i, g in enumerate(ids)}
        uf = UnionFind(len(ids))
        for a, b in pairs:
            uf.union(pos[recs[index[a]]["group"]], pos[recs[index[b]]["group"]])
        lab = uf.labels()
        for r in recs:
            r["group"] = lab[pos[r["group"]]]
        if verbose:
            print(f"[scb] {len(pairs)} same-picture pairs that are not byte-identical; "
                  f"{len(ids)} lesson groups -> {len(set(lab))} after joining them")
    for r in recs:
        r["dataset"] = "scb"
        r["group"] = f"scb:{r['group']}"
    return recs


def load_clabehavior(path, verbose=True):
    recs = [json.loads(line) for line in open(path)]
    groups, root = None, None
    for r in recs:
        here = Path(r["img"]).parents[2]
        if here != root:
            root = here
            gp = root / "groups.json"
            if not gp.is_file():
                raise SystemExit(f"{gp} not found: run classifiers.prepare_clabehavior first")
            groups = json.load(open(gp))
        r["dataset"] = "clabehavior"
        r["group"] = f"clabehavior:{groups[Path(r['img']).name]}"
    if verbose:
        print(f"[clabehavior] {len(recs)} images, {len({r['group'] for r in recs})} camera groups")
    return recs


def load(dataset, path, verbose=True, near_dups=True):
    if dataset == "scb":
        return load_scb(path, near_dups=near_dups, verbose=verbose)
    if dataset == "clabehavior":
        return load_clabehavior(path, verbose=verbose)
    raise SystemExit(f"unknown dataset {dataset!r}; known: {', '.join(DATASETS)}")
