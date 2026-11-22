"""Paired comparison of the crop baseline and the pose classifiers, one vote per student.

score.py and classifiers.train_all each report their own test set. They share folds and labels,
but not weights: train_all counts a student once per pose cache (found with and without tiles =
twice), and a box labelled twice appears twice in an older split.json. Here every labelled box
counts once:
  - pose classifiers: the out-of-fold probabilities of the pooled caches are averaged per box
  - crop model: duplicate entries of a box are averaged; a box whose labels disagree under the
    mapping (focused and distracted) is dropped
and both are scored on the boxes they share. The difference in AUC and in within-image AUC gets a
95% interval from a bootstrap over groups (lessons, camera views), since students of one lesson
are not independent.

No model runs here. Inputs: split.json and preds.jsonl from the crop baseline, and the
`classifiers.train_all --save-cv` file of each mapping, made with --folds split.json.

usage:
  python -m baselines.crop_cls.paired split.json preds.jsonl \\
      --cls thesis=cv_thesis.npz strict=cv_strict.npz lenient=cv_lenient.npz [--report paired.json]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from classifiers.common import mapping

from .score import p_distracted

FAMILY_KEYS = {"oof_gradient_boosted": "trees", "oof_mlp": "MLP", "oof_logistic": "logistic"}


def key(img, box):
    return (str(img), tuple(round(float(v), 1) for v in box))


def crop_scores(students, preds, focused, distracted):
    """{key: (y, p, group)} from the fold predictions, one entry per box."""
    acc = defaultdict(list)
    for p in preds:
        if not p["run"].startswith("fold"):
            continue
        s = students[p["id"]]
        if s["action"] not in focused | distracted:
            continue
        acc[key(s["img"], s["box"])].append((s["action"] in distracted,
                                             p_distracted(p["probs"], focused, distracted), s["group"], s["dataset"]))
    out = {}
    for k, v in acc.items():
        ys = {x[0] for x in v}
        if len(ys) == 1:
            out[k] = (v[0][0], float(np.mean([x[1] for x in v])), v[0][2], v[0][3])
    return out


def cls_scores(npz):
    """{family: {key: (y, p)}} averaged over the pooled caches."""
    z = np.load(npz, allow_pickle=True)
    keys = [key(i, b) for i, b in zip(z["images"], z["gt_boxes"])]
    out = {}
    for fk, name in FAMILY_KEYS.items():
        if fk not in z.files:
            continue
        acc = defaultdict(list)
        for k, y, p in zip(keys, z["y"], z[fk]):
            acc[k].append((bool(y), float(p)))
        out[name] = {k: (v[0][0], float(np.mean([x[1] for x in v]))) for k, v in acc.items()
                     if len({x[0] for x in v}) == 1}
    return out


def image_pairs(s, y, imgs):
    """Per image: (wins, pairs) of distracted-vs-focused comparisons, for the within-image AUC."""
    out = {}
    for im in set(imgs.tolist()):
        m = imgs == im
        if not (y[m].any() and (~y[m]).any()):
            continue
        d = s[m & y][:, None] - s[m & ~y][None, :]
        out[im] = ((d > 0).sum() + 0.5 * (d == 0).sum(), d.size)
    return out


def compare(y, a, b, groups, imgs, reps=1000, seed=0):
    """AUC and within-image AUC of scores a and b on the same students, and b - a with a group bootstrap."""
    pa, pb = image_pairs(a, y, imgs), image_pairs(b, y, imgs)
    ims = sorted(pa)
    img_group = {}
    for g, im in zip(groups, imgs):
        img_group.setdefault(im, g)
    G = sorted(set(groups.tolist()))
    gi = {g: i for i, g in enumerate(G)}
    sg = np.array([gi[g] for g in groups])
    ig = np.array([gi[img_group[im]] for im in ims])
    wa = np.array([pa[i][0] for i in ims]); ta = np.array([pa[i][1] for i in ims])
    wb = np.array([pb[i][0] for i in ims])

    def stats(w_students, w_imgs):
        r = {}
        for name, s, wins in (("a", a, wa), ("b", b, wb)):
            r[f"auc_{name}"] = roc_auc_score(y, s, sample_weight=w_students)
            r[f"within_{name}"] = (w_imgs * wins).sum() / (w_imgs * ta).sum()
        return r

    point = stats(np.ones(len(y)), np.ones(len(ims)))
    rng = np.random.default_rng(seed)
    diffs = {"auc": [], "within": []}
    for _ in range(reps):
        mult = np.bincount(rng.integers(0, len(G), len(G)), minlength=len(G)).astype(float)
        ws, wi = mult[sg], mult[ig]
        if ws[y].sum() == 0 or ws[~y].sum() == 0 or (wi * ta).sum() == 0:
            continue
        r = stats(ws, wi)
        diffs["auc"].append(r["auc_b"] - r["auc_a"])
        diffs["within"].append(r["within_b"] - r["within_a"])
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in diffs.items()}
    return {k: float(v) for k, v in point.items()}, ci


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("preds")
    ap.add_argument("--cls", nargs="+", required=True, metavar="MAPPING=NPZ")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--report", metavar="JSON")
    args = ap.parse_args(argv)

    S = json.load(open(args.split))
    students = {s["id"]: s for s in S["students"]}
    preds = [json.loads(line) for line in open(args.preds)]
    report = {}
    print(f"{'mapping':8} {'tested on':12} {'model':9} {'n':>6}  {'AUC crop / pose':>16}  {'diff [95% CI]':>22}  "
          f"{'within-img crop / pose':>23}  {'diff [95% CI]':>22}")
    for spec in args.cls:
        name, npz = spec.split("=", 1)
        focused, distracted = mapping.get(name)
        crop = crop_scores(students, preds, focused, distracted)
        fams = cls_scores(npz)
        parts = ["scb"] if name == "thesis" else ["pooled", "scb", "clabehavior"]
        for part in parts:
            for fam, pose in fams.items():
                ks = sorted(k for k in crop.keys() & pose.keys() if part == "pooled" or crop[k][3] == part)
                ks = [k for k in ks if crop[k][0] == pose[k][0]]
                if len(ks) < 50:
                    continue
                y = np.array([crop[k][0] for k in ks])
                a = np.array([crop[k][1] for k in ks])
                b = np.array([pose[k][1] for k in ks])
                groups = np.array([crop[k][2] for k in ks])
                imgs = np.array([k[0] for k in ks])
                pt, ci = compare(y, a, b, groups, imgs, args.reps)
                report[f"{name}/{part}/{fam}"] = {"n": len(ks), "groups": int(len(set(groups.tolist()))), **pt,
                                                  "ci_pose_minus_crop": ci}
                print(f"{name:8} {part:12} {fam:9} {len(ks):6d}  {pt['auc_a']:.3f} / {pt['auc_b']:.3f}  "
                      f"{pt['auc_b'] - pt['auc_a']:+.3f} [{ci['auc'][0]:+.3f}, {ci['auc'][1]:+.3f}]  "
                      f"{pt['within_a']:.3f} / {pt['within_b']:.3f}       "
                      f"{pt['within_b'] - pt['within_a']:+.3f} [{ci['within'][0]:+.3f}, {ci['within'][1]:+.3f}]")
    print("\ndiff = pose classifier minus crop model; an interval that excludes 0 is a real difference "
          "at the lesson level")
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
