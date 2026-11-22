"""Score the crop baseline like the pose classifiers: AUC, within-image AUC, balanced accuracy.

No model runs here (any machine). The crop model predicts the 8 shared actions; a mapping turns
that into a distracted probability:
    p = sum of P(distracted actions) / sum of P(focused + distracted actions)
and a student counts as flagged when p >= 0.5, i.e. when the model's own choice among the mapped
actions is a distracted one. Students whose action the mapping leaves out are not scored.

  thesis   SCB students only, bow head and discuss distracted (the thesis's SCB table)
  strict / lenient   both datasets, pooled and per dataset; plus the cross-dataset lines
                     (scb_only on ClaBehavior, cla_only on SCB)

The within-image AUC compares students of the same image only (same code as the classifiers).

usage:  python -m baselines.crop_cls.score split.json preds.jsonl [--report crop_cls.json]
"""
import argparse
import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from classifiers.common import mapping


def within_image_auc(s, y, imgs):
    wins = total = 0.0
    for im in set(imgs.tolist()):
        m = imgs == im
        if not (y[m].any() and (~y[m]).any()):
            continue
        diff = s[m & y][:, None] - s[m & ~y][None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += diff.size
    return wins / total if total else float("nan")


def p_distracted(probs, focused, distracted):
    d = sum(probs.get(a, 0.0) for a in distracted)
    f = sum(probs.get(a, 0.0) for a in focused)
    return d / (d + f) if d + f > 0 else 0.5


def metrics(rows, focused, distracted):
    rows = [r for r in rows if r["action"] in focused | distracted]
    if not rows:
        return None
    y = np.array([r["action"] in distracted for r in rows])
    s = np.array([p_distracted(r["probs"], focused, distracted) for r in rows])
    imgs = np.array([r["img"] for r in rows])
    two = y.any() and (~y).any()
    return {"n": len(rows), "true_distracted": float(y.mean()), "pred_distracted": float((s >= 0.5).mean()),
            "auc": float(roc_auc_score(y, s)) if two else float("nan"),
            "within_image_auc": float(within_image_auc(s, y, imgs)),
            "balanced_acc": float(balanced_accuracy_score(y, s >= 0.5)) if two else float("nan")}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("preds", nargs="+")
    ap.add_argument("--report", metavar="JSON")
    args = ap.parse_args(argv)

    S = json.load(open(args.split))
    st = {s["id"]: s for s in S["students"]}
    by_run = defaultdict(list)
    for f in args.preds:
        for line in open(f):
            p = json.loads(line)
            s = st[p["id"]]
            by_run[p["run"]].append(dict(s, probs=p["probs"]))
    cv = [r for run, rows in by_run.items() if run.startswith("fold") for r in rows]
    n_eval = sum(1 for s in S["students"] if s.get("eval"))
    print(f"{len(cv)} of {n_eval} evaluated students have a fold prediction")

    report = {"cv": {}, "cross": {}}
    print(f"\n{'mapping':9} {'tested on':12} {'n':>7} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9}")
    for name in ("thesis", "strict", "lenient"):
        focused, distracted = mapping.get(name)
        parts = {"scb": [r for r in cv if r["dataset"] == "scb"]} if name == "thesis" else \
            {"pooled": cv, "scb": [r for r in cv if r["dataset"] == "scb"],
             "clabehavior": [r for r in cv if r["dataset"] == "clabehavior"]}
        for part, rows in parts.items():
            m = metrics(rows, focused, distracted)
            if m:
                report["cv"][f"{name}/{part}"] = m
                print(f"{name:9} {part:12} {m['n']:7d} {m['auc']:6.3f} {m['within_image_auc']:15.3f} "
                      f"{m['balanced_acc']:8.1%}")
    print(f"\n{'mapping':9} {'train -> test':28} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9} "
          f"{'pred / true distracted':>23}")
    for name in ("strict", "lenient"):
        focused, distracted = mapping.get(name)
        for run, label in (("cla_only", "clabehavior -> scb"), ("scb_only", "scb -> clabehavior")):
            m = metrics(by_run.get(run, []), focused, distracted)
            if m:
                report["cross"][f"{name}/{label}"] = m
                print(f"{name:9} {label:28} {m['auc']:6.3f} {m['within_image_auc']:15.3f} {m['balanced_acc']:8.1%} "
                      f"{m['pred_distracted']:15.0%} / {m['true_distracted']:.0%}")
    if args.report:
        json.dump(report, open(args.report, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
