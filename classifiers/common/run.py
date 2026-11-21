"""Shared command line for the classifier scripts: cross-validate, then optionally export.

Every family script (trees/train.py, logistic/train.py, mlp/train.py) and train_all.py call
main() with the families they cover. Output, file names and exported JSON are the same as the
runtime's former research/scb/train_classifier.py.
"""
import argparse
import json
import pickle
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from hcpose.engagement.classifier import FEATURE_NAMES, export_model, load_model

from . import mapping
from .families import FAMILIES, prior_offset, shift
from .samples import build


def within_image_auc(s, smp):
    """AUC over pairs of students in the same image only: a model can't score here by
    recognising which image (lesson) a student is in."""
    y, imgs = smp.y, smp.images
    wins = total = 0.0
    for im in set(imgs[smp.mixed]):
        m = imgs == im
        diff = s[m & y][:, None] - s[m & ~y][None, :]
        wins += (diff > 0).sum() + 0.5 * (diff == 0).sum()
        total += diff.size
    return wins / total if total else float("nan")


def cross_validate(fam, smp, folds=5):
    """Out-of-fold probabilities (balanced scale) and metrics, folds grouped by lesson."""
    oof = np.zeros(len(smp.y))
    for tr, te in GroupKFold(n_splits=folds).split(smp.X, smp.y, smp.groups):
        m = fam.make()
        m.fit(smp.X[tr], smp.y[tr])
        oof[te] = shift(m.predict_proba(smp.X[te])[:, 1], prior_offset(fam.kind, smp.y[tr]))
    metrics = {"auc": roc_auc_score(smp.y, oof), "within_image_auc": within_image_auc(oof, smp),
               "balanced_acc": balanced_accuracy_score(smp.y, oof >= 0.5)}  # all on the balanced scale
    return oof, metrics


def export_final(fam, smp, path, cv_metrics, args):
    """Fit on all data, export JSON for hcpose, and check the NumPy runtime reproduces sklearn."""
    t0 = time.time()
    final = fam.make().fit(smp.X, smp.y)
    off = prior_offset(fam.kind, smp.y)
    meta = {
        "name": args.name,
        "kind": fam.kind,
        "trained_on": ", ".join(Path(p).name for p in args.poses),
        "students": int(len(smp.y)),
        "lesson_groups": int(len(set(smp.groups))),
        "det_thr": args.det_thr,
        "cv": {k: round(float(v), 4) for k, v in cv_metrics.items()},
        "sklearn": sklearn.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
    }
    if args.mapping != "thesis":
        meta["mapping"] = args.mapping
    path = Path(path)
    previous = None
    if path.is_file():
        try:
            previous = load_model(str(path)).predict_proba(smp.X)
        except (ValueError, KeyError) as e:
            print(f"  (existing {path.name} not comparable: {e})")
    path.parent.mkdir(parents=True, exist_ok=True)
    export_model(final, str(path), meta, logit_offset=off)
    rt = load_model(str(path))
    want = shift(final.predict_proba(smp.X)[:, 1], off)
    err = np.abs(rt.predict_proba(smp.X) - want).max()
    note = ""
    if previous is not None:
        d = np.abs(previous - rt.predict_proba(smp.X)).max()
        note = f"; vs previous {path.name}: max |diff| {d:.1e}" + (" (identical)" if d < 1e-9 else "")
    print(f"exported {path} ({path.stat().st_size / 1e6:.2f} MB, {time.time() - t0:.1f}s); "
          f"NumPy runtime vs sklearn max |diff| = {err:.1e}{note}")
    assert err < 1e-5, f"exported {fam.kind} model does not reproduce sklearn"
    assert len(FEATURE_NAMES) == smp.X.shape[1]


def parser(description):
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", nargs="+", help="pose caches; several are pooled (e.g. with and without tiles)")
    ap.add_argument("--export-dir", metavar="DIR",
                    help="fit on all data and export scb_<kind>.json here (e.g. the runtime's hcpose/engagement/models)")
    ap.add_argument("--out", metavar="JSON", help="export to this file instead (single family only)")
    ap.add_argument("--det-thr", type=float, default=0.3, help="detection confidence for students (default 0.3)")
    ap.add_argument("--kpt-thr", type=float, default=0.5)
    ap.add_argument("--mapping", default="thesis", help="label mapping, see common/mapping.py (default thesis)")
    ap.add_argument("--name", default="scb", help="model name stored in the JSON")
    ap.add_argument("--save-cv", metavar="NPZ",
                    help="save out-of-fold predictions + labels + metrics (for ROC curves and report tables)")
    ap.add_argument("--save-models", metavar="DIR", help="also pickle the fitted sklearn models here (for the record)")
    return ap


def main(kinds, description, argv=None):
    args = parser(description).parse_args(argv)
    if args.out and len(kinds) > 1:
        raise SystemExit("--out takes one family; use --export-dir for several")
    focused, distracted = mapping.get(args.mapping)
    smp = build(args.poses, focused, distracted, args.det_thr, args.kpt_thr)
    print(f"{len(smp.y)} students ({len(args.poses)} pose cache(s)), {len(set(smp.groups))} lesson groups, "
          f"distracted {smp.y.mean():.1%}")

    fams = [FAMILIES[k] for k in FAMILIES if k in kinds]  # fixed order: logistic, trees, mlp
    print(f"\n{'model':24} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9}  (5-fold, grouped by lesson)")
    cv, oofs = {}, {}
    for fam in fams:
        oofs[fam.title], cv[fam.title] = cross_validate(fam, smp)
        m = cv[fam.title]
        print(f"{fam.title:24} {m['auc']:6.3f} {m['within_image_auc']:15.3f} {m['balanced_acc']:8.1%}")
        if fam.kind == "trees":
            oof = oofs[fam.title]
            print("   per behavior, share predicted distracted:",
                  ", ".join(f"{c} {np.mean(oof[smp.labels == c] >= 0.5):.0%}" for c in
                            ["read", "write", "hand-raising", "TurnHead", "BowHead", "discuss"]
                            if (smp.labels == c).any()))

    if args.save_cv:
        np.savez_compressed(
            args.save_cv, y=smp.y, labels=smp.labels, groups=smp.groups, images=smp.images, mixed_image=smp.mixed,
            feature_names=np.array(FEATURE_NAMES), X=smp.X,
            **{f"oof_{k.split(' ')[0].lower().replace('-', '_')}": v for k, v in oofs.items()},
        )
        Path(args.save_cv).with_suffix(".json").write_text(json.dumps(
            {"poses": args.poses, "det_thr": args.det_thr, "mapping": args.mapping, "students": int(len(smp.y)),
             "lesson_groups": int(len(set(smp.groups))), "metrics": cv}, indent=1))
        print(f"\nsaved out-of-fold predictions to {args.save_cv} and metrics to {Path(args.save_cv).with_suffix('.json')}")

    if args.save_models:
        d = Path(args.save_models)
        d.mkdir(parents=True, exist_ok=True)
        for fam in fams:
            m = fam.make().fit(smp.X, smp.y)
            fn = d / (fam.title.split(" ")[0].lower().replace("-", "_") + ".pkl")
            pickle.dump({"model": m, "feature_names": FEATURE_NAMES, "sklearn": sklearn.__version__}, open(fn, "wb"))
            print(f"saved {fn}")

    if args.export_dir or args.out:
        print()
    if args.export_dir:
        for fam in fams:
            export_final(fam, smp, Path(args.export_dir) / f"scb_{fam.kind}.json", cv[fam.title], args)
    elif args.out:
        export_final(fams[0], smp, args.out, cv[fams[0].title], args)
    return cv
