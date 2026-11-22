"""Shared command line for the classifier scripts: cross-validate, then optionally export.

Every family script (trees/train.py, logistic/train.py, mlp/train.py) and train_all.py call
main() with the families they cover. With SCB caches only, the thesis mapping and
--no-near-dups, output, file names and exported JSON are the same as the runtime's former
research/scb/train_classifier.py.

  pooled cross-validation  5 folds grouped by lesson / camera view, over every given dataset,
                           with the out-of-fold scores also broken down per dataset
  --cross                  train on all of one dataset, test on all of the other, both ways:
                           does what is learned on one dataset carry over to another?
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

PRINT_ACTIONS = mapping.ACTIONS


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


def fold_splits(smp, folds=5, fold_map=None):
    """(train, test) index pairs. Default: GroupKFold over the groups. With fold_map (group name ->
    fold, e.g. from a baseline's split.json), every model is tested on exactly the same students."""
    if fold_map is None:
        return list(GroupKFold(n_splits=folds).split(smp.X, smp.y, smp.groups))
    missing = sorted({g for g in smp.group_names if g not in fold_map})
    if missing:
        raise SystemExit(f"{len(missing)} groups are not in the fold map (e.g. {missing[:3]})")
    f = np.array([fold_map[g] for g in smp.group_names])
    return [(np.nonzero(f != k)[0], np.nonzero(f == k)[0]) for k in sorted(set(f.tolist()))]


def cross_validate(fam, smp, folds=5, fold_map=None):
    """Out-of-fold probabilities (balanced scale) and metrics, folds grouped by lesson."""
    oof = np.zeros(len(smp.y))
    for tr, te in fold_splits(smp, folds, fold_map):
        m = fam.make()
        m.fit(smp.X[tr], smp.y[tr])
        oof[te] = shift(m.predict_proba(smp.X[te])[:, 1], prior_offset(fam.kind, smp.y[tr]))
    return oof, scores(oof, smp)


def scores(prob, smp):
    """Metrics of probabilities on the balanced scale (0.5 = the decision threshold)."""
    two = smp.y.any() and (~smp.y).any()
    return {"auc": roc_auc_score(smp.y, prob) if two else float("nan"),
            "within_image_auc": within_image_auc(prob, smp),
            "balanced_acc": balanced_accuracy_score(smp.y, prob >= 0.5) if two else float("nan"),
            "pred_distracted": float(np.mean(prob >= 0.5)), "true_distracted": float(np.mean(smp.y)),
            "students": int(len(smp.y))}


def cross_dataset(fam, train, test):
    """Fit on one dataset, score on another."""
    m = fam.make().fit(train.X, train.y)
    return scores(shift(m.predict_proba(test.X)[:, 1], prior_offset(fam.kind, train.y)), test)


def export_final(fam, smp, path, cv_metrics, args):
    """Fit on all data, export JSON for hcpose, and check the NumPy runtime reproduces sklearn."""
    t0 = time.time()
    final = fam.make().fit(smp.X, smp.y)
    off = prior_offset(fam.kind, smp.y)
    meta = {
        "name": args.name,
        "kind": fam.kind,
        "trained_on": ", ".join(Path(p).name for _, p in args.sources),
        "students": int(len(smp.y)),
        "lesson_groups": int(len(set(smp.groups))),
        "det_thr": args.det_thr,
        "cv": {k: round(float(cv_metrics[k]), 4) for k in ("auc", "within_image_auc", "balanced_acc")},
        "sklearn": sklearn.__version__,
        "created": time.strftime("%Y-%m-%d %H:%M"),
        "host": platform.node(),
    }
    if args.mapping != "thesis":
        meta["mapping"] = args.mapping
    if set(smp.datasets) != {"scb"}:
        meta["datasets"] = sorted(set(smp.datasets.tolist()))
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
    ap.add_argument("poses", nargs="*", help="SCB pose caches; several are pooled (e.g. with and without tiles)")
    ap.add_argument("--clabehavior", nargs="+", default=[], metavar="CACHE",
                    help="ClaBehavior pose caches (laid out with classifiers.prepare_clabehavior)")
    ap.add_argument("--export-dir", metavar="DIR",
                    help="fit on all data and export scb_<kind>.json here (e.g. the runtime's hcpose/engagement/models)")
    ap.add_argument("--out", metavar="JSON", help="export to this file instead (single family only)")
    ap.add_argument("--det-thr", type=float, default=0.3, help="detection confidence for students (default 0.3)")
    ap.add_argument("--kpt-thr", type=float, default=0.5)
    ap.add_argument("--mapping", default="thesis",
                    help="label mapping: thesis (SCB only, the default), strict or lenient; see common/mapping.py")
    ap.add_argument("--cross", action="store_true", help="also train on each dataset and test on the other(s)")
    ap.add_argument("--no-near-dups", action="store_true",
                    help="don't join SCB pictures saved twice into one group (reproduces the thesis's folds)")
    ap.add_argument("--report", metavar="JSON", help="write every metric to this file")
    ap.add_argument("--name", default="scb", help="model name stored in the JSON")
    ap.add_argument("--save-cv", metavar="NPZ",
                    help="save out-of-fold predictions + labels + metrics (for ROC curves and report tables)")
    ap.add_argument("--save-models", metavar="DIR", help="also pickle the fitted sklearn models here (for the record)")
    ap.add_argument("--folds", metavar="SPLIT_JSON",
                    help="use the fold of each group from this file (baselines/crop_cls/export_split.py) instead of "
                         "GroupKFold, so the classifiers and the crop baseline are tested on the same students")
    return ap


def main(kinds, description, argv=None):
    args = parser(description).parse_args(argv)
    if args.out and len(kinds) > 1:
        raise SystemExit("--out takes one family; use --export-dir for several")
    args.sources = [("scb", p) for p in args.poses] + [("clabehavior", p) for p in args.clabehavior]
    if not args.sources:
        raise SystemExit("no pose caches given")
    if args.mapping == "thesis" and args.clabehavior:
        raise SystemExit("the thesis mapping is SCB-only; use --mapping strict or lenient with ClaBehavior")
    focused, distracted = mapping.get(args.mapping)
    smp = build(args.sources, focused, distracted, args.det_thr, args.kpt_thr, near_dups=not args.no_near_dups)
    fold_map = json.load(open(args.folds))["folds"] if args.folds else None
    names = sorted(set(smp.datasets.tolist()))
    print(f"{len(smp.y)} students ({len(args.sources)} pose cache(s)), {len(set(smp.groups))} lesson groups, "
          f"distracted {smp.y.mean():.1%}" + (f"; mapping {args.mapping}" if args.mapping != "thesis" else ""))
    if len(names) > 1:
        for n in names:
            m = smp.datasets == n
            print(f"  {n}: {m.sum()} students, {len(set(smp.groups[m]))} groups, distracted {smp.y[m].mean():.1%}")

    fams = [FAMILIES[k] for k in FAMILIES if k in kinds]  # fixed order: logistic, trees, mlp
    print(f"\n{'model':24} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9}  (5-fold, grouped by lesson)")
    cv, oofs = {}, {}
    for fam in fams:
        oofs[fam.title], cv[fam.title] = cross_validate(fam, smp, fold_map=fold_map)
        m = cv[fam.title]
        print(f"{fam.title:24} {m['auc']:6.3f} {m['within_image_auc']:15.3f} {m['balanced_acc']:8.1%}")
        if len(names) > 1:
            for n in names:
                d = scores(oofs[fam.title][smp.datasets == n], smp.subset(smp.datasets == n))
                cv[fam.title][f"on_{n}"] = d
                print(f"  {'  on ' + n:22} {d['auc']:6.3f} {d['within_image_auc']:15.3f} {d['balanced_acc']:8.1%}")
        if fam.kind == "trees":
            oof = oofs[fam.title]
            print("   per behavior, share predicted distracted:",
                  ", ".join(f"{c} {np.mean(oof[smp.labels == c] >= 0.5):.0%}" for c in PRINT_ACTIONS
                            if (smp.labels == c).any()))

    cross = {}
    if args.cross:
        if len(names) < 2:
            raise SystemExit("--cross needs at least two datasets")
        print(f"\n{'train -> test':28} {'model':24} {'AUC':>6} {'within-img AUC':>15} {'bal. acc':>9} "
              f"{'pred / true distracted':>23}")
        for a in names:
            for b in names:
                if a == b:
                    continue
                for fam in fams:
                    r = cross_dataset(fam, smp.subset(smp.datasets == a), smp.subset(smp.datasets == b))
                    cross[f"{a}->{b}:{fam.kind}"] = r
                    print(f"{a + ' -> ' + b:28} {fam.title:24} {r['auc']:6.3f} {r['within_image_auc']:15.3f} "
                          f"{r['balanced_acc']:8.1%} {r['pred_distracted']:11.0%} / {r['true_distracted']:.0%}")

    if args.report:
        Path(args.report).write_text(json.dumps(
            {"sources": args.sources, "mapping": args.mapping, "det_thr": args.det_thr,
             "near_dups": not args.no_near_dups, "students": int(len(smp.y)),
             "groups": int(len(set(smp.groups))), "pooled_cv": cv, "cross": cross}, indent=1))
        print(f"\nwrote {args.report}")

    if args.save_cv:
        np.savez_compressed(
            args.save_cv, y=smp.y, labels=smp.labels, groups=smp.groups, images=smp.images, mixed_image=smp.mixed,
            datasets=smp.datasets, group_names=smp.group_names, gt_boxes=smp.gt_boxes,
            feature_names=np.array(FEATURE_NAMES), X=smp.X,
            **{f"oof_{k.split(' ')[0].lower().replace('-', '_')}": v for k, v in oofs.items()},
        )
        Path(args.save_cv).with_suffix(".json").write_text(json.dumps(
            {"poses": [p for _, p in args.sources], "det_thr": args.det_thr, "mapping": args.mapping,
             "students": int(len(smp.y)),
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
