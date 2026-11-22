"""Write split.json: every labelled student of SCB and ClaBehavior, with its fold and the pose box
hcpose found for it. Run where the pose caches are (pi5); no model is involved.

The crop baseline (YOLOv8s-cls on student crops) and the pose classifiers must be tested on the
same students in the same folds. This script fixes both:

  folds     group name -> fold (0-4). Computed with GroupKFold over the students the classifiers
            use (front view, matched to a pose detection, every action of the lenient mapping), so
            it is the classifiers' own grouping. Groups with no such student (side views, images
            where pose found nobody) are added to the fold with the fewest students: they only
            serve as training data for the crop model.
  students  one entry per labelled student (byte-identical SCB copies merged, as everywhere):
            dataset, image path, group, fold, shared action name, labelled box, and, if the
            classifiers use this student ("eval"), the matched pose box ("det_box"). The crop
            model is trained on labelled boxes and tested on pose boxes, as it would run live.

Pass the caches exactly as for classifiers.train_all (with and without tiles are pooled; a student
found in either is evaluated, once).

usage (pi5):
  python -m baselines.crop_cls.export_split ~/hcpose_paper/poses/scb_m.jsonl ~/hcpose_paper/poses/scb_m_tiles.jsonl \\
      --clabehavior ~/hcpose_paper/poses/cla_m.jsonl ~/hcpose_paper/poses/cla_m_tiles.jsonl -o split.json
"""
import argparse
import json
from collections import Counter

import numpy as np

from classifiers.common import mapping
from classifiers.common.datasets import load
from classifiers.common.run import fold_splits
from classifiers.common.samples import build

FOLDS = 5


def box_key(img, box):
    return (img, tuple(round(float(v), 1) for v in box))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("poses", nargs="+", help="SCB pose caches (pooled)")
    ap.add_argument("--clabehavior", nargs="*", default=[], metavar="CACHE")
    ap.add_argument("--det-thr", type=float, default=0.3)
    ap.add_argument("--kpt-thr", type=float, default=0.5)
    ap.add_argument("-o", "--out", default="split.json")
    args = ap.parse_args(argv)

    sources = [("scb", p) for p in args.poses] + [("clabehavior", p) for p in args.clabehavior]
    focused, distracted = mapping.get("lenient")  # every action, so the eval set covers all mappings
    smp = build(sources, focused, distracted, args.det_thr, args.kpt_thr)

    folds = {}
    for k, (_, te) in enumerate(fold_splits(smp, FOLDS)):
        for g in set(smp.group_names[te].tolist()):
            folds[g] = k
    det_of = {}
    for img, gb, db in zip(smp.images, smp.gt_boxes, smp.det_boxes):
        det_of.setdefault(box_key(img, gb), [round(float(v), 1) for v in db])

    sample_action = {box_key(i, b): a for i, b, a in zip(smp.images, smp.gt_boxes, smp.labels)}
    students, seen, dup = [], set(), 0
    for dataset, path in sources:
        if dataset in seen:  # pooled caches hold the same images: list each student once
            continue
        seen.add(dataset)
        for d in load(dataset, path, verbose=False):
            boxes = set()
            for g in d["gt"]:
                act = mapping.action(dataset, g[0])
                if act is None:
                    continue
                k = box_key(d["img"], g[1:5])
                if k in boxes:  # the same box labelled twice (e.g. raise_hand and read): keep one entry
                    dup += 1
                    continue
                boxes.add(k)
                act = sample_action.get(k, act)  # the action the classifiers used for this box
                students.append({"dataset": dataset, "img": d["img"], "group": d["group"], "action": act,
                                 "box": [round(float(v), 1) for v in g[1:5]], "eval": k in det_of,
                                 "det_box": det_of.get(k)})

    size = Counter(folds[s["group"]] for s in students if s["group"] in folds)
    by_group = Counter(s["group"] for s in students if s["group"] not in folds)
    for g, n in sorted(by_group.items(), key=lambda x: (-x[1], x[0])):  # largest first, to the emptiest fold
        k = min(range(FOLDS), key=lambda f: (size[f], f))
        folds[g] = k
        size[k] += n
    for i, s in enumerate(students):
        s["id"] = i
        s["fold"] = folds[s["group"]]

    n_eval = sum(s["eval"] for s in students)
    if dup:
        print(f"{dup} boxes labelled twice in one image kept once")
    if n_eval != len(det_of):
        print(f"[warn] {len(det_of) - n_eval} classifier students not found among the labelled boxes")
    json.dump({"folds": folds, "actions": list(mapping.ACTIONS), "students": students,
               "made_from": {"poses": args.poses, "clabehavior": args.clabehavior, "det_thr": args.det_thr,
                             "kpt_thr": args.kpt_thr, "fold_source": "GroupKFold(5) over lenient-mapping students"}},
              open(args.out, "w"))
    per = Counter((s["dataset"], s["fold"]) for s in students)
    print(f"{len(students)} labelled students, {n_eval} evaluated (matched to a pose box), "
          f"{len(folds)} groups in {FOLDS} folds -> {args.out}")
    for ds in sorted({s["dataset"] for s in students}):
        print(f"  {ds}: students per fold {[per[(ds, k)] for k in range(FOLDS)]}; actions "
              f"{dict(Counter(s['action'] for s in students if s['dataset'] == ds).most_common())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
