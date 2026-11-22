"""Run the trained crop models on the test crops and write every class probability (ubuntu3060, GPU).

For each fold k, the fold-k model (trained without fold k) classifies the pose-box crops of the
evaluated students in fold k. For the cross-dataset test, scb_only classifies every evaluated
ClaBehavior student and cla_only every evaluated SCB student. Weights are the last epoch's
(weights/last.pt), never "best": the val folder overlaps training (make_crops.py).

Output: one JSON line per (student, run): {"id", "run", "probs": {action: p}}.

usage:
  python -m baselines.crop_cls.predict split.json ~/hcpose/crop_cls ~/hcpose/crop_cls/runs -o preds.jsonl
"""
import argparse
import json
from pathlib import Path


def jobs(students, runs_dir):
    """[(run, weights, students)] for every run that has weights."""
    out = []
    for k in sorted({s["fold"] for s in students}):
        out.append((f"fold{k}", [s for s in students if s["fold"] == k]))
    out.append(("scb_only", [s for s in students if s["dataset"] == "clabehavior"]))
    out.append(("cla_only", [s for s in students if s["dataset"] == "scb"]))
    res = []
    for run, ss in out:
        w = Path(runs_dir) / run / "weights" / "last.pt"
        if ss and w.is_file():
            res.append((run, w, ss))
        elif ss:
            print(f"[skip] {run}: no {w}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split")
    ap.add_argument("crops_root", type=Path, help="the make_crops.py output folder")
    ap.add_argument("runs", type=Path, help="Ultralytics project folder holding fold0..4, scb_only, cla_only")
    ap.add_argument("-o", "--out", default="preds.jsonl")
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args(argv)
    from ultralytics import YOLO

    S = json.load(open(args.split))
    ev = [s for s in S["students"] if s.get("eval")]
    det = args.crops_root / "crops" / "det"
    ev = [s for s in ev if (det / f"{s['id']}.jpg").is_file()]
    with open(args.out, "w") as out:
        for run, w, ss in jobs(ev, args.runs):
            model = YOLO(str(w))
            names = model.names
            for i in range(0, len(ss), args.batch):
                chunk = ss[i:i + args.batch]
                res = model.predict([str(det / f"{s['id']}.jpg") for s in chunk], imgsz=args.imgsz,
                                    verbose=False, batch=len(chunk))
                for s, r in zip(chunk, res):
                    p = r.probs.data.tolist()
                    out.write(json.dumps({"id": s["id"], "run": run,
                                          "probs": {names[j]: round(v, 5) for j, v in enumerate(p)}}) + "\n")
            print(f"{run}: {len(ss)} students ({w})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
