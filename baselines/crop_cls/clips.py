"""Run the crop baseline on a classroom clip and write `hcpose analyze`-style outputs (ubuntu3060).

The person boxes and seats come from an existing `hcpose analyze -o` file of the same clip (made
on the Pi), so the baseline classifies exactly the students hcpose found; only the decision
differs. Each box is cropped from the video frame (same padding as in training), classified,
and mapped to a distracted probability (score.py). Two files are written:

  <out>_crop_raw.jsonl     state per frame: distracted when p >= 0.5 (how such models are usually run)
  <out>_crop_smooth.jsonl  p smoothed per seat with hcpose's own settings: EMA (tau 1 s), then
                           distracted after p > 0.6 for 2.5 s, focused again after p < 0.4 for 1.5 s

Both carry the input's header "source", so research/labels/score_labels.py and the clip summary
match them to the human labels like any hcpose output. The class share is the plain share of the
frame's students in the distracted state (no seat memory or zone weighting).

The video frames are minors' images for every clip but clip_4 and the exam hall: nothing is saved
but these JSON files.

usage:
  python -m baselines.crop_cls.clips ~/hcpose/crop_cls/runs/all/weights/last.pt clip_3.mp4 \\
      clip_3_m_tiles_rules.jsonl -o out/clip_3_m_tiles [--mapping strict]
"""
import argparse
import json
import math

from classifiers.common import mapping

from .score import p_distracted


class Smoother:
    """hcpose's per-seat smoothing: EMA, then hysteresis with hold times."""

    def __init__(self, tau=1.0, on=0.6, on_hold=2.5, off=0.4, off_hold=1.5):
        self.tau, self.on, self.on_hold, self.off, self.off_hold = tau, on, on_hold, off, off_hold
        self.seats = {}

    def update(self, seat, t, p):
        s = self.seats.get(seat)
        if s is None:
            s = self.seats[seat] = {"t": t, "ema": p, "state": "focused", "since": None}
        else:
            a = 1.0 - math.exp(-max(t - s["t"], 0.0) / self.tau)
            s["ema"] += a * (p - s["ema"])
            s["t"] = t
        flip = s["ema"] > self.on if s["state"] == "focused" else s["ema"] < self.off
        if not flip:
            s["since"] = None
        else:
            s["since"] = t if s["since"] is None else s["since"]
            hold = self.on_hold if s["state"] == "focused" else self.off_hold
            if t - s["since"] >= hold:
                s["state"] = "distracted" if s["state"] == "focused" else "focused"
                s["since"] = None
        return s["state"], s["ema"]


def class_block(persons):
    st = [(p.get("engagement") or {}).get("state") for p in persons]
    st = [x for x in st if x in ("focused", "distracted")]
    return {"visible": len(persons), "counted": len(st),
            "class_pct_distracted": round(100.0 * sum(x == "distracted" for x in st) / len(st), 1) if st else None}


def crop_bgr(frame, box, pad):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    x1, y1 = int(max(0, x1 - pad * bw)), int(max(0, y1 - pad * bh))
    x2, y2 = int(min(w, round(x2 + pad * bw))), int(min(h, round(y2 + pad * bh)))
    return frame[y1:y2, x1:x2] if x2 - x1 >= 2 and y2 - y1 >= 2 else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("weights")
    ap.add_argument("video")
    ap.add_argument("analysis", help="an `hcpose analyze -o` file of this clip (boxes and seats)")
    ap.add_argument("-o", "--out", required=True, help="output prefix")
    ap.add_argument("--mapping", default="strict", choices=["strict", "lenient"])
    ap.add_argument("--pad", type=float, default=0.1)
    ap.add_argument("--imgsz", type=int, default=224)
    args = ap.parse_args(argv)
    import cv2
    from ultralytics import YOLO

    focused, distracted = mapping.get(args.mapping)
    header, frames = None, []
    for line in open(args.analysis):
        d = json.loads(line)
        if d.get("type") == "header":
            header = d
        elif d.get("type") == "frame":
            frames.append(d)
    model = YOLO(args.weights)
    names = model.names
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    smoother = Smoother()
    head = {"type": "header", "source": (header or {}).get("source", args.video),
            "decision": f"crop_cls YOLOv8s-cls ({args.mapping} mapping)", "weights": args.weights}
    raw_out = open(f"{args.out}_crop_raw.jsonl", "w")
    smooth_out = open(f"{args.out}_crop_smooth.jsonl", "w")
    raw_out.write(json.dumps(dict(head, smoothing="none")) + "\n")
    smooth_out.write(json.dumps(dict(head, smoothing="EMA 1 s, hysteresis 0.6/2.5 s, 0.4/1.5 s")) + "\n")

    idx, frame = -1, None
    for fr in frames:
        want = int(round(fr["t"] * fps))
        while idx < want:
            ok, img = cap.read()
            if not ok:
                break
            idx, frame = idx + 1, img
        if frame is None or idx < want:  # video ended; an earlier or equal index reuses the current frame
            break
        persons = fr.get("persons", [])
        crops = [crop_bgr(frame, p["box"], args.pad) for p in persons]
        ps = [None] * len(persons)
        todo = [i for i, c in enumerate(crops) if c is not None]
        if todo:
            res = model.predict([crops[i] for i in todo], imgsz=args.imgsz, verbose=False, batch=len(todo))
            for i, r in zip(todo, res):
                probs = {names[j]: v for j, v in enumerate(r.probs.data.tolist())}
                ps[i] = p_distracted(probs, focused, distracted)
        raw_p, smooth_p = [], []
        for p, prob in zip(persons, ps):
            seat = (p.get("engagement") or {}).get("seat")
            base = {"box": p["box"], "score": p.get("score")}
            if prob is None:
                raw_p.append(dict(base, engagement={"seat": seat, "state": "unknown"}))
                smooth_p.append(dict(base, engagement={"seat": seat, "state": "unknown"}))
                continue
            raw_p.append(dict(base, engagement={"seat": seat, "state": "distracted" if prob >= 0.5 else "focused",
                                                "score": round(prob, 3)}))
            if seat is None:
                st, ema = ("distracted" if prob >= 0.5 else "focused"), prob
            else:
                st, ema = smoother.update(seat, fr["t"], prob)
            smooth_p.append(dict(base, engagement={"seat": seat, "state": st, "score": round(ema, 3)}))
        for out, P in ((raw_out, raw_p), (smooth_out, smooth_p)):
            out.write(json.dumps({"type": "frame", "t": fr["t"], "w": fr.get("w"), "h": fr.get("h"),
                                  "class": class_block(P), "persons": P}) + "\n")
    raw_out.close()
    smooth_out.close()
    print(f"{idx + 1} video frames read, {len(frames)} analysis frames -> {args.out}_crop_{{raw,smooth}}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
