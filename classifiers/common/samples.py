"""Turn pose caches into one feature vector per labelled, detected student.

Pose caches are JSON-lines files written by the runtime's research/scb/extract_hef.py (the
deployed HEF on the Pi) or extract_pt.py (PyTorch): one line per image with the ground-truth
boxes ("gt": [label, x1, y1, x2, y2]) and the detections ("dets": box, score, 13 keypoints).
Each source is a (dataset, cache) pair; labels are translated to the shared action names in
mapping.py, then to focused / distracted by the chosen mapping.

Features come from hcpose.engagement.classifier.feature_vector(), the same code the live
analyzer runs, so a model trained here sees exactly what it will see on the Pi. Context that
lets a model recognise the image (position in the frame, crowd size) is left out.
"""
from dataclasses import dataclass

import numpy as np
from hcpose.engagement.classifier import feature_vector
from hcpose.engagement.features import Baseline, KeypointIndex, compute
from hcpose.presets import UPPER_BODY13_NAMES

from . import mapping
from .datasets import load

IDX = KeypointIndex(UPPER_BODY13_NAMES)


@dataclass
class Samples:
    X: np.ndarray        # features, one row per student
    y: np.ndarray        # True = distracted
    groups: np.ndarray   # lesson group, for grouped cross-validation
    images: np.ndarray   # image path, for the within-image AUC
    labels: np.ndarray   # the shared action name (mapping.ACTIONS)
    mixed: np.ndarray    # True if the student's image holds both classes
    datasets: np.ndarray # which dataset the student comes from
    group_names: np.ndarray = None  # the group's name ("scb:12"), stable across runs and mappings
    gt_boxes: np.ndarray = None     # the labelled box [x1, y1, x2, y2]
    det_boxes: np.ndarray = None    # the pose detection matched to it

    FIELDS = ("X", "y", "groups", "images", "labels", "mixed", "datasets", "group_names", "gt_boxes", "det_boxes")

    def subset(self, mask):
        return Samples(*(None if getattr(self, f) is None else getattr(self, f)[mask] for f in self.FIELDS))


def is_front(K, T):
    """Front view: at least 3 students with both shoulders, and most of them show the face."""
    sh = [k for k in K if k[5, 2] >= T and k[6, 2] >= T]
    return len(sh) >= 3 and np.mean([k[0, 2] >= T and (k[1, 2] >= T or k[2, 2] >= T) for k in sh]) >= 0.6


def match(gt, dets):
    """Greedy one-to-one match of ground-truth boxes to detections (coverage >= 0.5 or IoU >= 0.3)."""
    pairs = []
    for gi, g in enumerate(gt):
        for di, x in enumerate(dets):
            b = x["box"]
            ix = max(0, min(g[3], b[2]) - max(g[1], b[0]))
            iy = max(0, min(g[4], b[3]) - max(g[2], b[1]))
            inter = ix * iy
            ga = (g[3] - g[1]) * (g[4] - g[2])
            da = (b[2] - b[0]) * (b[3] - b[1])
            cov, iou = inter / max(ga, 1e-9), inter / max(ga + da - inter, 1e-9)
            if cov >= 0.5 or iou >= 0.3:
                pairs.append((iou + cov, gi, di))
    pairs.sort(reverse=True)
    ug, ud = set(), set()
    for _, gi, di in pairs:
        if gi not in ug and di not in ud:
            ug.add(gi)
            ud.add(di)
            yield gi, di


def build(sources, focused, distracted, det_thr=0.3, kpt_thr=0.5, verbose=True, near_dups=True):
    """Samples from (dataset, pose cache) pairs. Several caches of one dataset are pooled, e.g. with
    and without tiles; they hold the same images, so their groups agree."""
    T = kpt_thr
    X, y, groups, imgs, labels, dsets, gboxes, dboxes = [], [], [], [], [], [], [], []
    seen = set()
    for dataset, path in sources:
        first = dataset not in seen
        seen.add(dataset)
        recs = load(dataset, path, verbose=verbose and first, near_dups=near_dups)
        for d in recs:
            dets = [x for x in d["dets"] if x["score"] >= det_thr]
            K = [np.asarray(x["kpts"], np.float32) for x in dets]
            if not is_front(K, T):
                continue
            feats = [compute(k, x["box"], IDX, T) for k, x in zip(K, dets)]
            base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)  # class median, as before calibration
            for gi, di in match(d["gt"], dets):
                act = mapping.action(dataset, d["gt"][gi][0])
                if act not in focused | distracted or not feats[di].valid:
                    continue
                X.append(feature_vector(K[di], dets[di]["box"], feats[di], base, IDX, T))
                y.append(act in distracted)
                groups.append(d["group"])  # pooled caches share images, so groups stay consistent
                imgs.append(f"{d['img']}")
                labels.append(act)
                dsets.append(dataset)
                gboxes.append([float(v) for v in d["gt"][gi][1:5]])
                dboxes.append([float(v) for v in dets[di]["box"]])
    if not y:
        raise SystemExit("no labelled front-view students found in the pose caches")
    # number groups by first appearance: for SCB alone this keeps the thesis's group order, so the
    # grouped folds come out the same; across datasets the "scb:"/"clabehavior:" names stay distinct
    code = {}
    names = np.array(groups)
    groups = np.array([code.setdefault(g, len(code)) for g in groups])
    X, y, imgs = np.vstack(X), np.array(y), np.array(imgs)
    labels, dsets = np.array(labels), np.array(dsets)
    mixed = np.zeros(len(y), bool)
    for im in set(imgs):
        m = imgs == im
        mixed[m] = y[m].any() and (~y[m]).any()
    return Samples(X, y, groups, imgs, labels, mixed, dsets, names, np.array(gboxes), np.array(dboxes))
