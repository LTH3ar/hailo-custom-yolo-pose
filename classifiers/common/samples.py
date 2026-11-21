"""Turn pose caches into one feature vector per labelled, detected student.

Pose caches are JSON-lines files written by the runtime's research/scb/extract_hef.py (the
deployed HEF on the Pi) or extract_pt.py (PyTorch): one line per image with the ground-truth
boxes ("gt": [label, x1, y1, x2, y2]) and the detections ("dets": box, score, 13 keypoints).

Features come from hcpose.engagement.classifier.feature_vector(), the same code the live
analyzer runs, so a model trained here sees exactly what it will see on the Pi. Context that
lets a model recognise the image (position in the frame, crowd size) is left out.
"""
from dataclasses import dataclass

import numpy as np
from hcpose.engagement.classifier import feature_vector
from hcpose.engagement.features import Baseline, KeypointIndex, compute
from hcpose.presets import UPPER_BODY13_NAMES

from . import scb

IDX = KeypointIndex(UPPER_BODY13_NAMES)


@dataclass
class Samples:
    X: np.ndarray        # features, one row per student
    y: np.ndarray        # True = distracted
    groups: np.ndarray   # lesson group, for grouped cross-validation
    images: np.ndarray   # image path, for the within-image AUC
    labels: np.ndarray   # the dataset's own action label
    mixed: np.ndarray    # True if the student's image holds both classes


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


def build(poses, focused, distracted, det_thr=0.3, kpt_thr=0.5, verbose=True):
    """Samples from one or more SCB pose caches (several caches are pooled, e.g. with and without tiles)."""
    T = kpt_thr
    X, y, groups, imgs, labels = [], [], [], [], []
    for src_i, path in enumerate(poses):
        for d in scb.load(path, verbose=verbose and src_i == 0):
            dets = [x for x in d["dets"] if x["score"] >= det_thr]
            K = [np.asarray(x["kpts"], np.float32) for x in dets]
            if not is_front(K, T):
                continue
            feats = [compute(k, x["box"], IDX, T) for k, x in zip(K, dets)]
            base = Baseline.from_samples([f for f in feats if f.valid], min_n=3)  # class median, as before calibration
            for gi, di in match(d["gt"], dets):
                cls = d["gt"][gi][0]
                if cls not in focused | distracted or not feats[di].valid:
                    continue
                X.append(feature_vector(K[di], dets[di]["box"], feats[di], base, IDX, T))
                y.append(cls in distracted)
                groups.append(d["group"])  # pooled caches share images, so groups stay consistent
                imgs.append(f"{d['img']}")
                labels.append(cls)
    if not y:
        raise SystemExit("no labelled front-view students found in the pose caches")
    X, y, groups, imgs, labels = np.vstack(X), np.array(y), np.array(groups), np.array(imgs), np.array(labels)
    mixed = np.zeros(len(y), bool)
    for im in set(imgs):
        m = imgs == im
        mixed[m] = y[m].any() and (~y[m]).any()
    return Samples(X, y, groups, imgs, labels, mixed)
