"""Checks of the crop baseline's data handling and scoring on the synthetic datasets (no model).

Run from the repository root:  python -m pytest baselines/crop_cls/tests
"""
import json
import subprocess
import sys

import pytest

from baselines.crop_cls import export_split, make_crops, score
from baselines.crop_cls.clips import Smoother, class_block
from classifiers.common import mapping
from classifiers.common.run import main as train_all
from classifiers.tests.synthetic import write_cache, write_clabehavior, write_clabehavior_cache


@pytest.fixture(scope="module")
def split(tmp_path_factory):
    root = tmp_path_factory.mktemp("crop")
    cache = write_cache(root / "scb")
    src, truth = write_clabehavior(root / "cla_src")
    out = root / "cla"
    subprocess.run([sys.executable, "-m", "classifiers.prepare_clabehavior", str(src), str(out)],
                   capture_output=True, check=True)
    cla = write_clabehavior_cache(out, truth, root / "poses_cla.jsonl")
    path = root / "split.json"
    export_split.main([str(cache), "--clabehavior", str(cla), "-o", str(path)])
    return root, path, str(cache), str(cla)


def test_split(split):
    _, path, _, _ = split
    S = json.loads(path.read_text())
    st = S["students"]
    assert {s["fold"] for s in st} == set(range(5))
    assert all(S["folds"][s["group"]] == s["fold"] for s in st)          # one fold per group
    assert {s["dataset"] for s in st} == {"scb", "clabehavior"}
    ev = [s for s in st if s["eval"]]
    assert ev and all(s["det_box"] for s in ev)
    assert all(s["action"] in mapping.ACTIONS for s in st)


def test_classifiers_use_the_split(split, tmp_path):
    _, path, cache, cla = split
    cv = train_all(["logistic"], "test", [cache, "--clabehavior", cla, "--mapping", "strict", "--folds", str(path)])
    assert next(iter(cv.values()))["auc"] > 0.7


def test_crops(split):
    root, path, _, _ = split
    out = root / "crops_out"
    make_crops.main([str(path), str(out)])
    S = json.loads(path.read_text())
    n = len(S["students"])
    assert len(list((out / "crops" / "gt").glob("*.jpg"))) == n
    assert len(list((out / "crops" / "det").glob("*.jpg"))) == sum(s["eval"] for s in S["students"])
    for run in ("fold0", "fold4", "all", "scb_only", "cla_only"):
        train = {p.name for p in (out / "ds" / run / "train").iterdir()}
        val = {p.name for p in (out / "ds" / run / "val").iterdir()}
        assert train and train == val                                   # every action has a val crop
    held = {int(p.stem) for p in (out / "ds" / "fold0" / "train").rglob("*.jpg")}
    assert not held & {s["id"] for s in S["students"] if s["fold"] == 0}  # the test fold is not trained on


def test_score_perfect_and_cross(split, tmp_path):
    _, path, _, _ = split
    S = json.loads(path.read_text())
    preds = tmp_path / "preds.jsonl"
    with open(preds, "w") as f:
        for s in S["students"]:
            if s["eval"]:
                probs = {a: (0.9 if a == s["action"] else 0.1 / 7) for a in mapping.ACTIONS}
                f.write(json.dumps({"id": s["id"], "run": f"fold{s['fold']}", "probs": probs}) + "\n")
                other = "scb_only" if s["dataset"] == "clabehavior" else "cla_only"
                f.write(json.dumps({"id": s["id"], "run": other, "probs": {a: 1 / 8 for a in mapping.ACTIONS}}) + "\n")
    rep = tmp_path / "rep.json"
    score.main([str(path), str(preds), "--report", str(rep)])
    R = json.loads(rep.read_text())
    assert R["cv"]["strict/pooled"]["auc"] == 1.0 and R["cv"]["strict/pooled"]["balanced_acc"] == 1.0
    assert "thesis/scb" in R["cv"] and "lenient/clabehavior" in R["cv"]
    assert R["cross"]["strict/scb -> clabehavior"]["auc"] == 0.5        # uninformative model


def test_p_distracted():
    f, d = mapping.get("strict")
    assert score.p_distracted({"turn_head": 0.6, "read": 0.2, "bow_head": 0.2}, f, d) == pytest.approx(0.75)


def test_smoother_holds():
    sm = Smoother()
    states = [sm.update(1, i / 10, 1.0)[0] for i in range(40)]            # 4 s of a clearly distracted seat
    first = states.index("distracted")
    assert 2.5 <= first / 10 <= 3.5                                       # EMA climb + 2.5 s hold
    back = [sm.update(1, 4 + i / 10, 0.0)[0] for i in range(40)]
    assert "focused" in back and back.index("focused") / 10 >= 1.5
    assert class_block([{"engagement": {"state": "distracted"}}, {"engagement": {"state": "focused"}},
                        {"engagement": {"state": "unknown"}}])["class_pct_distracted"] == 50.0


def _fake_ultralytics(monkeypatch, names):
    """A stand-in for ultralytics.YOLO that 'classifies' every crop as turn_head, so predict.py and
    clips.py run end to end without a model."""
    import types

    class Probs:
        def __init__(self, p):
            self.data = types.SimpleNamespace(tolist=lambda: p)

    class YOLO:
        def __init__(self, weights):
            self.names = dict(enumerate(names))

        def predict(self, sources, **kw):
            p = [0.9 if n == "turn_head" else 0.1 / (len(names) - 1) for n in names]
            return [types.SimpleNamespace(probs=Probs(p)) for _ in sources]

    mod = types.ModuleType("ultralytics")
    mod.YOLO = YOLO
    monkeypatch.setitem(sys.modules, "ultralytics", mod)


def test_predict_runs(split, tmp_path, monkeypatch):
    from baselines.crop_cls import predict
    root, path, _, _ = split
    _fake_ultralytics(monkeypatch, list(mapping.ACTIONS))
    runs = tmp_path / "runs"
    for r in ("fold0", "fold1", "fold2", "fold3", "fold4", "scb_only", "cla_only"):
        (runs / r / "weights").mkdir(parents=True)
        (runs / r / "weights" / "last.pt").write_bytes(b"")
    out = tmp_path / "preds.jsonl"
    predict.main([str(path), str(root / "crops_out"), str(runs), "-o", str(out), "--batch", "7"])
    P = [json.loads(x) for x in open(out)]
    S = json.loads(path.read_text())
    n_eval = sum(s["eval"] for s in S["students"])
    assert sum(p["run"].startswith("fold") for p in P) == n_eval
    assert {p["run"] for p in P} >= {"scb_only", "cla_only"}


def test_clips_runs(tmp_path, monkeypatch):
    import cv2
    import numpy as np
    from baselines.crop_cls import clips
    _fake_ultralytics(monkeypatch, list(mapping.ACTIONS))
    video = tmp_path / "clip_x.avi"
    vw = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (160, 120))
    for i in range(50):
        vw.write(np.full((120, 160, 3), i * 4, np.uint8))
    vw.release()
    ana = tmp_path / "clip_x_m_rules.jsonl"
    with open(ana, "w") as f:
        f.write(json.dumps({"type": "header", "source": "/videos/clip_x.mp4"}) + "\n")
        for i in range(50):
            f.write(json.dumps({"type": "frame", "t": i / 10, "w": 160, "h": 120, "persons": [
                {"box": [10, 10, 60, 90], "score": 0.9, "engagement": {"seat": 1, "state": "focused"}},
                {"box": [159.5, 119.5, 160, 120], "score": 0.5, "engagement": {"seat": 2}}]}) + "\n")
    clips.main(["w.pt", str(video), str(ana), "-o", str(tmp_path / "x")])
    raw = [json.loads(x) for x in open(tmp_path / "x_crop_raw.jsonl")]
    smooth = [json.loads(x) for x in open(tmp_path / "x_crop_smooth.jsonl")]
    assert raw[0]["source"] == "/videos/clip_x.mp4" and len(raw) == 51 and len(smooth) == 51
    assert raw[1]["persons"][0]["engagement"]["state"] == "distracted"          # turn_head every frame
    assert raw[1]["persons"][1]["engagement"]["state"] == "unknown"             # crop too small
    assert smooth[1]["persons"][0]["engagement"]["state"] == "focused"          # smoothing holds at first
    assert smooth[-1]["persons"][0]["engagement"]["state"] == "distracted"
    assert raw[-1]["class"]["class_pct_distracted"] == 100.0
