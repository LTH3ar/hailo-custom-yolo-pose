"""End-to-end checks of the classifier pipeline on synthetic data in the real formats.

Needs hcpose installed (pip install -e path/to/hailo-custom-yolo-pose-runtime).
Run from the repository root:  python -m pytest classifiers/tests
"""
import json
import subprocess
import sys

import numpy as np
import pytest

from classifiers.common import datasets, mapping, scb
from classifiers.common.families import FAMILIES
from classifiers.common.run import main
from classifiers.common.samples import build
from classifiers.tests.synthetic import write_cache, write_clabehavior, write_clabehavior_cache


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(write_cache(tmp_path_factory.mktemp("scb")))


@pytest.fixture(scope="module")
def cla(tmp_path_factory):
    root = tmp_path_factory.mktemp("cla")
    src, truth = write_clabehavior(root / "src")
    out = root / "yolo"
    r = subprocess.run([sys.executable, "-m", "classifiers.prepare_clabehavior", str(src), str(out)],
                       capture_output=True, text=True, check=True)
    return str(write_clabehavior_cache(out, truth, root / "poses_cla.jsonl")), out, r.stdout


def test_scb_duplicates(cache):
    recs = scb.load(cache, verbose=False)
    assert not any(r["img"].endswith("_copy.jpg") for r in recs)    # byte-identical copies merged
    assert len({r["group"] for r in recs}) == 12                       # one group per lesson
    recs = datasets.load("scb", cache, verbose=False)
    assert len({r["group"] for r in recs}) == 11                       # the re-encoded copy joins lessons 1 and 12
    assert len({r["group"] for r in datasets.load("scb", cache, verbose=False, near_dups=False)}) == 12


def test_prepare_clabehavior(cla):
    _, out, stdout = cla
    assert (out / "classes.txt").read_text().split() == ["lookup", "read", "turn_head", "raise_hand", "write",
                                                          "discuss", "stand"]
    groups = json.loads((out / "groups.json").read_text())
    assert len(groups) == 36 and len(set(groups.values())) == 3       # three fixed cameras
    assert all(groups[f"cam{c}_0000.jpg"] == groups[f"cam{c}_0011.jpg"] for c in range(3))
    label = next(out.glob("labels/train/*.txt")).read_text().split("\n")[0].split()
    assert len(label) == 5 and 0 <= int(label[0]) < 7 and all(0 < float(v) < 1 for v in label[1:])
    assert f"{out}:lookup,read,turn_head,raise_hand,write,discuss,stand" in stdout


def test_view_groups_vs_hash(cla):
    from classifiers.common.neardup import camera_groups, scan_view_groups, view_groups
    _, out, _ = cla
    paths = sorted(str(p) for p in out.glob("images/*/*.jpg"))
    assert len(set(view_groups(paths, 0.8))) == 3
    assert len(set(camera_groups(paths, max_bits=4))) > 3      # the old hash splits moving scenes
    scan = {t: n for t, n, _, _ in scan_view_groups(paths)}
    assert scan[0.95] >= scan[0.8] == scan[0.7] == 3


def test_mappings():
    assert mapping.action("scb", "hand-raising") == "raise_hand"
    assert mapping.action("clabehavior", "Lookup") == "lookup"
    f, d = mapping.get("strict")
    assert "lookup" in f and d == {"turn_head"}
    f, d = mapping.get("lenient")
    assert {"bow_head", "discuss", "stand"} <= d


def test_samples(cache, cla):
    f, d = mapping.get("strict")
    smp = build([("scb", cache), ("clabehavior", cla[0])], f, d, verbose=False)
    assert set(smp.datasets) == {"scb", "clabehavior"}
    assert "lookup" in set(smp.labels) and "bow_head" not in set(smp.labels)   # strict drops the ambiguous actions
    # groups never span datasets
    for g in set(smp.groups):
        assert len(set(smp.datasets[smp.groups == g])) == 1


def test_train_all_cv_and_export(cache, tmp_path):
    cv = main(list(FAMILIES), "test", [cache, "--export-dir", str(tmp_path), "--save-cv", str(tmp_path / "cv.npz")])
    assert set(cv) == {f.title for f in FAMILIES.values()}
    for m in cv.values():
        assert m["auc"] > 0.7  # the synthetic signal is easy (the MLP is weakest on small data)
    for kind in FAMILIES:
        meta = json.loads((tmp_path / f"scb_{kind}.json").read_text())["meta"]
        assert meta["kind"] == kind and set(meta["cv"]) == {"auc", "within_image_auc", "balanced_acc"}
    z = np.load(tmp_path / "cv.npz")
    assert {"oof_logistic", "oof_gradient_boosted", "oof_mlp"} <= set(z.files)


def test_cross_dataset(cache, cla, tmp_path):
    rep = tmp_path / "report.json"
    main(["trees", "logistic"], "test", [cache, "--clabehavior", cla[0], "--mapping", "strict", "--cross",
                                         "--report", str(rep), "--export-dir", str(tmp_path)])
    r = json.loads(rep.read_text())
    assert set(r["cross"]) == {f"{a}->{b}:{k}" for a, b in [("scb", "clabehavior"), ("clabehavior", "scb")]
                               for k in ("trees", "logistic")}
    assert "on_clabehavior" in r["pooled_cv"]["gradient-boosted trees"]
    meta = json.loads((tmp_path / "scb_trees.json").read_text())["meta"]
    assert meta["mapping"] == "strict" and meta["datasets"] == ["clabehavior", "scb"]


def test_guards(cache, cla, tmp_path):
    with pytest.raises(SystemExit):
        main(["trees", "mlp"], "test", [cache, "--out", str(tmp_path / "x.json")])
    with pytest.raises(SystemExit):   # the thesis mapping is SCB-only
        main(["trees"], "test", [cache, "--clabehavior", cla[0]])
    with pytest.raises(SystemExit):   # --cross needs two datasets
        main(["trees"], "test", [cache, "--cross"])
