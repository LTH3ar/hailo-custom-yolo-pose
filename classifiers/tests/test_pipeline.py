"""End-to-end check of the classifier pipeline on a synthetic pose cache.

Needs hcpose installed (pip install -e path/to/hailo-custom-yolo-pose-runtime).
Run from the repository root:  python -m pytest classifiers/tests
"""
import json

import numpy as np
import pytest

from classifiers.common import mapping, scb
from classifiers.common.families import FAMILIES
from classifiers.common.run import main
from classifiers.common.samples import build
from classifiers.tests.synthetic import write_cache


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    return str(write_cache(tmp_path_factory.mktemp("scb")))


def test_duplicates_are_merged(cache):
    recs = scb.load(cache, verbose=False)
    imgs = [r["img"] for r in recs]
    assert not any(i.endswith("_copy.jpg") for i in imgs)   # each copy merged into its original
    assert len({r["group"] for r in recs}) == 12               # one group per lesson


def test_samples(cache):
    f, d = mapping.get("thesis")
    smp = build([cache], f, d, det_thr=0.3, verbose=False)
    assert smp.X.shape[0] == len(smp.y) == len(smp.groups) > 100
    assert 0.2 < smp.y.mean() < 0.5
    assert smp.mixed.any()


def test_train_all_cv_and_export(cache, tmp_path):
    cv = main(list(FAMILIES), "test", [cache, "--export-dir", str(tmp_path), "--save-cv", str(tmp_path / "cv.npz")])
    assert set(cv) == {f.title for f in FAMILIES.values()}
    for m in cv.values():
        assert m["auc"] > 0.7  # the synthetic signal is easy (the MLP is weakest on 360 students)
    for kind in FAMILIES:
        meta = json.loads((tmp_path / f"scb_{kind}.json").read_text())
        assert meta["meta"]["kind"] == kind
    z = np.load(tmp_path / "cv.npz")
    assert {"oof_logistic", "oof_gradient_boosted", "oof_mlp"} <= set(z.files)


def test_single_family_out(cache, tmp_path):
    out = tmp_path / "trees.json"
    main(["trees"], "test", [cache, "--out", str(out)])
    assert out.is_file()
    with pytest.raises(SystemExit):
        main(["trees", "mlp"], "test", [cache, "--out", str(out)])
