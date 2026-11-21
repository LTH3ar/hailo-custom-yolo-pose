# Classifiers: focused or distracted, per student

These models decide, for one student in one frame, whether they look focused or distracted, from the student's upper-body keypoints. In hcpose they are an opt-in alternative to the calibrated rules (`hcpose run --classifier trees|logistic|mlp`).

They are trained with **supervised** learning on **stand-in (proxy) labels**: the datasets label *actions* (reading, writing, turning the head, ...), which are mapped to focused or distracted. The mapping is fixed before training and named in every result (`common/mapping.py`).

## Layout

| Path | What |
|---|---|
| `trees/train.py` | gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) |
| `logistic/train.py` | logistic regression |
| `mlp/train.py` | a small neural network, two hidden layers of 64 and 32 units |
| `train_all.py` | all three at once, one comparison table (the thesis procedure) |
| `common/` | shared code: dataset loading and duplicate handling (`scb.py`), label mapping, features, grouped cross-validation, export |
| `experiments/` | `learn.py` (is the task learnable from keypoints?) and `eval_tiling.py` (do the classifiers behave the same on tiled poses?) |
| `tests/` | an end-to-end test on a synthetic pose cache |

## Setup

```bash
python3 -m venv .venv_cls
.venv_cls/bin/pip install -r classifiers/requirements.txt
.venv_cls/bin/pip install -e ../hailo-custom-yolo-pose-runtime     # hcpose, for the features
.venv_cls/bin/python -m pytest classifiers/tests                   # check
```

The features come from `hcpose.engagement.classifier.feature_vector()`, the same code the analyzer runs on the Pi, so a model sees exactly the features it will see in use. Install the hcpose version you will deploy to.

## Inputs: pose caches

A pose cache is a JSON-lines file with one line per image: the image path, the dataset's ground-truth boxes and labels, and the detected people with their 13 keypoints. They are made by the runtime (it runs the pose model), on the Pi with the deployed HEF:

```bash
# on the Pi, in hailo-custom-yolo-pose-runtime (see its research/scb/README.md)
.venv/bin/python research/scb/extract_hef.py ~/models/upper_body_m.hef ~/scb/poses_hef_m.jsonl <SCB dirs...>
.venv/bin/python research/scb/extract_hef.py ~/models/upper_body_m.hef ~/scb/poses_hef_m_tiles.jsonl <SCB dirs...> --tiles auto
```

## Training

The thesis models (pooled poses with and without tiles, m INT8, detection threshold 0.3):

```bash
python -m classifiers.train_all ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl --det-thr 0.3 \
    --name scb-m-int8-tiles --export-dir ../hailo-custom-yolo-pose-runtime/hcpose/engagement/models \
    --save-cv ~/scb/cv_m_pooled.npz
```

One family only:

```bash
python -m classifiers.trees.train ~/scb/poses_hef_m.jsonl --det-thr 0.3 --out /tmp/trees.json
```

Each run prints, for every family, the AUC, the within-image AUC and the balanced accuracy over 5 folds grouped by lesson. The within-image AUC compares only students in the same photo, so a model can't score by recognising the lesson; it is the honest figure. Export checks that the NumPy model in hcpose reproduces scikit-learn to within 1e-5.

## What the thesis found

On SCB the classifiers rank actions well (within-image AUC 0.84 for trees and MLP), but on a real lecture they call 48-89% of attentive students distracted: SCB has no "listening" label, so a raised head reads as distracted. The rules stay the default in hcpose. Next: more datasets with a head-up attentive class, a strict and a lenient mapping, and cross-dataset tests.
