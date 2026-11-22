# Classifiers: focused or distracted, per student

These models decide, for one student in one frame, whether they look focused or distracted, from the student's upper-body keypoints. In hcpose they are an opt-in alternative to the calibrated rules (`hcpose run --classifier trees|logistic|mlp`).

They are trained with **supervised** learning on **stand-in (proxy) labels**: the datasets label *actions* (reading, writing, turning the head, ...), which are mapped to focused or distracted. The mapping is fixed before training and named in every result (`common/mapping.py`).

## Datasets

| Dataset | What | Split groups | Source |
|---|---|---|---|
| SCB | ~10,000 classroom photos, Chinese schools: read, write, hand-raising, TurnHead, BowHead, discuss | lessons, from the file names | [SCB-Dataset](https://github.com/Whiffe/SCB-Dataset); academic, non-commercial use |
| ClaBehavior | 400 frames (2560×1440, 3840×2160) from fixed classroom cameras: lookup, read, write, raise_hand, turn_head, discuss, stand | camera views, found by image similarity (`prepare_clabehavior.py`) | [Student-behavior-detection-system](https://github.com/CCNUZFW/Student-behavior-detection-system), Apache-2.0 |

ClaBehavior's **lookup** (head up, looking at the board) is the attentive state SCB lacks, which is why the SCB-trained classifiers called attentive lectures distracted. STBD-08 and HCCB are not used: their provenance or release could not be verified.

**Privacy.** These are classroom photos, mostly of minors. The images stay on the machines that hold them; only pose caches, labels and numbers move. No image or crop goes into a repository, a paper or a slide. Cite each dataset's paper.

**Duplicates.** SCB has pictures saved more than once. Byte-identical copies across its subsets are merged into one image with the union of their labels (as in the thesis). Pictures saved twice but re-encoded or resized are put in the same split group, so they can't sit on both sides of a split. SCB's shipped train/val splits are not used: 950 copy groups straddle them.

## Mappings

Labels are first translated to shared action names, then to focused or distracted:

| Mapping | Focused | Distracted | Left out |
|---|---|---|---|
| `thesis` (SCB only) | read, write, raise hand | turn head, bow head, discuss | - |
| `strict` | read, write, raise hand, look up | turn head | bow head, discuss, stand |
| `lenient` | read, write, raise hand, look up | turn head, bow head, discuss, stand | - |

Bow head looks like writing, discuss may be on-task group work, and stand may be answering a question, so `strict` leaves them out and `lenient` counts them as distracted. Report both.

## Layout

| Path | What |
|---|---|
| `trees/train.py` | gradient-boosted trees (scikit-learn `HistGradientBoostingClassifier`) |
| `logistic/train.py` | logistic regression |
| `mlp/train.py` | a small neural network, two hidden layers of 64 and 32 units |
| `train_all.py` | all three at once, one comparison table (the thesis procedure) |
| `prepare_clabehavior.py` | lays ClaBehavior out like SCB for the pose extractor, and writes its camera groups |
| `common/` | shared code: dataset loading (`datasets.py`, `scb.py`), duplicate and camera grouping (`neardup.py`), label mappings, features (`samples.py`), cross-validation and cross-dataset tests (`run.py`), export |
| `experiments/` | `learn.py` (is the task learnable from keypoints?) and `eval_tiling.py` (do the classifiers behave the same on tiled poses?) |
| `tests/` | end-to-end tests on synthetic data in the real formats |

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

ClaBehavior is laid out like SCB first (on the machine that holds the images), then extracted the same way:

```bash
python -m classifiers.prepare_clabehavior ~/datasets/clabehavior ~/datasets/clabehavior_yolo --scan
# prints the number of camera groups per --min-corr; ClaBehavior has no plateau, thesis v1.1 uses 0.4:
python -m classifiers.prepare_clabehavior ~/datasets/clabehavior ~/datasets/clabehavior_yolo --min-corr 0.4
# prints the extractor argument: ~/datasets/clabehavior_yolo:lookup,read,turn_head,...
.venv/bin/python research/scb/extract_hef.py ~/models/upper_body_m.hef ~/datasets/poses_cla_m.jsonl <that argument>
```

## Training

SCB and ClaBehavior, strict mapping, with cross-dataset tests:

```bash
python -m classifiers.train_all ~/scb/poses_m.jsonl ~/scb/poses_m_tiles.jsonl \
    --clabehavior ~/datasets/poses_cla_m.jsonl ~/datasets/poses_cla_m_tiles.jsonl \
    --mapping strict --cross --det-thr 0.3 --report report_strict.json
```

This prints the pooled cross-validation (folds grouped by lesson or camera view), broken down per dataset, and then `--cross`: each model trained on all of one dataset and tested on the other. Run it again with `--mapping lenient`. With `--folds split.json` (from `baselines/crop_cls/export_split.py`) the folds come from that file instead of GroupKFold, so the classifiers and the crop baseline are tested on exactly the same students.

The thesis models, reproduced exactly (`--no-near-dups` keeps the thesis's folds; the thesis mapping is the default):

```bash
python -m classifiers.train_all ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl --det-thr 0.3 --no-near-dups \
    --name scb-m-int8-tiles --export-dir ../hailo-custom-yolo-pose-runtime/hcpose/engagement/models \
    --save-cv ~/scb/cv_m_pooled.npz
```

One family only:

```bash
python -m classifiers.trees.train ~/scb/poses_hef_m.jsonl --det-thr 0.3 --out /tmp/trees.json
```

Each run prints, for every family, the AUC, the within-image AUC and the balanced accuracy over 5 folds grouped by lesson. The within-image AUC compares only students in the same photo, so a model can't score by recognising the lesson; it is the honest figure. Export checks that the NumPy model in hcpose reproduces scikit-learn to within 1e-5.

## What the thesis found

On SCB the classifiers rank actions well (within-image AUC 0.83 for trees, 0.82 for the MLP; thesis v1.1, INT8 m poses on the Pi), but on a real lecture they call 44-82% of attentive students distracted: SCB has no "listening" label, so a raised head reads as distracted. Adding ClaBehavior, whose lookup class is that missing state, barely helps: trees and MLP still flag 43-51% of the lecture, and models trained on one dataset score AUC 0.65-0.81 on the other. The rules stay the default in hcpose.
