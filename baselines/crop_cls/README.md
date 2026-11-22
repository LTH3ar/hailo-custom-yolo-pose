# Crop baseline: YOLOv8s-cls on student crops

A comparison for the paper, not part of hcpose. It answers: does a model that looks at the pixels of each student, trained on the same public data, do better than hcpose's pose features and rules?

**Why a crop classifier and not a behaviour detector.** The SCB copy used here labels only each subset's own actions: in a BowTurnHead image, only bowing and turning students have boxes, and the students who read or listen are unlabelled. A detector trained on that learns those students as background in some images and as "read" in others. A classifier trained only on labelled boxes avoids that, and by running on the person boxes hcpose itself found, it isolates the one thing that differs: the decision from pixels instead of pose.

**Same tests as the pose classifiers.**
- Datasets: SCB + ClaBehavior, byte-identical SCB copies merged, the same lesson and camera groups.
- Folds: `export_split.py` takes the classifiers' own 5-fold grouping and writes it to `split.json`. `classifiers.train_all --folds split.json` uses the same folds, so both are tested on exactly the same students.
- Students scored: those the classifiers use (front view, matched to a pose box), with the crop taken at the pose box. Training uses the labelled boxes of every student in the training folds.
- Metrics: AUC, within-image AUC, balanced accuracy, for the thesis, strict and lenient mappings, plus the cross-dataset test.
- Clips: the boxes and seats of an existing `hcpose analyze` output; the result is written in the same format, so `research/labels/score_labels.py` (blind labels), the clip summary and `research/staged/score.py` (clip_4) score it unchanged. Two variants: per frame, and smoothed with hcpose's own EMA and hysteresis.
- Precision: FP32 on the GPU, the baseline's best case. hcpose's results are INT8 on the Pi.

**Privacy.** SCB, ClaBehavior, the lecture, the study hall and clip_5 are minors' images. The crops and frames stay on ubuntu3060; only `split.json`, predictions, metrics and the JSON outputs leave it.

## Steps

| Step | Machine | Command |
|---|---|---|
| 1. Split | pi5 (pose caches) | `python -m baselines.crop_cls.export_split SCB_CACHES --clabehavior CLA_CACHES -o split.json` |
| 2. Crops | ubuntu3060 (images) | `python -m baselines.crop_cls.make_crops split.json OUT --path-map PI_PREFIX=LOCAL_PREFIX` |
| 3. Train | ubuntu3060 | `baselines/crop_cls/train.sh OUT` (fold0-4, all, scb_only, cla_only; 50 epochs, 224 px) |
| 4. Predict | ubuntu3060 | `python -m baselines.crop_cls.predict split.json OUT OUT/runs -o preds.jsonl` |
| 5. Score | any | `python -m baselines.crop_cls.score split.json preds.jsonl --report crop_cls.json` |
| 6. Clips | ubuntu3060 (videos) | `python -m baselines.crop_cls.clips OUT/runs/all/weights/last.pt VIDEO ANALYSIS.jsonl -o PREFIX` |

The models use the last epoch's weights: Ultralytics' val folder is a sample of the training crops (it needs one), so "best" would be chosen on training data.

Tests (no model): `python -m pytest baselines/crop_cls/tests`
