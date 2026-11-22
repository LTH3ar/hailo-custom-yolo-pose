# hailo-custom-yolo-pose

Training for [hcpose](https://github.com/LTH3ar/hailo-custom-yolo-pose-runtime), the privacy-first classroom engagement system on Raspberry Pi 5 + Hailo-8. Everything that is *trained* lives here; running the models (pose extraction, the live analyzer, the bundled classifier JSON) lives in the runtime repository.

| Folder | What it trains | Output |
|---|---|---|
| [`yolo/`](yolo/README.md) | The 13-keypoint upper-body YOLO pose models (n, s, m): Ultralytics training, ONNX export, Hailo INT8 compilation | `.hef` files for the Pi |
| [`classifiers/`](classifiers/README.md) | The per-student focused/distracted classifiers on pose features: `trees/`, `logistic/`, `mlp/` | `scb_<kind>.json` for `hcpose/engagement/models` |
| [`baselines/crop_cls/`](baselines/crop_cls/README.md) | A comparison baseline, not part of hcpose: YOLOv8s-cls on student image crops, scored on the same folds, labels and clips | results only |

Run every command from the repository root. The two parts use separate environments: `yolo/` needs PyTorch + Ultralytics and the Hailo Dataflow Compiler (Python 3.10), `classifiers/` needs scikit-learn and the `hcpose` package.

The models evaluated in the thesis were trained with the code as of runtime tag `v1.0.3`; the classifier scripts moved here from the runtime's `research/scb/` without changing their results (checked by running the old and new code on the same data).

## License & attribution
This project is licensed under the **GNU AGPLv3**.

> **Author:** LTH3ar  
> **Project:** hailo-custom-yolo-pose (2026)  
> **Link:** [https://github.com/LTH3ar/hailo-custom-yolo-pose](https://github.com/LTH3ar/hailo-custom-yolo-pose)

Dependencies: [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), [Hailo Dataflow Compiler](https://hailo.ai/developer-zone/), [scikit-learn](https://scikit-learn.org) (BSD-3).
