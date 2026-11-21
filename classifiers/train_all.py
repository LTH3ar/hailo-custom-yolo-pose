"""Compare all three classifier families and export them: the thesis's procedure.

Same as running trees/, logistic/ and mlp/ one by one, but builds the features once and
prints one comparison table. Replaces the runtime's research/scb/train_classifier.py.

usage (from the repository root), as for the thesis models (v1.0.3):
  python -m classifiers.train_all ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl --det-thr 0.3 \
      --name scb-m-int8-tiles --export-dir ../hailo-custom-yolo-pose-runtime/hcpose/engagement/models \
      --save-cv ~/scb/cv_m_pooled.npz
"""
from classifiers.common.families import FAMILIES
from classifiers.common.run import main

if __name__ == "__main__":
    main(list(FAMILIES), __doc__)
