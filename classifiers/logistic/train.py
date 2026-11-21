"""Train the logistic classifier: logistic regression (mean imputation, standardisation, balanced class weights).

Cross-validates (5 folds, grouped by lesson) on pose caches, then exports scb_logistic.json for hcpose
with --export-dir, or to a chosen file with --out.

usage (from the repository root):
  python -m classifiers.logistic.train ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl --det-thr 0.3 \
      --name scb-m-int8-tiles --export-dir ../hailo-custom-yolo-pose-runtime/hcpose/engagement/models
"""
from classifiers.common.run import main

if __name__ == "__main__":
    main(["logistic"], __doc__)
