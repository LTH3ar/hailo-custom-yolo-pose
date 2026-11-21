"""Train the mlp classifier: MLP with two hidden layers of 64 and 32 units (mean imputation, standardisation, early stopping).
Its logit is shifted by the training prior so its probability is on the same balanced
scale as the other two (see common/families.py).

Cross-validates (5 folds, grouped by lesson) on pose caches, then exports scb_mlp.json for hcpose
with --export-dir, or to a chosen file with --out.

usage (from the repository root):
  python -m classifiers.mlp.train ~/scb/poses_hef_m.jsonl ~/scb/poses_hef_m_tiles.jsonl --det-thr 0.3 \
      --name scb-m-int8-tiles --export-dir ../hailo-custom-yolo-pose-runtime/hcpose/engagement/models
"""
from classifiers.common.run import main

if __name__ == "__main__":
    main(["mlp"], __doc__)
