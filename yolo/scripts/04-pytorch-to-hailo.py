"""Compile a YOLO pose ONNX model to a Hailo HEF: translate, optimise (INT8), compile.

The nine raw head outputs are the end nodes: box distribution (DFL), class score and
keypoints at strides 8, 16 and 32. Decoding and NMS run on the host (hcpose), which keeps
the HEF independent of the model's keypoint count.

The model script (.alls) sets the input normalisation, a sigmoid on the class outputs, and
16-bit precision on the keypoint outputs. It names layers by their position (output_layer3,
conv71, ...), so check with hef_reader.py that those positions really are the keypoint and
class outputs for the model being compiled: a mismatch quantises the keypoints to 8 bits.

Calibration uses MAX_IMAGES random images from CALIB_DIR, drawn with --seed so a compile can
be repeated exactly. (The thesis models were compiled before the seed existed.)

usage:
  python yolo/scripts/04-pytorch-to-hailo.py <model.onnx> <calib_dir> <model_script.alls> [--seed 0] [--images 1024]
"""
import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv4.0/cv4.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv4.1/cv4.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv", "/model.22/cv4.2/cv4.2.2/Conv",
]


def dfc_version():
    try:
        from importlib.metadata import version
        return version("hailo_dataflow_compiler")
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("calib_dir", type=Path, help="folder of calibration images (e.g. data/images/train2017)")
    ap.add_argument("model_script", type=Path, help=".alls model script")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random calibration draw (default 0)")
    ap.add_argument("--images", type=int, default=1024, help="number of calibration images (default 1024)")
    ap.add_argument("--hw-arch", default="hailo8", help="hailo8 (default), hailo8l, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    name = args.onnx.stem
    out_dir = args.onnx.parent
    har = out_dir / f"{name}.har"
    har_opt = out_dir / f"{name}_optimized.har"
    hef = out_dir / f"{name}.hef"
    # the cache name carries everything that changes the draw, so different settings never share a cache
    cache = out_dir / f"calib_{args.calib_dir.name}_n{args.images}_seed{args.seed}_{args.imgsz}.npy"
    print(f"[info] DFC {dfc_version()}, hw_arch {args.hw_arch}, model script {args.model_script}")

    # 1. translate
    runner = ClientRunner(hw_arch=args.hw_arch)
    runner.translate_onnx_model(
        str(args.onnx),
        name,
        start_node_names=["images"],
        end_node_names=END_NODES,
        net_input_shapes={"images": [1, 3, args.imgsz, args.imgsz]},
    )
    runner.save_har(str(har))

    # 2. calibration set (seeded, cached)
    if cache.exists():
        print(f"[info] calibration data from cache {cache}")
        calib = np.load(cache)
    else:
        images = sorted(p for p in args.calib_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        random.Random(args.seed).shuffle(images)
        images = images[: args.images]
        calib = np.zeros((len(images), args.imgsz, args.imgsz, 3), dtype=np.float32)
        for i, p in enumerate(images):
            calib[i] = np.array(Image.open(p).convert("RGB").resize((args.imgsz, args.imgsz)))
        np.save(cache, calib)
        (cache.with_suffix(".txt")).write_text("\n".join(p.name for p in images) + "\n")
        print(f"[info] {len(images)} calibration images (seed {args.seed}) cached to {cache}; list in {cache.with_suffix('.txt')}")

    # 3. optimise (quantise) with the model script
    runner.load_model_script(str(args.model_script))
    runner.optimize(calib)
    runner.save_har(str(har_opt))

    # 4. compile
    print("[info] compiling to HEF...")
    hef.write_bytes(runner.compile())
    print(f"[success] {hef}")


if __name__ == "__main__":
    main()
