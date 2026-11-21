"""Compile a YOLO pose ONNX model to a Hailo HEF: translate, optimise (INT8), compile.

The nine raw head outputs are the end nodes: box distribution (DFL), class score and
keypoints at strides 8, 16 and 32. Decoding and NMS run on the host (hcpose), which keeps
the HEF independent of the model's keypoint count.

The model script (.alls) sets the input normalisation, a sigmoid on the class outputs, and
16-bit precision on the keypoint outputs. It names layers by their position (output_layer3,
conv71, ...), so check with hef_reader.py that those positions really are the keypoint and
class outputs for the model being compiled: a mismatch quantises the keypoints to 8 bits.

Calibration uses --images random images from CALIB_DIR, drawn with --seed so a compile can be
repeated exactly. (The thesis models were compiled before the seed existed, and one unlucky draw
cost the s model 26 AP points.) Different seeds give different HEFs: compare several.

Outputs go to --out-dir (default: next to the ONNX) as <name>.har, <name>_optimized.har and
<name>.hef, where <name> defaults to the ONNX file name. Next to them, <name>.build.json records
what went in (ONNX and model script checksums, seed, calibration list, versions) and what came
out (HEF checksum, stage times), so any HEF can be traced and rebuilt.

usage:
  python yolo/scripts/04-pytorch-to-hailo.py <model.onnx> <calib_dir> <model_script.alls> \
      [--seed 0] [--images 1024] [--name upper_body_n_seed0] [--out-dir build/]
"""
import argparse
import hashlib
import json
import platform
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from hailo_sdk_client import ClientRunner

END_NODES = [
    "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv4.0/cv4.0.2/Conv",
    "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv4.1/cv4.1.2/Conv",
    "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv", "/model.22/cv4.2/cv4.2.2/Conv",
]


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "unknown"


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("onnx", type=Path)
    ap.add_argument("calib_dir", type=Path, help="folder of calibration images (e.g. data/images/train2017)")
    ap.add_argument("model_script", type=Path, help=".alls model script")
    ap.add_argument("--seed", type=int, default=0, help="seed for the random calibration draw (default 0)")
    ap.add_argument("--images", type=int, default=1024, help="number of calibration images (default 1024)")
    ap.add_argument("--hw-arch", default="hailo8", help="hailo8 (default), hailo8l, ...")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", help="output name (default: the ONNX file name)")
    ap.add_argument("--out-dir", type=Path, help="output folder (default: next to the ONNX)")
    args = ap.parse_args()

    name = args.name or args.onnx.stem
    out_dir = args.out_dir or args.onnx.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    har = out_dir / f"{name}.har"
    har_opt = out_dir / f"{name}_optimized.har"
    hef = out_dir / f"{name}.hef"
    # the cache name carries everything that changes the draw, so different settings never share a cache
    cache = out_dir / f"calib_{args.calib_dir.name}_n{args.images}_seed{args.seed}_{args.imgsz}.npy"
    versions = {"hailo_dataflow_compiler": pkg_version("hailo_dataflow_compiler"),
                "hailo_model_zoo": pkg_version("hailo_model_zoo"), "python": platform.python_version()}
    print(f"[info] {name}: DFC {versions['hailo_dataflow_compiler']}, hw_arch {args.hw_arch}, "
          f"model script {args.model_script}, seed {args.seed}")
    record = {
        "name": name, "started": datetime.now().isoformat(timespec="seconds"), "host": platform.node(),
        "command": " ".join(sys.argv), "versions": versions, "hw_arch": args.hw_arch,
        "onnx": {"path": str(args.onnx.resolve()), "md5": md5(args.onnx)},
        "model_script": {"path": str(args.model_script.resolve()), "md5": md5(args.model_script),
                         "text": args.model_script.read_text()},
        "calibration": {"dir": str(args.calib_dir.resolve()), "images": args.images, "seed": args.seed,
                        "imgsz": args.imgsz, "list": str(cache.with_suffix(".txt"))},
        "seconds": {},
    }
    t = time.time()

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
    record["seconds"]["translate"] = round(time.time() - t)
    t = time.time()

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

    record["seconds"]["calibration_set"] = round(time.time() - t)
    t = time.time()

    # 3. optimise (quantise) with the model script
    runner.load_model_script(str(args.model_script))
    runner.optimize(calib)
    runner.save_har(str(har_opt))
    record["seconds"]["optimize"] = round(time.time() - t)
    t = time.time()

    # 4. compile
    print("[info] compiling to HEF...")
    hef.write_bytes(runner.compile())
    record["seconds"]["compile"] = round(time.time() - t)
    record["finished"] = datetime.now().isoformat(timespec="seconds")
    record["hef"] = {"path": str(hef.resolve()), "md5": md5(hef), "bytes": hef.stat().st_size}
    (out_dir / f"{name}.build.json").write_text(json.dumps(record, indent=1))
    print(f"[success] {hef} (build record {out_dir / f'{name}.build.json'})")


if __name__ == "__main__":
    main()
