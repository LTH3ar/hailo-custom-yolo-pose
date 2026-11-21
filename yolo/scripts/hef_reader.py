"""Show which layers a HAR's output layers and model-script entries point at.

The .alls model scripts name layers by position (conv71, output_layer3, ...). Positions can
differ between models and compiler versions, so before trusting a compile, check that the
sigmoid entries sit on the class-score convs (cv3) and the 16-bit entries on the keypoint
outputs (cv4).

usage (inside the Hailo venv):
  python yolo/scripts/hef_reader.py model_optimized.har [--alls yolo/config/yolov8s_pose.alls] [--raw]

--raw prints the whole HN dictionary, as the first version of this script did.
"""
import argparse
import json
import re

from hailo_sdk_client import ClientRunner


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("har")
    ap.add_argument("--alls", help="model script whose layer names to look up")
    ap.add_argument("--raw", action="store_true", help="print the whole HN dictionary")
    args = ap.parse_args()

    hn = ClientRunner(har=args.har).get_hn_dict()
    if args.raw:
        print(json.dumps(hn, indent=1, default=str))
        return
    layers = hn.get("layers", {})
    net = hn.get("name", "")

    def find(short):
        # HN layer names are usually "<network>/<short name>"
        for key in (short, f"{net}/{short}"):
            if key in layers:
                return key, layers[key]
        for key, val in layers.items():
            if key.split("/")[-1] == short:
                return key, val
        return None, None

    def describe(key, val):
        orig = val.get("original_names") or []
        return f"{key:40} {val.get('type', '?'):18} {', '.join(orig) if orig else '-'}"

    print("output layers (and the layer feeding each):")
    for key, val in layers.items():
        if val.get("type") == "output_layer":
            src = (val.get("input") or ["?"])[0]
            _, sv = find(src.split("/")[-1])
            print(" ", describe(key, val), "<-", describe(src, sv) if sv else src)

    if args.alls:
        names = sorted(set(re.findall(r"\b(conv\d+|output_layer\d+)\b", open(args.alls).read())),
                       key=lambda n: (n.rstrip("0123456789"), int(re.sub(r"\D", "", n))))
        print(f"\nlayers named in {args.alls}:")
        for n in names:
            key, val = find(n)
            print(" ", describe(key, val) if val else f"{n}: not in this HAR")
            if val and val.get("type") == "output_layer":
                src = (val.get("input") or ["?"])[0]
                _, sv = find(src.split("/")[-1])
                if sv:
                    print("    fed by", describe(src, sv))
    print("\ncv3 = class scores (should carry the sigmoid); cv4 = keypoints (should be 16-bit).")


if __name__ == "__main__":
    main()
