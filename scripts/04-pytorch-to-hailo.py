import sys
import os
import random
import numpy as np
from pathlib import Path
from PIL import Image
from hailo_sdk_client import ClientRunner

# 1. Setup Paths
if len(sys.argv) < 4:
    print("Usage: python convert.py <model.onnx> <calib_dir> <model_script.alls>")
    sys.exit(1)

ONNX_FILE = Path(sys.argv[1])
CALIBRATION_IMAGES_PATH = Path(sys.argv[2])
MODEL_SCRIPT = Path(sys.argv[3])

model_name = ONNX_FILE.stem
parent_dir = ONNX_FILE.parent

STANDARD_HAR = parent_dir / f"{model_name}.har"
OPTIMIZED_HAR = parent_dir / f"{model_name}_optimized.har"
HEF_OUTPUT = parent_dir / f"{model_name}.hef"
# Cache file named after the dataset folder to avoid mixing up different datasets
NPY_CACHE = parent_dir / f"calib_set_{CALIBRATION_IMAGES_PATH.name}.npy"

# 2. Translation Stage
runner = ClientRunner(hw_arch="hailo8")
runner.translate_onnx_model(
    ONNX_FILE,
    model_name,
    start_node_names=["images"],
    end_node_names=[
        "/model.22/cv2.0/cv2.0.2/Conv", "/model.22/cv3.0/cv3.0.2/Conv", "/model.22/cv4.0/cv4.0.2/Conv",
        "/model.22/cv2.1/cv2.1.2/Conv", "/model.22/cv3.1/cv3.1.2/Conv", "/model.22/cv4.1/cv4.1.2/Conv",
        "/model.22/cv2.2/cv2.2.2/Conv", "/model.22/cv3.2/cv3.2.2/Conv", "/model.22/cv4.2/cv4.2.2/Conv"
    ],
    net_input_shapes={"images": [1, 3, 640, 640]}
)
runner.save_har(str(STANDARD_HAR))

# 3. Calibration Dataset (with Caching & Randomization)
MAX_IMAGES = 1024

if NPY_CACHE.exists():
    print(f"[info] Loading calibration data from cache: {NPY_CACHE}")
    calib_dataset = np.load(NPY_CACHE)
else:
    print(f"[info] Cache not found. Preparing new calibration dataset...")
    all_images = [img for img in CALIBRATION_IMAGES_PATH.iterdir() if img.suffix.lower() in ('.jpg', '.jpeg', '.png')]
    
    # 1. Randomize
    random.shuffle(all_images)
    
    # 2. Select Subset
    selected_images = all_images[:MAX_IMAGES]
    calib_dataset = np.zeros((len(selected_images), 640, 640, 3), dtype=np.float32)

    for idx, img_path in enumerate(selected_images):
        img = Image.open(img_path).convert('RGB').resize((640, 640))
        calib_dataset[idx] = np.array(img)
    
    # 3. Save Cache
    np.save(NPY_CACHE, calib_dataset)
    print(f"[info] Calibration data cached to: {NPY_CACHE}")

# 4. Optimization Stage
runner.load_model_script(str(MODEL_SCRIPT))
runner.optimize(calib_dataset)
runner.save_har(str(OPTIMIZED_HAR))

# 5. Compilation Stage
print("[info] Compiling to HEF...")
hef_binary = runner.compile()
with open(HEF_OUTPUT, "wb") as f:
    f.write(hef_binary)

print(f"[success] Process complete. HEF: {HEF_OUTPUT}")