#!/bin/bash
# =============================================================================
# COCO Dataset Download Script for YOLO Pose Training
# =============================================================================
# Downloads COCO 2017 dataset with keypoint annotations
# Required for training YOLO Pose models
# =============================================================================

set -e

# Configuration
DATA_DIR="${1:-./data/coco}"
DOWNLOAD_TRAIN="${DOWNLOAD_TRAIN:-true}"
DOWNLOAD_VAL="${DOWNLOAD_VAL:-true}"
CLEANUP="${CLEANUP:-false}"  # Set to true for non-interactive cleanup

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Cleanup partial files on unexpected exit
trap_cleanup() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        log_error "Script failed with exit code $exit_code"
        log_warn "Partial downloads are preserved for resume. Re-run the script to continue."
    fi
}
trap trap_cleanup EXIT

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    if ! command -v wget &> /dev/null && ! command -v curl &> /dev/null; then
        log_error "Neither wget nor curl found. Please install one of them."
        exit 1
    fi
    
    if ! command -v unzip &> /dev/null; then
        log_error "unzip not found. Please install it."
        exit 1
    fi
    
    log_info "All dependencies satisfied."
}

# Download function with resume support
download_file() {
    local url=$1
    local output=$2
    
    if [ -f "$output" ]; then
        log_warn "File $output already exists, skipping download."
        return 0
    fi
    
    log_info "Downloading: $url"
    
    if command -v wget &> /dev/null; then
        wget -c --progress=bar:force "$url" -O "$output"
    else
        curl -L -C - "$url" -o "$output"
    fi
}

# Create directory structure
create_directories() {
    log_info "Creating directory structure..."
    mkdir -p "$DATA_DIR/images/train2017"
    mkdir -p "$DATA_DIR/images/val2017"
    mkdir -p "$DATA_DIR/annotations"
    mkdir -p "$DATA_DIR/labels/train2017"
    mkdir -p "$DATA_DIR/labels/val2017"
    mkdir -p "$DATA_DIR/downloads"
}

# Download COCO images and annotations
download_coco() {
    local DOWNLOAD_DIR="$DATA_DIR/downloads"
    
    # COCO 2017 URLs
    TRAIN_IMAGES_URL="http://images.cocodataset.org/zips/train2017.zip"
    VAL_IMAGES_URL="http://images.cocodataset.org/zips/val2017.zip"
    TEST_IMAGES_URL="http://images.cocodataset.org/zips/test2017.zip"
    ANNOTATIONS_URL="http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
    
    # Download annotations (always needed)
    log_info "Downloading COCO annotations..."
    download_file "$ANNOTATIONS_URL" "$DOWNLOAD_DIR/annotations_trainval2017.zip"
    
    # Download training images
    if [ "$DOWNLOAD_TRAIN" = true ]; then
        log_info "Downloading COCO train2017 images (~18GB)..."
        download_file "$TRAIN_IMAGES_URL" "$DOWNLOAD_DIR/train2017.zip"
    fi
    
    # Download validation images
    if [ "$DOWNLOAD_VAL" = true ]; then
        log_info "Downloading COCO val2017 images (~1GB)..."
        download_file "$VAL_IMAGES_URL" "$DOWNLOAD_DIR/val2017.zip"
    fi

}

# Extract a zip file with .done marker to track completion
extract_with_marker() {
    local zip_file=$1
    local dest_dir=$2
    local marker_name=$3
    local done_marker="$DATA_DIR/.${marker_name}.extracted"
    
    if [ -f "$done_marker" ]; then
        log_warn "$marker_name already extracted (marker found)."
        return 0
    fi
    
    if [ ! -f "$zip_file" ]; then
        log_warn "Zip file not found: $zip_file"
        return 0
    fi
    
    log_info "Extracting $marker_name (this may take a while)..."
    unzip -q "$zip_file" -d "$dest_dir"
    touch "$done_marker"
    log_info "$marker_name extraction complete."
}

# Extract downloaded files
extract_files() {
    # Extract annotations
    extract_with_marker \
        "$DATA_DIR/downloads/annotations_trainval2017.zip" \
        "$DATA_DIR" \
        "annotations"
    
    # Extract train images
    if [ "$DOWNLOAD_TRAIN" = true ]; then
        extract_with_marker \
            "$DATA_DIR/downloads/train2017.zip" \
            "$DATA_DIR/images" \
            "train2017"
    fi

    # Extract val images
    if [ "$DOWNLOAD_VAL" = true ]; then
        extract_with_marker \
            "$DATA_DIR/downloads/val2017.zip" \
            "$DATA_DIR/images" \
            "val2017"
    fi

}

# Cleanup downloaded zip files
cleanup() {
    local DOWNLOAD_DIR="$DATA_DIR/downloads"
    
    if [ "$CLEANUP" = true ]; then
        log_info "Removing zip files (CLEANUP=true)..."
        rm -rf "$DOWNLOAD_DIR"
        log_info "Cleanup complete."
        return 0
    fi
    
    # Only prompt if running interactively
    if [ -t 0 ]; then
        read -p "Do you want to remove downloaded zip files to save space? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            log_info "Removing zip files..."
            rm -rf "$DOWNLOAD_DIR"
            log_info "Cleanup complete."
        else
            log_info "Keeping zip files for potential re-extraction."
        fi
    else
        log_info "Non-interactive mode: keeping zip files. Set CLEANUP=true to auto-remove."
    fi
}

# Verify download
verify_download() {
    log_info "Verifying download..."
    
    local errors=0
    
    # Check annotations
    if [ ! -f "$DATA_DIR/annotations/person_keypoints_train2017.json" ]; then
        log_error "Missing: person_keypoints_train2017.json"
        errors=$((errors + 1))
    fi
    
    if [ ! -f "$DATA_DIR/annotations/person_keypoints_val2017.json" ]; then
        log_error "Missing: person_keypoints_val2017.json"
        errors=$((errors + 1))
    fi
    
    # Check images (using find instead of ls glob to handle 100K+ files)
    if [ "$DOWNLOAD_TRAIN" = true ]; then
        local train_count
        train_count=$(find "$DATA_DIR/images/train2017" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l)
        log_info "Train images found: $train_count"
        if [ "$train_count" -lt 100000 ]; then
            log_warn "Expected ~118K train images, found $train_count"
        fi
    fi
    
    if [ "$DOWNLOAD_VAL" = true ]; then
        local val_count
        val_count=$(find "$DATA_DIR/images/val2017" -maxdepth 1 -name "*.jpg" 2>/dev/null | wc -l)
        log_info "Val images found: $val_count"
        if [ "$val_count" -lt 4000 ]; then
            log_warn "Expected ~5K val images, found $val_count"
        fi
    fi
    
    if [ "$errors" -eq 0 ]; then
        log_info "Download verification passed!"
    else
        log_error "Download verification failed with $errors errors."
        exit 1
    fi
}

# Print summary
print_summary() {
    echo ""
    echo "=============================================="
    echo "COCO Dataset Download Complete!"
    echo "=============================================="
    echo "Data directory: $DATA_DIR"
    echo ""
    echo "Directory structure:"
    echo "  $DATA_DIR/"
    echo "  ├── images/"
    echo "  │   ├── train2017/    # Training images"
    echo "  │   └── val2017/      # Validation images"
    echo "  ├── annotations/      # COCO JSON annotations"
    echo "  │   ├── person_keypoints_train2017.json"
    echo "  │   └── person_keypoints_val2017.json"
    echo "  └── labels/           # YOLO format labels (to be generated)"
    echo ""
    echo "Next step: Run the dataset conversion script"
    echo "  python scripts/02_convert_to_upper_body.py"
    echo "=============================================="
}

# Main execution
main() {
    echo "=============================================="
    echo "COCO Dataset Download Script"
    echo "=============================================="
    echo ""
    
    check_dependencies
    create_directories
    download_coco
    extract_files
    verify_download
    print_summary
    
    # Optional cleanup
    if [ -d "$DATA_DIR/downloads" ]; then
        cleanup
    fi
}

# Run main
main
