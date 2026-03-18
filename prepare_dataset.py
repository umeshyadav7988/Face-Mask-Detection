"""
Face Mask Detection - Dataset Preparation
Downloads and organizes a face mask dataset for training.

Dataset source: GitHub-hosted curated face mask dataset
Fallback: Creates a synthetic mini-dataset for quick testing.
"""

import os
import urllib.request
import zipfile
import shutil
import sys
import numpy as np
from PIL import Image, ImageDraw


# ─── Configuration ──────────────────────────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
WITH_MASK_DIR = os.path.join(DATASET_DIR, "with_mask")
WITHOUT_MASK_DIR = os.path.join(DATASET_DIR, "without_mask")

# Public dataset URL (Prajna Bhandary's face mask dataset)
DATASET_URL = (
    "https://github.com/prajnasb/observations/raw/master/experiements/"
    "data/with_and_without_mask_data.zip"
)


def download_dataset():
    """Download the face mask dataset from GitHub."""
    os.makedirs(DATASET_DIR, exist_ok=True)
    zip_path = os.path.join(DATASET_DIR, "dataset.zip")

    print("📥 Downloading face mask dataset...")
    print(f"   Source: {DATASET_URL}")

    try:
        urllib.request.urlretrieve(DATASET_URL, zip_path, _download_progress)
        print("\n✅ Download complete!")

        print("📦 Extracting dataset...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(DATASET_DIR)

        # Organize into with_mask / without_mask folders
        _organize_extracted_data()

        # Clean up zip
        os.remove(zip_path)
        print("✅ Dataset ready!")
        return True

    except Exception as e:
        print(f"\n⚠️  Download failed: {e}")
        print("   Falling back to synthetic dataset generation...")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        return False


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 // total_size)
        bar = "█" * (percent // 2) + "░" * (50 - percent // 2)
        sys.stdout.write(f"\r   [{bar}] {percent}%")
        sys.stdout.flush()


def _organize_extracted_data():
    """Move extracted images into the right folder structure."""
    os.makedirs(WITH_MASK_DIR, exist_ok=True)
    os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

    # Walk through extracted directories and find image folders
    for root, dirs, files in os.walk(DATASET_DIR):
        folder_name = os.path.basename(root).lower()

        for f in files:
            if not f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                continue

            src = os.path.join(root, f)

            if "with_mask" in folder_name or "with_mask" in root.lower():
                dst = os.path.join(WITH_MASK_DIR, f)
            elif "without_mask" in folder_name or "without_mask" in root.lower():
                dst = os.path.join(WITHOUT_MASK_DIR, f)
            else:
                continue

            if src != dst and not os.path.exists(dst):
                shutil.copy2(src, dst)

    # Clean up any extra extracted directories
    for item in os.listdir(DATASET_DIR):
        item_path = os.path.join(DATASET_DIR, item)
        if os.path.isdir(item_path) and item not in ("with_mask", "without_mask"):
            shutil.rmtree(item_path)


def generate_synthetic_dataset(num_per_class=200):
    """
    Generate a small synthetic dataset for testing when the real dataset
    can't be downloaded. Creates simple face-like images with/without mask regions.
    """
    print(f"🎨 Generating synthetic dataset ({num_per_class} images per class)...")

    os.makedirs(WITH_MASK_DIR, exist_ok=True)
    os.makedirs(WITHOUT_MASK_DIR, exist_ok=True)

    for i in range(num_per_class):
        # --- Face WITHOUT mask ---
        img = _create_face_image(with_mask=False)
        img.save(os.path.join(WITHOUT_MASK_DIR, f"no_mask_{i:04d}.png"))

        # --- Face WITH mask ---
        img = _create_face_image(with_mask=True)
        img.save(os.path.join(WITH_MASK_DIR, f"mask_{i:04d}.png"))

        if (i + 1) % 50 == 0:
            print(f"   Generated {i + 1}/{num_per_class} pairs...")

    print("✅ Synthetic dataset created!")


def _create_face_image(with_mask=False, size=224):
    """Create a simple synthetic face image."""
    # Random background color
    bg_color = tuple(np.random.randint(180, 240, 3).tolist())
    img = Image.new("RGB", (size, size), bg_color)
    draw = ImageDraw.Draw(img)

    # Skin tone
    skin_tones = [
        (255, 224, 189), (255, 205, 148), (234, 192, 134),
        (255, 173, 96), (198, 134, 66), (141, 85, 36)
    ]
    skin = skin_tones[np.random.randint(0, len(skin_tones))]

    cx, cy = size // 2, size // 2
    rx, ry = 55 + np.random.randint(-10, 10), 70 + np.random.randint(-10, 10)

    # Face oval
    draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=skin)

    # Eyes
    eye_y = cy - 15 + np.random.randint(-5, 5)
    eye_offset = 20 + np.random.randint(-3, 3)
    draw.ellipse([cx - eye_offset - 6, eye_y - 4, cx - eye_offset + 6, eye_y + 4], fill="white")
    draw.ellipse([cx + eye_offset - 6, eye_y - 4, cx + eye_offset + 6, eye_y + 4], fill="white")
    draw.ellipse([cx - eye_offset - 3, eye_y - 2, cx - eye_offset + 3, eye_y + 2], fill=(50, 50, 50))
    draw.ellipse([cx + eye_offset - 3, eye_y - 2, cx + eye_offset + 3, eye_y + 2], fill=(50, 50, 50))

    if with_mask:
        # Draw a mask covering nose and mouth area
        mask_colors = [(0, 120, 215), (255, 255, 255), (100, 180, 100), (70, 70, 70)]
        mask_color = mask_colors[np.random.randint(0, len(mask_colors))]
        mask_top = cy + 2
        mask_bottom = cy + ry - 5
        draw.polygon(
            [
                (cx - rx + 10, mask_top),
                (cx + rx - 10, mask_top),
                (cx + rx - 15, mask_bottom),
                (cx - rx + 15, mask_bottom),
            ],
            fill=mask_color,
        )
    else:
        # Draw nose and mouth
        draw.polygon(
            [(cx, cy + 5), (cx - 5, cy + 18), (cx + 5, cy + 18)],
            fill=tuple(max(0, c - 30) for c in skin),
        )
        mouth_y = cy + 30 + np.random.randint(-3, 3)
        draw.arc(
            [cx - 15, mouth_y - 5, cx + 15, mouth_y + 5],
            0, 180, fill=(200, 50, 50), width=2,
        )

    # Add slight noise for realism
    img_array = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 5, img_array.shape)
    img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(img_array)


def get_dataset_stats():
    """Print dataset statistics."""
    with_mask_count = len([
        f for f in os.listdir(WITH_MASK_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]) if os.path.exists(WITH_MASK_DIR) else 0

    without_mask_count = len([
        f for f in os.listdir(WITHOUT_MASK_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    ]) if os.path.exists(WITHOUT_MASK_DIR) else 0

    print("\n📊 Dataset Statistics:")
    print(f"   With Mask:    {with_mask_count} images")
    print(f"   Without Mask: {without_mask_count} images")
    print(f"   Total:        {with_mask_count + without_mask_count} images")

    return with_mask_count, without_mask_count


if __name__ == "__main__":
    print("=" * 60)
    print("  🎭 Face Mask Detection - Dataset Preparation")
    print("=" * 60)

    # Try downloading first; fall back to synthetic
    if not download_dataset():
        generate_synthetic_dataset(num_per_class=300)

    get_dataset_stats()
    print("\n🎉 Dataset preparation complete!")
