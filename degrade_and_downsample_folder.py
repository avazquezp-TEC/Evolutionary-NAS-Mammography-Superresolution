"""
degrade_and_downsample_folder.py
=================================
Generates low-resolution (LR) images from a directory of high-resolution (HR)
images by applying the six downsampling methods used in the MST framework:

Nearest Neighbor · Bilinear · Bicubic · Lanczos · Box · Hamming

For each HR image, six LR images are generated (one per downsampling method).
Images are processed in their entirety (no patch extraction is performed).

The implementation uses PIL, matching the preprocessing pipeline employed during
MST fine-tuning (02_finetune_mammo_SR.py).

Supported formats: PNG, JPEG, BMP, TIFF/TIF, and PGM (8-bit and 16-bit,
grayscale and color images).

The downsampling method applied to each generated image is recorded in a CSV file.

Updated: 2026-06-16
"""

import os
import csv
import numpy as np
from PIL import Image

# ── MST method definitions (the same 6 methods used during fine-tuning) ─────
MST_METHODS = [
    (Image.NEAREST, "nearest"),
    (Image.BILINEAR, "bilinear"),
    (Image.BICUBIC, "bicubic"),
    (Image.LANCZOS, "lanczos"),
    (Image.BOX, "box"),
    (Image.HAMMING, "hamming"),
]
MST_NAMES = [name for _, name in MST_METHODS]

VALID_EXTENSIONS = ('.tif', '.tiff', '.pgm', '.png', '.jpg', '.jpeg', '.bmp')


def _load_pil(path: str) -> Image.Image:
    """Loads an image with PIL, preserving its bit depth."""
    img = Image.open(path)
    # Force full load before closing the file
    img.load()
    return img


def _downsample_pil(img: Image.Image, new_w: int, new_h: int, resample_method) -> Image.Image:
    """Downsizes an image using the selected resampling method."""
    if img.mode == 'I':
        # PIL mode 'I' = int32; HAMMING/BOX don't support mode 'I' directly
        arr = np.array(img, dtype=np.int32)
        pil_tmp = Image.fromarray(arr, mode='I')
        pil_lr = pil_tmp.resize((new_w, new_h), resample=resample_method)
        return pil_lr  # stays in mode 'I'
    else:
        return img.resize((new_w, new_h), resample=resample_method)


def _save_pil(img: Image.Image, path: str) -> None:
    """Saves the PIL image to disk, handling mode 'I' (16-bit)."""
    if img.mode == 'I':
        # Convert from int32 to uint16 to save as 16-bit TIFF
        arr = np.array(img, dtype=np.int32).clip(0, 65535).astype(np.uint16)
        Image.fromarray(arr, mode='I;16').save(path)
    else:
        img.save(path)


def generate_lr_images(
    input_dir: str,
    output_dir: str,
    scale_factor: float = 0.5,
    csv_path: str = 'scaling_log.csv'
) -> None:
    """
    For each HR image in `input_dir`, generates six LR versions (one for each
    MST downsampling method) and stores them in subdirectories under `output_dir`:
    output_dir/
        nearest/   image.png
        bilinear/  image.png
        bicubic/   image.png
        lanczos/   image.png
        box/       image.png
        hamming/   image.png

    Parameters
    ----------
    input_dir    : Directory containing the original HR images.
    output_dir   : Root directory where the generated LR images will be saved.
    scale_factor : Downsampling factor (e.g. 0.5 reduces image dimensions by half).
    csv_path     : Path to the CSV file used to log processed images.
    """
    # Create one subdirectory per method
    for name in MST_NAMES:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    with open(csv_path, mode='w', newline='', encoding='utf-8') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['Filename', 'Method', 'Scale Factor'])

        for filename in sorted(os.listdir(input_dir)):
            if not filename.lower().endswith(VALID_EXTENSIONS):  # skip non-image files
                continue

            img_path = os.path.join(input_dir, filename)
            try:
                img_hr = _load_pil(img_path)
            except Exception as e:
                print(f"  [SKIP] Can't open file {filename}: {e}")
                continue

            # Compute new dimensions
            w_hr, h_hr = img_hr.size
            new_w = max(1, int(w_hr * scale_factor))
            new_h = max(1, int(h_hr * scale_factor))

            print(f"  {filename}  ({w_hr}x{h_hr} -> {new_w}x{new_h})")

            # Generate LR images for all methods
            for resample, method_name in MST_METHODS:
                try:
                    img_lr = _downsample_pil(img_hr, new_w, new_h, resample)
                    output_img_path = os.path.join(output_dir, method_name, filename)
                    _save_pil(img_lr, output_img_path)

                    writer.writerow([filename, method_name, scale_factor])
                    print(f"    OK {method_name}")
                except Exception as e:
                    print(f"    FAIL {method_name}: {e}")

    print(f"\nLog saved to: {csv_path}")


# ── Example usage ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    high_res_dir = 'Data/High/Set5'
    low_res_dir = 'Data/Lowx2/Set5'
    generate_lr_images(
        high_res_dir,
        low_res_dir,
        scale_factor=0.5,
        csv_path='scaling_log_2x.csv'
    )
