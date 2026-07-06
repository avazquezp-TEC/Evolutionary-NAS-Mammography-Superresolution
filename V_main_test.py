"""
This script tests Single Image Super-Resolution (SISR) models by calculating
the PSNR and SSIM between the generated HR images and the original ground truth.

Requirements:
- Models must be generated using '../SRIR/trainSR.py'.
- Define the model paths to evaluate in the 'EXPERIMENTS' list.
- 'BASE_PATH' is the root directory where the generated models are located.
- High-Resolution (HR) and Low-Resolution (LR) image paths must be defined.
- If 'LR_PATH' is empty, LR images will be automatically generated from HR images
  using the MST degradation framework (6 PIL-based methods):
    Nearest Neighbor · Bilinear · Bicubic · Lanczos · Box · Hamming
  Each method's LR images are saved in their own subdirectory, e.g.:
        Data/Lowx2/Set5/nearest/
        Data/Lowx2/Set5/bilinear/

Set MST = False to skip the 6-method comparison and evaluate a single
pre-generated LR set located directly in LR_PATH (e.g. classic bicubic only).

Output:
- 'output_path' is the folder where the generated SR images and the
  PSNR/SSIM CSV log are saved for each experiment (and, if MST is enabled,
  for each downsampling method).

Updated 2026-06-16: SSIM added. Downsampling methods are evaluated one per image.
"""

from degrade_and_downsample_folder import generate_lr_images, MST_NAMES
from model_test import run_super_resolution_and_psnr
import os

if __name__ == "__main__":
    # --- 1. Configuration constants ────────────────────────────────────────
    SET = "mammo_val"
    SCALE_FACTOR = 4  # Scale factor (e.g. 2 or 4), depending on the model
    MST = True  # If False, the script only looks for images directly in LR_PATH
                # (e.g. a single pre-generated bicubic set), skipping the 6-method comparison

    # --- 2. Model folders ────────────────────────────────────────────────────
    BASE_PATH = f"../SRIR/outputs{SCALE_FACTOR}x/mammo_ft_blindSR"

    # --- 3. Image paths ──────────────────────────────────────────────────────
    HR_PATH = f"Data/High/{SET}"
    # Leave LR_PATH empty ("") to automatically generate low-resolution images
    LR_PATH = f"Data/Lowx{SCALE_FACTOR}/{SET}"
    OUT_BASE_PATH = f"Data/Outx{SCALE_FACTOR}/{SET}"

    # --- 4. Define experiments, one entry per generation ────────────────────
    EXPERIMENTS = [
        {"model_path": f"{BASE_PATH}/gene_001/mammo_ft_best.keras", "output_path": f"{OUT_BASE_PATH}/gene_001"},
        {"model_path": f"{BASE_PATH}/gene_002/mammo_ft_best.keras", "output_path": f"{OUT_BASE_PATH}/gene_002"},
        {"model_path": f"{BASE_PATH}/gene_003/mammo_ft_best.keras", "output_path": f"{OUT_BASE_PATH}/gene_003"},
        {"model_path": f"{BASE_PATH}/gene_004/mammo_ft_best.keras", "output_path": f"{OUT_BASE_PATH}/gene_004"},
        {"model_path": f"{BASE_PATH}/gene_005/mammo_ft_best.keras", "output_path": f"{OUT_BASE_PATH}/gene_005"},
    ]

    # --- 5. Generate LR images if LR_PATH is empty ──────────────────────────
    if LR_PATH == "":
        LR_PATH = HR_PATH.replace("High", f"Lowx{SCALE_FACTOR}")
        # Check whether all MST method subfolders already exist
        methods_generated = all(os.path.isdir(os.path.join(LR_PATH, name)) for name in MST_NAMES)
        if not methods_generated:
            print(f"\nGenerating LR images (MST 6 methods) at: {LR_PATH}")
            generate_lr_images(
                HR_PATH,
                LR_PATH,
                scale_factor=1 / SCALE_FACTOR,
                csv_path=f"scaling_log_{SCALE_FACTOR}x.csv"
            )
        else:
            print(f"LR images found at: {LR_PATH} (skipping generation)")

    # --- 6. Evaluation per experiment and MST method ────────────────────────
    for exp in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Model: {exp['model_path']}")
        print(f"{'='*60}")
        if MST:
            for method in MST_NAMES:
                lr_method_path = os.path.join(LR_PATH, method)
                out_method_path = os.path.join(exp["output_path"], method)

                if not os.path.isdir(lr_method_path):
                    print(f"  [SKIP] Does not exist: {lr_method_path}")
                    continue

                print(f"\n  -> MST method: {method}")
                run_super_resolution_and_psnr(
                    model_path=exp["model_path"],
                    low_res_path=lr_method_path,
                    ground_truth_path=HR_PATH,
                    output_image_path=out_method_path,
                    scale_factor=SCALE_FACTOR,
                    csv_path=f"registro_psnr_ssim_{method}.csv"
                )
        else:
            run_super_resolution_and_psnr(
                model_path=exp["model_path"],
                low_res_path=LR_PATH,
                ground_truth_path=HR_PATH,
                output_image_path=exp["output_path"],
                scale_factor=SCALE_FACTOR,
                csv_path="registro_psnr_ssim.csv"
            )
