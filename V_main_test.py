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
        Data/Lowx2/mammo_val/nearest/
        Data/Lowx2/mammo_val/bilinear/

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
import glob
import argparse
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser( description=("Super-resolution Test uning improve Med-BSR degradation and 6-method MST downsampling."))
    p.add_argument("--dataset", default="Data/High/mammo_val", type=str,
                   help="Path to the directory containing High-Resolution (HR) ground truth images.")
    p.add_argument("--scale",   default=4, type=int,
                   help="Upsampling scale factor for Super-Resolution (e.g., 2, 4).")
    p.add_argument("--genes_dir",  metavar="DIR", required=True,
                   help="Root directory containing the trained NAS sub-models (e.g., gene_001/, gene_002/).")
    p.add_argument("--out_dir",   default="Data/Outx4/mammo_val",
                   help="Root directory where evaluation outputs, checkpoints, and logs will be saved.")
    p.add_argument("--bicubic", action="store_false",
                   help="Disable the full pipeline and only execute bicubic downsampling (default: False).")
    p.add_argument("--lr_path",  metavar="DIR", default="",
                   help="Optional path to pre-degraded Low-Resolution (LR) images. If not provided, LR images will be generated on-the-fly from the HR dataset.")
    return p.parse_args()
def main() -> None:
    args = parse_args()
    # --- 1. Configuration constants ────────────────────────────────────────
    SCALE_FACTOR = args.scale
    MST = args.bicubic
    BASE_PATH = args.genes_dir
    HR_PATH = args.dataset
    LR_PATH = args.lr_path
    OUT_BASE_PATH = args.out_dir

    # --- 4. Define experiments, one entry per generation ────────────────────
    pattern_genes = os.path.join(BASE_PATH, "gene_*")
    gene_paths = sorted(glob.glob(pattern_genes))

    EXPERIMENTS = []
    for gene_path in gene_paths: #Search genes folder in base path
        if os.path.isdir(gene_path):
            pattern_keras = os.path.join(gene_path, "*best*.keras") # find any best keras file
            matching_files = glob.glob(pattern_keras)
            if matching_files:
                keras_file = matching_files[0] #get complete path
                gene_name = os.path.basename(gene_path)
                EXPERIMENTS.append({
                    "model_path": keras_file,
                    "output_path": f"{OUT_BASE_PATH}/{gene_name}"
                })

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
                    csv_path=f"register_psnr_ssim_{method}.csv"
                )
        else:
            run_super_resolution_and_psnr(
                model_path=exp["model_path"],
                low_res_path=LR_PATH,
                ground_truth_path=HR_PATH,
                output_image_path=exp["output_path"],
                scale_factor=SCALE_FACTOR,
                csv_path="register_psnr_ssim.csv"
            )

if __name__ == "__main__":
    main()
