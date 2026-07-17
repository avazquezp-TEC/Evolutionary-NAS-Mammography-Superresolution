"""
Computes the average PSNR (dB) and SSIM per gene_{id} folder, grouped by
interpolation method, separately for each super-resolution scale factor.
Results are saved to CSV, and the boxplot figures from box_plot.py are
generated for each scale.

Expected structure (relative to the project root):
  Data/Outx{scale}/mammo_val/
    gene_{id}/
      bicubic/register_psnr_ssim_bicubic.csv
      bilinear/register_psnr_ssim_bilinear.csv
      box/register_psnr_ssim_box.csv
      hamming/register_psnr_ssim_hamming.csv
      lanczos/register_psnr_ssim_lanczos.csv
      nearest/register_psnr_ssim_nearest.csv

Run this script from the project root (the folder that contains "Data/",
alongside main.py).
"""

import os
import glob
import pandas as pd
import argparse
from box_plot import plot_method_comparison, plot_dataset_impact

# ──────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser( description=("Super-resolution analysis"))
    p.add_argument("--gene", default="gene_005", type=str,
                   help="Gen that will be evaluated")
    p.add_argument("--method",   default="nearest", type=str, choices=["bicubic", "bilinear", "box", "hamming", "lanczos", "nearest"],
                   help="Method that will be evaluated")
    p.add_argument("--out_dir",   default="Analysis",
                   help="Root directory where evaluation outputs, checkpoints, and logs will be saved.")
    p.add_argument("--format",   default="png",
                   help="Format to save figures.")
    return p.parse_args()

def find_genes(root: str) -> list[str]:
    """Returns the sorted list of gene_XXX folders."""
    pattern = os.path.join(root, "gene_[0-9][0-9][0-9]")
    return sorted(glob.glob(pattern))


def read_csv(path: str, gene_id: str, method: str, COL_PSNR: str, COL_SSIM: str) -> pd.DataFrame | None:
    """Reads a CSV and returns the relevant columns, or None on failure."""
    if not os.path.isfile(path):
        print(f"  [WARN] File not found: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [ERROR] Could not read {path}: {e}")
        return None

    for col in (COL_PSNR, COL_SSIM):
        if col not in df.columns:
            print(f"  [WARN] Column '{col}' missing in {path}")
            return None

    df = df[[COL_PSNR, COL_SSIM]].copy()
    df["gene"] = gene_id
    df["metodo"] = method
    return df

def main():
    args = parse_args()
    SCALES = [2, 4]
    COL_PSNR = "PSNR (dB)"
    COL_SSIM = "SSIM"
    METHODS = ["bicubic", "bilinear", "box", "hamming", "lanczos", "nearest"]
    DATASET_IMPACT_GENE = args.gene
    DATASET_IMPACT_METHOD = args.method
    format = args.format
    for scale in SCALES:
        """Computes and saves the averages for a single scale factor, then
        generates the corresponding boxplot figures."""
        root = os.path.join("Data", f"Outx{scale}", "mammo_val")
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"SCALE {scale}x  (root: {root})")
        print(sep)

        if not os.path.isdir(root):
            print(f"  [WARN] Folder '{root}' does not exist, skipping scale {scale}x")
            continue

        genes = find_genes(root)
        if not genes:
            print(f"  [WARN] No gene_### folders found in '{root}', skipping scale {scale}x")
            return
        print(f"Genes found: {len(genes)}\n")

        rows = []
        for gene_path in genes:
            gene_id = os.path.basename(gene_path)  # e.g. "gene_001"
            for method in METHODS:
                csv_name = f"register_psnr_ssim_{method}.csv"
                csv_path = os.path.join(gene_path, method, csv_name)
                df = read_csv(csv_path, gene_id, method, COL_PSNR, COL_SSIM)
                if df is not None:
                    rows.append(df)

        if not rows:
            print(f"  [ERROR] Could not read any CSV file for scale {scale}x")
            return

        data = pd.concat(rows, ignore_index=True)

        # ── Averages per gene and method ─────────────────────────────────────────
        mean_by_gene_method = (
            data
            .groupby(["gene", "metodo"])[[COL_PSNR, COL_SSIM]]
            .mean()
            .round(6)
            .reset_index()
            .rename(columns={COL_PSNR: "PSNR_mean", COL_SSIM: "SSIM_mean"})
        )

        # ── Global average per gene (all methods combined) ──────────────────────
        mean_by_gene = (
            data
            .groupby("gene")[[COL_PSNR, COL_SSIM]]
            .mean()
            .round(6)
            .reset_index()
            .rename(columns={COL_PSNR: "PSNR_mean", COL_SSIM: "SSIM_mean"})
        )

        # ── Save results ──────────────────────────────────────────────────────────

        os.makedirs(args.out_dir, exist_ok=True)
        out_detail = f"{args.out_dir}/mean_per_gene_method_x{scale}.csv"
        out_global = f"{args.out_dir}/mean_per_gene_gene_x{scale}.csv"

        mean_by_gene_method.to_csv(out_detail, index=False)
        mean_by_gene.to_csv(out_global, index=False)

        # ── Print to screen ───────────────────────────────────────────────────────
        line = "-" * 62
        print(line)
        print(f"AVERAGES PER GENE AND METHOD ({scale}x)")
        print(line)
        print(mean_by_gene_method.to_string(index=False))

        print(f"\n{line}")
        print(f"GLOBAL AVERAGE PER GENE ({scale}x, all methods)")
        print(line)
        print(mean_by_gene.to_string(index=False))

        print(f"\nResults saved to:")
        print(f"   - {out_detail}")
        print(f"   - {out_global}")

        # ── Generate figures for this scale ─────────────────────────────────────
        plot_method_comparison(out_detail, scale, format)

        dataset_impact_csv = os.path.join(
            root, DATASET_IMPACT_GENE, DATASET_IMPACT_METHOD,
            f"register_psnr_ssim_{DATASET_IMPACT_METHOD}.csv"
        )
        plot_dataset_impact(dataset_impact_csv, scale, format)

if __name__ == "__main__":
    main()
