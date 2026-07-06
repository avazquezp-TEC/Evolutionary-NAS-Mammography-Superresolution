"""
Computes the average PSNR (dB) and SSIM per gene_{id} folder, grouped by
interpolation method, separately for each super-resolution scale factor.
Results are saved to CSV, and the boxplot figures from bigotes.py are
generated for each scale.

Expected structure (relative to the project root):
  Data/Outx{scale}/mammo_val/
    gene_{id}/
      bicubic/registro_psnr_ssim_bicubic.csv
      bilinear/registro_psnr_ssim_bilinear.csv
      box/registro_psnr_ssim_box.csv
      hamming/registro_psnr_ssim_hamming.csv
      lanczos/registro_psnr_ssim_lanczos.csv
      nearest/registro_psnr_ssim_nearest.csv

Run this script from the project root (the folder that contains "Data/",
alongside main.py).
"""

import os
import glob
import pandas as pd

from box_plot import plot_method_comparison, plot_dataset_impact

# ── Configuration ────────────────────────────────────────────────────────────
SCALES = [2, 4]
METHODS = ["bicubic", "bilinear", "box", "hamming", "lanczos", "nearest"]
COL_PSNR = "PSNR (dB)"
COL_SSIM = "SSIM"
# Gene/method used for the "impact of source dataset" plot
DATASET_IMPACT_GENE = "gene_005"
DATASET_IMPACT_METHOD = "nearest"
# ──────────────────────────────────────────────────────────────────────────────


def find_genes(root: str) -> list[str]:
    """Returns the sorted list of gene_XXX folders."""
    pattern = os.path.join(root, "gene_[0-9][0-9][0-9]")
    return sorted(glob.glob(pattern))


def read_csv(path: str, gene_id: str, method: str) -> pd.DataFrame | None:
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


def process_scale(scale: int) -> None:
    """Computes and saves the averages for a single scale factor, then
    generates the corresponding boxplot figures."""
    root = os.path.join("Data", f"Outx{scale}", "mammo_val")
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"SCALE {scale}x  (root: {root})")
    print(sep)

    if not os.path.isdir(root):
        print(f"  [WARN] Folder '{root}' does not exist, skipping scale {scale}x")
        return

    genes = find_genes(root)
    if not genes:
        print(f"  [WARN] No gene_### folders found in '{root}', skipping scale {scale}x")
        return
    print(f"Genes found: {len(genes)}\n")

    rows = []
    for gene_path in genes:
        gene_id = os.path.basename(gene_path)  # e.g. "gene_001"
        for method in METHODS:
            csv_name = f"registro_psnr_ssim_{method}.csv"
            csv_path = os.path.join(gene_path, method, csv_name)
            df = read_csv(csv_path, gene_id, method)
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
    os.makedirs('results', exist_ok=True)
    out_detail = f"results/mean_per_gene_method_x{scale}.csv"
    out_global = f"results/mean_per_gene_gene_x{scale}.csv"

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
    plot_method_comparison(out_detail, scale)

    dataset_impact_csv = os.path.join(
        root, DATASET_IMPACT_GENE, DATASET_IMPACT_METHOD,
        f"registro_psnr_ssim_{DATASET_IMPACT_METHOD}.csv"
    )
    plot_dataset_impact(dataset_impact_csv, scale)


def main():
    for scale in SCALES:
        process_scale(scale)


if __name__ == "__main__":
    main()
