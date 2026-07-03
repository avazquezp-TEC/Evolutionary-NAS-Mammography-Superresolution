"""
Generates boxplot figures summarizing PSNR and SSIM results:

1. plot_method_comparison — average PSNR/SSIM per interpolation method,
   using the per-gene-per-method averages CSV produced by calcular_promedios.py.
2. plot_dataset_impact — PSNR/SSIM spread per source dataset, using a
   per-image results CSV (one gene/method combination).

Both functions are tagged with the scale factor (2x or 4x) and save their
output as PNG files (e.g. degradation_comparison_x4.eps). They are meant to
be called from calcular_promedios.py, once per scale factor, but can also be
run directly (see __main__ block below) against pre-generated CSVs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
# ==========================================
# GLOBAL STYLE CONFIGURATION (Times New Roman)
# ==========================================
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['font.size'] = 22

METHOD_ORDER = ['bicubic', 'bilinear', 'box', 'hamming', 'lanczos', 'nearest']
DATABASES = ['cdd-cesm', 'cmmd', 'ddsm', 'dmid', 'inbreast', 'kau-bcmd']
DATABASE_LABELS = ['CDD-CESM', 'CMMD', 'DDSM', 'DMID', 'INBreast', 'KAU-BCMD']


def extract_database(filename: str) -> str:
    """Returns the dataset name matching the start of the filename, or 'Other'."""
    for db in DATABASES:
        if str(filename).lower().startswith(db):
            return db
    return 'Other'


def plot_method_comparison(csv_path: str, scale: int, output_path: str | None = None) -> None:
    """
    Boxplots of average PSNR and SSIM by interpolation method.

    Parameters
    ----------
    csv_path    : CSV with columns 'metodo', 'PSNR_mean', 'SSIM_mean'
                  (output of calcular_promedios.py).
    scale       : Scale factor (2 or 4), used to tag the output filename.
    output_path : Output PNG path. Defaults to 'degradation_comparison_x{scale}.eps'.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  [WARN] File not found: {csv_path}, skipping method-comparison plot ({scale}x)")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: PSNR_mean ---
    sns.boxplot(ax=axes[0], data=df, x='metodo', y='PSNR_mean', order=METHOD_ORDER, palette='Set2')
    axes[0].set_title('Average PSNR by Method', weight='bold')
    axes[0].set_xlabel('Interpolation kernel')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].tick_params(axis='x', rotation=45)

    # --- Plot 2: SSIM_mean ---
    sns.boxplot(ax=axes[1], data=df, x='metodo', y='SSIM_mean', order=METHOD_ORDER, palette='Set2')
    axes[1].set_title('Average SSIM by Method', weight='bold')
    axes[1].set_xlabel('Interpolation kernel')
    axes[1].set_ylabel('SSIM')
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = f'results/degradation_comparison_x{scale}.eps'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}")


def plot_dataset_impact(csv_path: str, scale: int, output_path: str | None = None) -> None:
    """
    Boxplots of PSNR and SSIM per source dataset, for a single per-image
    results CSV (e.g. one gene/method combination).

    Parameters
    ----------
    csv_path    : CSV with columns 'Filename', 'PSNR (dB)', 'SSIM'
                  (output of model_test.py's run_super_resolution_and_psnr).
    scale       : Scale factor (2 or 4), used to tag the output filename.
    output_path : Output PNG path. Defaults to 'analysis_database_x{scale}.eps'.
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"  [WARN] File not found: {csv_path}, skipping dataset-impact plot ({scale}x)")
        return
    print(df)
    df['base_datos'] = df['Filename'].apply(extract_database)
    database_order = sorted(DATABASES)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Plot 1: PSNR (dB) ---
    sns.boxplot(ax=axes[0], data=df, x='base_datos', y='PSNR (dB)', order=database_order, palette='Accent')
    axes[0].set_title('Impact of the Source Dataset on PSNR', weight='bold')
    axes[0].set_xlabel('Data Base')
    axes[0].set_ylabel('PSNR (dB)')
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].set_xticklabels(DATABASE_LABELS)

    # --- Plot 2: SSIM ---
    sns.boxplot(ax=axes[1], data=df, x='base_datos', y='SSIM', order=database_order, palette='Accent')
    axes[1].set_title('Impact of the Source Dataset on SSIM', weight='bold')
    axes[1].set_xlabel('Data Base')
    axes[1].set_ylabel('SSIM')
    axes[1].tick_params(axis='x', rotation=30)
    axes[1].set_xticklabels(DATABASE_LABELS)

    plt.tight_layout()

    if output_path is None:
        os.makedirs('results', exist_ok=True)
        output_path = f'results/analysis_database_x{scale}.eps'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Figure saved: {output_path}")


# ── Standalone usage (normally these functions are called from calcular_promedios.py) ──
if __name__ == "__main__":
    plot_method_comparison('mean_per_gene_method_x4.csv', scale=4)
    plot_dataset_impact('Data/Outx4/mammo_val/gene_005/nearest/registro_psnr_ssim_nearest.csv', scale=4)
