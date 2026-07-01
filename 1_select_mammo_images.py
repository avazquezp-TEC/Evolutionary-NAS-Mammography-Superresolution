"""
source ~/model_test/venv/bin/activate
select_mammo_images.py
=========================
Selects balanced images from mammo-bench.csv for SR finetuning 
and PSNR benchmarking.

Sampling Strategy:
  - 6 databases: inbreast, kau-bcmd, cmmd, cdd-cesm, dmid, ddsm
  - TRAIN : 480 images → 80 per dataset → 20 per density (A, B, C, D)
  - VAL   :  96 images → 16 per dataset →  4 per density (A, B, C, D)
  - cmmd has no density info → sampled directly without density subgroups.
  - If any cell (dataset × density) has fewer images than required,
    all available images are taken and a warning is logged.

Outputs:
  mammo_train.csv          → list of 480 paths (original_source_path)
  mammo_val.csv            → list of 96 paths (distinct from train)
  mammo_split_summary.txt  → sampling summary with warnings
  DATASET/mammo_train/     → copy of training images
  DATASET/mammo_val/       → copy of validation images

Usage:
  python select_mammo_images.py --out_dir DATASET --copy

  # 15/06/26 initial version - cmmd has no density classification, samples randomly.
  cdd-cesm has 8 images of density A instead of the 24 required. It takes the 8 
  available for train and 0 for val.
"""

import argparse
import csv
import os
import random
import shutil
from pathlib import Path
import pandas as pd

# ── Configuration Constants ──────────────────────────────────────────────────
DATASETS = ["cdd-cesm", "cmmd", "ddsm", "dmid", "inbreast", "kau-bcmd"]
DENSITIES = ["A", "B", "C", "D"]
TRAIN_PER_DENSITY = 20   # per dataset × density
VAL_PER_DENSITY = 4      # per dataset × density

TRAIN_PER_DS = TRAIN_PER_DENSITY * len(DENSITIES)   # = 80
VAL_PER_DS = VAL_PER_DENSITY * len(DENSITIES)       # = 16

# For cmmd (no density info available), use the same totals
TRAIN_CMMD = TRAIN_PER_DS   # 80
VAL_CMMD = VAL_PER_DS       # 16
NO_DENSITY_DS = {"cmmd"}    # datasets without density metadata


def parse_args() -> argparse.Namespace:
    """Parses command line arguments."""
    p = argparse.ArgumentParser(description="Balanced mammography selection for SR finetuning")
    p.add_argument("--csv", default="mammo-bench.csv",
                   help="Path to the mammo-bench CSV file")
    p.add_argument("--base_dir", default="",
                   help="Root directory where images live "
                        "(prepended to preprocessed_image_path). "
                        "Leave empty if the CSV already contains absolute paths.")
    p.add_argument("--out_dir", default="DATASET",
                   help="Output directory for train/val subfolders")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="If set, COPIES the images to out_dir/mammo_train and mammo_val")
    p.add_argument("--path_col", default="preprocessed_image_path",
                   help="CSV column containing the relative image path")
    return p.parse_args()


def build_full_path(row: pd.Series, path_col: str, base_dir: str) -> str:
    """Constructs the full absolute or relative path for an image."""
    rel = str(row[path_col])
    if base_dir:
        return os.path.join(base_dir, rel)
    return rel


def sample_balanced_with_density(
    df_ds: pd.DataFrame,
    path_col: str,
    base_dir: str,
    n_per_density: int,
    excluded: set,
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """
    Samples n_per_density images per density (A, B, C, D) avoiding 'excluded' paths.
    
    Returns:
        tuple: (list of dicts of selected rows, list of warning strings)
    """
    selected = []
    warnings = []
    ds_name = df_ds["source_dataset"].iloc[0]

    for dens in DENSITIES:
        pool = df_ds[df_ds["density"] == dens]
        # Exclude already selected paths
        pool = pool[~pool[path_col].isin(excluded)]
        available = pool.to_dict("records")
        rng.shuffle(available)

        if len(available) < n_per_density:
            warnings.append(
                f"  WARNING [{ds_name}/density={dens}]: "
                f"only {len(available)} available (needed {n_per_density}). "
                f"Taking all available."
            )
        chosen = available[:n_per_density]
        for row in chosen:
            row["_full_path"] = build_full_path(row, path_col, base_dir)
            excluded.add(row[path_col])
        selected.extend(chosen)

    return selected, warnings


def sample_no_density(
    df_ds: pd.DataFrame,
    path_col: str,
    base_dir: str,
    n_total: int,
    excluded: set,
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """Samples n_total random images without using density subgroups."""
    pool = df_ds[~df_ds[path_col].isin(excluded)]
    available = pool.to_dict("records")
    rng.shuffle(available)
    warnings = []
    ds_name = df_ds["source_dataset"].iloc[0]

    if len(available) < n_total:
        warnings.append(
            f"  WARNING [{ds_name}/no density]: "
            f"only {len(available)} available (needed {n_total}). "
            f"Taking all available."
        )
    chosen = available[:n_total]
    for row in chosen:
        row["_full_path"] = build_full_path(row, path_col, base_dir)
        excluded.add(row[path_col])
    return chosen, warnings


def write_csv(rows: list[dict], path_col: str, out_path: str, base_dir: str) -> None:
    """Writes the selected image metadata records into a CSV file."""
    fieldnames = ["source_dataset", "density", path_col, "full_path",
                  "classification", "BIRADS", "laterality", "view"]
    
    # Only include keys that actually exist in the rows
    all_keys = set(rows[0].keys()) if rows else set()
    fieldnames = [f for f in fieldnames if f in all_keys or f == "full_path"]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            row["full_path"] = row.get("_full_path", "")
            w.writerow(row)
    print(f"  → {out_path}  ({len(rows)} images)")


def copy_images(rows: list[dict], dest_dir: str) -> None:
    """Copies the chosen images to the destination directory, renaming to avoid collisions."""
    os.makedirs(dest_dir, exist_ok=True)
    missing = 0
    for row in rows:
        src = row.get("_full_path", "")
        if not os.path.isfile(src):
            missing += 1
            continue
        
        # Preserve original filename but prefix it with the dataset name to avoid collisions
        ds = row.get("source_dataset", "unk")
        fname = f"{ds}_{os.path.basename(src)}"
        dst = os.path.join(dest_dir, fname)
        shutil.copy2(src, dst)
        
    if missing:
        print(f"    ⚠ {missing} files not found on disk (paths from CSV)")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    path_col = args.path_col

    print(f"\n{'='*60}")
    print(" BALANCED SELECTION MAMMO-BENCH → SR FINETUNING")
    print(f"{'='*60}")
    print(f"  CSV      : {args.csv}")
    print(f"  base_dir : {args.base_dir or '(absolute paths in CSV)'}")
    print(f"  out_dir  : {args.out_dir}")
    print(f"  seed     : {args.seed}")
    print(f"  copy     : {args.copy}")
    print()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"[CSV] {len(df)} rows | datasets: {sorted(df['source_dataset'].unique())}")

    # Normalize density column to uppercase and clear NaN entries for valid datasets
    df["density"] = df["density"].astype(str).str.strip().str.upper()
    df.loc[df["density"] == "NAN", "density"] = None

    # Verify that all target datasets exist within the CSV file
    found_ds = set(df["source_dataset"].unique())
    missing_ds = set(DATASETS) - found_ds
    if missing_ds:
        print(f"  ⚠ Datasets not found in CSV: {missing_ds}")

    # ── TRAIN Sampling ────────────────────────────────────────────────────────
    print("\n[TRAIN] Sampling images...")
    all_warnings = []
    excluded_paths: set = set()   # Track selected paths to prevent Train ↔ Val duplicates
    train_rows = []

    for ds in DATASETS:
        df_ds = df[df["source_dataset"] == ds].copy()
        if df_ds.empty:
            all_warnings.append(f"  WARNING: dataset '{ds}' not found in CSV.")
            continue

        if ds in NO_DENSITY_DS:
            rows, warns = sample_no_density(
                df_ds, path_col, args.base_dir,
                TRAIN_CMMD, excluded_paths, rng
            )
        else:
            rows, warns = sample_balanced_with_density(
                df_ds, path_col, args.base_dir,
                TRAIN_PER_DENSITY, excluded_paths, rng
            )

        print(f"  {ds:12s}: {len(rows):3d} images selected for TRAIN")
        all_warnings.extend(warns)
        train_rows.extend(rows)

    print(f"  TOTAL TRAIN: {len(train_rows)} images")

    # ── VAL Sampling ──────────────────────────────────────────────────────────
    print("\n[VAL] Sampling images (distinct from TRAIN)...")
    val_rows = []

    for ds in DATASETS:
        df_ds = df[df["source_dataset"] == ds].copy()
        if df_ds.empty:
            continue

        if ds in NO_DENSITY_DS:
            rows, warns = sample_no_density(
                df_ds, path_col, args.base_dir,
                VAL_CMMD, excluded_paths, rng
            )
        else:
            rows, warns = sample_balanced_with_density(
                df_ds, path_col, args.base_dir,
                VAL_PER_DENSITY, excluded_paths, rng
            )

        print(f"  {ds:12s}: {len(rows):3d} images selected for VAL")
        all_warnings.extend(warns)
        val_rows.extend(rows)

    print(f"  TOTAL VAL: {len(val_rows)} images")

    # ── Save CSV Outputs ──────────────────────────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    train_csv = os.path.join(args.out_dir, "mammo_train.csv")
    val_csv = os.path.join(args.out_dir, "mammo_val.csv")

    print("\n[OUTPUT] Saving CSV files...")
    write_csv(train_rows, path_col, train_csv, args.base_dir)
    write_csv(val_rows,   path_col, val_csv,   args.base_dir)

    # ── Copy Images (Optional) ────────────────────────────────────────────────
    if args.copy:
        train_img_dir = os.path.join(args.out_dir, "mammo_train")
        val_img_dir = os.path.join(args.out_dir, "mammo_val")
        print(f"\n[COPY] Copying images to {train_img_dir} and {val_img_dir}...")
        copy_images(train_rows, train_img_dir)
        copy_images(val_rows,   val_img_dir)
        print(f"  Train dir: {train_img_dir}")
        print(f"  Val   dir: {val_img_dir}")

    # ── Generate Summary Report ───────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "mammo_split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAMPLING SUMMARY — MAMMO-BENCH SR FINETUNING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Seed : {args.seed}\n")
        f.write(f"CSV  : {args.csv}\n\n")

        f.write("── TRAIN ──\n")
        for ds in DATASETS:
            ds_train = [r for r in train_rows if r.get("source_dataset") == ds]
            f.write(f"  {ds:12s}: {len(ds_train):3d} images\n")
            if ds not in NO_DENSITY_DS:
                for dens in DENSITIES:
                    cnt = sum(1 for r in ds_train if r.get("density") == dens)
                    f.write(f"    density {dens}: {cnt}\n")
        f.write(f"  {'TOTAL':12s}: {len(train_rows)}\n\n")

        f.write("── VAL ──\n")
        for ds in DATASETS:
            ds_val = [r for r in val_rows if r.get("source_dataset") == ds]
            f.write(f"  {ds:12s}: {len(ds_val):3d} images\n")
            if ds not in NO_DENSITY_DS:
                for dens in DENSITIES:
                    cnt = sum(1 for r in ds_val if r.get("density") == dens)
                    f.write(f"    density {dens}: {cnt}\n")
        f.write(f"  {'TOTAL':12s}: {len(val_rows)}\n\n")

        if all_warnings:
            f.write("── WARNINGS ──\n")
            for w in all_warnings:
                f.write(w + "\n")
        else:
            f.write("── No warnings. Sampling complete ──\n")

    print(f"\n[SUMMARY] {summary_path}")
    if all_warnings:
        print("  Warnings:")
        for w in all_warnings:
            print(w)
    else:
        print("  ✓ No warnings. Sampling complete.")

    print(f"\n{'='*60}")
    print(f" Done. Train={len(train_rows)}  Val={len(val_rows)}")
    print(f" Output folder: {os.path.abspath(args.out_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()