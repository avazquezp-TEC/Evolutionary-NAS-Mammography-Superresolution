"""
source ~/model_test/SRIR/tf_venv/bin/activate
sftp://10.48.9.141/home/angel/
select_mammo_images.py
=========================
Selects balanced images from mammo-bench.csv for SR finetuning (TRAIN / VAL)
and for the final PSNR benchmark against other architectures (TEST).

Sampling Strategy:
  - 6 databases: inbreast, kau-bcmd, cmmd, cdd-cesm, dmid, ddsm
  - TRAIN : 480 images → 80 per dataset → 20 per density (A, B, C, D)
  - VAL   :  96 images → 16 per dataset →  4 per density (A, B, C, D)
            (used during finetuning, e.g. early stopping)
  - TEST  : --test_per_density per dataset × density (default 10 → 40 per
            dataset, 240 images). Held-out benchmark, never used for training
            or model selection.
  - cmmd has no density info → sampled directly without density subgroups.
  - If any cell (dataset × density) has fewer images than required,
    all available images are taken and a warning is logged.
  - Patient-level split: no patient (source_dataset + source_subjectID) can
    appear in more than one of TRAIN, VAL and TEST, to avoid data leakage
    caused by the high correlation between images of the same patient.
    No image can appear in more than one split either.
  - TEST backfill: if a (dataset × density) cell cannot provide the requested
    number of images (patients/images already used by TRAIN/VAL), the deficit
    is filled with images of the SAME density from the other datasets, so the
    density balance of the whole TEST set is preserved. Backfilled images are
    drawn from the dataset with the fewest TEST images so far.

Outputs:
  mammo_train.csv          → list of 480 paths (original_source_path)
  mammo_val.csv            → list of 96 paths (distinct from train)
  mammo_test.csv           → list of benchmark paths (distinct from train/val)
  mammo_split_summary.txt  → sampling summary with warnings and leakage checks
  DATASET/mammo_train/     → copy of training images
  DATASET/mammo_val/       → copy of validation images
  DATASET/mammo_test/      → copy of test images

Usage:
  python II_select_mammo_images.py --out_dir DATASET --copy
  python II_select_mammo_images.py --out_dir DATASET --test_per_density 8

  # 15/06/26 initial version - cmmd has no density classification, samples randomly.
  cdd-cesm has 8 images of density A instead of the 24 required. It takes the 8 
  available for train and 0 for val.
  # Added patient-level exclusion (train patients are forbidden in val) and an
  explicit leakage check using the "source_subjectID" column.
  # 02/10/26 Added the TEST split (patient- and image-disjoint from TRAIN/VAL)
  with density backfill from other datasets. TRAIN and VAL sampling is
  unchanged (same RNG stream); TEST uses its own RNG (seed + 1). If
  mammo_train.csv / mammo_val.csv already exist in out_dir and the new
  selection differs, the script aborts (use --allow_split_change to override).
"""

import argparse
import csv
import os
import random
import shutil
from itertools import combinations
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
                   help="Output directory for train/val/test subfolders")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--copy", action="store_true",
                   help="If set, COPIES the images to out_dir/mammo_train, mammo_val and mammo_test")
    p.add_argument("--path_col", default="preprocessed_image_path",
                   help="CSV column containing the relative image path")
    p.add_argument("--patient_col", default="source_subjectID",
                   help="CSV column containing the patient/subject identifier")
    p.add_argument("--test_per_density", type=int, default=4,
                   help="TEST images per dataset × density (cmmd: 4× this value)")
    p.add_argument("--allow_split_change", action="store_true",
                   help="Allow overwriting existing mammo_train.csv / mammo_val.csv "
                        "even if the new selection differs from them")
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
    excluded_patients: set,   # patients that must NOT be sampled (e.g. those in TRAIN)
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """
    Samples n_per_density images per density (A, B, C, D) avoiding 'excluded' paths
    and any image belonging to a patient in 'excluded_patients'.
    
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
        # Patient-level exclusion to avoid leakage between splits
        pool = pool[~pool["_patient_key"].isin(excluded_patients)]
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
    excluded_patients: set,   # patients that must NOT be sampled (e.g. those in TRAIN)
    rng: random.Random,
) -> tuple[list[dict], list[str]]:
    """Samples n_total random images without density subgroups (patient-aware)."""
    pool = df_ds[~df_ds[path_col].isin(excluded)]
    # Patient-level exclusion to avoid leakage between splits
    pool = pool[~pool["_patient_key"].isin(excluded_patients)]
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


def sample_test_with_backfill(
    df: pd.DataFrame,
    path_col: str,
    base_dir: str,
    n_per_density: int,
    excluded: set,
    excluded_patients: set,   # patients already used in TRAIN and VAL
    rng: random.Random,
) -> tuple[list[dict], list[str], list[str]]:
    """
    Samples the TEST split.
      Phase 1: same dataset × density grid as TRAIN/VAL (cmmd without density).
      Phase 2: for every density with a deficit, fill it with images of the
               same density from the other datasets (patient-aware), always
               taking from the dataset with the fewest TEST images so far.

    Returns:
        tuple: (selected rows, warnings, backfill log lines)
    """
    rows, warnings, backfill_log = [], [], []

    # Phase 1: regular grid
    for ds in DATASETS:
        df_ds = df[df["source_dataset"] == ds]
        if df_ds.empty:
            continue
        if ds in NO_DENSITY_DS:
            r, w = sample_no_density(
                df_ds, path_col, base_dir,
                n_per_density * len(DENSITIES), excluded, excluded_patients, rng
            )
        else:
            r, w = sample_balanced_with_density(
                df_ds, path_col, base_dir,
                n_per_density, excluded, excluded_patients, rng
            )
        print(f"  {ds:12s}: {len(r):3d} images selected for TEST (phase 1)")
        rows.extend(r)
        warnings.extend(w)

    # Phase 2: backfill density deficits from other datasets
    density_ds = [d for d in DATASETS
                  if d not in NO_DENSITY_DS and d in set(df["source_dataset"].unique())]
    for dens in DENSITIES:
        expected = n_per_density * len(density_ds)
        got = sum(1 for r in rows if r.get("density") == dens)
        deficit = expected - got
        if deficit <= 0:
            continue

        # Build the remaining candidate pool of this density for every dataset
        pools = {}
        for ds in density_ds:
            pool = df[(df["source_dataset"] == ds) & (df["density"] == dens)]
            pool = pool[~pool[path_col].isin(excluded)]
            pool = pool[~pool["_patient_key"].isin(excluded_patients)]
            cand = pool.to_dict("records")
            rng.shuffle(cand)
            pools[ds] = cand

        counts = {ds: sum(1 for r in rows if r.get("source_dataset") == ds)
                  for ds in density_ds}
        added = {ds: 0 for ds in density_ds}
        filled = 0
        while filled < deficit:
            cands = [ds for ds in density_ds if pools[ds]]
            if not cands:
                break
            ds = min(cands, key=lambda d: counts[d])   # fewest TEST images so far
            row = pools[ds].pop()
            row["_full_path"] = build_full_path(row, path_col, base_dir)
            excluded.add(row[path_col])
            rows.append(row)
            counts[ds] += 1
            added[ds] += 1
            filled += 1

        detail = ", ".join(f"{ds}={n}" for ds, n in added.items() if n)
        backfill_log.append(
            f"  density {dens}: deficit {deficit} → backfilled {filled}"
            + (f" ({detail})" if detail else "")
        )
        if filled < deficit:
            warnings.append(
                f"  WARNING [TEST/density={dens}]: deficit of {deficit - filled} "
                f"could not be backfilled (no more candidates)."
            )

    return rows, warnings, backfill_log


def check_split_overlap(splits: dict, path_col: str) -> dict:
    """
    For every pair of splits returns (shared patient keys, shared image paths).
    Both lists must be empty for a leakage-free partition.
    """
    result = {}
    for a, b in combinations(splits, 2):
        pat = {r["_patient_key"] for r in splits[a]} & {r["_patient_key"] for r in splits[b]}
        img = {r[path_col] for r in splits[a]} & {r[path_col] for r in splits[b]}
        result[(a, b)] = (sorted(pat), sorted(img))
    return result


def compare_with_existing(csv_path: str, rows: list[dict], path_col: str) -> tuple[bool, int, int]:
    """
    Compares a new selection with an existing CSV (by image path).
    Returns (file exists, #only in old file, #only in new selection).
    """
    if not os.path.isfile(csv_path):
        return False, 0, 0
    old = set(pd.read_csv(csv_path)[path_col].astype(str))
    new = {str(r[path_col]) for r in rows}
    return True, len(old - new), len(new - old)


def write_csv(rows: list[dict], path_col: str, out_path: str, base_dir: str) -> None:
    """Writes the selected image metadata records into a CSV file."""
    fieldnames = ["source_dataset", "source_subjectID", "density", path_col, "full_path",
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


def write_split_summary(f, title: str, rows: list[dict], with_totals: bool = False) -> None:
    """Writes per-dataset (and optionally overall) density counts of one split."""
    f.write(f"── {title} ──\n")
    for ds in DATASETS:
        ds_rows = [r for r in rows if r.get("source_dataset") == ds]
        f.write(f"  {ds:12s}: {len(ds_rows):3d} images\n")
        if ds not in NO_DENSITY_DS:
            for dens in DENSITIES:
                cnt = sum(1 for r in ds_rows if r.get("density") == dens)
                f.write(f"    density {dens}: {cnt}\n")
    f.write(f"  {'TOTAL':12s}: {len(rows)}\n")
    if with_totals:
        f.write("  Density totals (datasets with density info):\n")
        for dens in DENSITIES:
            cnt = sum(1 for r in rows if r.get("density") == dens)
            f.write(f"    density {dens}: {cnt}\n")
    f.write("\n")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    rng_test = random.Random(args.seed + 1)   # separate stream: TRAIN/VAL stay unchanged
    path_col = args.path_col

    print(f"\n{'='*60}")
    print(" BALANCED SELECTION MAMMO-BENCH → SR FINETUNING + BENCHMARK")
    print(f"{'='*60}")
    print(f"  CSV         : {args.csv}")
    print(f"  base_dir    : {args.base_dir or '(absolute paths in CSV)'}")
    print(f"  out_dir     : {args.out_dir}")
    print(f"  seed        : {args.seed}")
    print(f"  copy        : {args.copy}")
    print(f"  patient_col : {args.patient_col}")
    print(f"  test/dens.  : {args.test_per_density}")
    print()

    # ── Load CSV ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv)
    print(f"[CSV] {len(df)} rows | datasets: {sorted(df['source_dataset'].unique())}")

    # Normalize density column to uppercase and clear NaN entries for valid datasets
    df["density"] = df["density"].astype(str).str.strip().str.upper()
    df.loc[df["density"] == "NAN", "density"] = None

    # Make sure the patient column exists before using it
    if args.patient_col not in df.columns:
        raise KeyError(f"Patient column '{args.patient_col}' not found in {args.csv}. "
                       f"Available columns: {list(df.columns)}")

    # Build a dataset-aware patient key (IDs may collide across datasets)
    df["_patient_key"] = (df["source_dataset"].astype(str) + "::"
                          + df[args.patient_col].astype(str).str.strip())
    # Rows without subject ID: use the image path so they are not wrongly grouped together
    no_id = df[args.patient_col].isna()
    if no_id.any():
        print(f"  ⚠ {no_id.sum()} rows without {args.patient_col}; "
              f"treated as unique patients (leakage cannot be guaranteed for them)")
        df.loc[no_id, "_patient_key"] = "NOID::" + df.loc[no_id, path_col].astype(str)

    # Verify that all target datasets exist within the CSV file
    found_ds = set(df["source_dataset"].unique())
    missing_ds = set(DATASETS) - found_ds
    if missing_ds:
        print(f"  ⚠ Datasets not found in CSV: {missing_ds}")

    # ── TRAIN Sampling ────────────────────────────────────────────────────────
    print("\n[TRAIN] Sampling images...")
    all_warnings = []
    excluded_paths: set = set()   # Track selected paths to prevent duplicates across splits
    train_rows = []

    for ds in DATASETS:
        df_ds = df[df["source_dataset"] == ds].copy()
        if df_ds.empty:
            all_warnings.append(f"  WARNING: dataset '{ds}' not found in CSV.")
            continue

        # No patient restriction for TRAIN (it is sampled first)
        if ds in NO_DENSITY_DS:
            rows, warns = sample_no_density(
                df_ds, path_col, args.base_dir,
                TRAIN_CMMD, excluded_paths, set(), rng
            )
        else:
            rows, warns = sample_balanced_with_density(
                df_ds, path_col, args.base_dir,
                TRAIN_PER_DENSITY, excluded_paths, set(), rng
            )

        print(f"  {ds:12s}: {len(rows):3d} images selected for TRAIN")
        all_warnings.extend(warns)
        train_rows.extend(rows)

    print(f"  TOTAL TRAIN: {len(train_rows)} images")

    # Patients used in TRAIN are forbidden in VAL
    train_patients = {r["_patient_key"] for r in train_rows}

    # ── VAL Sampling ──────────────────────────────────────────────────────────
    print("\n[VAL] Sampling images (distinct images AND patients from TRAIN)...")
    val_rows = []

    for ds in DATASETS:
        df_ds = df[df["source_dataset"] == ds].copy()
        if df_ds.empty:
            continue

        if ds in NO_DENSITY_DS:
            rows, warns = sample_no_density(
                df_ds, path_col, args.base_dir,
                VAL_CMMD, excluded_paths, train_patients, rng
            )
        else:
            rows, warns = sample_balanced_with_density(
                df_ds, path_col, args.base_dir,
                VAL_PER_DENSITY, excluded_paths, train_patients, rng
            )

        print(f"  {ds:12s}: {len(rows):3d} images selected for VAL")
        all_warnings.extend(warns)
        val_rows.extend(rows)

    print(f"  TOTAL VAL: {len(val_rows)} images")

    # ── TEST Sampling ─────────────────────────────────────────────────────────
    # Patients used in TRAIN or VAL are forbidden in TEST
    val_patients = {r["_patient_key"] for r in val_rows}
    print("\n[TEST] Sampling images (distinct images AND patients from TRAIN and VAL)...")
    test_rows, test_warns, backfill_log = sample_test_with_backfill(
        df, path_col, args.base_dir, args.test_per_density,
        excluded_paths, train_patients | val_patients, rng_test
    )
    all_warnings.extend(test_warns)
    for line in backfill_log:
        print(line)
    for ds in DATASETS:
        n = sum(1 for r in test_rows if r.get("source_dataset") == ds)
        print(f"  {ds:12s}: {n:3d} images in TEST (after backfill)")
    print(f"  TOTAL TEST: {len(test_rows)} images")

    # ── Leakage checks (patient and image level, all pairs of splits) ─────────
    splits = {"train": train_rows, "val": val_rows, "test": test_rows}
    overlaps = check_split_overlap(splits, path_col)
    print("\n[CHECK] Unique patients → " + " | ".join(
        f"{k}: {len({r['_patient_key'] for r in v})}" for k, v in splits.items()))
    for (a, b), (pat, img) in overlaps.items():
        print(f"  {a} ∩ {b}: patient overlap = {len(pat)} | image overlap = {len(img)}")
    bad = {k: v for k, v in overlaps.items() if v[0] or v[1]}
    if bad:
        raise RuntimeError(f"Leakage detected between splits: "
                           f"{ {k: (v[0][:5], v[1][:5]) for k, v in bad.items()} }")

    # ── Protect already-used TRAIN/VAL files ──────────────────────────────────
    os.makedirs(args.out_dir, exist_ok=True)
    train_csv = os.path.join(args.out_dir, "mammo_train.csv")
    val_csv = os.path.join(args.out_dir, "mammo_val.csv")
    test_csv = os.path.join(args.out_dir, "mammo_test.csv")

    changed = []
    for name, rows, csv_path in [("train", train_rows, train_csv), ("val", val_rows, val_csv)]:
        exists, only_old, only_new = compare_with_existing(csv_path, rows, path_col)
        if exists and (only_old or only_new):
            changed.append(f"{name}: {only_old} images only in the existing file, "
                           f"{only_new} only in the new selection")
        elif exists:
            print(f"  ✓ {name} selection is identical to the existing {os.path.basename(csv_path)}")
    if changed and not args.allow_split_change:
        raise RuntimeError(
            "The new TRAIN/VAL selection differs from the existing CSV files, so the TEST "
            "set would be built against images the models were NOT trained on:\n  "
            + "\n  ".join(changed)
            + "\nCheck --csv/--seed/--base_dir, or pass --allow_split_change to overwrite."
        )

    # ── Save CSV Outputs ──────────────────────────────────────────────────────
    print("\n[OUTPUT] Saving CSV files...")
    write_csv(train_rows, path_col, train_csv, args.base_dir)
    write_csv(val_rows,   path_col, val_csv,   args.base_dir)
    write_csv(test_rows,  path_col, test_csv,  args.base_dir)

    # ── Copy Images (Optional) ────────────────────────────────────────────────
    if args.copy:
        train_img_dir = os.path.join(args.out_dir, "mammo_train")
        val_img_dir = os.path.join(args.out_dir, "mammo_val")
        test_img_dir = os.path.join(args.out_dir, "mammo_test")
        print(f"\n[COPY] Copying images to {train_img_dir}, {val_img_dir} and {test_img_dir}...")
        copy_images(train_rows, train_img_dir)
        copy_images(val_rows,   val_img_dir)
        copy_images(test_rows,  test_img_dir)
        print(f"  Train dir: {train_img_dir}")
        print(f"  Val   dir: {val_img_dir}")
        print(f"  Test  dir: {test_img_dir}")

    # ── Generate Summary Report ───────────────────────────────────────────────
    summary_path = os.path.join(args.out_dir, "mammo_split_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("SAMPLING SUMMARY — MAMMO-BENCH SR FINETUNING\n")
        f.write("="*60 + "\n\n")
        f.write(f"Seed : {args.seed}\n")
        f.write(f"CSV  : {args.csv}\n")
        f.write(f"Test images per dataset × density: {args.test_per_density}\n\n")

        write_split_summary(f, "TRAIN", train_rows)
        write_split_summary(f, "VAL", val_rows)
        write_split_summary(f, "TEST", test_rows, with_totals=True)

        f.write("── TEST BACKFILL (density deficits filled from other datasets) ──\n")
        if backfill_log:
            for line in backfill_log:
                f.write(line + "\n")
        else:
            f.write("  No backfill needed.\n")
        f.write("\n")

        f.write("── LEAKAGE CHECK (patient and image level) ──\n")
        for k, v in splits.items():
            f.write(f"  Unique patients {k:5s}: {len({r['_patient_key'] for r in v})}\n")
        for (a, b), (pat, img) in overlaps.items():
            f.write(f"  {a} ∩ {b}: patient overlap = {len(pat)} | image overlap = {len(img)}\n")
        f.write("\n")

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
    print(f" Done. Train={len(train_rows)}  Val={len(val_rows)}  Test={len(test_rows)}")
    print(f" Output folder: {os.path.abspath(args.out_dir)}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
