# Medical Super-Resolution Repository

This repository contains the official implementation for searching, training, and fine-tuning optimized Super-Resolution (SR) models via NSGA-III under the BASS (Blind and Anisotropic Search Space) framework. The pipeline targets both natural image domains and specialized medical image transfer learning, specifically optimized for mammography scans using advanced blind degradation models.

## 📂 Project Structure
```
├── DATASET/
│   ├── div2k/                # HR and LR training subsets (Natural Domain)
│   ├── mammo_train/          # Sampled Mammo balanced training patches
│   └── mammo_val/            # Sampled Mammo balanced validation patches
├── Data/
│   └── High/                 # High resolution datasets
│   │   └── mammo_val/        # HR mammography images
│   └── Lowx2/                # LR resolution datasets (scale x2)
│   │   └── mammo_val/        # LR mammography images available after executing 5_min_test.py
│   └── Lowx4/                # LR resolution datasets (scale x2)
│       └── mammo_val/        # LR mammography images available after executing 5_min_test.py
├── pareto_models.csv         # Architecture configuration definitions
├── top_5_pareto_global.csv   # Comprehensive performance description
│                               metrics for Pareto frontiers
├── 1_get_data.py             # Download the DATASET and Data folders
├── 2_select_mammo_images.py  # Balanced dataset sampler for Mammography
│                               targets
├── mammo_degradation.py      # Blind degradation execution pipeline
│                              (Med-BSR model adaptation)
├── 3_TrainSR_GPU.py          # Stage 1 & Stage 2 Coarse Trainer for
│                               Natural Domains (DIV2K)
└── 4_finetune_mammo_SR.py    # Domain Adaptation Fine-Tuning Controller
│                               for Medical Scans
└── 5_main_test.py            # Creation of LR images for testing. And inference of algorithms
└── 6_mean_psnr_per_seed.py   # Generates figures for paper
```

## Dataset Setup

* **Option 1 (Manual):** Download the file from this [Google Drive Link](https://drive.google.com/file/d/1uzvq4-Ad-sBzQQvt8yui9aOwM_SfxVya/view?usp=sharing) and unzip it directly into the repository's root folder.
* **Option 2 (Automatic):** Run the provided script to download, unzip, and clean up the environment automatically:
  ```bash
  python 1_getData.py
  
### 🚀 Pipeline Overview
1. **Dataset Sampling & Balancing:** Filter and split medical domain scans (`mammo-bench`) while ensuring perfect demographic group representation over density classifications.

2. **Coarse Pre-Training (Natural Domain):** Train highly efficient candidate models discovered by the NSGA-III algorithm on natural rich images (`DIV2K`).

3. **Medical Domain Fine-Tuning:** Inject multi-stage blind degradation filters to perform rigorous transfer learning targeting grayscale medical mammographies.

### 📊 Data Structure & Domains

Once extracted, the data is organized into two main directories acting as the root for different workflows:

#### 1. `DATASET/`
This directory contains the core data used for training, validation, and fine-tuning across two domains:
* **Natural Domain:** Includes standard **DIV2K** datasets used to construct deep feature extraction benchmarks (split into training and validation).
* **Medical Domain:** Contains balanced datasets selected from the **mammo-bench** collection, specifically structured for model fine-tuning (split into training and validation).

#### 2. `Data/`
This is a separate directory dedicated exclusively to evaluation and benchmarking:
* **Medical SR Benchmark:** Contains a distinct set of medical domain images designed specifically to serve as the super-resolution (SR) benchmark.

#### Replicating the Mammography Split
Although sampled splits are readily provided, you can completely replicate or modify the sampling layout using `1_select_mammo_images.py`.

This module enforces a Balanced Sampling Strategy over 6 distinct source databases (inbreast, kau-bcmd, cmmd, cdd-cesm, dmid, ddsm), assigning 80 training images and 16 validation images per dataset evenly divided across density levels (A, B, C, D).

To run the replication selection filter:
1. Download the original data set here: [Mammo-Bench](https://india-data.org/dataset-details/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada)
2. Then run the script:
```python 2_select_mammo_images.py --csv mammo-bench.csv --out_dir DATASET --copy```


## 🧬 NSGA-III Search Space Configurations

> ⚠️ **Review Note:** This repository contains the architecture definitions and evaluation results for the discovered models. The full NSGA-III evolutionary search framework is hosted in a separate repository to maintain anonymity during the peer-review process. It will be linked upon publication.

The repository includes the structural descriptions for the top 5 models discovered by the NSGA-III evolutionary algorithm optimizing trade-offs inside the BASS search space:

* `pareto_models.csv`: Contains the genetic representation of each discovered model, formatted as an **84-bit string (gene chain)** that encodes its specific structural operations.
* `top_5_pareto_global.csv`: Contains the evaluation metadata for the selected models. For each solution, it details:
  * The encoded **gene**.
  * The **predicted PSNR**.
  * The total **number of parameters**.
  * The evolutionary **generation** in which it was found.
  * The random **seed** that generated the solution.

You can modify the architecture according to the BASS search space specification:

![Search space](figs/BASS2.svg)


## 🏋️‍♂️ Training Phase 1: Natural Images (DIV2K)

The candidate architectures extracted from the genetic search are built as highly performant RCAN (Residual Channel Attention Network) models. Initial coarse training is conducted via `3_TrainSR_GPU.py` inside a 2-Stage optimization loop:

* **Stage 1:** Coarse feature training mapping L1 loss functions on targeted crops.
* **Stage 2:** Context scale expansion (doubling spatial window dimensions to $128 \times 128$) to improve global context awareness.

### 🚀 Running the Experiments (Paper Benchmarks)
To replicate the exact training routine used in our paper, trigger the script by specifying the upscale factor:

```bash
# For 2x Super-Resolution
python 3_TrainSR_GPU.py --upscale_factor 2

# For 4x Super-Resolution
python 3_TrainSR_GPU.py --upscale_factor 4
```

### ⚙️ Custom Configuration
If you want to customize the training process, the script supports several arguments. You can modify the behavior directly from the command line:
|Argument |Type | Default Description |
| ------- | --- | ------------------- |
--upscale_factor | int | 4 | Super-resolution scale factor (2 or 4). |
--directory_train | str | "DATASET/DIV2K_train_HR" | Path to the training images directory
--directory_val | str | "DATASET/DIV2K_valid_HR" | Path to the validation images directory.
--genes_csv | str | "pareto_models.csv" | CSV file containing the model genes to train
--base_seed | int | 1 | Random seed for reproducibility.
--batch_size | int | 64 | Training batch size
--overlap_val | float | 0.1 | Validation overlap fraction.
--max_epochs1 | int | 200 | Maximum number of epochs for Stage 1.
--max_epochs2 | int | 120 | Maximum number of epochs for Stage 2.


### 💾 Outputs & Logs
Upon running the script, the training process creates Keras model checkpoints and execution logs. These are automatically organized into structured directories:

Output Directory Structure:

```
outputs{scale}x/a100_fast_bestpsnr_{run_id}/gene_{gen_id}/
├── stage1_p64_best.keras
├── stage1_p64_last.keras
├── stage1_p64_log.csv
├── stage2_p128_best.keras
├── stage2_p128_last.keras
└── stage2_p128_log.csv
```
## 🩺 Finetunning el Phase 2: Medical Fine-Tuning

Domain adaptation transfer learning targets medical diagnostics through 

`4_finetune_mammo_SR.py`.

Because low-resolution medical observations suffer from unpredictable hardware conditions, this script applies a customized Blind Degradation Architecture modeled directly after `mammo_degradation.py` (Med-BSR pipeline variant):

### Degradation Workflow (`Med-BSR` + `MST` Integration)

Every patch processing loop undergoes a randomized permutation sequence containing:

1. __Blur Filters:__ Randomly selects between Isotropic Gaussian ($B_{iso}$) or Anisotropic Gaussian ($B_{aniso}$) variants.

2. __Noise Infusion:__ Adds Additive Gaussian variations (Grayscale, independent Per-Channel, or Generalized scaling models).

3. __Downsampling Filters:__ Executes downscaling operations utilizing one of the 6 MST (Multi-Scale Transformation) methods (`Nearest`, `Bilinear`, `Bicubic`, `Lanczos`, `Box`, `Hamming`).

### Fine-Tuning Execution
The training sequence randomizes both degradation parameters and scheduling configurations to maintain domain invariance. Conversely, validation steps apply a stable __Deterministic Round-Robin__ sequence over the 6 MST filters to evaluate generalizability with strict reproducibility.

To start the target fine-tuning adaptation run:

```
python 4_finetune_mammo_SR.py \
    --model_path /path/to/your/stage2_p128_best.keras \
    --scale 4 \
    --train_csv DATASET/mammo_train.csv \
    --val_csv DATASET/mammo_val.csv
```

The description of degrade process is ilustrated here
![workflow](figs/degrade_process.svg)
### 📈 Tracking & Outputs
Metrics such as Peak Signal-to-Noise Ratio (PSNR) and Mean Absolute Error (MAE) are saved automatically into organized result summary ledgers:

* DATASET/mammo_split_summary.txt: Details dataset distribution tallies and demographic safety assertions.

* FINETUNE_RESULTS/ft_run_[timestamp]/finetune_summary.csv: Master dashboard tracking historical evaluation parameters for the Pareto-frontier models.

## Testing
Configure and Execute the script `5_main_test.py`
You can use anydata set, for example `Set5` or `Set14`. Plance the original dataset in `Data/High` and chage the line `SET = "dataset"`
if `LR_PATH = ""' is empty the program will compute the LR image from the data set.
Also if `MST=False`the program will create LR images only using bicubic degradation. if `MST=True` the program will excecute the Med-BSR degradation process.
if `LR_PATH = "directory"' means that you have a LR datasets. In that case the program skips the degrade process. Dont forguet to change `MST=False`

The program `6_mean_psnr_per_seed.py` generates the images for the paper only.
## 🛠️ Evironmente Requirements
The requirementes are listed in the `requirements.txt` file
