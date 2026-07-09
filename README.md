# Medical Super-Resolution Repository

This repository contains the official implementation for searching, training, and fine-tuning optimized Super-Resolution (SR) models via NSGA-III under the BASS (Blind and Anisotropic Search Space) framework. The pipeline targets both natural image domains and specialized medical image transfer learning, specifically optimized for mammography scans using advanced blind degradation models.
This repository requieres Python 3.11

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

* **Option 1 (Manual):** Download the file from this [Google Drive Link](https://drive.google.com/file/d/1cJa_I6f3CyoyCOQMqXjbBzBkr35Uv-eG/view?usp=sharing) and unzip it directly into the repository's root folder.
* **Option 2 (Automatic):** Run the provided script to download, unzip, and clean up the environment automatically:
  ```bash
  python I_getData.py
  ```
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
Although sampled splits are readily provided, you can completely replicate or modify the sampling layout using `II_select_mammo_images.py`.

This module enforces a Balanced Sampling Strategy over 6 distinct source databases (inbreast, kau-bcmd, cmmd, cdd-cesm, dmid, ddsm), assigning 80 training images and 16 validation images per dataset evenly divided across density levels (A, B, C, D).

To run the replication selection filter:
1. Download the original data set here: [Mammo-Bench](https://india-data.org/dataset-details/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada)
2. Then run the script:
```python II_select_mammo_images.py --csv mammo-bench.csv --out_dir DATASET --copy```


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

The candidate architectures extracted from the genetic search are built as highly performant RCAN (Residual Channel Attention Network) models. Initial coarse training is conducted via `III_TrainSR_GPU.py` inside a 2-Stage optimization loop:

* **Stage 1:** Coarse feature training mapping L1 loss functions on targeted crops.
* **Stage 2:** Context scale expansion (doubling spatial window dimensions to $128 \times 128$) to improve global context awareness.

### 🚀 Running the Experiments (Paper Benchmarks)
To replicate the exact training routine used in our paper, trigger the script by specifying the upscale factor:

```bash
# For 2x Super-Resolution
python III_TrainSR_GPU.py --upscale_factor 2

# For 4x Super-Resolution
python III_TrainSR_GPU.py --upscale_factor 4
```

### ⚙️ Custom Configuration
If you want to customize the training process, the script supports several arguments. You can modify the behavior directly from the command line:
|Argument |Type | Default | Description |
| ------- | --- | ------- | --------- |
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
📦 Pretrained Models Note: For your convenience and to ensure reproducibility, the models trained with the DIV2K dataset under our paper's configuration have already been generated and included inside the following directories:

* For 2x Scale: `outputs2x/a100_fast_bestpsnr_20260526_105721/`

* For 4x Scale: `outputs4x/a100_fast_bestpsnr_20260519_131528/`

## 🩺 Fine-Tuning Phase 2: Medical Domain (Mammo-Bench)

The script `IV_finetune_mammo_SR.py` adapts the pretrained natural domain models to specialized medical imaging. Because low-resolution medical observations suffer from unpredictable hardware conditions, this script applies a customized **Blind Degradation Architecture** modeled directly after `mammo_degradation.py` (Med-BSR pipeline variant).

### 🔄 Degradation Workflow (`Med-BSR` + `MST` Integration)

Every patch processing loop undergoes a randomized permutation sequence containing:
1. **Blur Filters:** Randomly selects between Isotropic Gaussian ($B_{iso}$) or Anisotropic Gaussian ($B_{aniso}$) variants.
2. **Noise Infusion:** Adds Additive Gaussian variations (Grayscale, independent Per-Channel, or Generalized scaling models).
3. **Downsampling Filters:** Executes downscaling operations utilizing one of the 6 MST (Multi-Scale Transformation) methods (`Nearest`, `Bilinear`, `Bicubic`, `Lanczos`, `Box`, `Hamming`).

> 💡 **Training vs. Validation Behavior:** The training sequence randomizes both degradation parameters and scheduling configurations to maintain domain invariance. Conversely, validation steps apply a stable **Deterministic Round-Robin** sequence over the 6 MST filters to evaluate generalizability with strict reproducibility.



### 🚀 Running the Fine-Tuning

The script supports two execution modes via mutually exclusive arguments: you can either fine-tune a **single specific model** or an **entire directory of NAS-discovered genes**.

#### Option A: Fine-tune all NAS genes inside a directory (Default Paper Setup)
```bash
python IV_finetune_mammo_SR.py \
    --genes_dir outputs4x/a100_fast_bestpsnr_20260519_131528 \
    --out_dir outputs \
    --upscale 4 \
    --grayscale
```
#### Option B: Fine-tune a single specific model checkpoint
```bash
python IV_finetune_mammo_SR.py \
    --model_path outputs4x/a100_fast_bestpsnr_20260519_131528/gene_001/stage2p_128_best.keras \
    --out_dir outputs \
    --upscale 4 \
    --grayscale
```
### 💾 Outputs & Logs
Upon running the script, the training process creates Keras model checkpoints and execution logs. These are automatically organized into structured directories:

Output Directory Structure:

```
outputs{scale}x/mammo_blindSR{run_id}/gene_{gen_id}/
├── mammo_ft_best.keras
├── mammo_ft_last.keras
└── mammo_ft_log.csv
```

📦 Pretrained Models Note: For your convenience and to ensure reproducibility, the models trained with the DIV2K dataset under our paper's configuration have already been generated and included inside the following directories:

* For 2x Scale: `outputs2x/mammo_blindSR2020/`

* For 4x Scale: `outputs4x/mammo_blindSR20260706_162533/`

The description of degrade process is ilustrated here:
![workflow](figs/degrade_process.svg)


## Testing

Configure and execute the evaluation script `V_main_test.py`. 
You can use any dataset, such as `Set5`, `Set14`, or `BSD100`. Place the original high-resolution dataset in the `Data/High/` directory.

### Usage Example
```bash
python V_main_test.py --scale 4 --genes_dir outputs4x/mammo_blindSR_20260620/
````
### Arguments Reference
where: 
Argument   |Type  | Default                | Description 
 -------   | ---  | ---------------------- | ----------- 
--dataset  | str  | "Data/High/mammo_val"  | Path to HR ground truth images. 
--scale    | int  | 4                      | Upsampling scale factor for Super-Resolution (2,4).
--genes_dir| str  | Required               | Root directory containing the trained NAS models
--out_dir  | str  | "Data/Outx4/mammo_val" | Root directory where evaluation outputs, checkpoints, and logs will be saved.
--bicubic  | bool | False                  | Execute bicubic downsampling only.
--lr_path  | str  | ""                     | Path to pre-degraded Low-Resolution (LR) images. 

### Degradation and LR Data Behavior

* Low-Resolution Data (`--lr_path`):
  * If `--lr_path` is not provided (left empty), the program will automatically generate the Low-Resolution (LR) images on-the-fly from the HR dataset.
  * If a directory is provided via --lr_path, the program assumes you already have a pre-degraded LR dataset and will skip the internal degradation process entirely.
* Degradation Method (`--bicubic`):
  * Default behavior: The program executes the advanced Med-BSR degradation process to evaluate the models.
  * Bicubic Only: Adding the `--bicubic` flag disables Med-BSR, forcing the program to generate LR images using standard bicubic downsampling only.

### Output Directory Structure

By default, the evaluation results are saved in `Data/Outx4/mammo_val`. Inside this root output directory, the program automatically creates a structured hierarchy for each discovered NAS sub-model and degradation method:

```text
Data/Outx4/mammo_val/
├── gene_001/
│   ├── bicubic/
│   │   ├── restored_image_001.png
│   │   ├── restored_image_002.png
│   │   └── register_psnr_ssim_bicubic.csv
│   ├── bilinear/
│   │   └── ...
│   └── nearest/
│       └── ...
├── gene_002/
└── ...
└── gene_005/
```

Inside each `gene_XXX` folder:
* Subdirectories by Method: Separate folders are created for each evaluation method: `bicubic`, `bilinear`, `box`, `hamming`, `lanczos`, and `nearest`. Each folder contains the images reconstructed/super-resolved by that specific sub-model.
* Evaluation Metrics (CSV Reports):
  * Standard Execution: Each method folder contains a summary report named `register_psnr_ssim_{method}.csv` (e.g., register_psnr_ssim_bilinear.csv) tracking the PSNR and SSIM metrics for all processed images.

  * Bicubic-Only Execution (`--bicubic flag`): If the `--bicubic` flag is used, only the bicubic degradation is executed, and the evaluation report inside the bicubic directory will be named simply `register_psnr_ssim.csv` and the subdirectories by method will not be generated.


## Result Analysis

The script `VI_mean_psnr_per_seed.py` is used to generate the statistical analysis and plots for the paper. 

### Usage Example
To run the analysis with the default configuration, execute:
```bash
python VI_mean_psnr_per_seed.py
````

You can modify the arguments as follows:

Argument  | Type | Default    | Description 
 -------  | ---- | ---------- | ----------- 
--gene    | str  | "gene_005" | The specific gene model to be evaluated in the analysis.
--method  | str  | "nearest"  | The degradation method to be analyzed.
--out_dir | str  | "Analysis" | Directory where evaluation plots and summary reports will be saved.
--format  | str  | "png"      | File format for the generated figures (e.g., png, eps, svg).

### Results and Outputs

All generated files are stored in the directory specified by `--out_dir` (default: `Analysis/`). The script produces both visual plots and tabular data:

📊 Graphical Plots
* `analysis_database_{scale}.{format}`: Comprehensive plot analyzing the behavior across the dataset.
* `degradation_comparison_{scale}.{format}`: Visual comparison of the model's performance against different degradation methods.

📋 Summary Reports (CSV)
* `mean_per_gene_{scale}.csv`: Contains the average metrics (PSNR/SSIM) calculated per gene across all tests.
* `mean_per_gene_method_{scale}.csv`: Detailed breakdown of the average metrics per gene, grouped by each specific degradation method.

## 🛠️ Environment Requirements

The project dependencies and environmental requirements are listed in the `requirements.txt` file. To install them, run:
```bash
pip install -r requirements.txt
```

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. You are free to use, modify, and distribute this software for academic and commercial purposes, provided the original copyright notice is included.

## ✍️ Citation
This paper is currently under review. The complete citation and BibTeX entry will be updated and made available here as soon as the manuscript is formally accepted.

In the meantime, if you use this code or dataset in your research, please link back to this repository.
