# Medical Super-Resolution Repository

This repository contains the official implementation for searching, training, and fine-tuning optimized Super-Resolution (SR) models via NSGA-III under the BASS (Blind and Anisotropic Search Space) framework. The pipeline targets both natural image domains and specialized medical image transfer learning, specifically optimized for mammography scans using advanced blind degradation models.

## 📂 Project Structure
```
├── DATASET/
│   ├── div2k/                # HR and LR training subsets (Natural Domain)
│   ├── mammo_train/          # Sampled Mammo balanced training patches
│   └── mammo_val/            # Sampled Mammo balanced validation patches
├── top_5_pareto_models.csv   # Architecture configuration definitions
├── top_5_pareto_global.csv   # Comprehensive performance description
│                               metrics for Pareto frontiers
├── 1_select_mammo_images.py  # Balanced dataset sampler for Mammography
│                               targets
├── mammo_degradation.py      # Blind degradation execution pipeline
│                              (Med-BSR model adaptation)
├── 3_TrainSR_GPU.py          # Stage 1 & Stage 2 Coarse Trainer for
│                               Natural Domains (DIV2K)
└── 4_finetune_mammo_SR.py    # Domain Adaptation Fine-Tuning Controller
                                for Medical Scans
```

## Dataset

Download and unzip from: [DATASET]([https://drive.google.com/drive/folders/16BRmpJfa_fV5vXMs7WLQdSEpQv-qMkeo?usp=sharing](https://drive.google.com/file/d/1_lU1tOPnjN4B6phRV5Mx05pJ5KfKDqNE/view?usp=sharing))

Or execute the script `0_getData.py` to download the file, unzip and delete de zip file.
## 🚀 Pipeline Overview
1. **Dataset Sampling & Balancing:** Filter and split medical domain scans (`mammo-bench`) while ensuring perfect demographic group representation over density classifications.

2. **Coarse Pre-Training (Natural Domain):** Train highly efficient candidate models discovered by the NSGA-III algorithm on natural rich images (`DIV2K`).

3. **Medical Domain Fine-Tuning:** Inject multi-stage blind degradation filters to perform rigorous transfer learning targeting grayscale medical mammographies.

## 📊 Dataset Configuration & Replication

The `DATASET/` directory acts as the data root for all workflows:

* Natural Domain: Contains standard DIV2K datasets used to construct deep feature extraction benchmarks.

* Medical Domain: Contains balanced datasets selected from the mammo-bench collection.

### Replicating the Mammography Split
Although sampled splits are readily provided, you can completely replicate or modify the sampling layout using `1_select_mammo_images.py`.

This module enforces a Balanced Sampling Strategy over 6 distinct source databases (inbreast, kau-bcmd, cmmd, cdd-cesm, dmid, ddsm), assigning 80 training images and 16 validation images per dataset evenly divided across density levels (A, B, C, D).

To run the replication selection filter:
1. Download the original data set here: [Mammo-Bench](https://india-data.org/dataset-details/c86fb00c-0fb8-4e0e-85a2-4d415f9c1ada)
2. Then run the script:
```python 1_select_mammo_images.py --csv mammo-bench.csv --out_dir DATASET --copy```

## 🧬 NSGA-III Search Space Configurations
The repository includes the structural descriptions for the top 5 models discovered by the NSGA-III evolutionary algorithm optimizing trade-offs inside the BASS search space:
* `pareto_models.csv`: Defines structural gene variables (number of residual groups, filter capacities, channel attention reduction, etc.) for each model.
* `top_5_pareto_global.csv`: Contains metadata descriptions, performance parameters, and efficiency metrics supporting the multi-objective selection choices.

## 🏋️‍♂️ Training Phase 1: Natural Images (DIV2K)

The candidate architectures extracted from the genetic search are built as highly performant RCAN (Residual Channel Attention Network) models. Initial coarse training is conducted via 3_TrainSR_GPU.py inside a 2-Stage optimization loop:
* __Stage 1:__ Coarse feature training mapping L1 loss functions on targeted crops.
* **Stage 2:** Context scale expansion (doubling spatial window dimensions to $128 \times 128$) to improve global context awareness.
To trigger the natural training routine:


`python 3_TrainSR_GPU.py --upscale_factor 2`
or
`python 3_TrainSR_GPU.py --upscale_factor 4`

## 🩺 Training Phase 2: Medical Fine-Tuning

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

### 📈 Tracking & Outputs
Metrics such as Peak Signal-to-Noise Ratio (PSNR) and Mean Absolute Error (MAE) are saved automatically into organized result summary ledgers:

* DATASET/mammo_split_summary.txt: Details dataset distribution tallies and demographic safety assertions.

* FINETUNE_RESULTS/ft_run_[timestamp]/finetune_summary.csv: Master dashboard tracking historical evaluation parameters for the Pareto-frontier models.

## 🛠️ Evironmente Requirements
The requirementes are listed in the `requirements.txt` file
