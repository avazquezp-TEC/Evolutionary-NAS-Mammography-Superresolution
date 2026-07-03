
from __future__ import annotations

import os
import copy
import csv
import datetime
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import tensorflow as tf
import argparse
import sys

keras = tf.keras
layers = tf.keras.layers
AUTOTUNE = tf.data.AUTOTUNE
ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# ============================================================
# 0) Keras-serializable layers (NO python lambda)
# ============================================================
@tf.keras.utils.register_keras_serializable(package="sr")
class PixelShuffle(tf.keras.layers.Layer):
    """Safe PixelShuffle wrapper (depth_to_space) for Keras 3 serialization."""
    def __init__(self, upscale_factor: int, **kwargs):
        super().__init__(**kwargs)
        self.upscale_factor = int(upscale_factor)

    def call(self, x):
        return tf.nn.depth_to_space(x, self.upscale_factor)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"upscale_factor": self.upscale_factor})
        return cfg


# ============================================================
# 1) Runtime: CPU/GPU + mixed precision + threads
# ============================================================
def configure_runtime(cpu_mode: bool) -> None:
    if cpu_mode:
        tf.config.set_visible_devices([], "GPU")
        return

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for g in gpus:
                tf.config.experimental.set_memory_growth(g, True)
            print("[GPU] memory_growth enabled")
        except Exception as e:
            print("[GPU] memory_growth warning:", e)


def configure_threads() -> None:
    # Let TF choose. Often improves throughput.
    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass


def configure_mixed_precision(cpu_mode: bool) -> None:
    # GPU: mixed_float16; CPU: float32 for numerical safety.
    try:
        from tensorflow.keras import mixed_precision
        policy = "float32" if cpu_mode else "mixed_float16"
        mixed_precision.set_global_policy(policy)
        print("[MP] policy:", mixed_precision.global_policy())
    except Exception as e:
        print("[MP] could not set policy:", e)


def enable_xla() -> None:
    # XLA can help on GPUs; if it hurts, disable.
    try:
        tf.config.optimizer.set_jit(True)
        print("[XLA] JIT enabled")
    except Exception as e:
        print("[XLA] could not enable:", e)


def get_strategy_auto() -> tf.distribute.Strategy:
    if len(tf.config.list_logical_devices("GPU")) >= 2:
        return tf.distribute.MirroredStrategy()
    return tf.distribute.get_strategy()


# ============================================================
# 2) Data config + stable step counting (for "epoch = full dataset")
# ============================================================
@dataclass
class DataCfg:
    ratio: int = 2
    patch_size: int = 64
    overlap: float = 0.1              # used ONLY for deterministic validation sliding patches
    batch_size: int = 64


def _is_image_file(path: str) -> bool:
    p = path.lower()
    return any(p.endswith(ext) for ext in ALLOWED_EXTS)


def list_images(directory: str) -> List[str]:
    files: List[str] = []
    for name in os.listdir(directory):
        p = os.path.join(directory, name)
        if os.path.isfile(p) and _is_image_file(p):
            files.append(p)
    return sorted(files)


def get_image_hw(path: str) -> Optional[Tuple[int, int]]:
    try:
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        if img.shape.rank != 3:
            return None
        return int(img.shape[0]), int(img.shape[1])
    except Exception:
        return None


def count_patches_for_image(h: int, w: int, patch_size: int, overlap: float) -> int:
    if h < patch_size or w < patch_size:
        return 0
    step = int(patch_size * (1.0 - overlap))
    step = max(1, step)
    ny = ((h - patch_size) // step) + 1
    nx = ((w - patch_size) // step) + 1
    return int(ny * nx)


def compute_steps_per_epoch(directory: str, cfg: DataCfg) -> Tuple[int, int, int]:
    """
    Returns (steps, total_patches, images_used)
    Using floor(total_patches / batch_size) matches "drop remainder".
    This makes epochs comparable across CPU/GPU and across runs.
    """
    files = list_images(directory)
    total_patches = 0
    used = 0
    for p in files:
        hw = get_image_hw(p)
        if hw is None:
            continue
        h, w = hw
        n = count_patches_for_image(h, w, cfg.patch_size, cfg.overlap)
        if n > 0:
            total_patches += n
            used += 1
    steps = max(1, total_patches // cfg.batch_size)
    return steps, total_patches, used


# ============================================================
# 3) TRAIN pipeline (FAST GPU): tf.data RANDOM CROP per step
# ============================================================
def _decode_image(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img.set_shape([None, None, 3])
    return img


def _stateless_augment(hr: tf.Tensor, seed: tf.Tensor) -> tf.Tensor:
    """
    SR-safe augmentation: flips + rot90, stateless.
    """
    # Flip LR/HR equivalently => do on HR before downsample
    hr = tf.image.stateless_random_flip_left_right(hr, seed=seed)
    hr = tf.image.stateless_random_flip_up_down(hr, seed=seed + tf.constant([1, 1], tf.int32))

    k = tf.random.stateless_uniform([], seed=seed + tf.constant([2, 2], tf.int32), minval=0, maxval=4, dtype=tf.int32)
    hr = tf.image.rot90(hr, k=k)
    return hr


def make_train_dataset_randomcrop(directory: str, cfg: DataCfg, seed: int) -> tf.data.Dataset:
    """
    Training dataset:
    - reads images
    - random-crops HR patches per sample
    - augmentation
    - creates LR by bicubic downsample
    - batch + prefetch
    """
    paths = list_images(directory)
    if not paths:
        raise ValueError(f"No images found in {directory}")

    ps = cfg.patch_size
    r = cfg.ratio
    lr_ps = ps // r

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.shuffle(buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.repeat()  # infinite stream

    # Decode with parallelism
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE)

    # Convert each image into an infinite stream of random crops by repeating each image a few times
    # This is a key trick: it increases crop diversity without Python loops.
    # You can increase repeats_per_image if you want more variety per epoch.
    repeats_per_image = 8
    ds = ds.flat_map(lambda img: tf.data.Dataset.from_tensors(img).repeat(repeats_per_image))

    # Add an index to build stateless seeds per element
    counter = tf.data.experimental.Counter()
    ds = tf.data.Dataset.zip((ds, counter))

    def _make_pair(img: tf.Tensor, idx: tf.Tensor):
        # Ensure the image is large enough; otherwise, skip by returning dummy and filtering.
        h = tf.shape(img)[0]
        w = tf.shape(img)[1]
        ok = tf.logical_and(h >= ps, w >= ps)

        # Stateless seed per sample
        s = tf.stack([tf.cast(seed, tf.int32), tf.cast(idx, tf.int32)], axis=0)

        def _do():
            hr = tf.image.stateless_random_crop(img, size=[ps, ps, 3], seed=s)
            hr = _stateless_augment(hr, seed=s + tf.constant([3, 3], tf.int32))

            lr = tf.image.resize(hr, [lr_ps, lr_ps], method="bicubic", antialias=True)
            lr = tf.clip_by_value(lr, 0.0, 1.0)
            hr2 = tf.clip_by_value(hr, 0.0, 1.0)
            return lr, hr2

        # If too small, return zeros; we'll filter them out.
        lr0 = tf.zeros([lr_ps, lr_ps, 3], tf.float32)
        hr0 = tf.zeros([ps, ps, 3], tf.float32)
        lr, hr = tf.cond(ok, _do, lambda: (lr0, hr0))
        return lr, hr, ok

    ds = ds.map(_make_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.filter(lambda lr, hr, ok: ok)
    ds = ds.map(lambda lr, hr, ok: (lr, hr), num_parallel_calls=AUTOTUNE)

    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    ds = ds.prefetch(AUTOTUNE)
    return ds


# ============================================================
# 4) VAL pipeline (DETERMINISTIC): sliding patches, no repeat
#     Kept as generator for clarity + deterministic behavior.
# ============================================================
def val_patch_generator(paths: List[str], cfg: DataCfg):
    """
    Deterministic sliding-window patches:
    - fixed order of images
    - fixed y/x scan
    - fixed overlap
    - yields batches, drops remainder
    """
    ps = cfg.patch_size
    r = cfg.ratio
    lr_ps = ps // r
    step = int(ps * (1.0 - cfg.overlap))
    step = max(1, step)

    batch_lr, batch_hr = [], []

    for p in paths:
        try:
            img = _decode_image(tf.constant(p)).numpy()  # VAL only (ok)
        except Exception:
            continue

        h, w = img.shape[:2]
        if h < ps or w < ps:
            continue

        for y in range(0, h - ps + 1, step):
            for x in range(0, w - ps + 1, step):
                hr = img[y:y + ps, x:x + ps, :]
                lr = tf.image.resize(hr, [lr_ps, lr_ps], method="bicubic", antialias=True).numpy()
                batch_lr.append(lr.astype(np.float32))
                batch_hr.append(hr.astype(np.float32))
                if len(batch_lr) == cfg.batch_size:
                    yield np.asarray(batch_lr, np.float32), np.asarray(batch_hr, np.float32)
                    batch_lr, batch_hr = [], []


def make_val_dataset_deterministic(directory: str, cfg: DataCfg) -> Tuple[tf.data.Dataset, int]:
    paths = list_images(directory)
    if not paths:
        raise ValueError(f"No images found in {directory}")

    val_steps, _, _ = compute_steps_per_epoch(directory, cfg)

    ps = cfg.patch_size
    r = cfg.ratio
    lr_ps = ps // r

    ds = tf.data.Dataset.from_generator(
        lambda: val_patch_generator(paths, cfg),
        output_signature=(
            tf.TensorSpec(shape=(cfg.batch_size, lr_ps, lr_ps, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(cfg.batch_size, ps, ps, 3), dtype=tf.float32),
        ),
    )

    # IMPORTANT: no repeat; fixed number of batches each epoch
    ds = ds.take(val_steps).prefetch(AUTOTUNE)
    return ds, val_steps


# ============================================================
# 5) PSNR metric (float32 stable)
# ============================================================
@tf.keras.utils.register_keras_serializable(package="sr")
def psnr(y_true, y_pred):
    y_true = tf.cast(tf.clip_by_value(y_true, 0.0, 1.0), tf.float32)
    y_pred = tf.cast(tf.clip_by_value(y_pred, 0.0, 1.0), tf.float32)
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


# ============================================================
# 6) Gene -> Model (your op space) safe for Keras 3
# ============================================================
from collections import namedtuple
Genotype = namedtuple("Genotype", ["Branch1", "Branch2", "Branch3"])

PRIMITIVES = [
    "conv",
    "dil_conv_d2",
    "dil_conv_d3",
    "dil_conv_d4",
    "Dsep_conv",
    "invert_Bot_Conv_E2",
    "conv_transpose",
    "identity",
]
CHANNELS = [16, 32, 48, 64, 16, 32, 48, 64]
REPEAT   = [1, 2, 3, 4, 1, 2, 3, 4]
K        = [1, 3, 5, 7, 1, 3, 5, 7]


def convert_cell(cell_bit_string):
    tmp = [cell_bit_string[i:i + 3] for i in range(0, len(cell_bit_string), 3)]
    return [tmp[i:i + 3] for i in range(0, len(tmp), 3)]


def convert(bit_string):
    b1 = convert_cell(bit_string[:len(bit_string) // 3])
    b2 = convert_cell(bit_string[len(bit_string) // 3:(len(bit_string) // 3) * 2])
    b3 = convert_cell(bit_string[(len(bit_string) // 3) * 2:])
    return [b1, b2, b3]


def decode_gene_to_genotype(gene: List[int]) -> Genotype:
    g = copy.deepcopy(gene)
    ch_idx = g.pop(0)
    b1, b2, b3 = convert(g)

    branch1 = [("channels", CHANNELS[ch_idx])]
    branch2 = [("channels", CHANNELS[ch_idx])]
    branch3 = [("channels", CHANNELS[ch_idx])]

    for block in b1:
        for unit in block:
            branch1.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))
    for block in b2:
        for unit in block:
            branch2.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))
    for block in b3:
        for unit in block:
            branch3.append((PRIMITIVES[unit[0]], [K[unit[1]], K[unit[1]]], REPEAT[unit[2]]))

    return Genotype(Branch1=branch1, Branch2=branch2, Branch3=branch3)


def get_branches(genotype: Genotype):
    gens = copy.deepcopy(genotype)
    conv_args = {"activation": "relu", "padding": "same"}

    channels = []
    for element in gens:
        channels.append(element.pop(0))  # ('channels', C)

    branches = [[], [], []]
    for i in range(len(gens)):
        for (op, kernel, rep) in gens[i]:
            if op == "conv":
                for _ in range(rep):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, **conv_args))
            elif op == "dil_conv_d2":
                for _ in range(rep):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=2, **conv_args))
            elif op == "dil_conv_d3":
                for _ in range(rep):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=3, **conv_args))
            elif op == "dil_conv_d4":
                for _ in range(rep):
                    branches[i].append(layers.Conv2D(channels[i][1], kernel, dilation_rate=4, **conv_args))
            elif op == "Dsep_conv":
                for _ in range(rep):
                    branches[i].extend([
                        layers.DepthwiseConv2D(kernel, **conv_args),
                        layers.Conv2D(channels[i][1], 1, **conv_args),
                    ])
            elif op == "invert_Bot_Conv_E2":
                expand = int(channels[i][1]) * 2
                for _ in range(rep):
                    branches[i].extend([
                        layers.Conv2D(expand, 1, **conv_args),
                        layers.DepthwiseConv2D(kernel, **conv_args),
                        layers.Conv2D(channels[i][1], kernel, **conv_args),
                    ])
            elif op == "conv_transpose":
                for _ in range(rep):
                    branches[i].append(layers.Conv2DTranspose(channels[i][1], kernel, **conv_args))
            elif op == "identity":
                branches[i].append(layers.Identity())
            else:
                raise ValueError(f"Unknown op: {op}")

    channels_mod = channels[0][1]
    return branches[0], branches[1], branches[2], channels_mod


def build_model_from_genotype(genotype: Genotype, upscale_factor: int = 2) -> tf.keras.Model:
    """
    Keras 3 safe, serializable.
    Output float32 sigmoid for stable PSNR.
    """
    branch1, branch2, branch3, channels_mod = get_branches(genotype)
    conv_args = {"activation": "relu", "padding": "same"}

    inputs = layers.Input(shape=(None, None, 3), name="lr")
    stem = layers.Conv2D(channels_mod, 3, **conv_args, name="stem")(inputs)

    b1 = stem
    for l in branch1:
        b1 = l(b1)
    b2 = stem
    for l in branch2:
        b2 = l(b2)
    b3 = stem
    for l in branch3:
        b3 = l(b3)

    x = layers.Add(name="merge")([b1, b2, b3])

    x = layers.Conv2D(3 * (upscale_factor ** 2), 3, **conv_args, name="pre_shuffle")(x)
    x = PixelShuffle(upscale_factor, name="pixel_shuffle")(x)

    out = layers.Conv2D(3, 3, padding="same", activation="sigmoid", dtype="float32", name="sr")(x)
    return keras.Model(inputs, out, name="SR_gene")


# ============================================================
# 7) Genes loader
# ============================================================
def load_genes_csv(path: str, expected_len: int = 28) -> List[List[int]]:
    genes: List[List[int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            vals = [v for v in line.strip().split(",") if v != ""]
            if not vals:
                continue
            g = [int(v) for v in vals]
            if expected_len and len(g) != expected_len:
                raise ValueError(f"Expected {expected_len} ints per line, got {len(g)}")
            genes.append(g)
    return genes


# ============================================================
# 8) Training helpers (robust early stop for both stages)
# ============================================================
def compile_model(model: tf.keras.Model, lr: float, wd: float, eps: float) -> None:
    opt = tf.keras.optimizers.AdamW(learning_rate=lr, weight_decay=wd, epsilon=eps)
    model.compile(
        optimizer=opt,
        loss=tf.keras.losses.MeanSquaredError(),
        metrics=[psnr],
        steps_per_execution=100, 
    )



def make_callbacks(stage_dir: str, stage_name: str, es_patience: int, es_min_delta: float) -> List[tf.keras.callbacks.Callback]:
    os.makedirs(stage_dir, exist_ok=True)
    return [
        tf.keras.callbacks.CSVLogger(os.path.join(stage_dir, f"{stage_name}_log.csv"), append=False),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(stage_dir, f"{stage_name}_best.keras"),
            monitor="val_psnr",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(stage_dir, f"{stage_name}_last.keras"),
            save_best_only=False,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_psnr",
            mode="max",
            factor=0.5,
            patience=max(2, es_patience // 4),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_psnr",
            mode="max",
            patience=es_patience,
            min_delta=es_min_delta,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.TerminateOnNaN(),
    ]


def fit_stage(
    model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    steps_per_epoch: int,
    val_steps: int,
    max_epochs: int,
    stage_dir: str,
    stage_name: str,
    es_patience: int,
    es_min_delta: float,
) -> tf.keras.callbacks.History:
    callbacks = make_callbacks(stage_dir, stage_name, es_patience, es_min_delta)
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=max_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=callbacks,
        verbose=0,
    )


# ============================================================
# 9) Main
# ============================================================
def main():
    args = parse_args()
    directory_train = args.directory_train
    directory_val = args.directory_val
    genes_csv = args.genes_csv
    base_seed = args.base_seed
    batch_size = args.batch_size
    overlap_val = args.overlap_val
    upscale_factor = args.upscale_factor
    max_epochs_st1 = args.max_epochs1
    max_epochs_st2 = args.max_epochs2

    # VM + A100:
    CPU_MODE = False  # MUST be False in GPU VM/Docker

    configure_threads()
    configure_runtime(cpu_mode=CPU_MODE)
    configure_mixed_precision(cpu_mode=CPU_MODE)
    enable_xla()

    print("TF:", tf.__version__)
    print("tf.keras:", tf.keras.__version__)
    print("CPU_MODE:", CPU_MODE)
    print("GPUs:", len(tf.config.list_logical_devices("GPU")))

    # Reproducibility (CPU vs GPU not bit-identical)
    random.seed(base_seed)
    np.random.seed(base_seed)
    tf.random.set_seed(base_seed)
    ratio = upscale_factor
    outputspath = f"outputs{upscale_factor}"
    


    # A100 recommended starting point (fast + usually good PSNR):
    # - Stage1 bigger batch (improves throughput)
    # - Stage2 a bit smaller batch (more stable fine-tuning)
    st1 = DataCfg(ratio=ratio, patch_size=64,  overlap=overlap_val, batch_size=batch_size)
    st2 = DataCfg(ratio=ratio, patch_size=128, overlap=overlap_val, batch_size=batch_size)

    # Steps per epoch based on full dataset sliding-window count (keeps your "epoch=dataset-sized" semantics)
    st1_train_steps, st1_train_patches, st1_imgs = compute_steps_per_epoch(directory_train, st1)
    st2_train_steps, st2_train_patches, st2_imgs = compute_steps_per_epoch(directory_train, st2)

    # Deterministic validation (stable PSNR + stable early stop)
    val_ds_st1, st1_val_steps = make_val_dataset_deterministic(directory_val, st1)
    val_ds_st2, st2_val_steps = make_val_dataset_deterministic(directory_val, st2)

    print(f"[ST1] p64  train_imgs={st1_imgs} patches={st1_train_patches:,} steps={st1_train_steps:,} | val_steps={st1_val_steps:,}")
    print(f"[ST2] p128 train_imgs={st2_imgs} patches={st2_train_patches:,} steps={st2_train_steps:,} | val_steps={st2_val_steps:,}")

    # FAST GPU training datasets (random crop per step)
    train_ds_st1 = make_train_dataset_randomcrop(directory_train, st1, seed=base_seed + 11)
    train_ds_st2 = make_train_dataset_randomcrop(directory_train, st2, seed=base_seed + 22)

    # Genes
    genes = load_genes_csv(genes_csv, expected_len=28)
    print("[GENES] total:", len(genes))

    run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    root = os.path.join(outputspath, f"a100_fast_bestpsnr_{run_id}")
    os.makedirs(root, exist_ok=True)

    summary_path = os.path.join(root, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "gene_index", "params",
                "st1_best_val_psnr", "st2_best_val_psnr",
                "st1_best_model", "st2_best_model",
                "best_overall_psnr", "best_overall_model",
            ],
        )
        w.writeheader()

    strategy = get_strategy_auto()
    print("[STRATEGY]", type(strategy).__name__, "| replicas:", strategy.num_replicas_in_sync)

    for gene_index, gene in enumerate(genes, start=1):
        print(f"\n================ Gene {gene_index}/{len(genes)} ================")
        gene_dir = os.path.join(root, f"gene_{gene_index:03d}")
        os.makedirs(gene_dir, exist_ok=True)

        # ---------- Stage 1 ----------
        with strategy.scope():
            genotype = decode_gene_to_genotype(gene)
            model = build_model_from_genotype(genotype, upscale_factor=upscale_factor)
            params = int(model.count_params())

            # LR scaling: batch 256 is 4x batch 64 -> start moderately higher than 3e-4
            # (Too aggressive LR can hurt PSNR. This is a safe starting point.)
            compile_model(model, lr=5e-4, wd=1e-8, eps=1e-7)

        print("[MODEL] params:", params)

        hist1 = fit_stage(
            model=model,
            train_ds=train_ds_st1,
            val_ds=val_ds_st1,
            steps_per_epoch=st1_train_steps,
            val_steps=st1_val_steps,
            max_epochs=max_epochs_st1,              # high cap; ES decides
            stage_dir=gene_dir,
            stage_name="stage1_p64",
            es_patience=14,
            es_min_delta=0.02,
        )
        st1_best_val = float(np.nanmax(hist1.history.get("val_psnr", [np.nan])))
        stage1_best_path = os.path.join(gene_dir, "stage1_p64_best.keras")

        # ---------- Stage 2 ----------
        with strategy.scope():
            model2 = tf.keras.models.load_model(stage1_best_path, compile=False)
            # Fine-tune lower LR
            compile_model(model2, lr=2e-5, wd=1e-8, eps=1e-7)

        hist2 = fit_stage(
            model=model2,
            train_ds=train_ds_st2,
            val_ds=val_ds_st2,
            steps_per_epoch=st2_train_steps,
            val_steps=st2_val_steps,
            max_epochs=max_epochs_st2,              # high cap; ES decides
            stage_dir=gene_dir,
            stage_name="stage2_p128",
            es_patience=8,
            es_min_delta=0.01,
        )
        st2_best_val = float(np.nanmax(hist2.history.get("val_psnr", [np.nan])))
        stage2_best_path = os.path.join(gene_dir, "stage2_p128_best.keras")

        # Choose best overall (Stage2 can sometimes hurt; we never accept a worse final)
        if np.isnan(st2_best_val) or (st1_best_val >= st2_best_val):
            best_overall_psnr = st1_best_val
            best_overall_model = stage1_best_path
        else:
            best_overall_psnr = st2_best_val
            best_overall_model = stage2_best_path

        with open(summary_path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "gene_index", "params",
                    "st1_best_val_psnr", "st2_best_val_psnr",
                    "st1_best_model", "st2_best_model",
                    "best_overall_psnr", "best_overall_model",
                ],
            )
            w.writerow({
                "gene_index": gene_index,
                "params": params,
                "st1_best_val_psnr": st1_best_val,
                "st2_best_val_psnr": st2_best_val,
                "st1_best_model": stage1_best_path,
                "st2_best_model": stage2_best_path,
                "best_overall_psnr": best_overall_psnr,
                "best_overall_model": best_overall_model,
            })

        tf.keras.backend.clear_session()

    print("\nDONE. Outputs at:", root)
    print("Summary:", summary_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Training configuration")

    parser.add_argument("--directory_train",type=str,default="DATASET/DIV2K_train_HR",help="Training images directory")
    parser.add_argument("--directory_val",type=str,default="DATASET/DIV2K_valid_HR",help="Validation images directory")
    parser.add_argument("--genes_csv",type=str,default="pareto_models.csv",help="CSV containing model genes")
    parser.add_argument("--base_seed",type=int,default=1,help="Random seed")
    parser.add_argument("--batch_size",type=int,default=64,help="Batch size")
    parser.add_argument("--overlap_val",type=float,default=0.1,help="Validation overlap fraction")
    parser.add_argument("--upscale_factor",type=int,default=4,choices=[2, 4],help="Super-resolution scale factor")
    parser.add_argument("--max_epochs1", type=int, default=200, help="Define the maximun epochs for st1")
    parser.add_argument("--max_epochs2", type=int, default=120, help="Define the maximun epochs for st2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
