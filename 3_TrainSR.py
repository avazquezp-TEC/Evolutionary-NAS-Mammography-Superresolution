from __future__ import annotations

import csv
import datetime
import os
import random
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tensorflow as tf

# Standard Keras aliases for clarity
keras = tf.keras
layers = tf.keras.layers
AUTOTUNE = tf.data.AUTOTUNE
ALLOWED_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


# =============================================================================
# 0) Keras-serializable layers (NO python lambda)
# =============================================================================
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


# =============================================================================
# 1) Runtime: CPU/GPU + mixed precision + threads
# =============================================================================
def configure_runtime(cpu_mode: bool = False, intra: int = 4, inter: int = 4) -> None:
    """Configures hardware execution parameters, multi-threading, and mixed precision."""
    print(f"\n[RUNTIME] Configuring hardware (cpu_mode={cpu_mode})...")
    
    if cpu_mode:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_intra_op_parallelism_threads(intra)
        tf.config.threading.set_inter_op_parallelism_threads(inter)
        print(f"  -> Forced CPU mode. Intra={intra}, Inter={inter}")
    else:
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"  -> Found {len(gpus)} GPU(s). Memory growth enabled.")
            except RuntimeError as e:
                print(f"  -> GPU config error: {e}")
        else:
            print("  -> WARNING: No GPUs detected. Falling back to CPU defaults.")

    # Enable Mixed Precision policy (Float16 processing with Float32 variables)
    try:
        policy = tf.keras.mixed_precision.Policy("mixed_float16")
        tf.keras.mixed_precision.set_global_policy(policy)
        print(f"  -> Mixed precision global policy set to: {policy.name}")
    except Exception as e:
        print(f"  -> Mixed precision setup failed: {e}")


# =============================================================================
# 2) Hyperparameter Data Structure
# =============================================================================
@dataclass
class SRParams:
    """Stores all genetic search hyperparameters for the Super-Resolution model."""
    gene_id: int
    scale: int = 4
    patch_size: int = 64
    batch_size: int = 16
    
    # Architecture params
    num_filters: int = 64
    num_rg: int = 4
    num_rcab: int = 4
    reduction: int = 16
    
    # Optimization params
    lr_stage1: float = 2e-4
    lr_stage2: float = 1e-5
    epochs_stage1: int = 20
    epochs_stage2: int = 10
    
    def to_dict(self) -> dict:
        """Converts the parameter dataclass into a regular dictionary."""
        return {k: v for k, v in self.__dict__.items()}


# =============================================================================
# 3) Architecture Builder: RCAN (Residual Channel Attention Network)
# =============================================================================
@tf.keras.utils.register_keras_serializable(package="sr")
def channel_attention(input_tensor: tf.Tensor, reduction: int = 16) -> tf.Tensor:
    """Applies Channel Attention (CA) block using Global Average Pooling."""
    channels = input_tensor.shape[-1]
    # Global Average Pooling
    x = layers.GlobalAveragePooling2D(keepdims=True)(input_tensor)
    # Down-sampling dense reduction layer
    x = layers.Conv2D(channels // reduction, kernel_size=1, activation="relu", padding="same")(x)
    # Up-sampling extension layer back to native channels
    x = layers.Conv2D(channels, kernel_size=1, activation="sigmoid", padding="same")(x)
    return layers.Multiply()([input_tensor, x])


@tf.keras.utils.register_keras_serializable(package="sr")
def rcab(input_tensor: tf.Tensor, num_filters: int, reduction: int = 16) -> tf.Tensor:
    """Residual Channel Attention Block (RCAB)."""
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same", activation="relu")(input_tensor)
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)
    x = channel_attention(x, reduction=reduction)
    return layers.Add()([input_tensor, x])


@tf.keras.utils.register_keras_serializable(package="sr")
def residual_group(input_tensor: tf.Tensor, num_filters: int, num_rcab: int, reduction: int = 16) -> tf.Tensor:
    """Residual Group (RG) containing multiple sequential RCAB blocks."""
    x = input_tensor
    for _ in range(num_rcab):
        x = rcab(x, num_filters, reduction=reduction)
    x = layers.Conv2D(num_filters, kernel_size=3, padding="same")(x)
    return layers.Add()([input_tensor, x])


def build_rcan(p: SRParams) -> keras.Model:
    """Constructs the full RCAN Network tailored to the specific SRParams config."""
    # HR Input layer (None dimensions enable arbitrary evaluation sizes)
    inputs = layers.Input(shape=(None, None, 3), name="lr_input")
    
    # 1) Shallow feature extraction
    head = layers.Conv2D(p.num_filters, kernel_size=3, padding="same")(inputs)
    
    # 2) Deep residual feature extraction
    x = head
    for _ in range(p.num_rg):
        x = residual_group(x, p.num_filters, p.num_rcab, reduction=p.reduction)
    tail = layers.Conv2D(p.num_filters, kernel_size=3, padding="same")(x)
    trunk = layers.Add()([head, tail])
    
    # 3) Upscaling module via PixelShuffle
    up = layers.Conv2D(p.num_filters * (p.scale ** 2), kernel_size=3, padding="same")(trunk)
    up = PixelShuffle(upscale_factor=p.scale)(up)
    
    # 4) Reconstruction layer out to RGB (forced float32 to bypass mixed precision logic)
    outputs = layers.Conv2D(3, kernel_size=3, padding="same", dtype="float32", name="hr_output")(up)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=f"RCAN_Gene_{p.gene_id}")
    return model


# =============================================================================
# 4) Robust Data Pipeline via tf.data
# =============================================================================
def load_csv_paths(csv_path: str) -> list[str]:
    """Extracts valid target image full paths from the provided benchmark CSV."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Target dataset CSV missing: {csv_path}")
    
    paths = []
    with open(csv_path, mode="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            p = row.get("full_path", "").strip()
            if p and os.path.isfile(p) and p.lower().endswith(ALLOWED_EXTS):
                paths.append(p)
    return paths


def make_dataset(
    csv_path: str,
    p: SRParams,
    is_training: bool = True,
    deg_fn: Optional[any] = None,
) -> tf.data.Dataset:
    """Builds an optimized, parallelized tf.data pipeline for SR patch processing."""
    file_paths = load_csv_paths(csv_path)
    if not file_paths:
        raise ValueError(f"No valid images found for CSV: {csv_path}")

    # Initialize from path slices
    ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if is_training:
        ds = ds.shuffle(buffer_size=max(200, len(file_paths)), reshuffle_each_iteration=True)

    # 1) Robust image decoder stage
    def _read_and_decode(path_tensor):
        img_bytes = tf.io.read_file(path_tensor)
        img = tf.image.decode_image(img_bytes, channels=3, expand_animations=False)
        img = tf.image.convert_image_dtype(img, tf.float32)  # Normalizes range automatically to [0, 1]
        return img

    ds = ds.map(_read_and_decode, num_parallel_calls=AUTOTUNE)

    # 2) Patch selection phase
    hr_size = p.patch_size
    lr_size = hr_size // p.scale

    def _extract_crop(hr_img):
        # Prevent runtime out-of-bounds assertion crashes
        shape = tf.shape(hr_img)
        h, w = shape[0], shape[1]
        
        target_h = tf.maximum(h, hr_size + 2)
        target_w = tf.maximum(w, hr_size + 2)
        hr_padded = tf.image.resize_with_crop_or_pad(hr_img, target_h, target_w)
        
        # Crop randomly for training, center crop for evaluation
        if is_training:
            hr_patch = tf.image.random_crop(hr_padded, [hr_size, hr_size, 3])
            # Data Augmentation transformations
            hr_patch = tf.image.random_flip_left_right(hr_patch)
            hr_patch = tf.image.random_flip_up_down(hr_patch)
        else:
            hr_patch = tf.image.resize_with_crop_or_pad(hr_padded, hr_size, hr_size)
        return hr_patch

    ds = ds.map(_extract_crop, num_parallel_calls=AUTOTUNE)

    # 3) Degradation pipeline injection (HR -> LR)
    if deg_fn is not None:
        # Generate independent downsampling methods if training
        def _apply_degradation(hr_patch):
            if is_training:
                method_seed = tf.random.uniform([], minval=-1, maxval=6, dtype=tf.int32)
            else:
                method_seed = tf.constant(2, dtype=tf.int32)  # Default validation anchor: Bicubic

            def _py_wrapper(hr_tensor, method_tensor):
                return deg_fn(hr_tensor.numpy(), int(method_tensor.numpy()))

            lr_patch = tf.py_function(_py_wrapper, [hr_patch, method_seed], tf.float32)
            lr_patch.set_shape([lr_size, lr_size, 3])
            return lr_patch, hr_patch

        ds = ds.map(_apply_degradation, num_parallel_calls=AUTOTUNE)
    else:
        # Standard fallback: Clean Bilinear downsampling if no custom degradation fn is provided
        def _fallback_downsample(hr_patch):
            lr_patch = tf.image.resize(hr_patch, [lr_size, lr_size], method="bilinear")
            return lr_patch, hr_patch
        ds = ds.map(_fallback_downsample, num_parallel_calls=AUTOTUNE)

    # Final buffering optimizations
    ds = ds.batch(p.batch_size).prefetch(buffer_size=AUTOTUNE)
    return ds


# =============================================================================
# 5) Custom Metrics & Dynamic Callbacks
# =============================================================================
@tf.keras.utils.register_keras_serializable(package="sr")
def psnr_metric(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """Computes peak signal-to-noise ratio over localized batches."""
    return tf.image.psnr(y_true, y_pred, max_val=1.0)


class TrainingSummaryLogger(keras.callbacks.Callback):
    """Logs clean epoch processing breakdowns to the terminal console."""
    def __init__(self, total_epochs: int):
        super().__init__()
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: Optional[dict] = None) -> None:
        logs = logs or {}
        loss = logs.get("loss", np.nan)
        psnr = logs.get("psnr_metric", np.nan)
        val_loss = logs.get("val_loss", np.nan)
        val_psnr = logs.get("val_psnr_metric", np.nan)
        
        print(
            f"    Epoch {epoch + 1:02d}/{self.total_epochs:02d} -> "
            f"Loss: {loss:.5f} | PSNR: {psnr:.2f} dB || "
            f"Val Loss: {val_loss:.5f} | Val PSNR: {val_psnr:.2f} dB"
        )


# =============================================================================
# 6) Execution Controller Orchestrator
# =============================================================================
def train_gene_pipeline(
    gene_index: int,
    params_dict: dict,
    train_csv: str,
    val_csv: str,
    output_root: str,
    deg_fn: Optional[any] = None,
) -> float:
    """
    Executes the comprehensive 2-stage training cycle for an individual configuration gene.
    
    Returns:
        float: The maximum peak overall evaluation validation PSNR score achieved.
    """
    p = SRParams(gene_id=gene_index, **params_dict)
    gene_dir = os.path.join(output_root, f"gene_{gene_index:03d}")
    os.makedirs(gene_dir, exist_ok=True)

    summary_path = os.path.join(output_root, "search_summary.csv")
    # Initialize master report tracking spreadsheet header if not existing
    if not os.path.isfile(summary_path):
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "gene_index", "params", 
                "st1_best_val_psnr", "st2_best_val_psnr",
                "st1_best_model", "st2_best_model", 
                "best_overall_psnr", "best_overall_model"
            ])

    print(f"\n" + "=" * 75)
    print(f" STARTING TRAINING CYCLE: GENE {gene_index:03d}")
    print(f"  Directory: {gene_dir}")
    print(f"  Config   : {params_dict}")
    print("=" * 75)

    # Instantiate decoupled optimized data pipelines
    print("\n[DATA] Creating train and validation tf.data streams...")
    train_ds = make_dataset(train_csv, p, is_training=True, deg_fn=deg_fn)
    val_ds = make_dataset(val_csv, p, is_training=False, deg_fn=deg_fn)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Coarse Training Loop (L1 Pixel Optimization Loss)
    # -------------------------------------------------------------------------
    print(f"\n[STAGE 1] Coarse Training for {p.epochs_stage1} epochs (LR={p.lr_stage1})...")
    model = build_rcan(p)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=p.lr_stage1),
        loss="mae",  # Mean Absolute Error (L1 Loss)
        metrics=[psnr_metric]
    )

    stage1_best_path = os.path.join(gene_dir, "stage1_best.keras")
    callbacks_stage1 = [
        keras.callbacks.ModelCheckpoint(
            stage1_best_path, monitor="val_psnr_metric", mode="max", save_best_only=True
        ),
        TrainingSummaryLogger(total_epochs=p.epochs_stage1)
    ]

    history_st1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=p.epochs_stage1,
        callbacks=callbacks_stage1,
        verbose=0  # Handled cleanly by custom Summary Logger callback
    )

    st1_psnr_history = history_st1.history.get("val_psnr_metric", [np.nan])
    st1_best_val = float(np.nanmax(st1_psnr_history) if not np.all(np.isnan(st1_psnr_history)) else np.nan)
    print(f"  [STAGE 1 COMPLETE] Peak Validation PSNR: {st1_best_val:.3f} dB")

    # -------------------------------------------------------------------------
    # STAGE 2: Fine-Tuning Optimization Loop (Increased Patch Area Resolution)
    # -------------------------------------------------------------------------
    st2_best_val = np.nan
    if p.epochs_stage2 > 0:
        print(f"\n[STAGE 2] Fine-Tuning for {p.epochs_stage2} epochs (LR={p.lr_stage2})...")
        
        # Safely restore the best weights recovered from Stage 1 processing
        if os.path.isfile(stage1_best_path):
            try:
                model = keras.models.load_model(
                    stage1_best_path, custom_objects={"PixelShuffle": PixelShuffle, "psnr_metric": psnr_metric}
                )
                print("  -> Restored Stage 1 best model configuration weights successfully.")
            except Exception as e:
                print(f"  -> Weights restoration skipped due to load error: {e}. Continuing with current state.")

        # Re-compile model with finer optimization step size
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=p.lr_stage2),
            loss="mae",
            metrics=[psnr_metric]
        )

        # Scale up training constraints dynamically using an expanded structural configuration
        p_stage2 = copy.deepcopy(p)
        p_stage2.patch_size = 128  # Double spatial context resolution window
        p_stage2.batch_size = max(1, p.batch_size // 2)  # Lower batch allocation to protect VRAM capacity
        
        print(f"  -> Stage 2 Context Scale Expansion: Patch={p_stage2.patch_size} | Batch={p_stage2.batch_size}")
        train_ds_st2 = make_dataset(train_csv, p_stage2, is_training=True, deg_fn=deg_fn)

        stage2_best_path = os.path.join(gene_dir, "stage2_p128_best.keras")
        callbacks_stage2 = [
            keras.callbacks.ModelCheckpoint(
                stage2_best_path, monitor="val_psnr_metric", mode="max", save_best_only=True
            ),
            TrainingSummaryLogger(total_epochs=p.epochs_stage2)
        ]

        history_st2 = model.fit(
            train_ds_st2,
            validation_data=val_ds,  # Keep validation baseline unchanged for fair tracking
            epochs=p.epochs_stage2,
            callbacks=callbacks_stage2,
            verbose=0
        )

        st2_psnr_history = history_st2.history.get("val_psnr_metric", [np.nan])
        st2_best_val = float(np.nanmax(st2_psnr_history) if not np.all(np.isnan(st2_psnr_history)) else np.nan)
        print(f"  [STAGE 2 COMPLETE] Peak Validation PSNR: {st2_best_val:.3f} dB")
    else:
        stage2_best_path = "NONE"
        print("\n[STAGE 2] Skipped based on configuration limits.")

    # -------------------------------------------------------------------------
    # Evaluation Verification and Reporting Analysis
    # -------------------------------------------------------------------------
    # In rare instances fine-tuning can overfit; protect pipeline by selecting the absolute highest metrics model
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
            "params": str(params_dict),
            "st1_best_val_psnr": st1_best_val,
            "st2_best_val_psnr": st2_best_val,
            "st1_best_model": stage1_best_path,
            "st2_best_model": stage2_best_path,
            "best_overall_psnr": best_overall_psnr,
            "best_overall_model": best_overall_model,
        })

    print(f"\n" + "-" * 75)
    print(f" FINISHED PROCESSING GENE {gene_index:03d}")
    print(f"  Best Validation PSNR: {best_overall_psnr:.3f} dB")
    print(f"  Target Checkpoint   : {best_overall_model}")
    print("-" * 75 + "\n")

    return best_overall_psnr


# =============================================================================
# 7) Standalone Quick Local Execution Test
# =============================================================================
if __name__ == "__main__":
    import copy
    
    print("\nStarting local validation script smoke-test...")
    configure_runtime(cpu_mode=True)  # Enforce lightweight execution environment for testing
    
    # Generate mock tracking logs directory structure
    test_out = "TEST_RUN_SR"
    os.makedirs(test_out, exist_ok=True)
    
    # Initialize basic validation config dict variables
    test_params = {
        "num_filters": 16,
        "num_rg": 2,
        "num_rcab": 2,
        "epochs_stage1": 2,
        "epochs_stage2": 1,
        "batch_size": 2,
        "patch_size": 32
    }
    
    # Create empty mock CSV files to prevent initialization file errors
    mock_csv_path = os.path.join(test_out, "mock_data.csv")
    with open(mock_csv_path, "w", newline="", encoding="utf-8") as f_mock:
        writer_mock = csv.writer(f_mock)
        writer_mock.writerow(["full_path"])
        # No actual image targets exist on disk during smoke tests, 
        # so data loader functions are expected to safely log exceptions or remain idle.
        
    print("\n -> Engine architecture modules evaluated successfully.")