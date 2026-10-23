#!/usr/bin/env python3
"""
benchmark_sr.py
===============
Computational cost of SR models using the SAME definitions in PyTorch and TensorFlow:
parameters, FLOPs/MACs, latency, end-to-end inference time, and peak memory.

Run one method at a time (each lives in its own repository/environment); all rows are APPENDED
to the same CSV (--csv), so you can combine EDSR+MHCA, DLSR, EMNAS, FALSR, and Keras
models in a single table.
Copy this file to the root of each repository (next to its finetune_*.py files), or use --repo_dir.

    # PyTorch (torch environment)
    python benchmark_sr.py --method mhca  --scale 2 --weights best_mhca_x2_finetuned.pth
    python benchmark_sr.py --method rfdn  --scale 2 --weights checkpoints/Best/best_rfdn_beta21_x2_finetuned.pth
    python benchmark_sr.py --method emnas --scale 2 --variant a --n_resgroups 4 --weights best_emnas_a_x2_finetuned.pth
    # TensorFlow (tf environment)
    python benchmark_sr.py --method falsr --weights outputs2x/mammo_blindSR/falsr_a/mammo_ft_best.pb
    python benchmark_sr.py --method keras --scale 2 --weights "../SRIR/outputs2x/mammo_blindSR/gene_001/mammo_ft_best.keras"
    python benchmark_sr.py --method keras --scale 2 --weights "../SRIR/outputs2x/mammo_blindSR/gene_002/mammo_ft_best.keras"
    python benchmark_sr.py --method keras --scale 2 --weights "../SRIR/outputs2x/mammo_blindSR/gene_003/mammo_ft_best.keras"
    python benchmark_sr.py --method keras --scale 2 --weights "../SRIR/outputs2x/mammo_blindSR/gene_004/mammo_ft_best.keras"
    python benchmark_sr.py --method keras --scale 2 --weights "../SRIR/outputs2x/mammo_blindSR/gene_005/mammo_ft_2best.keras"
    # Add --lr_dir DATA/Lowx2/mammo_test to also measure inference time on real images.

Definitions (identical across both frameworks; batch = 1, fp32, model as used during testing):
  params        Parameters of the layers EXECUTED in the forward pass (e.g., excludes the
                `spatial_attention_2` layers of the two-branch MHCA model and unused MeanShift).
                params_stored = everything stored in the model.
  GMACs/GFLOPs  Convolutions only (standard, depthwise, transposed) and matrix multiplications;
                FLOPs = 2 x MACs. Activations, additions, pooling, resize, and pixel shuffle are excluded.
                Computed for an HR output of --hr_sizes (default: 720x1280 = 1280x720, the
                lightweight-SR literature standard); the LR input is HR/scale.
  latency       Time for ONE model forward pass with random input at the specified size, after --warmup,
                over --iters repetitions, synchronizing the GPU on each iteration (ms; mean, std., median, p95).
                In TensorFlow, synchronization is performed by returning a scalar (tf.reduce_max), without
                copying the image.
  e2e           Time per real image (with --lr_dir): file read + preprocessing + forward pass (with --tile,
                if requested) + returning the SR output to CPU. The first images are discarded as warmup
                (up to 2 and at most one third of the total).
  peak_mem      Peak GPU memory during a forward pass (weights + input + activations + workspace),
                in MB; act_mem = peak minus memory already reserved before the forward pass.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FRAMEWORK = {"mhca": "torch", "rfdn": "torch", "emnas": "torch", "falsr": "tf", "keras": "tf"}
FIELDS = ["timestamp", "method", "name", "scale", "hr_size", "lr_size", "framework", "device", "gpu", "lib",
          "precision", "params", "params_stored", "gmacs", "gflops", "latency_ms_mean", "latency_ms_std",
          "latency_ms_median", "latency_ms_p95", "latency_ms_min", "fps", "peak_mem_mb", "act_mem_mb",
          "e2e_ms_mean", "e2e_ms_median", "e2e_n_images", "weights", "notes"]


# ==============================================================================
# Common utilities
# ==============================================================================
def parse_size(txt: str):
    h, w = (int(v) for v in txt.lower().replace("*", "x").split("x"))
    return h, w


def stats(times_ms):
    t = np.asarray(times_ms, dtype=np.float64)
    return dict(mean=float(t.mean()), std=float(t.std(ddof=1)) if len(t) > 1 else 0.0,
                median=float(np.median(t)), p95=float(np.percentile(t, 95)), min=float(t.min()))


def append_csv(path, rows):
    path = Path(path)
    new = not path.exists() or path.stat().st_size == 0
    if not new:
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
        if header != FIELDS:
            raise SystemExit(f"{path} exists with a different header; use a different --csv file.")
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def print_table(rows):
    cols = [("name", "model", 22), ("lr_size", "LR", 9), ("params", "params(M)", 10), ("gmacs", "GMACs", 9),
            ("gflops", "GFLOPs", 9), ("latency_ms_mean", "lat ms", 8), ("latency_ms_std", "±", 6),
            ("peak_mem_mb", "peak MB", 9), ("e2e_ms_mean", "e2e ms", 9)]
    print("\n" + " | ".join(f"{h:>{wd}}" for _, h, wd in cols))
    print("-+-".join("-" * wd for _, _, wd in cols))
    for r in rows:
        cells = []
        for k, _, wd in cols:
            v = r.get(k, "")
            if k == "params" and v != "":
                v = f"{float(v) / 1e6:.4f}"
            elif isinstance(v, float):
                v = f"{v:.2f}"
            cells.append(f"{str(v):>{wd}}")
        print(" | ".join(cells))


def build_row(args, name, hr, lr, framework, device, gpu, lib, params, params_stored, macs, lat, mem, e2e, notes):
    row = dict(timestamp=datetime.datetime.now().isoformat(timespec="seconds"), method=args.method, name=name,
               scale=args.scale, hr_size=f"{hr[0]}x{hr[1]}", lr_size=f"{lr[0]}x{lr[1]}", framework=framework,
               device=device, gpu=gpu, lib=lib, precision="fp32", params=params, params_stored=params_stored,
               gmacs=round(macs / 1e9, 6) if macs is not None else "",
               gflops=round(2 * macs / 1e9, 6) if macs is not None else "",
               weights=str(args.weights or ""), notes="; ".join(n for n in notes if n))
    if lat:
        s = stats(lat)
        row.update(latency_ms_mean=round(s["mean"], 4), latency_ms_std=round(s["std"], 4),
                   latency_ms_median=round(s["median"], 4), latency_ms_p95=round(s["p95"], 4),
                   latency_ms_min=round(s["min"], 4), fps=round(1000.0 / s["mean"], 3))
    if mem:
        row.update(peak_mem_mb=round(mem[0], 2), act_mem_mb=round(mem[1], 2))
    if e2e:
        s = stats(e2e)
        row.update(e2e_ms_mean=round(s["mean"], 3), e2e_ms_median=round(s["median"], 3), e2e_n_images=len(e2e))
    return row


# ==============================================================================
# PyTorch
# ==============================================================================
def build_torch_model(args):
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    notes = []
    if args.method == "mhca":
        from degraded_data import load_pretrained
        from finetune_mhca import MODEL_CFG, build_model
        model, name = build_model(args.scale), "EDSR+MHCA"
        if args.weights:
            ignore = () if MODEL_CFG["use_mhca_3"] else ("spatial_attention_2",)
            load_pretrained(model, args.weights, ignore_missing=ignore)
    elif args.method == "rfdn":
        if args.scale != 2:
            raise SystemExit("RFDN beta21 only supports x2.")
        from degraded_data import load_pretrained
        from finetune_rfdn_beta21 import RFDNGrayAdapter, build_core
        core = build_core()
        if args.weights:
            load_pretrained(core, args.weights)
        model, name = RFDNGrayAdapter(core, args.pixel_range), "RFDN-beta21 (DLSR)"
    else:  # emnas
        from degraded_data import load_pretrained
        from finetune_emnas import GrayAdapter, build_net, infer_arch, read_sd
        if not args.weights or args.n_resgroups is None:
            raise SystemExit("EMNAS requires --weights (used to infer the architecture) and --n_resgroups.")
        arch = infer_arch(read_sd(args.weights), args.scale)
        core = build_net(args.scale, args.variant, arch, args.n_resgroups)
        load_pretrained(core, args.weights)
        model, name = GrayAdapter(core, args.pixel_range), f"EMNAS-{args.variant}"
        notes.append(f"n_resgroups={args.n_resgroups}")
    if not args.weights:
        notes.append("no weights (random initialization; does not affect computational cost)")
    return model, args.name or name, notes


def torch_hook_macs(model, x):
    """Compute MACs for Conv/ConvTranspose/Linear layers through hooks (cross-check for the PyTorch counter)."""
    import torch.nn as nn
    total = [0]
    hooks = []

    def conv(m, i, o):
        total[0] += o.numel() * (m.in_channels // m.groups) * int(np.prod(m.kernel_size))

    def convT(m, i, o):
        total[0] += i[0].numel() * (m.out_channels // m.groups) * int(np.prod(m.kernel_size))

    def lin(m, i, o):
        total[0] += o.numel() * m.in_features

    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d):
            hooks.append(m.register_forward_hook(convT))
        elif isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin))
    import torch
    with torch.inference_mode():
        model(x)
    for h in hooks:
        h.remove()
    return total[0]


def torch_params_and_flops(model, x):
    import torch
    executed = set()
    hooks = [m.register_forward_hook(lambda mod, i, o: executed.add(mod)) for m in model.modules()]
    notes, macs = [], None
    try:
        from torch.utils.flop_counter import FlopCounterMode
        with torch.inference_mode(), FlopCounterMode(display=False) as fc:
            model(x)
        macs = fc.get_total_flops() / 2.0
    except ImportError:
        notes.append("torch<2.1: MACs computed only from Conv/Linear hooks")
    for h in hooks:
        h.remove()
    hook_macs = torch_hook_macs(model, x)
    if macs is None:
        macs = float(hook_macs)
    elif abs(macs - hook_macs) / max(macs, 1.0) > 0.01:
        notes.append(f"MACs dispatch={macs / 1e9:.3f}G vs hooks={hook_macs / 1e9:.3f}G (convolutions outside nn.Conv2d detected)")
    stored = sum(p.numel() for p in model.parameters())
    used = sum(p.numel() for m in executed for p in m.parameters(recurse=False))
    return used, stored, macs, notes


def run_torch(args, sizes):
    import torch
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but PyTorch cannot detect a GPU.")
    torch.backends.cudnn.benchmark = args.cudnn_benchmark
    model, name, notes0 = build_torch_model(args)
    model.to(device).eval()
    gpu = torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor() or "cpu"
    lib = f"torch {torch.__version__}" + (f" cuda {torch.version.cuda}" if device.type == "cuda" else
                                          f" threads {torch.get_num_threads()}")
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    rows = []
    for hr in sizes:
        lr = (hr[0] // args.scale, hr[1] // args.scale)
        x = torch.rand(1, 1, *lr, device=device) * args.pixel_range
        print(f"\n[{name}] HR {hr[0]}x{hr[1]} -> LR {lr[0]}x{lr[1]}")
        params, stored, macs, notes = torch_params_and_flops(model, x)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(x)
            sync()
            lat = []
            for _ in range(args.iters):
                sync()
                t0 = time.perf_counter()
                model(x)
                sync()
                lat.append((time.perf_counter() - t0) * 1e3)
        mem = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            with torch.inference_mode():
                model(x)
            sync()
            peak = torch.cuda.max_memory_allocated()
            mem = (peak / 2**20, (peak - base) / 2**20)
        e2e = torch_e2e(model, args, device, sync) if args.lr_dir else None
        rows.append(build_row(args, name, hr, lr, "torch", device.type, gpu, lib, params, stored, macs,
                              lat, mem, e2e, notes0 + notes))
        del x
    return rows


def torch_e2e(model, args, device, sync):
    import torch
    from degraded_data import list_images, load_gray
    from sr_inference import super_resolve
    files = list_images(args.lr_dir)[:args.n_images]
    if not files:
        raise SystemExit(f"No images found in {args.lr_dir}")
    times = []
    with torch.inference_mode():
        for f in files:
            sync()
            t0 = time.perf_counter()
            lr = torch.from_numpy(load_gray(f))[None, None].to(device) * args.pixel_range
            sr = super_resolve(model, lr, args.scale, args.tile, args.tile_overlap).clamp(0, args.pixel_range)
            _ = sr[0, 0].float().cpu().numpy()
            sync()
            times.append((time.perf_counter() - t0) * 1e3)
    return times[min(2, len(times) // 3):]


# ==============================================================================
# TensorFlow
# ==============================================================================
def tf_graph_macs(concrete_fn):
    """Compute MACs for convolution and matmul operations in a concrete function with a FIXED input shape."""
    import tensorflow as tf
    from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
    _, gd = convert_variables_to_constants_v2_as_graph(concrete_fn)
    total, skipped = 0, Counter()
    with tf.Graph().as_default() as g:
        tf.graph_util.import_graph_def(gd, name="")
        for op in g.get_operations():
            t = op.type
            if t not in ("Conv2D", "DepthwiseConv2dNative", "Conv2DBackpropInput", "MatMul", "BatchMatMul",
                         "BatchMatMulV2", "Einsum"):
                continue
            try:
                out = op.outputs[0].shape
                if not out.is_fully_defined():
                    skipped[t] += 1
                    continue
                if t == "Conv2D":
                    f = op.inputs[1].shape.as_list()                      # [kh, kw, cin/groups, cout]
                    total += out.num_elements() * f[0] * f[1] * f[2]
                elif t == "DepthwiseConv2dNative":
                    f = op.inputs[1].shape.as_list()
                    total += out.num_elements() * f[0] * f[1]
                elif t == "Conv2DBackpropInput":                          # transposed convolution
                    f = op.inputs[1].shape.as_list()                      # [kh, kw, cin_fwd, cout_fwd]
                    total += op.inputs[2].shape.num_elements() * f[0] * f[1] * f[2]
                elif t == "MatMul":
                    a = op.inputs[0].shape.as_list()
                    k = a[0] if op.get_attr("transpose_a") else a[1]
                    total += out.num_elements() * k
                elif t in ("BatchMatMul", "BatchMatMulV2"):
                    a = op.inputs[0].shape.as_list()
                    k = a[-2] if op.get_attr("adj_x") else a[-1]
                    total += out.num_elements() * k
                else:                                                     # Einsum
                    eq = op.get_attr("equation").decode().replace(" ", "")
                    ins = eq.split("->")[0].split(",")
                    sizes = {}
                    for sub, inp in zip(ins, op.inputs):
                        for ch, d in zip(sub, inp.shape.as_list()):
                            sizes[ch] = d
                    total += int(np.prod(list(sizes.values())))
            except Exception:
                skipped[t] += 1
    return total, skipped


def build_tf_model(args):
    import tensorflow as tf
    sys.path.insert(0, str(Path(args.repo_dir).resolve()))
    if not args.weights or not os.path.isfile(args.weights):
        raise SystemExit("Specify --weights with the FALSR .pb or a .keras file.")
    if args.method == "falsr":
        if args.scale != 2:
            raise SystemExit("FALSR only supports x2.")
        from finetune_falsr import FrozenFalsr
        net = FrozenFalsr(args.weights)
        name = args.name or f"FALSR ({Path(args.weights).parent.name}/{Path(args.weights).stem})"
        return net.forward_y, 1, net.n_params, net.n_params, name, ["Y branch of the graph (excluding final color conversion)"]
    from model_test import PixelShuffle, psnr
    model = tf.keras.models.load_model(args.weights, custom_objects={"PixelShuffle": PixelShuffle, "psnr": psnr})
    name = args.name or f"Keras ({Path(args.weights).parent.name})"
    p = int(model.count_params())
    return (lambda x: model(x, training=False)), 3, p, p, name, ["3-channel input (replicated grayscale), as used during testing"]


def run_tf(args, sizes):
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    import tensorflow as tf
    if args.device == "cpu" or args.cpu:
        tf.config.set_visible_devices([], "GPU")
    else:
        for g in tf.config.list_physical_devices("GPU"):
            try:
                tf.config.experimental.set_memory_growth(g, True)
            except Exception as exc:
                print(f"[GPU] {exc}")
    gpus = tf.config.list_logical_devices("GPU")
    device = "cuda" if gpus else "cpu"
    gpu = tf.config.experimental.get_device_details(tf.config.list_physical_devices("GPU")[0]).get(
        "device_name", "GPU") if gpus else (platform.processor() or "cpu")
    lib = f"tensorflow {tf.__version__}"
    forward, in_ch, params, stored, name, notes0 = build_tf_model(args)
    rows = []
    for hr in sizes:
        lr = (hr[0] // args.scale, hr[1] // args.scale)
        print(f"\n[{name}] HR {hr[0]}x{hr[1]} -> LR {lr[0]}x{lr[1]}")
        spec = tf.TensorSpec([1, lr[0], lr[1], in_ch], tf.float32)
        macs, skipped = tf_graph_macs(tf.function(forward).get_concrete_function(spec))
        notes = list(notes0) + ([f"unaccounted operations: {dict(skipped)}"] if skipped else [])

        @tf.function(input_signature=[spec])
        def run(x):
            return tf.reduce_max(forward(x))              # synchronizes without copying the image to the host

        x = tf.random.uniform([1, lr[0], lr[1], in_ch])
        for _ in range(args.warmup):
            run(x).numpy()
        lat = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            run(x).numpy()
            lat.append((time.perf_counter() - t0) * 1e3)
        mem = None
        if gpus:
            tf.config.experimental.reset_memory_stats("GPU:0")
            base = tf.config.experimental.get_memory_info("GPU:0")["current"]
            run(x).numpy()
            peak = tf.config.experimental.get_memory_info("GPU:0")["peak"]
            mem = (peak / 2**20, (peak - base) / 2**20)
        e2e = tf_e2e(args) if args.lr_dir else None
        rows.append(build_row(args, name, hr, lr, "tf", device, gpu, lib, params, stored, macs, lat, mem, e2e, notes))
    return rows


def tf_e2e(args):
    from V_main_test_flat import list_images, load_image_universal, load_predictor, super_resolve
    import numpy as _np
    files = list_images(args.lr_dir)[:args.n_images]
    if not files:
        raise SystemExit(f"No images found in {args.lr_dir}")
    predict, _handle = load_predictor(args.weights)
    times = []
    for f in files:
        t0 = time.perf_counter()
        lr = load_image_universal(f)
        sr = _np.clip(super_resolve(predict, lr, args.scale, args.tile, args.tile_overlap), 0, 1)
        _ = sr.shape
        times.append((time.perf_counter() - t0) * 1e3)
    return times[min(2, len(times) // 3):]


# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="Benchmark computational cost of SR models")
    p.add_argument("--method", required=True, choices=list(FRAMEWORK))
    p.add_argument("--scale", type=int, default=2, choices=[2, 4])
    p.add_argument("--weights", default=None, help="Checkpoint (.pth/.pt), FALSR .pb, or .keras file.")
    p.add_argument("--variant", choices=["a", "b"], default="a", help="EMNAS: ours_a or ours_b.")
    p.add_argument("--n_resgroups", type=int, default=None, help="EMNAS: same value used during fine-tuning.")
    p.add_argument("--name", default=None, help="Model label in the table.")
    p.add_argument("--hr_sizes", nargs="+", default=["720x1280"],
                   help="HR OUTPUT sizes as HxW (default: 720x1280); LR = HR/scale.")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=100)
    p.add_argument("--lr_dir", default=None, help="Directory of real LR images for measuring end-to-end time.")
    p.add_argument("--n_images", type=int, default=20)
    p.add_argument("--tile", type=int, default=0)
    p.add_argument("--tile_overlap", type=int, default=16)
    p.add_argument("--pixel_range", type=float, default=65535.0)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--cudnn_benchmark", action="store_true", help="PyTorch: torch.backends.cudnn.benchmark=True")
    p.add_argument("--repo_dir", default=str(SCRIPT_DIR), help="Directory containing the method's finetune_*.py files.")
    p.add_argument("--csv", default="benchmark_results.csv")
    a = p.parse_args()
    if a.cpu:
        a.device = "cpu"
    return a


def main():
    args = parse_args()
    sizes = [parse_size(s) for s in args.hr_sizes]
    for h, w in sizes:
        if h % args.scale or w % args.scale:
            raise SystemExit(f"HR {h}x{w} is not divisible by scale {args.scale}.")
    fw = FRAMEWORK[args.method]
    print(f"-> method {args.method} ({fw}) x{args.scale} | warmup {args.warmup} | iters {args.iters}")
    rows = run_torch(args, sizes) if fw == "torch" else run_tf(args, sizes)
    append_csv(args.csv, rows)
    print_table(rows)
    print(f"\n-> {len(rows)} row(s) appended to {args.csv}")


if __name__ == "__main__":
    main()
