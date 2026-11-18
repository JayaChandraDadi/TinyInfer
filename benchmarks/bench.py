"""Latency comparison: TinyInfer vs ONNX Runtime vs PyTorch eager, on CPU.

    python benchmarks/bench.py [--iters 30] [--out benchmarks/results.md]

Every engine runs the same ONNX file (PyTorch runs the source module) on the
same input, after warm-up, with matched thread counts. Outputs are checked
against PyTorch before timing.
"""

import argparse
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import tinyinfer  # noqa: E402
from models import build, export  # noqa: E402


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def models():
    import torchvision

    torch.manual_seed(0)
    yield "LeNet-5", build("lenet5"), (1, 1, 28, 28)
    yield "SmallCNN", build("small_cnn"), (1, 3, 32, 32)
    yield "MiniResNet", build("mini_resnet"), (8, 3, 32, 32)
    yield "MobileNetV3-Small", torchvision.models.mobilenet_v3_small(weights=None).eval(), (1, 3, 224, 224)
    yield "ResNet-18", torchvision.models.resnet18(weights=None).eval(), (1, 3, 224, 224)


def main():
    import onnxruntime as ort

    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results.md"))
    args = ap.parse_args()

    work = ROOT / "benchmarks" / "models"
    work.mkdir(exist_ok=True)
    rows = []
    for name, model, shape in models():
        path = work / f"{name.lower().replace('-', '_')}.onnx"
        export(model, shape, path)
        x = torch.randn(*shape)
        xn = x.numpy()
        with torch.no_grad():
            ref = model(x).numpy()
        for t in args.threads:
            tinyinfer.set_num_threads(t)
            torch.set_num_threads(t)
            ti = tinyinfer.Session(str(path))
            so = ort.SessionOptions()
            so.intra_op_num_threads = t
            so.inter_op_num_threads = 1
            rt = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            err = float(np.abs(ti.run(xn)[0] - ref).max())
            t_ti = timeit(lambda: ti.run(xn), args.iters)
            t_ort = timeit(lambda: rt.run(None, {"input": xn}), args.iters)
            with torch.no_grad():
                t_pt = timeit(lambda: model(x), args.iters)
            rows.append((name, shape, t, t_ti, t_ort, t_pt, err))
            print(f"{name:18s} {str(shape):18s} threads={t}  tinyinfer {t_ti:8.2f} ms  ort {t_ort:8.2f} ms  "
                  f"torch {t_pt:8.2f} ms  max|diff| {err:.1e}", flush=True)

    cpu = platform.machine()
    try:
        cpu = next(l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name"))
    except (OSError, StopIteration):
        pass
    lines = [
        f"CPU: {cpu}, {torch.get_num_threads()} threads max, AVX2+FMA: {tinyinfer.simd_available()}  ",
        f"onnxruntime {ort.__version__}, torch {torch.__version__}; median of {args.iters} runs after warm-up.",
        "",
        "| model | input | threads | TinyInfer ms | ONNX Runtime ms | PyTorch ms | max abs diff vs PyTorch |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, shape, t, a, b, c, e in rows:
        lines.append(f"| {name} | {'x'.join(map(str, shape))} | {t} | {a:.2f} | {b:.2f} | {c:.2f} | {e:.1e} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
