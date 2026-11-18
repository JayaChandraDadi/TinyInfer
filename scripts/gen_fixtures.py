"""Export the reference models to ONNX and record PyTorch outputs for GoogleTest.

Writes tests/fixtures/<name>/{model.onnx, input.bin, expected.bin}.
Run from the repo root:  python scripts/gen_fixtures.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from models import MODELS, build, export  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def write_tensor(path: Path, arr: np.ndarray):
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    with path.open("wb") as f:
        f.write(np.int32(arr.ndim).tobytes())
        f.write(np.asarray(arr.shape, dtype=np.int64).tobytes())
        f.write(arr.tobytes())


def main():
    out_root = ROOT / "tests" / "fixtures"
    for name, (_, shape) in MODELS.items():
        d = out_root / name
        d.mkdir(parents=True, exist_ok=True)
        model = build(name)
        export(model, shape, d / "model.onnx")
        torch.manual_seed(123)
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x).numpy()
        write_tensor(d / "input.bin", x.numpy())
        write_tensor(d / "expected.bin", y)
        print(f"{name:14s} input {tuple(shape)} -> output {y.shape}")


if __name__ == "__main__":
    main()
