"""pytest: the pybind11 module against PyTorch on the reference models and on
torchvision ResNet-18 / MobileNetV3-Small."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

import tinyinfer  # noqa: E402
from models import MODELS, build, export  # noqa: E402


def _check(model, shape, tmp_path, atol, dynamic_batch=False):
    path = tmp_path / "m.onnx"
    export(model, shape, path, dynamic_batch=dynamic_batch)
    sess = tinyinfer.Session(str(path))
    torch.manual_seed(7)
    x = torch.randn(*shape)
    with torch.no_grad():
        ref = model(x).numpy()
    out = sess.run(x.numpy())[0]
    assert out.shape == ref.shape
    np.testing.assert_allclose(out, ref, atol=atol, rtol=1e-4)
    return sess


@pytest.mark.parametrize("name", list(MODELS))
def test_reference_models(name, tmp_path):
    _, shape = MODELS[name]
    _check(build(name), shape, tmp_path, atol=1e-4)


@pytest.mark.parametrize("arch", ["resnet18", "mobilenet_v3_small"])
def test_torchvision_models(arch, tmp_path):
    torchvision = pytest.importorskip("torchvision")
    torch.manual_seed(0)
    model = getattr(torchvision.models, arch)(weights=None).eval()
    _check(model, (1, 3, 224, 224), tmp_path, atol=2e-3)


def test_dynamic_batch(tmp_path):
    model = build("small_cnn")
    path = tmp_path / "dyn.onnx"
    export(model, (1, 3, 32, 32), path, dynamic_batch=True)
    sess = tinyinfer.Session(str(path))
    for n in (1, 3, 8):
        x = torch.randn(n, 3, 32, 32)
        with torch.no_grad():
            np.testing.assert_allclose(sess.run(x.numpy())[0], model(x).numpy(), atol=1e-4)


@pytest.mark.parametrize("kernel", ["naive", "blocked", "simd"])
def test_gemm_kernels_match_numpy(kernel):
    rng = np.random.default_rng(0)
    a = rng.standard_normal((37, 129), dtype=np.float32)
    b = rng.standard_normal((129, 301), dtype=np.float32)
    tinyinfer.set_gemm_kernel(kernel)
    try:
        np.testing.assert_allclose(tinyinfer.gemm(a, b), a @ b, atol=1e-3)
    finally:
        tinyinfer.set_gemm_kernel("simd")


def test_profile_and_metadata(tmp_path):
    sess = _check(build("mini_resnet"), MODELS["mini_resnet"][1], tmp_path, atol=1e-4)
    assert sess.input_names == ["input"]
    assert "Conv" in sess.op_types
    sess.enable_profiling(True)
    sess.run(np.zeros((2, 3, 32, 32), dtype=np.float32))
    prof = sess.profile()
    assert prof[0]["total_ms"] >= prof[-1]["total_ms"]


def test_unsupported_input_shape_raises(tmp_path):
    path = tmp_path / "l.onnx"
    export(build("lenet5"), (1, 1, 28, 28), path)
    sess = tinyinfer.Session(str(path))
    with pytest.raises(RuntimeError):
        sess.run(np.zeros((1, 3, 28, 28), dtype=np.float32))
