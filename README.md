# TinyInfer: lightweight neural network inference engine

A dependency-free **C++17** engine that loads **ONNX**-exported CNNs and runs them on CPU. Convolutions are lowered to **im2col + GEMM**, the GEMM is a hand-written **AVX2/FMA (SIMD)** micro-kernel, work is spread across cores with **OpenMP**, and the engine is exposed to Python through **pybind11**. Every model is checked against **PyTorch** in a **GoogleTest** suite.

```python
import tinyinfer, numpy as np
sess = tinyinfer.Session("resnet18.onnx")
logits = sess.run(np.random.randn(1, 3, 224, 224).astype(np.float32))[0]
```

## What's inside

| piece | file | notes |
|---|---|---|
| ONNX loader | [`src/onnx_reader.cpp`](src/onnx_reader.cpp) | decodes the protobuf wire format directly (nodes, attributes, initializers, I/O shapes): no protobuf or onnx library is linked |
| Graph runtime | [`src/session.cpp`](src/session.cpp) | validates every op at load, folds `Constant` nodes, runs in topological order, frees each intermediate after its last use, optional per-op profiler |
| Conv | [`src/ops.cpp`](src/ops.cpp) | im2col → GEMM with stride, padding, dilation and groups; 1×1 convs skip the unfold; depthwise convs use a dedicated vectorised kernel |
| GEMM | [`src/gemm.cpp`](src/gemm.cpp) | 4×16 register-blocked AVX2/FMA micro-kernel, cache-blocked (64×256×256), OpenMP over output tiles; scalar fallbacks kept for comparison |
| Python | [`python/bindings.cpp`](python/bindings.cpp) | `Session`, numpy in/out, GIL released during inference, kernel/thread switches |

**Supported ops:** Conv (incl. grouped/depthwise), MaxPool, AveragePool, GlobalAveragePool, Gemm, MatMul, Relu, Clip (ReLU6), Sigmoid, HardSigmoid, HardSwish, BatchNormalization, Add/Sub/Mul/Div with numpy broadcasting, Flatten, Reshape, Softmax, Dropout/Identity and Constant. That covers LeNet, VGG-style nets, ResNets and MobileNetV3, including torchvision's `resnet18` and `mobilenet_v3_small`.

## Correctness

```
$ ctest --test-dir build        # 47 GoogleTest cases
$ pytest -q tests               # 12 pytest cases
```

- **Kernel tests:** the blocked and SIMD GEMMs match the naive GEMM on odd sizes that exercise every edge tile. Results are bit-identical across thread counts. im2col conv matches the direct 7-loop conv across stride, padding, dilation, pointwise, grouped and depthwise cases.
- **PyTorch parity (GoogleTest):** four reference CNNs ([`scripts/models.py`](scripts/models.py)) are exported to ONNX together with PyTorch's output for a fixed input. The C++ engine must match it within 1e-4 under all six conv/GEMM kernel combinations.
- **PyTorch parity (pytest, through pybind11):** the same models plus torchvision **ResNet-18** and **MobileNetV3-Small** at 224×224, with dynamic batch sizes.

## Performance

All runs below used a 2-vCPU Intel Xeon @ 2.10 GHz (AVX2 + FMA).

**What each optimisation buys** on ResNet-18 at 1×3×224×224 (`build/ti_bench`):

| configuration | threads | median ms | speed-up |
|---|---:|---:|---:|
| direct conv (7 nested loops) | 1 | 2343 | 1.0× |
| im2col + naive GEMM | 1 | 1890 | 1.2× |
| im2col + cache-blocked GEMM (scalar) | 1 | 1115 | 2.1× |
| im2col + AVX2/FMA GEMM | 1 | 110 | **21×** |
| im2col + AVX2/FMA GEMM + OpenMP | 2 | 65 | **36×** |

**Against production runtimes**, on the same ONNX file and input with matched thread counts (`benchmarks/bench.py`; full table in [`benchmarks/results.md`](benchmarks/results.md)):

| model | input | threads | TinyInfer | ONNX Runtime | PyTorch eager |
|---|---|---:|---:|---:|---:|
| LeNet-5 | 1×1×28×28 | 1 | 0.07 ms | 0.03 ms | 0.20 ms |
| SmallCNN | 1×3×32×32 | 1 | 0.94 ms | 0.25 ms | 1.15 ms |
| MobileNetV3-Small | 1×3×224×224 | 2 | 7.9 ms | 2.5 ms | 7.7 ms |
| ResNet-18 | 1×3×224×224 | 2 | 69 ms | 15 ms | 29 ms |

TinyInfer beats PyTorch eager on small batch-1 models, where framework overhead dominates. With 2 threads it roughly matches PyTorch on MobileNetV3-Small. ONNX Runtime is still 2–4.5× faster, mainly because of operator fusion (Conv+activation), blocked NCHWc layouts with pre-packed weights, and wider AVX-512 kernels on CPUs that have them. Those are the natural next steps.

**Profiling-driven fix.** Running the [EdgeVision](https://github.com/JayaChandraDadi/EdgeVision) CIFAR-10 MobileNetV3 through `ti_run --profile` showed the classifier `Gemm` taking 6 ms of an 11 ms forward pass. It was transposing the 1024×576 weight matrix on every call. A dot-product path for small-batch `transB` layers cut that op to 0.1 ms. Vectorising the depthwise conv brought the whole model to 4.5 ms (1 thread), with predictions identical to ONNX Runtime on all 10,000 test images.

## Build

Requires CMake ≥ 3.18, a C++17 compiler with OpenMP, and Python with pybind11 for the bindings. GoogleTest is fetched automatically if it isn't installed.

```bash
pip install -r requirements.txt            # pybind11 + (for tests/benchmarks) torch, onnx, onnxruntime
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build                      # C++ tests
python -m pytest -q tests                   # Python tests (module is built into python/tinyinfer/)

./build/ti_run model.onnx --profile         # run once, print outputs + per-op timings
./build/ti_bench model.onnx --threads 4     # kernel ablation table
python scripts/gen_fixtures.py              # regenerate GoogleTest fixtures from PyTorch
python benchmarks/bench.py                  # TinyInfer vs ONNX Runtime vs PyTorch
```

Options: `-DTINYINFER_NATIVE=OFF` builds for generic AVX2 instead of `-march=native`; `-DTINYINFER_BUILD_TESTS=OFF` and `-DTINYINFER_BUILD_PYTHON=OFF` skip those targets.

### Python API

```python
import tinyinfer
tinyinfer.set_num_threads(4)
tinyinfer.set_conv_algo("im2col")        # or "direct"
tinyinfer.set_gemm_kernel("simd")        # or "blocked", "naive"
sess = tinyinfer.Session("model.onnx")
sess.input_names, sess.input_shapes, sess.op_types
out = sess.run(x)                        # or sess.run({"input": x})
sess.enable_profiling(); sess.run(x); sess.profile()   # [{op, calls, total_ms}, ...]
```

## Layout

```
include/tinyinfer/  tensor.h · onnx_reader.h · session.h · ops.h · gemm.h · tensor_io.h
src/                onnx_reader.cpp · session.cpp · ops.cpp · gemm.cpp
python/             bindings.cpp · tinyinfer/__init__.py
tools/              ti_run.cpp · ti_bench.cpp
tests/              GoogleTest suites, pytest suite, fixtures/ (ONNX + PyTorch outputs)
scripts/            reference models + fixture generator
benchmarks/         bench.py + results.md
```

Scope: float32, NCHW, 2-D convolutions, static shapes per run (dynamic batch is fine). Models must be exported with the TorchScript exporter (`dynamo=False`) or otherwise avoid `Shape`/`Gather` shape arithmetic.
