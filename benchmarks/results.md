CPU: Intel(R) Xeon(R) Processor @ 2.10GHz, 2 threads max, AVX2+FMA: True  
onnxruntime 1.29.0, torch 2.15.0+cu132; median of 30 runs after warm-up.

| model | input | threads | TinyInfer ms | ONNX Runtime ms | PyTorch ms | max abs diff vs PyTorch |
|---|---|---:|---:|---:|---:|---:|
| LeNet-5 | 1x1x28x28 | 1 | 0.07 | 0.03 | 0.20 | 3.0e-08 |
| LeNet-5 | 1x1x28x28 | 2 | 0.09 | 0.04 | 0.21 | 3.0e-08 |
| SmallCNN | 1x3x32x32 | 1 | 0.94 | 0.25 | 1.15 | 7.5e-09 |
| SmallCNN | 1x3x32x32 | 2 | 0.91 | 0.23 | 0.88 | 7.5e-09 |
| MiniResNet | 8x3x32x32 | 1 | 5.70 | 1.88 | 4.22 | 6.7e-08 |
| MiniResNet | 8x3x32x32 | 2 | 6.25 | 1.03 | 3.09 | 6.7e-08 |
| MobileNetV3-Small | 1x3x224x224 | 1 | 10.55 | 2.53 | 7.85 | 8.0e-16 |
| MobileNetV3-Small | 1x3x224x224 | 2 | 7.86 | 2.48 | 7.72 | 8.0e-16 |
| ResNet-18 | 1x3x224x224 | 1 | 116.33 | 28.42 | 54.34 | 1.9e-06 |
| ResNet-18 | 1x3x224x224 | 2 | 68.89 | 15.27 | 29.22 | 1.9e-06 |
