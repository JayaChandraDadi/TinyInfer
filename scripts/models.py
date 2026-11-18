"""Reference PyTorch CNNs used to exercise TinyInfer.

Each model targets a different slice of the op set so that a failure points at
a specific kernel.
"""

import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """Classic conv -> pool -> conv -> pool -> 3 dense layers (MNIST-sized)."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
            nn.Linear(120, 84), nn.ReLU(), nn.Linear(84, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class SmallCNN(nn.Module):
    """VGG-style CIFAR net: padding, stride-2 conv, ceil-mode max pool, avg pool, dropout."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=2, dilation=2), nn.ReLU(),
            nn.AvgPool2d(2),
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(64 * 4 * 4, 128), nn.ReLU(), nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)


class BasicBlock(nn.Module):
    def __init__(self, c_in, c_out, stride):
        super().__init__()
        self.c1 = nn.Conv2d(c_in, c_out, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(c_out)
        self.c2 = nn.Conv2d(c_out, c_out, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(c_out)
        self.short = nn.Identity() if stride == 1 and c_in == c_out else nn.Sequential(
            nn.Conv2d(c_in, c_out, 1, stride, bias=False), nn.BatchNorm2d(c_out))

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        return torch.relu(self.b2(self.c2(y)) + self.short(x))


class MiniResNet(nn.Module):
    """Residual adds, 1x1 projection shortcuts, BatchNorm, global average pool."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU())
        self.layers = nn.Sequential(BasicBlock(16, 16, 1), BasicBlock(16, 32, 2), BasicBlock(32, 64, 2))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        return self.fc(torch.flatten(self.pool(self.layers(self.stem(x))), 1))


class DepthwiseSE(nn.Module):
    """MobileNet-style ops: depthwise/grouped conv, ReLU6 (Clip), HardSwish, SE (Sigmoid * Mul)."""

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU6())
        self.dw = nn.Conv2d(16, 16, 5, 1, 2, groups=16)
        self.grouped = nn.Conv2d(16, 32, 3, 1, 1, groups=4)
        self.se_fc1 = nn.Conv2d(32, 8, 1)
        self.se_fc2 = nn.Conv2d(8, 32, 1)
        self.head = nn.Linear(32, 10)

    def forward(self, x):
        x = self.stem(x)
        x = nn.functional.hardswish(self.dw(x))
        x = self.grouped(x)
        s = torch.sigmoid(self.se_fc2(torch.relu(self.se_fc1(nn.functional.adaptive_avg_pool2d(x, 1)))))
        x = x * s
        x = nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        return torch.softmax(self.head(x), dim=1)


MODELS = {
    "lenet5": (LeNet5, (1, 1, 28, 28)),
    "small_cnn": (SmallCNN, (2, 3, 32, 32)),
    "mini_resnet": (MiniResNet, (2, 3, 32, 32)),
    "depthwise_se": (DepthwiseSE, (1, 3, 64, 64)),
}


def build(name: str, seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    cls, _ = MODELS[name]
    model = cls()
    # Non-trivial BatchNorm statistics so the BN path is actually tested.
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.uniform_(-0.5, 0.5)
            m.running_var.uniform_(0.5, 2.0)
            m.weight.data.uniform_(0.5, 1.5)
            m.bias.data.uniform_(-0.2, 0.2)
    return model.eval()


def export(model: nn.Module, shape, path, opset: int = 17, dynamic_batch: bool = False):
    import warnings

    x = torch.randn(*shape)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            model, (x,), str(path), dynamo=False, opset_version=opset,
            input_names=["input"], output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None,
        )
