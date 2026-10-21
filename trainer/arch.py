"""实验用架构：BC-ResNet（B1）。

B2（统计池化）已经进了 `models.CNN`，`head` 也已经换成 openWakeWord 底座，
这里只剩 B1，留着给对照实验复现用：

    python scripts/experiment_arch.py --folds 3 --epochs 40
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models import OWWHead, StatsAttentionPool  # noqa: F401  （实验脚本复用）


class BCBlock(nn.Module):
    """BC-ResNet 的基本块：1×1 → 深度可分离(时间/频率) → 1×1 + 残差。"""

    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.dw_t = nn.Conv2d(c_out, c_out, (3, 1), padding=(1, 0), groups=c_out, bias=False)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.dw_f = nn.Conv2d(c_out, c_out, (1, 3), padding=(0, 1), groups=c_out, bias=False)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(c_out)
        self.down = None
        if stride != 1 or c_in != c_out:
            self.down = nn.Sequential(
                # 25 这种奇数边长：卷积 stride=2 是 ceil，池化必须同样 ceil 才对得上
                nn.AvgPool2d(stride, ceil_mode=True) if stride != 1 else nn.Identity(),
                nn.Conv2d(c_in, c_out, 1, bias=False),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.dw_t(y)))
        y = F.relu(self.bn3(self.dw_f(y)))
        y = self.bn4(self.conv2(y))
        return F.relu(y + (self.down(x) if self.down is not None else x))


class BCResNet(nn.Module):
    """BC-ResNet-tiny：步进 2 的 stem + 4 段残差块。"""

    def __init__(self, n_cls=2, widths=(16, 16, 24, 32, 48), blocks=(1, 2, 2, 2),
                 pool="gap"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(widths[0]), nn.ReLU(),
        )
        stages = []
        for i, n in enumerate(blocks):
            c_in, c_out = widths[i], widths[i + 1]
            stages.append(BCBlock(c_in, c_out, stride=2 if i else 1))
            stages += [BCBlock(c_out, c_out) for _ in range(n - 1)]
        self.stages = nn.Sequential(*stages)
        c = widths[-1]
        if pool == "stats":
            self.pool = StatsAttentionPool(c)
            feat = c * 3
        else:
            self.pool = nn.AdaptiveAvgPool2d(1)
            feat = c
        self.head = nn.Linear(feat, n_cls)

    def forward(self, x):
        x = self.stages(self.stem(x))
        p = self.pool(x)
        return self.head(p.flatten(1) if p.dim() == 4 else p)


class LinearHead(nn.Module):
    """对照用的纯线性头（386 参数，实验里被证明不够）。"""

    def __init__(self, dim=192, n_cls=2, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, n_cls))

    def forward(self, x):
        return self.net(x)


ARCHS = ("mlp", "cnn", "cnn_stats", "bcres", "bcres_stats", "oww", "oww_mlp")


def build_arch(name, in_dim=192):
    from .models import MLP, CNN

    if name == "mlp":
        return MLP()
    if name in ("cnn", "cnn_stats"):          # cnn_stats 已是生产口径
        return CNN()
    if name == "bcres":
        return BCResNet(pool="gap")
    if name == "bcres_stats":
        return BCResNet(pool="stats")
    if name == "oww":
        return LinearHead(in_dim)
    if name == "oww_mlp":
        return OWWHead(in_dim)                # 与生产 head 同一结构
    raise ValueError(name)
