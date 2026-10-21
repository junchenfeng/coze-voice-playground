"""四个模型：简单 CNN / 复杂 CNN / 预训练底座 + 判断头 / MLP（仅 CLI 兼容）。

- **简单 CNN**（`simple`）：一层卷积 + 一层池化，参数约 3k。和复杂 CNN 只差
  卷积块数量，其余（池化、头、增强、优化器）完全一致 —— 这样课上比较的
  「唯一变量」就是卷积层数。
- **复杂 CNN**（`cnn`）：三层卷积 + 两个池化 + 统计/注意力池化，参数约 1.4 万。
- **head**：openWakeWord 冻结底座 + 小判断头（部署包会大 40 倍）。
- **MLP**：拉平后两层全连接，效果最弱，页面已下线，只留给 `trainer.train --model mlp`
  复现早期实验。
"""
import torch
import torch.nn as nn

from .features import N_MELS, T_FRAMES


class StatsAttentionPool(nn.Module):
    """mean + std + attention 三路拼接，替代全局平均（B2）。

    GAP 会把"哪一帧出现了唤醒词"平均掉；std 保留"有没有起伏"，
    attention 保留"哪一帧最像"。32 维瓶颈 → 96 维，参数只多 33 个。
    """

    def __init__(self, c):
        super().__init__()
        self.att = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        mean = x.mean(dim=(2, 3))
        std = x.std(dim=(2, 3), unbiased=False)
        w = torch.softmax(self.att(x).flatten(2), dim=2)      # (B,1,H*W)
        att = (x.flatten(2) * w).sum(dim=2)                   # (B,C)
        return torch.cat([mean, std, att], dim=1)


class MLP(nn.Module):
    """把 40×100 的图拉平成 4000 个数字，直接接两层全连接。"""

    def __init__(self, n_cls=2, hidden=(256, 128), drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(N_MELS * T_FRAMES, hidden[0]), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(hidden[1], n_cls),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    """三层小窗卷积 + 统计池化 + 判断头。"""

    def __init__(self, n_cls=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
        )
        self.pool = StatsAttentionPool(32)
        self.head = nn.Linear(96, n_cls)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class SimpleCNN(nn.Module):
    """一层卷积 + 一层池化（"极简 CNN"）。

    结构刻意与 CNN 同构，只是少了两层卷积：
        Conv2d(1,16,3,p1) → BatchNorm → ReLU → MaxPool(2) → 统计/注意力池化 → Linear(48,2)
    参数约 3k（复杂 CNN 约 14k），感受野只有 3×3，看不到长时结构 —— 正是课上
    要让学生观察到的「简单模型也能认，但边界更脆」。
    """

    def __init__(self, n_cls=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.pool = StatsAttentionPool(16)
        self.head = nn.Linear(48, n_cls)

    def forward(self, x):
        return self.head(self.pool(self.features(x)))


class OWWHead(nn.Module):
    """L3：openWakeWord 底座冻结，只训 192 → 64 → 2 这个小头。

    底座（mel + Google speech embedding）在 trainer/backbone.py 里跑，
    这里拿到的已经是每条音频 192 维的池化向量（mean + std）。
    """

    def __init__(self, dim=192, hidden=64, n_cls=2, drop=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(drop), nn.Linear(dim, hidden), nn.ReLU(),
            nn.Dropout(drop), nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


def build(name):
    if name == "mlp":
        return MLP()
    if name == "simple":
        return SimpleCNN()
    if name == "cnn":
        return CNN()
    if name == "head":
        return OWWHead()
    raise SystemExit(f"未知模型: {name}")


def count_params(model, trainable_only=False):
    ps = model.parameters()
    if trainable_only:
        ps = (p for p in ps if p.requires_grad)
    return sum(p.numel() for p in ps)
