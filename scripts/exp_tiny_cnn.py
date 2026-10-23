"""U4 L2 备课用：最简网络（1 层卷积 + 1 层池化）vs 成熟 CNN（3 层）。

    python scripts/exp_tiny_cnn.py --seeds 42,7,2026

口径（与 L1 课堂一致，且可复现）：
    训练集 = 内置合成语料 350 条（纯合成，课前造好，固定不变）
    考卷   = 线上语料池里的全部真人录音（19 正 / 20 负，随时会变）
    指标   = 正样本唤醒率 + AUC（AUC 不受正负比例影响，乱猜 = 0.5）

唯一变量是卷积层数：池化方式、判断头、增强、优化器、epoch 全部相同。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.data import load_dataset                                  # noqa: E402
# 这个脚本是最早的"极简 CNN"对照实验；结构已经沉淀成 trainer.models.SimpleCNN
# （页面上那张「简单 CNN」卡）。这里只保留别名，别再复制一份结构 —— 两份实现
# 迟早会漂移，课上"唯一变量是卷积层数"的说法就不成立了。
from trainer.models import CNN, SimpleCNN, count_params                 # noqa: E402
from trainer.train import augment, soft_ce                              # noqa: E402

TinyCNN = SimpleCNN

CORPUS = "data/corpus"
# 预置语料本体在 OSS（清单驱动）。本地缓存优先，其次兼容旧布局 data/builtin。
BUILTIN = next((p for p in (".cache/seed/builtin", "data/builtin")
                if os.path.isdir(p)), ".cache/seed/builtin")


def is_synth(group):
    name = group.rsplit("/", 1)[-1]
    return name.startswith("ws-") or "-synth-" in name


def is_builtin(group):
    return group.startswith("builtin/")


def build_model(name):
    if name == "tiny":
        return TinyCNN()
    if name == "cnn":
        return CNN()
    raise SystemExit("未知模型: " + name)


def roc_auc(scores, labels):
    """rank 法算 AUC，不依赖 sklearn。"""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = labels == 1
    neg = ~pos
    n_pos, n_neg = pos.sum(), neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 并列取平均名次
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(model, loader):
    model.eval()
    npos = nneg = hitpos = hitneg = 0
    scores, labels = [], []
    with torch.no_grad():
        for xb, yb, _tb in loader:
            logits = model(xb)
            prob = torch.softmax(logits, dim=1)[:, 1]
            pred = logits.argmax(1)
            scores.extend(prob.tolist())
            labels.extend(yb.tolist())
            for p, t in zip(pred.tolist(), yb.tolist()):
                if t == 1:
                    npos += 1
                    hitpos += int(p == 1)
                else:
                    nneg += 1
                    hitneg += int(p == 0)
    return {
        "pos_recall": round(hitpos / npos, 4) if npos else None,
        "neg_recall": round(hitneg / nneg, 4) if nneg else None,
        "auc": round(roc_auc(scores, labels), 4) if roc_auc(scores, labels) is not None else None,
        "n_pos": npos,
        "n_neg": nneg,
    }


def train_one(X, y, t, tr_idx, va_idx, name, args, seed):
    torch.set_num_threads(args.threads)
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(name)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    xt = torch.from_numpy(X[tr_idx]).unsqueeze(1)
    xv = torch.from_numpy(X[va_idx]).unsqueeze(1)
    tr_ld = DataLoader(TensorDataset(xt, torch.from_numpy(y[tr_idx]), torch.from_numpy(t[tr_idx])),
                       batch_size=args.batch, shuffle=True)
    va_ld = DataLoader(TensorDataset(xv, torch.from_numpy(y[va_idx]), torch.from_numpy(t[va_idx])),
                       batch_size=64, shuffle=False)

    t0 = time.time()
    last = None
    for ep in range(args.epochs):
        model.train()
        tot = n = 0
        for xb, _yb, tb in tr_ld:
            if not args.no_aug:
                xb = augment(xb)
            opt.zero_grad()
            loss = soft_ce(model(xb), tb, torch.where(tb > 0.5, 4.0, 1.0))
            loss.backward()
            opt.step()
            tot += float(loss) * len(xb)
            n += len(xb)
        last = round(tot / n, 4)
    elapsed = round(time.time() - t0, 1)

    return {
        "model": name, "seed": seed,
        "params_trainable": count_params(model, trainable_only=True),
        "elapsed_s": elapsed, "final_loss": last,
        "train": evaluate(model, tr_ld),
        "val": evaluate(model, va_ld),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tiny,cnn")
    ap.add_argument("--seeds", default="42,7,2026")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--out", default="runs/exp-tiny-cnn.json")
    args = ap.parse_args()

    X, y, t, files, groups, meta = load_dataset(
        CORPUS, [BUILTIN], model="cnn", max_pos=100000, max_neg=100000)

    g_builtin = np.array([is_builtin(g) for g in groups])
    g_synth = np.array([is_synth(g) for g in groups])
    tr_idx = np.where(g_builtin)[0]
    va_idx = np.where((~g_synth) & (~g_builtin))[0]
    print("窗口 %d | 训练(内置合成) %d 条 / %d 窗口 | 考卷(真人) %d 条 / %d 窗口"
          % (len(groups),
             len({groups[i] for i in tr_idx}), len(tr_idx),
             len({groups[i] for i in va_idx}), len(va_idx)))

    out = {"created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
           "train_source": "builtin(350 synth)", "val_source": "corpus human",
           "n_windows": int(len(groups)),
           "n_clips_train": len({groups[i] for i in tr_idx}),
           "n_clips_val": len({groups[i] for i in va_idx}),
           "runs": []}

    for name in args.models.split(","):
        for seed in [int(s) for s in args.seeds.split(",")]:
            r = train_one(X, y, t, tr_idx, va_idx, name, args, seed)
            out["runs"].append(r)
            print("[%-4s s=%d] %5.1fs loss %.3f 参数 %-6d | 训练 %s auc %s | 考卷 %s auc %s | 考卷负 %s"
                  % (name, seed, r["elapsed_s"], r["final_loss"], r["params_trainable"],
                     r["train"]["pos_recall"], r["train"]["auc"],
                     r["val"]["pos_recall"], r["val"]["auc"], r["val"]["neg_recall"]))

    # 汇总：均值
    summary = {}
    for name in args.models.split(","):
        rs = [r for r in out["runs"] if r["model"] == name]
        summary[name] = {
            "n": len(rs),
            "params": rs[0]["params_trainable"],
            "train_recall_mean": round(float(np.mean([r["train"]["pos_recall"] for r in rs])), 4),
            "train_auc_mean": round(float(np.mean([r["train"]["auc"] for r in rs])), 4),
            "val_recall_mean": round(float(np.mean([r["val"]["pos_recall"] for r in rs])), 4),
            "val_auc_mean": round(float(np.mean([r["val"]["auc"] for r in rs])), 4),
            "val_neg_mean": round(float(np.mean([r["val"]["neg_recall"] for r in rs])), 4),
            "elapsed_mean": round(float(np.mean([r["elapsed_s"] for r in rs])), 1),
        }
    out["summary"] = summary
    print("\n=== 汇总（%d 个种子）===" % len(args.seeds.split(",")))
    for k, v in summary.items():
        print("%-5s 参数 %-6d 训练 %s (auc %s) | 考卷 %s (auc %s) | 负 %s | %ss"
              % (k, v["params"], v["train_recall_mean"], v["train_auc_mean"],
                 v["val_recall_mean"], v["val_auc_mean"], v["val_neg_mean"], v["elapsed_mean"]))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("写入 " + args.out)


if __name__ == "__main__":
    main()
