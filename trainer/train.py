"""训练入口。

    python -m trainer.train --data data/corpus --model simple --out runs/simple.json
    python -m trainer.train --data data/corpus --model cnn    --out runs/cnn.json
    python -m trainer.train --data data/corpus --model head   --out runs/head.json
    python -m trainer.train --data data/corpus --model mlp    --out runs/mlp.json   # 仅兼容旧实验

所有模型共用一套样本口径：1 s 窗口、按与唤醒词的重叠给软标签（见 data.make_windows）。
head 的特征不是梅尔谱，而是 openWakeWord 预训练底座的 192 维池化向量。

数据来源（可叠加）：
    --data        用户自己用麦克风录的语料（data/corpus）
    --builtin-dir 预置语料（.cache/seed/builtin，OSS 清单驱动的缓存）
    --extra-dir   额外根，可重复（.cache/seed/corpus 这类历史语料）
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from .data import (HUMAN_TRAIN_NEG, HUMAN_TRAIN_POS, MAX_NEG_CLIPS, MAX_POS_CLIPS,
                   load_dataset, make_split)
from .models import build, count_params
from .showcase import PRESETS, VAL_PRESETS, load_or_create_plan, select_sets


def load_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(sd, dict) or "state_dict" in sd:
        return sd.get("state_dict", sd)
    return sd


def augment(x):
    """时间平移 ±10 帧 + 频率平移 ±2 格：对抗"录音早晚一秒"和音色差异。

    平移后补零（不是 torch.roll 的环绕）——环绕会把 7.6 kHz 卷到 20 Hz，
    造出物理上不存在的频谱。
    """
    dx = int(torch.randint(-10, 11, (1,)).item())
    if dx:
        x = torch.roll(x, shifts=dx, dims=3)
        if dx > 0:
            x[..., :dx] = 0
        else:
            x[..., dx:] = 0
    dy = int(torch.randint(-2, 3, (1,)).item())
    if dy:
        x = torch.roll(x, shifts=dy, dims=2)
        if dy > 0:
            x[..., :dy, :] = 0
        else:
            x[..., dy:, :] = 0
    return x


def soft_ce(logits, target, weight=None):
    """软标签交叉熵：target=0.5 表示"只有半个唤醒词"。"""
    logp = torch.log_softmax(logits, dim=1)
    loss = -(target * logp[:, 1] + (1 - target) * logp[:, 0])
    return (loss * weight).mean() if weight is not None else loss.mean()


def _weight(tb):
    """正例只占少数（负样本池更大），按 4 倍加权；软标签 0.5 的窗口算子类。"""
    return torch.where(tb > 0.5, 4.0, 1.0)


def evaluate(model, loader, balanced=False):
    model.eval()
    loss, correct, n = 0.0, 0, 0
    by_class = {0: [0.0, 0, 0], 1: [0.0, 0, 0]}
    with torch.no_grad():
        for xb, yb, tb in loader:
            out = model(xb)
            loss += soft_ce(out, tb, _weight(tb)).item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            n += len(yb)
            if balanced:
                for lab in (0, 1):
                    mask = yb == lab
                    count = int(mask.sum())
                    if count:
                        by_class[lab][0] += soft_ce(out[mask], tb[mask],
                                                     _weight(tb[mask])).item() * count
                        by_class[lab][1] += int((out[mask].argmax(1) == yb[mask]).sum())
                        by_class[lab][2] += count
    if balanced:
        if not all(by_class[lab][2] for lab in (0, 1)):
            raise ValueError("均衡验证需要正负两类录音")
        return (sum(by_class[lab][0] / by_class[lab][2] for lab in (0, 1)) / 2,
                sum(by_class[lab][1] / by_class[lab][2] for lab in (0, 1)) / 2)
    return loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="simple", choices=["mlp", "simple", "cnn", "head"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-ratio", type=float, default=0.12)
    ap.add_argument("--split-by", default="clip", choices=["clip", "speaker", "human"],
                    help="clip=随机抽条；speaker=留出几个人当考卷；"
                         "human=训练时留出的真人录音当考卷（某类无真人时回落 clip）")
    ap.add_argument("--seed", type=int, default=42,
                    help="切考卷用的种子：同一个 seed + 同一份语料 = 同一张卷")
    # 训练装载上限（页面上「训练数据集大小」两个滑杆）：上限是总条数，真人先占配额
    ap.add_argument("--max-pos", type=int, default=MAX_POS_CLIPS,
                    help=f"正例最多多少条（默认 {MAX_POS_CLIPS}）")
    ap.add_argument("--max-neg", type=int, default=MAX_NEG_CLIPS,
                    help=f"负例最多多少条（默认 {MAX_NEG_CLIPS}）")
    # 真人录音的分配：每类抽出这么多条进训练集，其余固定当考卷（默认 30 / 50）。
    # 传 0 = 老口径（真人全部只做考卷）。
    ap.add_argument("--human-train-pos", type=int, default=HUMAN_TRAIN_POS,
                    help=f"真人正例进训练集的条数（默认 {HUMAN_TRAIN_POS}）")
    ap.add_argument("--human-train-neg", type=int, default=HUMAN_TRAIN_NEG,
                    help=f"真人负例进训练集的条数（默认 {HUMAN_TRAIN_NEG}）")
    # 初始化/打乱顺序用的种子。默认跟 --seed 一致（CLI 行为不变）；
    # web 的"再学一遍"会给一个新的 init-seed：考卷不动，但模型从头换个起点，
    # 否则同数据同 seed 训两遍结果一模一样，看起来像"没重新训练"。
    ap.add_argument("--init-seed", type=int, default=None)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-ckpt", default=None)
    ap.add_argument("--threads", type=int, default=2)
    # 以下两个参数是给 web 层用的，缺省值保持原有 CLI 行为完全不变：
    #   --progress-jsonl  逐 epoch 追加一行 JSON（web 用 SSE 尾随它画实时曲线）
    #   --split-json      每个 run 一张独立考卷，避免复用 <data>/split.json 让
    #                     val_ratio / seed 静默失效
    ap.add_argument("--progress-jsonl", default=None)
    ap.add_argument("--split-json", default=None)
    # 内置合成语料：默认取 --data 同级的 builtin/（web 层就是 data/builtin），
    # 学生可以在页面上取消勾选；--no-builtin 优先级最高。
    ap.add_argument("--builtin-dir", default=None,
                    help="预置语料目录（pos/neg），默认 <data 的父目录>/builtin")
    ap.add_argument("--no-builtin", action="store_true", help="不使用预置语料")
    # 额外语料根（可重复）：OSS 上下载到 .cache/seed/corpus 的历史人工语料走这里。
    # 命名口径与 --builtin-dir 相同：用目录 basename 做前缀，不跟主池重名。
    ap.add_argument("--extra-dir", action="append", default=None,
                    help="额外语料根（pos/neg），可重复；文件名带 <basename>/ 前缀")
    # 自定义唤醒词：正例只从 --pos-dir 来，--data / --builtin-dir 下的 pos 全部改标
    # 负例（老词的正例是新词的难负例）。缺省 None 时口径与以前逐字节一致。
    ap.add_argument("--pos-dir", default=None,
                    help="自定义唤醒词目录（读 <pos-dir>/pos/*.wav 作正例）")
    ap.add_argument("--wake-word", default=None,
                    help="唤醒词原文，只写进 result.json 便于溯源，不影响训练")
    ap.add_argument("--dataset-preset", choices=PRESETS, default=None)
    ap.add_argument("--val-preset", choices=VAL_PRESETS, default="standard")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    init_seed = args.seed if args.init_seed is None else args.init_seed
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)

    t0 = time.time()
    builtin_dir = None
    if not args.no_builtin:
        builtin_dir = args.builtin_dir or os.path.join(
            os.path.dirname(os.path.abspath(args.data)), "builtin")
        if not os.path.isdir(builtin_dir):
            builtin_dir = None
    # 额外根顺序固定为：预置语料在前、其余（历史人工语料）在后。顺序会进入
    # files 的前缀，进而影响 split.json 的指纹 —— 不要随手调换。
    extra_roots = [builtin_dir] if builtin_dir else []
    extra_roots += [d for d in (args.extra_dir or []) if d and os.path.isdir(d)]
    stats: dict = {}
    X, y, target, files, groups, meta = load_dataset(
        args.data, extra_roots, model=args.model,
        pos_dir=args.pos_dir,
        max_pos=100000 if args.dataset_preset else args.max_pos,
        max_neg=100000 if args.dataset_preset else args.max_neg, stats=stats,
        human_train=(100000, 100000) if args.dataset_preset else
                    (args.human_train_pos, args.human_train_neg))
    builtin_prefix = (os.path.basename(os.path.normpath(builtin_dir)) + "/") if builtin_dir else ""

    split_path = args.split_json or os.path.join(args.data, f"split-{args.split_by}.json")
    os.makedirs(os.path.dirname(split_path) or ".", exist_ok=True)
    human_groups = stats.get("human_groups") or []
    holdout = stats.get("val_groups") or []      # 装载时切好的真人考卷（固定不变）
    if args.dataset_preset:
        manifest = os.path.join(args.data, "showcase-dataset.json")
        plan = load_or_create_plan(groups, y, manifest)
        train_groups, val_groups, synth_groups = select_sets(
            plan, groups, y, args.dataset_preset, args.val_preset)
        selected_groups = set(train_groups) | set(val_groups) | set(synth_groups)
        keep = np.array([g in selected_groups for g in groups], dtype=bool)
        X, y, target = X[keep], y[keep], target[keep]
        files = [f for f, selected_row in zip(files, keep) if selected_row]
        groups = [g for g, selected_row in zip(groups, keep) if selected_row]
        val_set = set(val_groups)
        val_idx = np.flatnonzero(np.array([g in val_set for g in groups]))
        synth_idx = (np.flatnonzero(np.array([g in set(synth_groups) for g in groups]))
                     if args.val_preset != "synthetic" else np.array([], dtype=int))
        holdout = val_groups
        from .data import is_synth
        label = dict(zip(groups, map(int, y)))
        stats["human_split"] = {
            "train": {lab: sum(label[g] == lab and not is_synth(g)
                               for g in train_groups) for lab in (0, 1)},
            "val": {lab: sum(label[g] == lab for g in val_groups) for lab in (0, 1)},
        }
        stats["pos"] = {"after": 50, "limit": 50, "human": stats["human_split"]["train"][1],
                        "synth": 50 - stats["human_split"]["train"][1]}
        stats["neg"] = {"after": 150, "limit": 150, "human": stats["human_split"]["train"][0],
                        "synth": 150 - stats["human_split"]["train"][0]}
        with open(split_path, "w", encoding="utf-8") as f:
            json.dump({"train_groups": train_groups, "val_groups": val_groups,
                       "synth_val_groups": synth_groups if len(synth_idx) else [],
                       "train": [file for file, group in zip(files, groups)
                                 if group in set(train_groups)],
                       "val": [files[i] for i in val_idx],
                       "val_preset": args.val_preset,
                       "dataset_preset": args.dataset_preset},
                      f, ensure_ascii=False, indent=2)
    else:
        synth_idx = np.array([], dtype=int)
        val_idx = make_split(y, files, meta, args.val_ratio, args.seed,
                             out=split_path, split_by=args.split_by, groups=groups,
                             human_groups=human_groups, val_groups=holdout)
    if args.split_by == "human":
        hs = stats.get("human_split") or {"train": {}, "val": {}}
        tr, va = hs["train"], hs["val"]
        print(f"[data] split: 考卷 {len(holdout)} 条真人（正 {va.get(1, 0)} / "
              f"负 {va.get(0, 0)}），训练集里的真人 正 {tr.get(1, 0)} / "
              f"负 {tr.get(0, 0)}"
              + ("" if holdout else "（没有真人考卷，回落随机 12%）"))

    progress_fp = None
    if args.progress_jsonl:
        os.makedirs(os.path.dirname(args.progress_jsonl) or ".", exist_ok=True)
        progress_fp = open(args.progress_jsonl, "a", encoding="utf-8")
    val_mask = np.zeros(len(y), dtype=bool)
    val_mask[val_idx] = True
    synth_mask = np.zeros(len(y), dtype=bool)
    synth_mask[synth_idx] = True

    if args.model == "head":
        xt = torch.from_numpy(X)                       # (N,192)
    else:
        xt = torch.from_numpy(X).unsqueeze(1)          # (N,1,40,100)
    yt = torch.from_numpy(y)
    tt = torch.from_numpy(target)
    tr_ds = TensorDataset(xt[~(val_mask | synth_mask)], yt[~(val_mask | synth_mask)],
                          tt[~(val_mask | synth_mask)])
    va_ds = TensorDataset(xt[val_mask], yt[val_mask], tt[val_mask])
    synth_ds = TensorDataset(xt[synth_mask], yt[synth_mask], tt[synth_mask])

    # CNN 的 BatchNorm 遇到 batch=1 会直接报错：训练集除以 batch 余 1 时丢掉最后一条。
    # 简单 CNN 同样带 BatchNorm，规则一致。
    train_n = len(tr_ds)
    drop_last = (args.model in ("simple", "cnn") and args.batch > 1
                 and train_n % args.batch == 1)
    tr_ld = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, drop_last=drop_last)
    va_ld = DataLoader(va_ds, batch_size=64, shuffle=False)
    synth_ld = DataLoader(synth_ds, batch_size=64, shuffle=False) if len(synth_idx) else None

    model = build(args.model)
    params_total = count_params(model)
    params_trainable = count_params(model, trainable_only=True)
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

    history = []
    best_state, best_epoch, best_val_loss = None, None, float("inf")
    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            ep_t0 = time.time()
            loss_sum, correct, n = 0.0, 0, 0
            for xb, yb, tb in tr_ld:
                if not args.no_aug:
                    if args.model == "head":
                        xb = xb + torch.randn_like(xb) * 0.02
                    else:
                        xb = augment(xb)
                opt.zero_grad()
                out = model(xb)
                loss = soft_ce(out, tb, _weight(tb))
                loss.backward()
                opt.step()
                loss_sum += loss.item() * len(yb)
                correct += (out.argmax(1) == yb).sum().item()
                n += len(yb)

            tr_loss, tr_acc = loss_sum / n, correct / n
            va_loss, va_acc = evaluate(model, va_ld, balanced=bool(args.dataset_preset))
            synth_loss, synth_acc = (evaluate(model, synth_ld, balanced=True)
                                     if synth_ld is not None else (None, None))
            epoch_s = round(time.time() - ep_t0, 2)
            row = {"epoch": epoch, "epoch_s": epoch_s, "train_loss": round(tr_loss, 4),
                            "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                            "val_acc": round(va_acc, 4)}
            if synth_ld is not None:
                row.update(synth_val_loss=round(synth_loss, 4),
                           synth_val_acc=round(synth_acc, 4))
            history.append(row)
            if args.dataset_preset and va_loss < best_val_loss:
                best_state, best_epoch, best_val_loss = copy.deepcopy(model.state_dict()), epoch, va_loss
            print(f"epoch {epoch:3d}  loss {tr_loss:.4f}  acc {tr_acc:.3f}  "
                  f"val_loss {va_loss:.4f}  val_acc {va_acc:.3f}", flush=True)
            if progress_fp:
                # 每 epoch 落一行，web 的 SSE 尾随这个文件画实时曲线
                progress_fp.write(json.dumps(history[-1], ensure_ascii=False) + "\n")
                progress_fp.flush()
    finally:
        if progress_fp:
            progress_fp.close()

    if best_state is not None:
        model.load_state_dict(best_state)
    chosen = history[(best_epoch or len(history)) - 1]

    result = {
        "model": args.model,
        "split_by": args.split_by,
        "dataset_preset": args.dataset_preset,
        "val_preset": args.val_preset if args.dataset_preset else None,
        "best_epoch": best_epoch,
        "params_total": params_total,
        "params_trainable": params_trainable,
        "n_total": int(len(y)),
        "n_clips": int(len(set(groups))),
        "n_train": int((~(val_mask | synth_mask)).sum()),
        "n_val": int(val_mask.sum()),
        "n_synth_val": int(synth_mask.sum()),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "n_pos_windows": int((target >= 0.9).sum()),
        "n_partial_windows": int(((target > 0) & (target < 0.9)).sum()),
        "builtin_dir": builtin_dir,
        "extra_dirs": extra_roots[1:] if builtin_dir else extra_roots,
        "pos_dir": args.pos_dir,
        "wake_word": args.wake_word,
        # 装载口径：正/负各最多多少条、抽样前后多少条、真人是否存在
        "sampled": {key: stats.get(key) for key in ("pos", "neg")},
        "human_split": stats.get("human_split"),
        "val_source": (args.val_preset if args.dataset_preset else
                       "human" if args.split_by == "human" and holdout
                       else args.split_by),
        "n_builtin": sum(1 for f in files if builtin_prefix and f.startswith(builtin_prefix)),
        "elapsed_s": round(time.time() - t0, 2),
        "avg_epoch_s": round(sum(h["epoch_s"] for h in history) / max(1, len(history)), 2),
        "final": {"train_acc": chosen["train_acc"],
                  "val_acc": chosen["val_acc"],
                  # 训练过程中的主指标是 loss：页面实时卡与曲线都读这两个字段
                  "train_loss": chosen["train_loss"],
                  "val_loss": chosen["val_loss"],
                  **({"synth_val_loss": chosen["synth_val_loss"],
                      "synth_val_acc": chosen["synth_val_acc"]}
                     if synth_ld is not None else {}),
                  "gap": round(chosen["train_acc"] - chosen["val_acc"], 4)},
        "history": history,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)

    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_ckpt)

    print(json.dumps({k: result[k] for k in
                      ("model", "params_total", "params_trainable", "n_clips", "n_total",
                       "n_train", "n_val", "elapsed_s", "final")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
