"""Three fixed-size teaching datasets sharing human and synthetic exams.

The manifest freezes clip identities across runs. The exam is balanced by
*windows*: 10 positive recordings x 9 windows = 30 negative x 3 windows.
"""
import itertools
import json
import os
import glob
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data import is_synth

PRESETS = ("synthetic", "few_humans", "balanced_humans")
VAL_PRESETS = ("synthetic", "standard", "standard_plus_recordings")
TRAIN_POS = 50
TRAIN_NEG = 150
VAL_POS = 10
VAL_NEG = 30
SYNTH_VAL_POS = 9
SYNTH_VAL_NEG = 27


def _owner(name):
    base = os.path.basename(name)
    if "@" in base:
        return base.split("@", 1)[0]
    match = re.match(r"(s\d+)_", base)
    return match.group(1) if match else base


def _round_robin(by_owner, count, seed):
    rng = np.random.default_rng(seed)
    slots = {o: sorted(v) for o, v in by_owner.items()}
    for clips in slots.values():
        rng.shuffle(clips)
    out = []
    while len(out) < count and any(slots.values()):
        for o in sorted(slots):
            if slots[o] and len(out) < count:
                out.append(slots[o].pop())
    if len(out) < count:
        raise ValueError(f"需要 {count} 条真人录音，现有 {len(out)} 条")
    return out


def build_plan(groups, labels):
    by_owner = defaultdict(lambda: {0: [], 1: []})
    synth = {0: [], 1: []}
    group_label = dict(zip(groups, map(int, labels)))
    for g, lab in sorted(group_label.items()):
        # The local machine's microphone clips are personal exam material,
        # never part of the frozen seed-based standard exam.
        if g.startswith(("pos/", "neg/")) and not is_synth(g):
            continue
        (synth[lab] if is_synth(g) else by_owner[_owner(g)][lab]).append(g)
    corpus_synth = {lab: sorted(g for g in synth[lab] if g.startswith("corpus/"))
                    for lab in (0, 1)}
    builtin_synth = {lab: sorted(g for g in synth[lab] if g.startswith("builtin/"))
                     for lab in (0, 1)}
    if (len(corpus_synth[1]) < SYNTH_VAL_POS or
            len(corpus_synth[0]) < SYNTH_VAL_POS or
            len(builtin_synth[1]) < TRAIN_POS or
            len(builtin_synth[0]) < TRAIN_NEG + SYNTH_VAL_NEG - SYNTH_VAL_POS):
        raise ValueError("双考卷演示需要语料合成正/负各 9 条、内置合成正例 50 条和负例 168 条")
    owners = sorted(o for o, rows in by_owner.items() if rows[0] and rows[1])
    if len(owners) < 6:
        raise ValueError("三种演示数据集需要至少 6 个各有正负录音的真人账号")
    first = tuple(sorted(np.random.default_rng(43).choice(owners, 4, replace=False).tolist()))
    candidates = [first] + [c for c in itertools.combinations(owners, 4) if c != first]
    held = None
    for candidate in candidates:
        remain = [o for o in owners if o not in candidate]
        if (sum(len(by_owner[o][1]) for o in candidate) >= VAL_POS
                and sum(len(by_owner[o][0]) for o in candidate) >= VAL_NEG
                and sum(len(by_owner[o][1]) for o in remain) >= 30
                and sum(len(by_owner[o][0]) for o in remain) >= 50):
            held = candidate
            break
    if held is None:
        raise ValueError("无法留出均衡真人考卷，同时凑齐 30 正 / 50 负真人训练录音")
    remain = [o for o in owners if o not in held]
    pairs = [(a, b) for a, b in itertools.combinations(remain, 2)
             if len(by_owner[a][1]) + len(by_owner[b][1]) >= 10
             and len(by_owner[a][0]) + len(by_owner[b][0]) >= 15]
    if not pairs:
        raise ValueError("缺少两个合计有 10 正 / 15 负录音的真人账号")
    pair = min(pairs, key=lambda p: (sum(len(by_owner[o][1]) + len(by_owner[o][0])
                                        for o in p), p))
    val = (_round_robin({o: by_owner[o][1] for o in held}, VAL_POS, 1701)
           + _round_robin({o: by_owner[o][0] for o in held}, VAL_NEG, 1702))
    few = (_round_robin({o: by_owner[o][1] for o in pair}, 10, 1801)
           + _round_robin({o: by_owner[o][0] for o in pair}, 15, 1800))
    balanced = (_round_robin({o: by_owner[o][1] for o in remain}, 30, 1901)
                + _round_robin({o: by_owner[o][0] for o in remain}, 50, 1900))
    rng = np.random.default_rng(20261023)
    rng.shuffle(builtin_synth[0])
    rng.shuffle(builtin_synth[1])
    val_synthetic = (corpus_synth[1][:SYNTH_VAL_POS]
                     + corpus_synth[0][:SYNTH_VAL_POS]
                     + builtin_synth[0][:SYNTH_VAL_NEG - SYNTH_VAL_POS])
    synth_pos = builtin_synth[1][:TRAIN_POS]
    synth_neg = builtin_synth[0][SYNTH_VAL_NEG - SYNTH_VAL_POS:
                                  SYNTH_VAL_NEG - SYNTH_VAL_POS + TRAIN_NEG]
    train = {
        "synthetic": synth_pos + synth_neg,
        "few_humans": few + synth_pos[:40] + synth_neg[:135],
        "balanced_humans": balanced + synth_pos[:20] + synth_neg[:100],
    }
    return {"version": 2, "held_owners": list(held), "few_owners": list(pair),
            "val": sorted(val), "val_synthetic": sorted(val_synthetic),
            "train": {k: sorted(v) for k, v in train.items()}}


def load_or_create_plan(groups, labels, path):
    path = Path(path)
    if path.exists():
        plan = json.loads(path.read_text(encoding="utf-8"))
    if not path.exists() or plan.get("version") != 2:
        plan = build_plan(groups, labels)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, path)
    available = set(groups)
    used = set(plan["val"]) | set(plan["val_synthetic"]) | set().union(
        *map(set, plan["train"].values()))
    missing = used - available
    if missing:
        raise ValueError(f"固定演示数据集有 {len(missing)} 条录音已缺失；需重新生成考卷清单")
    val = set(plan["val"])
    val_synthetic = set(plan["val_synthetic"])
    if val & val_synthetic or len(val_synthetic) != SYNTH_VAL_POS + SYNTH_VAL_NEG:
        raise ValueError("演示数据集清单损坏：合成考卷")
    for preset in PRESETS:
        train = plan["train"][preset]
        if (len(train) != TRAIN_POS + TRAIN_NEG or len(set(train)) != len(train)
                or (val | val_synthetic).intersection(train)):
            raise ValueError(f"演示数据集清单损坏：{preset}")
    return plan


def select_sets(plan, groups, labels, train_preset, val_preset, owner=None):
    """Choose a fixed training set and one exam, without clip overlap.

    The personal exam includes all non-synthetic recordings entered under the
    current slug. If any of those were in the fixed train set, refill from the
    remaining pool with the same label, preferring human recordings.
    """
    if train_preset not in PRESETS or val_preset not in VAL_PRESETS:
        raise ValueError("未知的训练集或验证集")
    label = dict(zip(groups, map(int, labels)))
    standard = set(plan["val"])
    synth = set(plan["val_synthetic"])
    own = {g for g in label if g.startswith(("pos/", "neg/")) and not is_synth(g)}
    if val_preset == "standard_plus_recordings":
        if not any(label[g] == 1 for g in own) or not any(label[g] == 0 for g in own):
            raise ValueError("标准＋录音验证集需要该昵称至少一条正例和一条负例录音")
        val = standard | own
    elif val_preset == "standard":
        val = standard
    else:
        val = synth
    train = set(plan["train"][train_preset]) - val
    if val_preset == "standard_plus_recordings":
        # Leave this speaker entirely unseen by the model, including clips
        # that were previously assigned to the fixed training manifest.
        train -= own
    reserved = standard | synth | own if val_preset == "standard_plus_recordings" else standard | synth
    for lab, total in ((1, TRAIN_POS), (0, TRAIN_NEG)):
        need = total - sum(label[g] == lab for g in train)
        pool = [g for g in label if label[g] == lab and g not in train | reserved]
        pool.sort(key=lambda g: (is_synth(g), g))
        if len(pool) < need:
            raise ValueError(f"排除本昵称录音后无法补足训练集类别 {lab}，缺 {need - len(pool)} 条")
        train.update(pool[:need])
    if len(train) != TRAIN_POS + TRAIN_NEG or train & val:
        raise ValueError("训练集与验证集清单发生重叠")
    return sorted(train), sorted(val), sorted(synth)


def prepare_plan(root, builtin, corpus_seed=None, dataset_seed=None,
                 pos_dir=None, train_preset=None, val_preset="standard"):
    """Preflight the exact audio files before accepting a website training job."""
    from .data import has_speech, read_wav

    groups, labels = [], []
    sources = [(root, "", sub, 0 if pos_dir else int(sub == "pos"))
               for sub in ("pos", "neg")]
    if builtin:
        prefix = os.path.basename(os.path.normpath(builtin)) + "/"
        sources += [(builtin, prefix, sub, 0 if pos_dir else int(sub == "pos"))
                    for sub in ("pos", "neg")]
    for extra in (corpus_seed, dataset_seed):
        if extra:
            prefix = os.path.basename(os.path.normpath(extra)) + "/"
            sources += [(extra, prefix, sub, 0 if pos_dir else int(sub == "pos"))
                        for sub in ("pos", "neg")]
    if pos_dir:
        sources.append((pos_dir, "word/", "pos", 1))
    for base, prefix, sub, lab in sources:
        for path in sorted(glob.glob(os.path.join(str(base), sub, "*.wav"))):
            if has_speech(read_wav(path)):
                groups.append(prefix + os.path.relpath(path, base))
                labels.append(lab)
    manifest = os.path.join(root, "showcase-dataset.json")
    plan = load_or_create_plan(groups, labels, manifest)
    if train_preset:
        select_sets(plan, groups, labels, train_preset, val_preset)
    return plan
