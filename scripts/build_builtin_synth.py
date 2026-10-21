"""生成"内置合成语料"：默认随训练数据一起用，学生可在页面上取消勾选。

和 scripts/build_dataset.py 的区别：
- build_dataset 造的是"模拟全班"（25 人 × 3 正 5 负），用来跑教案表格；
- 这个脚本造的是**通用多样性**一批，和线上真人录音混着训练：
  50 条正例（唤醒词 × 音色 × 语速）+ 300 条负例（普通句 / 近音词 / 纯英文）。

产出目录结构与 pos/ + neg/ 一致，文件名沿用
`builtin@<label>-synth-<时间>-<随机>.wav`（`-synth-` 是训练侧判合成语料的唯一依据）。

    python scripts/build_builtin_synth.py --out .cache/seed/builtin --workers 4

**默认写进 `.cache/seed/builtin`（不是 `data/`）**：本项目约定 data/ 不存预置音频，
种子语料由 OSS 清单驱动。生成完可以直接训练（`--builtin-dir` 就指向这里）；要把这批
发布给别的机器，把它们放到某台机器的 `<repo>/data/builtin/` 下再跑
`python scripts/oss_push.py --source <repo> --roots builtin --apply`。

幂等：同名文件已存在就跳过；meta.json 记录每条的 voice / text / speed。
"""
import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainer.data import write_wav                    # noqa: E402
from trainer.tts import EN_VOICES, ZH_VOICES, tts_pcm  # noqa: E402

WAKE_WORD = "buddy助手"

# 正样本文本变体：只改标点/空格，不改变读出来的词，用来拉开韵律
POS_TEXTS = ["buddy助手", "buddy助手", "buddy助手", "Buddy助手", "buddy 助手"]

NEG_ZH = [
    "今天天气不错", "打开台灯", "我想吃冰淇淋", "帮我找一下书包", "这道题我不会",
    "老师再见", "把音量调小一点", "明天要下雨吗", "给我讲个故事", "我的铅笔不见了",
    "我们一起去操场", "妈妈在做饭", "这个字怎么读", "我想喝水", "现在几点了",
    "今天的作业好多", "我要去洗手间", "请再说一遍", "这个游戏好玩吗", "我喜欢画画",
    "播放音乐", "打开客厅的灯", "明天早上七点叫我", "导航去学校", "这道题怎么写",
    "帮我订一个闹钟", "今天有什么作业", "把声音关掉", "我想听故事", "我们下课了吗",
]

# 近音负样本：最容易被误唤醒的一批，必须包含"半个唤醒词"
NEG_HARD = [
    "buddy", "助手", "小助手", "我的助手", "巴迪", "八弟", "帮弟", "邦迪",
    "buddy 助", "帮帮我", "帮我", "buddy 在吗", "你好助手", "语音助手",
]

NEG_EN = [
    "turn on the light", "what time is it", "i am hungry", "play some music",
    "how are you today", "open the window", "where is my bag", "good morning",
    "what is your name", "let us go outside", "i like this song", "stop the music",
    "hey buddy", "my buddy", "body", "buddy system",
]


def plan(n_pos, n_neg, seed):
    """返回 [(rel_path, text, voice, rid, speed)] 与 meta 行。"""
    rng = random.Random(seed)
    tasks, meta = [], []

    def add(label, text, voice, rid, speed):
        idx = sum(1 for m in meta if m["label"] == label) + 1
        rel = f"{label}/builtin@{label}-synth-{idx:03d}.wav"
        meta.append({"file": rel, "label": label, "speaker": f"synth-{voice}",
                     "voice": voice, "text": text, "speed": speed})
        tasks.append((rel, text, voice, rid, speed))

    # 正样本：15 个中文音色轮转，语速 -40~+40 铺开
    for i in range(n_pos):
        voice, rid = ZH_VOICES[i % len(ZH_VOICES)]
        speed = [-40, -30, -20, -10, 0, 10, 20, 30, 40][i % 9]
        add("pos", POS_TEXTS[i % len(POS_TEXTS)], voice, rid, speed)

    # 负样本配额：普通句 ~65%，近音词 ~20%，纯英文 ~15%
    n_en = max(1, int(n_neg * 0.15))
    n_hard = max(1, int(n_neg * 0.20))
    n_plain = n_neg - n_en - n_hard

    for i in range(n_plain):
        voice, rid = ZH_VOICES[i % len(ZH_VOICES)]
        add("neg", NEG_ZH[i % len(NEG_ZH)], voice, rid,
            [-20, -10, 0, 10, 20][i % 5])

    for i in range(n_hard):
        voice, rid = ZH_VOICES[(i * 3) % len(ZH_VOICES)]
        add("neg", NEG_HARD[i % len(NEG_HARD)], voice, rid,
            [-10, 0, 10][i % 3])

    for i in range(n_en):
        voice, rid = EN_VOICES[i % len(EN_VOICES)]
        add("neg", NEG_EN[i % len(NEG_EN)], voice, rid,
            [-10, 0, 10][i % 3])

    rng.shuffle(tasks)
    return tasks, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=".cache/seed/builtin",
                    help="输出目录（默认写进缓存；data/ 不存预置音频）")
    ap.add_argument("--pos", type=int, default=50)
    ap.add_argument("--neg", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    tasks, meta = plan(args.pos, args.neg, args.seed)
    meta_by_rel = {m["file"]: m for m in meta}
    os.makedirs(args.out, exist_ok=True)

    todo = [(rel, text, voice, rid, speed) for rel, text, voice, rid, speed in tasks
            if args.force or not os.path.exists(os.path.join(args.out, rel))]
    print(f"计划 {len(tasks)} 条，待生成 {len(todo)} 条 → {args.out}")

    def run(task):
        rel, text, voice, rid, speed = task
        pcm = tts_pcm(text, voice, rid, speed)
        write_wav(os.path.join(args.out, rel), pcm)
        return rel

    done, failed = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [(t, pool.submit(run, t)) for t in todo]
        for i, (t, fut) in enumerate(futures, 1):
            try:
                done.append(fut.result())
            except Exception as exc:                    # noqa: BLE001
                failed.append((t[0], str(exc)[:140]))
            if i % 25 == 0:
                print(f"  {i}/{len(futures)}", flush=True)

    # meta 只保留磁盘上真实存在的条目（含之前跑过的）
    keep = []
    for rel in sorted(meta_by_rel):
        if os.path.exists(os.path.join(args.out, rel)):
            keep.append(meta_by_rel[rel])
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(keep, f, ensure_ascii=False, indent=1)

    print(f"本次成功 {len(done)}，失败 {len(failed)}；目录内共 {len(keep)} 条")
    for name, err in failed[:10]:
        print("  失败:", name, err)
    if todo and len(done) < 0.9 * len(todo):
        raise SystemExit("成功率不足 90%，先查 TTS 凭证/限流")


if __name__ == "__main__":
    main()
