# 三个模型，同一场考试

## 摘要

同一份语料（350 条录音切出的 1350 个 1 秒窗口）、同一张考卷（随机留 12% 做验证）、同样的 30 轮训练，我们在本地把三张模型卡各训了一遍：简单 CNN、复杂 CNN、预训练底座 + 小判断头。把成绩单摆在一起，能看到两条很清楚的规律：**参数越多，训练越慢，成绩越好**；而预训练底座训练飞快、成绩也不差，秘密是——**它的绝大部分参数根本没参与训练**。

## 一、实验怎么做的

- 语料：唤醒词正例 50 条 + 背景负例 300 条，共 350 条录音；
- 每条录音按 1 秒滑窗切样，一共得到 1350 个窗口，随机留 162 个（12%）当考卷，其余 1188 个用来训练；
- 三个模型完全同卷：30 轮（epoch）、batch 16、学习率 1e-3、随机种子 42，谁也不许开小灶；
- 评分口径也一样：训练准确率看训练集，验证准确率只看没训过的考卷——它才是模型真正的成绩。

## 二、成绩单

| 模型 | 总参数 | 实际训练的参数 | 每 epoch 平均时长 | 最终训练准确率 | 最终验证准确率 | 最好验证准确率 |
| --- | --- | --- | --- | --- | --- | --- |
| 简单 CNN | 307 | 307（全部） | 0.74 秒 | 71.4% | 70.4% | 74.1% |
| 复杂 CNN | 14,435 | 14,435（全部） | 1.69 秒 | 90.3% | 84.6% | 85.8% |
| 预训练底座 + 小头 | 约 61.5 万 | 12,482（仅约 2%） | 0.07 秒 | 81.7% | 78.4% | 84.6% |

两个 CNN 的「总参数」和「训练的参数」一模一样——它们是白纸，每个参数都从头学起。预训练底座不一样：openWakeWord 底座约 60 万参数（2.4 MB）全部冻结，真正在动的只有 192 → 64 → 2 的那个 1.2 万参数小判断头。

## 三、发现一：参数越大，越慢，也越准

从简单 CNN 到复杂 CNN：参数量 ×47（307 → 14,435），每 epoch 平均时长 ×2.3（0.74 秒 → 1.69 秒），验证准确率 +14 分（70.4% → 84.6%）。

为什么会这样？简单 CNN 只有一层 3×3 卷积，一次只能看到很小的局部，听不出「先这个音、再那个音」的先后顺序；复杂 CNN 三层卷积叠起来，感受野更大，能抓住音节之间的长时结构——代价是每一步要做的乘加多了几十倍，训练自然变慢。

参数量、训练时长、模型能力，这三件事是绑在一起的：想要更好的成绩，就得付更多时间。

## 四、发现二：预训练底座为什么又快又不差

<div class="note-hand">
<p>把三行数据连起来看：</p>
<p>· 复杂 CNN：14,435 个参数全部在学，每轮 1.69 秒；</p>
<p>· 底座 + 小头：约 61.5 万个参数里只有 1.25 万（约 2%）在学，每轮只要 0.07 秒，是复杂 CNN 的 1/24；</p>
<p>· 30 轮下来总共 3.8 秒，复杂 CNN 要 52.3 秒，差了约 14 倍；</p>
<p>· 但它的最好验证准确率有 84.6%，和复杂 CNN（85.8%）几乎打平，远超简单 CNN（74.1%）。</p>
<p>为什么快？因为它不是从零开始学「什么是人声」。openWakeWord 底座在海量语音上提前练过，本站把它整个冻住（不训练、不更新），只训练最后那个小判断头——训练的参数少了，每一步自然就快。</p>
<p>为什么成绩不差？因为底座「见多识广」，声音的通用规律它早就学会了，小头只需要专心学一件事：这个声音像不像我们的唤醒词。</p>
</div>

## 给 Agent 的指令

- 数据来自 2026-10-23 本地 CLI 训练（与页面训练完全同口径：`--data data/corpus`（空）+ `--builtin-dir .cache/seed/builtin` + `--extra-dir .cache/seed/corpus`，epochs 30 / batch 16 / lr 1e-3 / seed 42 / split-by clip / threads 2 / max-pos 50 / max-neg 300）。
- `trainer/train.py` 已新增逐 epoch 计时：`history[].epoch_s` 与 `result.avg_epoch_s`；更新本笔记数字时重跑附录命令，替换「成绩单」表格即可。
- 底座参数量按 `models/backbone/openwakeword/` 两个 onnx（melspectrogram 1.09 MB + embedding 1.33 MB）按 float32 估算，约 60 万；正文表述「约 61.5 万」为底座 + 小头合计。

## 附录

复现命令（在仓库根目录）：

```bash
for m in simple cnn head; do
  .venv/bin/python -m trainer.train \
    --data data/corpus --model $m --epochs 30 --batch 16 --lr 1e-3 \
    --val-ratio 0.12 --seed 42 --split-by clip \
    --max-pos 50 --max-neg 300 --threads 2 \
    --out /tmp/note-train/$m.json \
    --progress-jsonl /tmp/note-train/$m.progress.jsonl \
    --split-json /tmp/note-train/$m.split.json \
    --builtin-dir .cache/seed/builtin --extra-dir .cache/seed/corpus
done
```

result.json 关键字段：`params_total` / `params_trainable`（参数口径）、`avg_epoch_s`（每 epoch 平均时长）、`final`（最终 train/val accuracy）、`history`（逐 epoch 曲线）。本次三个 run 的完整产物在 `/tmp/note-train/`。
