# data/ 里到底放什么

原则：**git 里没有一句声音**。音频要么在 OSS（预置语料 / 合成语料），要么是使用者
自己录的（只在这台机器上）。

```
data/
├── seed-manifest.json    # 预置语料 + 预训练底座的清单（key / sha256 / 大小 / 时长）—— 进 git
├── research-notes.json   # 研究笔记（本轮是 4 条空壳）—— 进 git
├── corpus/               # 麦克风录音：pos/ neg/（唯一的本地音频）—— 不进 git
├── words/<wid>/          # 自定义唤醒词：word.json + pos/*.wav —— 不进 git
├── runs/<run_id>/        # 训练产物：model.pt / result.json / progress.jsonl / train.log —— 不进 git
├── synth.json            # 合成音频的元数据索引（wav 在 .cache/synth）—— 不进 git
└── active.json           # 当前应用的模型指针 —— 不进 git
```

三个概念别混：

| 东西 | 在哪 | 谁产生 | 删了会怎样 |
| --- | --- | --- | --- |
| 预置语料（builtin / 历史人工语料） | OSS → `.cache/seed/` | `scripts/oss_push.py` 上传 | `python scripts/fetch_seed.py` 重下 |
| 合成语料 | `.cache/synth/` + OSS `synth/` | 页面上的「合成语料」 | 重新合成即可 |
| 麦克风录音 | `data/corpus/` 与 `data/words/` | 使用者自己 | **没了就是没了**，要重录 |

清单口径（`seed-manifest.json`）：

```json
{
  "schema": 1,
  "prefix": "classroom/voice-playground",
  "files": [
    {"path": "seed/builtin/pos/builtin@pos-synth-001.wav",
     "root": "builtin", "label": "pos", "name": "builtin@pos-synth-001.wav",
     "key": "classroom/voice-playground/seed/builtin/pos/builtin@pos-synth-001.wav",
     "sha256": "…", "size": 12345, "duration": 1.02, "sample_rate": 16000, "synth": true}
  ],
  "models": [
    {"path": "models/backbone/openwakeword/melspectrogram.onnx",
     "key": "classroom/voice-playground/models/backbone/openwakeword/melspectrogram.onnx",
     "sha256": "…", "size": 1234}
  ],
  "deduped": [{"path": "seed/online-20261020/pos/x.wav", "same_as": "seed/corpus/pos/x.wav"}]
}
```

`path` 是**项目内相对路径**：语料落到 `.cache/<path>`，模型落到项目根的 `<path>`。
