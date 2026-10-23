# Nova Voice Lab · 扣子编程版

教机器听懂一句话。录几条自己的声音，训一个很小很小的模型，然后喊一声看它认不认。

从 `nova-voice-playground`（线上 voice.aibeaver.cn）迁移而来，变化：

- **没有账户 / 工作区**：每台扣子编程机器就是一套独立的数据域，打开即用；
- **音频分层**：预置语料与合成音频在 OSS，仓库只留清单；只有你自己录的声音放在本地；
- **多了一种模型**：简单 CNN（一层卷积 + 一层池化）与复杂 CNN（三层）并排对比；
- **研究笔记**：`/notes` 四份笔记 + 只读的 `/note-log/{1..4}`；
- **一键初始化**：`bootstrap.md` + `scripts/bootstrap.sh` 负责环境、底座、语料。

## 快速开始

```bash
bash scripts/bootstrap.sh                              # 装依赖 + 拉底座 + 拉语料
.venv/bin/python scripts/fetch_projenv.py --token ...  # 领 .env（合成语料/OSS 需要）
bash scripts/dev.sh                                    # http://localhost:5000
```

没有 `.env` 也能用：录音、训练、测试全可用，只有「合成语料」与「预置语料下载」会不可用。

## 三个房间怎么走

1. **录入**：对着麦克风喊 4 次唤醒词（正例），再说 4 句普通的话（负例）；也可以点「合成语料」
   用豆包 TTS 造一批。
2. **教它**：训练集选纯合成、少量真人或多人均衡（均为 50 正 / 150 负）；
   验证集选纯合成、标准真人或标准＋本机录音。后者加入本机录入的全部正负例，
   录音从训练集中排除，loss 和准确率按正负类别等权统计。选模型卡、调 epoch，
   看曲线逐轮长出来。
3. **喊一声试试**：浏览器实时录音，每 300ms 打一次 `/api/predict`，看它认不认。
4. **新唤醒词**（可选）：换个词，正例只收这个词的录音，负例自动用全部语料。
5. **我的模型**：应用 / 删除 / 重训 / 导出部署包（`buddy-wake-<run_id>.zip`，约 60 KB）。

## 目录与约束

见 `AGENTS.md`（给 AI/新同事）与 `data/README.md`（音频到底放哪）。

要点：仓库里**没有一句声音**——预置语料与底座在 OSS，按 `data/seed-manifest.json`
下载到 `.cache/`；固定训练方案还使用清单中的 `dataset` 归档根。
只有麦克风录音（`data/corpus/`、`data/words/`）是本地资产。

## 命令行

```bash
python scripts/oss_push.py --source ~/git/nova-voice-playground          # dry-run 看一眼
python scripts/oss_push.py --source ~/git/nova-voice-playground --apply  # 上传 + 写清单
python scripts/fetch_seed.py --check                                     # 体检本地缓存
python scripts/fetch_seed.py --all                                       # 连归档根一起下
bash scripts/validate.sh                                                 # pytest
```

OSS 凭据只从 `.env` 或 `~/git/ai-gallery/config.json` 的 `oss` 块读，不复制、不入库。

## 测试

```bash
.venv/bin/python -m pytest tests -q
```

没装 torch 时训练/推理相关用例自动 skip；接口、存储、清单、笔记用例不依赖 torch。
