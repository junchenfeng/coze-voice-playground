# AGENTS.md

一台「教机器听懂自己声音」的语音训练台：录几条自己的声音 → 训一个小模型 → 喊一声
看它认不认 → 换一个唤醒词重训。每个扣子编程项目都是独立的一台机器，**没有账户、
没有工作区**：全站一份语料、一份唤醒词、一份训练记录。

## 技术栈

- Python 3.12 + FastAPI + Uvicorn + Jinja2（原生 JS/CSS，没有前端构建）
- PyTorch 2.6 CPU（训练，子进程）· onnxruntime（openWakeWord 底座）· numpy（导出包纯 numpy 前向）
- 存储：目录即状态，没有数据库；音频在本地或 OSS，仓库里只有清单

## 目录

```
web/            FastAPI 应用
  app.py        页面 + JSON API + SSE（唯一路由入口）
  store.py      单数据域存储：语料 / runs / words / active 指针
  seed.py       预置语料与底座的清单驱动缓存（.cache/seed）
  oss.py        OSS 最小客户端（V1 签名，纯标准库）
  jobs.py       训练子进程 + progress.jsonl 尾随 + SSE
  synth.py      豆包 TTS 合成语料（本地缓存 + 尽力上传 OSS）
  export.py     部署包（np_cnn.py + model.npz + meta.json，含阈值标定）
  notes.py      研究笔记读取 / 渲染 / note-log
  words.py      自定义唤醒词数据集
  predictor.py  active 模型推理（按 mtime 缓存）
  audio.py      WAV 解码 + 音量门控
  settings.py   全部路径与上限（环境变量覆盖）
  templates/    base / studio / notes
  static/       app.js / charts.js / recorder.js / style.css
trainer/        训练与特征
  models.py     SimpleCNN（1 层卷积）/ CNN（3 层）/ OWWHead / MLP（仅 CLI）
  np_cnn.py     纯 numpy 前向，原样进部署包（不许 import torch）
  data.py       1 s 窗口 + 软标签 + 分组切分 + 特征缓存（.cache/features，按内容 sha256）
  train.py      训练入口（--model simple|cnn|head|mlp）
  backbone.py   openWakeWord mel + embedding（onnx）
scripts/        oss_push / fetch_seed / fetch_projenv / prepare / dev / build / start / validate / bootstrap
data/           只放元数据与使用者自己录的音频（见 data/README.md）
note-logs/      GET /note-log/{1..4} 的原始日志（AI 学习伙伴写，本站只读）
reports/        研究笔记的 markdown 源（发布流程见其 README）
```

## 硬约束

1. **不提交：** `.env`、`.cache/`、`.venv/`、`data/corpus/`、`data/runs/`、`data/words/`、
   `data/synth.json`、`models/backbone/`、任何 `*.wav`。
2. **`data/` 不存预置/合成音频。** 预置语料在 OSS（清单 `data/seed-manifest.json`），
   合成音频在 `.cache/synth` + OSS `synth/`；`data/` 只留元数据和麦克风录音。
3. **`trainer/np_cnn.py` 只能依赖 numpy**：它会被原样打进部署包，由桌宠从包目录 import。
4. **密钥永不回显**：日志、错误、接口响应里都不能出现 `OSS_ACCESS_KEY_SECRET` 或 token。
5. **文件名即状态**，所以所有外部名字都要过 `store._safe_child`（白名单正则 + resolve
   前缀校验 + 拒绝 symlink）。
6. **训练是子进程**（`python -m trainer.train`），不要改成线程里跑 torch。
7. 单训练并发默认 1（`NOVA_MAX_RUNNING_TRAININGS`）；改大要先确认机器核数。

## 三张模型卡

| 键 | 页面名 | 结构 | 约参数 | 能否导出 |
| --- | --- | --- | --- | --- |
| `simple` | 简单 CNN | 1×Conv+BN+ReLU+MaxPool → 统计/注意力池化 → Linear(48,2) | 3k | 能 |
| `cnn` | 复杂 CNN | 3×Conv（2 次池化）→ 同一套池化 → Linear(96,2) | 14k | 能 |
| `head` | 预训练底座 | openWakeWord 冻结 mel+embedding → 小判断头 | 底座 2.4 MB | 可以（小包携带 OSS 底座清单） |

`mlp` 仍留在 `trainer.train --model` 里复现早期实验，但页面不暴露。

## 可写根（FaaS 部署形态）

veFaaS 把应用目录挂成**只读**（如 `/opt/bytefaas`），在 BASE_DIR 下 mkdir 直接
`Errno 30`。`web/settings.py` 用「可写根」统一处理，优先级：

1. 显式 `NOVA_WRITABLE_ROOT`；
2. BASE_DIR 可写（开发/自托管）→ 一切照旧；
3. 否则 → `/tmp/nova-runtime-<BASE_DIR指纹>/`。

只读形态下：`DATA_ROOT`/`CACHE_ROOT` 及其派生（corpus/runs/words/active/synth、
seed/synth/特征缓存）全部落在可写根；随代码打包的只读资源
（`seed-manifest.json`、`research-notes.json`）仍从 `BASE_DIR/data` 读。
运行期数据随容器重启丢失（种子/缓存会自动重建，麦克风录音不持久）。
训练子进程由 `jobs.py` 注入 `NOVA_FEATURE_CACHE` 与 `NOVA_OWW_DIR`
（trainer 侧变量名与 web 侧不同，见 `trainer/data.py`/`trainer/backbone.py`）。
给运行时新增任何落盘路径时，必须挂在 `settings.WRITABLE_ROOT` 派生下。

## 接口速查

页面：`/`（→ 303 `/studio`）、`/studio`、`/notes`、`/healthz`

语料：`GET /api/corpus`、`POST /api/corpus/{pos|neg}`、`GET|DELETE /api/corpus/{label}/{name}`、
`POST /api/corpus/synth`、`GET /api/voices`

唤醒词：`GET|POST /api/words`、`GET /api/words/{wid}`、`POST /api/words/{wid}/clips`、
`GET|DELETE /api/words/{wid}/clips/{name}`、`POST /api/words/{wid}/synth[/auto]`、
`GET /api/words/{wid}/synth/progress`

训练：`POST|GET /api/runs`、`GET /api/runs/{id}`、`GET /api/runs/{id}/stream`（SSE）、
`POST /api/runs/{id}/apply|stop|rerun`、`DELETE /api/runs/{id}`、`GET /api/runs/{id}/export`

推理：`POST /api/predict`（上传 16k wav，返回 `p_wake`）

只读外部接口：`GET /note-log/{1|2|3|4}`（契约见 `note-logs/README.md`）

## 数据从哪里来

- **预置语料**：`scripts/oss_push.py` 上传 → `data/seed-manifest.json` 记录 sha256/时长 →
  `scripts/fetch_seed.py` 下载到 `.cache/seed/`（训练前 `jobs._prefetch()` 再兜一次）；
- **合成语料**：豆包 TTS，wav 写 `.cache/synth/`，元数据写 `data/synth.json`，尽力上传 OSS；
- **麦克风录音**：浏览器转 16k/16bit/单声道 wav，`data/corpus/{pos,neg}` 或 `data/words/<wid>/pos`。

训练时三层一起用：`--data data/corpus`、`--builtin-dir .cache/seed/builtin`、
`--extra-dir .cache/seed/corpus`。

## 开发与验收

```bash
bash scripts/prepare.sh          # 首次
bash scripts/dev.sh              # 5000 端口，日志 logs/server.log
bash scripts/validate.sh         # pytest
.venv/bin/python -m pytest tests -q -k export
```

**沙箱装依赖注意**：本沙箱会掐 Python socket 的大文件下载（TLS 握手正常、请求发出后
收不到数据，pip 会卡死），curl 与 Rust 系工具不受影响。若 `prepare.sh` 卡在
pip install，改用 uv 直装（已装在 /root/.local/bin）：

```bash
uv pip install --python .venv/bin/python -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
uv pip install --python .venv/bin/python "torch==2.6.0+cpu" \
  --find-links https://mirrors.aliyun.com/pytorch-wheels/cpu/ \
  -i https://mirrors.aliyun.com/pypi/simple/
```

系统缺 python3.12-venv 时先 `apt-get update && apt-get install -y python3.12-venv`。

正因为沙箱掐 Python socket 下载，训练前的种子补下载有总预算
（`NOVA_SEED_PREFETCH_BUDGET_S`，默认 30s，见 `web/settings.py`）：缓存缺失时逐条
下载各等满超时会把 POST /api/runs 拖上几个小时；超预算自动停手、只用本地已有的
（warning 记进 run 的 status）。不要把这个预算改回去掉。

特征缓存（`trainer/data.py`，目录 `.cache/features`，`NOVA_FEATURE_CACHE` 可覆盖）：
按「窗口口径指纹 + 模型 + 软标签 + 文件内容 sha256」存 npz。同一条录音只提一次
特征——head 的 ONNX 逐窗口前向从 ~6.8s 降到 ~0.1s。目录可随时整个删除（退回首次
提取耗时）；改滑窗口径（SR/WIN/offsets）会因指纹变化自动全部失效，不用手动清。

改 `.coze` 里的入口脚本、`web/settings.py` 的路径口径、`trainer/np_cnn.py` 的权重键时，
记得同步 `tests/`（`test_export.py` 会逐位比对 torch 与 numpy）。

## 研究笔记

页面 `/notes` 读 `data/research-notes.json`（当前 1 篇已发布：三模型对比实验）。
正文只渲染 `## 摘要` ~ `## 附录`，并强制剥掉「给 Agent 的指令」。发布流程见
`reports/README.md`（源文件 `reports/note-1.md` 与 JSON 一起换）；原始日志由外部写入
`note-logs/note-{1..4}.json`，本站只读。

**当用户说「引导我完成（声音）研究笔记」时**：不要现场翻仓库找规则，直接打开
`reports/note-1.md` 读「给 Agent 的指令」一节并严格照做——取数优先级（成绩以学生
现场训练为准，Agent 参考值不主动报）、剧透/封存规则、报数宽容（大致对就行）、
倍数选择题（10/50/500 倍）、分站流程、挑战题 4 轮收束、静默附录、发布三步
（md + JSON + 原始记录必写 `note-logs/note-1.json`）全在那儿。
