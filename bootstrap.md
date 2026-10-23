# Bootstrap：把这台扣子编程机器准备好

给 AI / 新同事的执行清单。按顺序做，每步都有验收命令；失败按「排障」处理，别跳过。

## 0. 先读三份文件

- `AGENTS.md` —— 项目结构与约定（哪些文件能改、哪些是生成物）
- `data/README.md` —— 音频分层：什么在 OSS、什么在本地、清单长什么样
- `note-logs/README.md` —— 研究笔记日志的接口契约

## 1. 检查运行时

```bash
python3 --version    # 需要 ≥ 3.10（.coze 声明 python-3.12）
node --version       # 仅 dev.sh 的 detached 启动用；没有 node 就前台跑 uvicorn
```

两条都没有时：`.coze` 里已经声明 `requires = ["python-3.12", "nodejs-24"]`，在扣子编程
环境里重建一次运行时会自动装好；不要在项目里塞二进制安装包。

## 2. 装依赖（含 torch CPU）

```bash
bash scripts/bootstrap.sh
```

它等价于依次执行：

1. `scripts/prepare.sh` —— 建 `.venv`（在 `/Coze/Drive` 下会落到 `/tmp` 再 symlink）、
   装 `requirements.txt`（阿里云镜像）、再装 `requirements-torch.txt`（官方 CPU 源，约 200 MB）；
2. `scripts/fetch_seed.py --models-only` —— 拉 2 个 ONNX 底座到 `models/backbone/openwakeword/`；
3. 提示怎么拿 `.env`；
4. `scripts/fetch_seed.py` —— 拉种子语料 `builtin` + `corpus` 到 `.cache/seed/`；
5. 自检打印 data 根 / 缓存根 / 语料条数 / 底座状态 / OSS 凭证状态。

验收：

```bash
.venv/bin/python -c "import torch, numpy, onnxruntime, fastapi, markdown; print('deps ok')"
```

## 3. 领 `.env`

```bash
.venv/bin/python scripts/fetch_projenv.py --token <PROJENV_TOKEN>
```

- 接口是线上 `GET https://voice.aibeaver.cn/projenv`，**和桌面端的 `/credential` 不是一个东西**：
  这里一次给全 ASR + TTS + OSS 四类键；
- 写盘后权限 0600，旧文件备份成 `.env.bak`；
- token 找老师要。**没有 .env 也能用**：录音/训练/测试照常，只有合成语料和预置语料下载不可用。

验收：

```bash
test -f .env && ls -l .env          # 权限应为 -rw-------
.venv/bin/python scripts/fetch_projenv.py --token <...> --check
```

## 4. 起服务

```bash
bash scripts/dev.sh
curl -s localhost:5000/healthz      # {"ok":true,...}
curl -s localhost:5000/notes >/dev/null && echo notes ok
```

- `dev.sh` 会把 uvicorn detached 到独立进程组（coze-daemon 会清理 runtime shell 的原进程组），
  日志在 `logs/server.log`，PID 在 `logs/server.pid`；
- 生产用 `bash scripts/start.sh`（前台、不开 reload）。

## 5. 跑验收

```bash
bash scripts/validate.sh            # pytest
```

期望全绿；没装 torch 时 torch 相关用例自动 skip（`NOVA_SKIP_TORCH=1` 同理）。

## 6. 不该提交的东西

`git status` 里出现这些就说明 `.gitignore` 被绕过了，先修再继续：

```
.env  .cache/  .venv/  data/corpus/  data/runs/  data/words/
data/synth.json  models/backbone/  *.wav
```

## 排障

| 现象 | 原因 / 处理 |
| --- | --- |
| `prepare.sh` 报「找不到 Python ≥ 3.10」 | 扣子编程环境缺 python 运行时；确认 `.coze` 的 requires 生效后重新 build |
| torch 装不上 | 换源重试：`pip install -r requirements-torch.txt -i https://mirrors.aliyun.com/pypi/simple/ --extra-index-url https://download.pytorch.org/whl/cpu`；仍失败就先用 CNN 以外功能（训练会报错） |
| `fetch_seed.py` 全部 404 | `.env` 里 `OSS_PREFIX` 与清单不一致，或 OSS 上的对象还没上传（在有钥匙的机器上 `python scripts/oss_push.py --apply`） |
| `/notes` 500 | `data/research-notes.json` 不是合法 JSON（注意中文引号别写成英文双引号） |
| 训练报 `预训练底座文件缺失` | 只影响 head 模型；跑 `scripts/fetch_seed.py --models-only` |
| 训练进程立刻失败 | 看 `data/runs/<id>/train.log` 尾部；多半是 torch 没装或语料条数不足 |
| 端口被占 | `dev.sh` 会自己清端口；生产用 `DEPLOY_RUN_PORT=5001 bash scripts/start.sh` |
