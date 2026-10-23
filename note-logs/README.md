# note-logs/：研究笔记日志（只读）

本站不写这里。日志由「AI 学习伙伴」在每轮实验后生成，仓库只提供读取接口；
页面本身读的是 `data/research-notes.json`（人工整理过的版本）。

## 接口契约

```
GET /note-log/1     200  原样返回 note-logs/note-1.json 的内容（application/json, no-store）
GET /note-log/5     400  {"detail": "笔记编号只能是 1 / 2 / 3 / 4"}
GET /note-log/2     404  {"detail": "这份日志还没生成"}     —— 文件不存在
GET /note-log/3     500  {"detail": "日志不是合法 JSON"}    —— 文件存在但坏了
```

- 不做任何二次加工：接口返回的就是文件原文（字节级一致），方便比对与回放；
- 文件名固定 `note-{1..4}.json`，白名单校验 + 目录 resolve 校验，不接受任何路径片段；
- 写完不必重启服务（每次请求都读盘）。

## 文件结构（约定，暂未生成）

```json
{
  "id": 3,
  "title": "简单 CNN 与复杂 CNN",
  "generated_at": "2026-10-21T10:00:00+08:00",
  "model": "ai-learning-buddy",
  "entries": [
    {"at": "2026-10-21T09:12:00+08:00", "kind": "observation", "text": "…"},
    {"at": "2026-10-21T09:30:00+08:00", "kind": "next", "text": "…"}
  ]
}
```

字段只约束到「能被 `json.load` 解析」；`entries` 的细节留给写入方演进，读取端不校验。
