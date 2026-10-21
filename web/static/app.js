/* Nova Voice Lab 前端：五个房间（录入 · 教它 · 喊一声试试 · 新唤醒词 · 我的模型）
 *
 * 没有登录、没有工作区：打开就是这台机器自己的训练台，顶部用一句可忽略的
 * 建议条提示下一步。研究笔记在独立的 /notes 页。
 */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const STEPS = ["record", "train", "check", "word", "models"];
  // 模型卡与历史行的显示名：配置里存的是 simple / cnn / head
  const MODEL_LABEL = { simple: "简单 CNN", cnn: "复杂 CNN", head: "预训练底座", mlp: "MLP" };
  const modelLabel = (name) => MODEL_LABEL[String(name || "")] || String(name || "?").toUpperCase();
  // 能导出给桌宠的：两种 CNN（head 要带 2.4 MB 底座，包会大 40 倍）
  const EXPORTABLE = ["simple", "cnn"];
  const STATE_LABEL = {
    idle: "待选择", queued: "排队中", running: "训练中", succeeded: "已完成",
    failed: "失败", stopped: "已暂停", interrupted: "已中断", unknown: "未知",
  };

  /* ------------------------------------------------------------ 基础 */
  const toastEl = $("#toast");
  let toastTimer = null;
  function toast(message, kind = "") {
    if (!toastEl) return;
    toastEl.textContent = message;
    toastEl.className = "toast is-show" + (kind ? ` is-${kind}` : "");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => { toastEl.className = "toast"; }, 3800);
  }

  /* 必须让孩子看见的失败（录音被拒、麦克风没权限）：toast 3.8 s 就消失，
   * 而且会被下一条 toast 顶掉，所以单独做一个模态。 */
  const alertEl = $("#alert");
  let alertOnClose = null;

  function showAlert(title, message, { hint = "", kind = "error", onClose = null } = {}) {
    if (!alertEl) { window.alert([title, message, hint].filter(Boolean).join("\n")); return; }
    alertEl.querySelector(".alert__title").textContent = title || "出错了";
    alertEl.querySelector(".alert__body").textContent = message || "";
    const hintEl = alertEl.querySelector(".alert__hint");
    if (hintEl) hintEl.textContent = hint || "";
    alertEl.className = `alert${kind === "warn" ? " is-warn" : ""}`;
    alertEl.hidden = false;
    alertOnClose = typeof onClose === "function" ? onClose : null;
    const ok = alertEl.querySelector(".alert__ok");
    if (ok) ok.focus();
  }

  function hideAlert() {
    if (!alertEl || alertEl.hidden) return;
    alertEl.hidden = true;
    const cb = alertOnClose;
    alertOnClose = null;
    if (cb) cb();
  }

  /* 删除这类"点了就回不来"的操作：弹一次确认，必须再点一次「确认删除」。 */
  const confirmEl = $("#confirm");
  let confirmResolve = null;

  function askConfirm({ title, message, hint = "", okText = "确认删除" } = {}) {
    if (!confirmEl) {                       // 老模板兜底
      return Promise.resolve(window.confirm([title, message, hint].filter(Boolean).join("\n")));
    }
    $("#confirm-title").textContent = title || "确定要删除吗";
    $("#confirm-body").textContent = message || "";
    $("#confirm-hint").textContent = hint || "";
    $("#confirm-ok").textContent = okText;
    confirmEl.hidden = false;
    $("#confirm-ok").focus();
    return new Promise((resolve) => { confirmResolve = resolve; });
  }

  function settleConfirm(answer) {
    if (!confirmEl || confirmEl.hidden) return;
    confirmEl.hidden = true;
    const resolve = confirmResolve;
    confirmResolve = null;
    if (resolve) resolve(answer);
  }

  if (confirmEl) {
    $("#confirm-ok").addEventListener("click", () => settleConfirm(true));
    $("#confirm-cancel").addEventListener("click", () => settleConfirm(false));
    confirmEl.addEventListener("click", (event) => {
      if (event.target === confirmEl) settleConfirm(false);   // 点遮罩 = 取消
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") settleConfirm(false);
    });
  }

  if (alertEl) {
    alertEl.querySelector(".alert__ok").addEventListener("click", hideAlert);
    alertEl.addEventListener("click", (event) => {
      if (event.target === alertEl) hideAlert();        // 点遮罩关闭
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") hideAlert();
    });
  }

  /* 录音失败统一出口：能拿到 AudioKit 的中文解释就用它，拿不到就退回原始消息。 */
  function alertMicFailure(err) {
    if (window.AudioKit && window.AudioKit.micErrorMessage) {
      const info = window.AudioKit.micErrorMessage(err);
      showAlert(info.title, info.message, { hint: info.hint });
      return;
    }
    showAlert("打不开麦克风", (err && err.message) || "浏览器没有给出原因", {
      hint: "刷新页面再试。",
    });
  }

  function detailOf(data, fallback) {
    let detail = data && data.detail;
    if (Array.isArray(detail)) {
      detail = detail.map((d) => (d && (d.msg || d.message)) || JSON.stringify(d)).join("；");
    }
    return detail || fallback;
  }

  function friendly(message) {
    if (/同一时间只跑一个/.test(message)) return "前面还有同学在训练，等 TA 跑完再点（大概半分钟）";
    return message;
  }

  async function api(path, { method = "GET", body } = {}) {
    const init = { method, credentials: "same-origin", headers: {} };
    if (body !== undefined) {
      init.headers["Content-Type"] = "application/json";
      init.body = JSON.stringify(body);
    }
    const res = await fetch(path, init);
    let data = null;
    const text = await res.text();
    if (text) {
      try { data = JSON.parse(text); } catch { data = { detail: text }; }
    }
    if (!res.ok) {
      // 没有登录态：401/403 和其它错误一样就地提示，不再跳转
      throw new Error(friendly(detailOf(data, `请求失败（${res.status}）`)));
    }
    return data;
  }

  async function upload(path, blob, filename = "clip.wav") {
    const form = new FormData();
    form.append("file", blob, filename);
    const res = await fetch(path, { method: "POST", body: form, credentials: "same-origin" });
    let data = null;
    const text = await res.text();
    if (text) {
      try { data = JSON.parse(text); } catch { data = { detail: text }; }
    }
    if (!res.ok) {
      if (res.status === 401) window.location.href = "/";
      throw new Error(friendly(detailOf(data, `上传失败（${res.status}）`)));
    }
    return data;
  }

  function setBusy(btn, busy) {
    if (!btn) return;
    btn.classList.toggle("is-busy", !!busy);
    btn.disabled = !!busy;
  }

  const pct = (v) => (Number.isFinite(v) ? `${Math.round(v * 100)}%` : "--");
  const esc = (s) => String(s == null ? "" : s).replace(/[<>&"]/g,
    (c) => ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));
  const fmtTime = (s) => (!Number.isFinite(s) ? "--" : s < 60 ? `${s.toFixed(1)}s` : `${(s / 60).toFixed(1)}m`);
  const todayKey = () => new Date().toLocaleDateString("sv-SE");

  /* ------------------------------------------------------------ 工作台 */
  function initStudio(boot) {
    const state = {
      step: "",
      minPos: Number(boot.dataset.minPos) || 4,
      minNeg: Number(boot.dataset.minNeg) || 4,
      minWordPos: Number(boot.dataset.minWordPos) || 15,
      mine: { pos: 0, neg: 0 },
      wakeWord: boot.dataset.wakeWord || "buddy助手",
      activeRun: boot.dataset.activeRun || "",
      activeModel: boot.dataset.activeModel || "",
      activeWord: "",                 // 当前模型认的那个词（自定义词时和默认词不同）
      // 自定义唤醒词：word = 正在编辑的数据集；dataset = 训练时选的数据集（""=默认词）
      words: [],
      word: null,
      wordDraft: false,
      wordClips: [],
      wordRecording: false,
      wordPending: false,
      wordToken: 0,
      wordTimer: null,
      wordRecorder: null,
      wordJob: null,
      wordJobTimer: 0,
      wordSynthBusy: false,
      dataset: "",
      selectedRun: "",
      detail: null,
      runs: [],
      corpus: null,
      counts: { pos: 0, neg: 0 },
      stage: "pos",
      chart: null,
      mode_: "loss",
      es: null,
      recorder: null,
      recordingLabel: "",
      recordPending: false,
      recordWanted: true,
      recordTimer: null,
      recordSamples: [],
      listening: false,
      listeningPending: false,
      micToken: 0,
      hitTimer: 0,
      hitOnce: false,
      noHitTimer: 0,
      segmenter: null,
      hits: [],
      threshold: Number(localStorage.getItem("nova.threshold") || 0.7),
      raf: 0,
      epochStart: 0,
      voiceLoaded: false,
    };

    if ($("#chart") && window.MetricChart) state.chart = new window.MetricChart($("#chart"));
    // 主舞台那张"一边学一边长"的曲线（引导态也能看到，不藏在进阶里）
    if ($("#stage-chart") && window.MetricChart) {
      state.stageChart = new window.MetricChart($("#stage-chart"), { mode: "acc" });
    }

    /* -------------------------------------------------- 房间切换 */
    function setStep(step, { remember = true } = {}) {
      if (!STEPS.includes(step)) step = "train";
      if (state.step === "word" && step !== "word") {
        stopWordProgress();
        // 离开房间时别留一路麦克风在后台：已开录就正常收尾上传，还在等授权就取消
        if (state.wordRecording || state.wordPending) stopWordRecording();
      }
      state.step = step;
      $$(".view").forEach((v) => v.classList.toggle("is-active", v.id === `view-${step}`));
      $$(".room").forEach((b) => {
        const on = b.dataset.step === step;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-current", on ? "page" : "false");
      });
      if (step === "check") {
        refreshActive();
        if (!state.activeRun) $("#mic-status").textContent = "还没有模型 · 先去「教它」一次";
      } else {
        stopListening();
      }
      if (step === "word") enterWordRoom();
      if (step === "models") {
        renderModels();          // 先把手上这份画出来，再向服务端要一次最新的
        loadRuns();
      }
      if (step === "train" && state.chart) requestAnimationFrame(() => state.chart.resize());
      if (remember) {
        localStorage.setItem("nova.step", step);
        const url = new URL(window.location.href);
        if (url.searchParams.get("step") !== step) {
          url.searchParams.set("step", step);
          window.history.replaceState(null, "", url);
        }
      }
      updateSuggestion();
    }

    $$(".room").forEach((b) => b.addEventListener("click", () => setStep(b.dataset.step)));

    /* -------------------------------------------------- 当前模型 */
    // 模型名前面挂唤醒词：同一份语料能训出好几个词的模型，只写 CNN 会分不清是谁
    // 模型名一律带唤醒词：自定义词用 config 里记的，默认词回落到工作区的默认词
    function wordOf(cfg) {
      const w = cfg && cfg.wake_word;
      return w && w.text ? w.text : (state.wakeWord || "buddy助手");
    }

    function modelBadge(cfg) {
      const model = modelLabel(cfg && cfg.model);
      const text = wordOf(cfg);
      const safe = text.replace(/[<>&]/g, "");
      return text ? `<i class="run-row__word">${safe}</i>${model}` : model;
    }

    function wordTextOf(id) {
      const row = state.words.find((w) => w.id === id);
      return row ? row.text : "";
    }

    function currentWordText() {
      return state.activeWord || state.wakeWord;
    }

    function paintActive() {
      const word = state.activeWord || state.wakeWord;
      $("#active-model").textContent = state.activeModel
        ? `${word ? `${word} · ` : ""}${state.activeModel.toUpperCase()}` : "未应用";
      const sub = $("#check-sub");
      if (sub) {
        sub.textContent = `点一下麦克风，然后喊「${currentWordText()}」`;
      }
      state.runs.forEach((r) => { r.active = r.id === state.activeRun; });
      renderRuns();
      renderModelSwitch();
      renderModels();
    }

    async function refreshActive() {
      try {
        const me = await api("/api/me");
        state.activeRun = me.active ? me.active.run_id : "";
        state.activeModel = me.active ? me.active.model : "";
        state.activeWord = (me.active && me.active.wake_word && me.active.wake_word.text) || "";
        paintActive();
      } catch { /* 读不到当前模型就按"未应用"显示 */ }
    }

    /* -------------------------------------------------- 语料 */
    function renderCorpus(data, { notify = false } = {}) {
      state.corpus = data;
      state.counts = data.counts || { pos: 0, neg: 0 };
      state.mine = {
        pos: (data.pos || []).filter((c) => c.source !== "synth").length,
        neg: (data.neg || []).filter((c) => c.source !== "synth").length,
      };

      paintCounts();

      const builtin = data.builtin || {};
      const label = $("#builtin-count");
      if (label) {
        label.textContent = builtin.available
          ? `（正 ${builtin.pos || 0} / 负 ${builtin.neg || 0}）` : "（本机未装）";
      }
      const toggle = $("#use-builtin");
      if (toggle) {
        toggle.disabled = !builtin.available;
        if (!builtin.available) toggle.checked = false;
      }

      renderClips("pos", data.pos || []);
      renderClips("neg", data.neg || []);

      autoStage(notify);
      updateRecordUi();
      updateTrainIdle();
      updateSuggestion();
    }

    // 录入进度就是本地语料的条数（麦克风 + 合成，不分"谁的"）
    function progressOf() {
      return state.counts;
    }

    function goalOf(stage) {
      return stage === "pos" ? state.minPos : state.minNeg;
    }

    function paintCounts() {
      const c = progressOf();
      $("#count-pos").textContent = c.pos;
      $("#count-neg").textContent = c.neg;
      $("#goal-pos").textContent = goalOf("pos");
      $("#goal-neg").textContent = goalOf("neg");
      paintDataChips();
      $$(".stage-tab").forEach((b) => {
        b.classList.toggle("is-done", c[b.dataset.stage] >= goalOf(b.dataset.stage));
      });
    }

    // 右上角：当前数据集的语料规模，人工 / 合成 分开，各自（正例/负例）。
    // 自定义词的正例只算词目录，负例按训练口径 = 共享池全部 + 内置语料（内置算合成）。
    function datasetCounts() {
      const corpus = state.corpus || { pos: [], neg: [] };
      const mic = (list) => (list || []).filter((c) => c.source !== "synth").length;
      const synth = (list) => (list || []).filter((c) => c.source === "synth").length;
      const word = state.dataset ? state.words.find((w) => w.id === state.dataset) : null;
      if (word) {
        const c = word.counts || { mic: 0, synth: 0 };
        const builtin = corpus.builtin || {};
        const useBuiltin = $("#use-builtin") ? $("#use-builtin").checked : true;
        const extra = useBuiltin && builtin.available
          ? (builtin.pos || 0) + (builtin.neg || 0) : 0;
        return {
          human: { pos: c.mic || 0, neg: mic(corpus.pos) + mic(corpus.neg) },
          synth: { pos: c.synth || 0, neg: synth(corpus.pos) + synth(corpus.neg) + extra },
        };
      }
      return {
        human: { pos: mic(corpus.pos), neg: mic(corpus.neg) },
        synth: { pos: synth(corpus.pos), neg: synth(corpus.neg) },
      };
    }

    function paintDataChips() {
      const human = $("#chip-human");
      const synth = $("#chip-synth");
      if (!human || !synth) return;
      const c = datasetCounts();
      human.textContent = `人工（${c.human.pos}/${c.human.neg}）`;
      synth.textContent = `合成（${c.synth.pos}/${c.synth.neg}）`;
      human.title = `真人录音：正例 ${c.human.pos} 条 / 负例 ${c.human.neg} 条（只做考卷）`;
      synth.title = `合成语料：正例 ${c.synth.pos} 条 / 负例 ${c.synth.neg} 条（预置语料算合成）`;
    }

    function autoStage(notify) {
      const c = progressOf();
      // 没录够的一类自动浮到前面；两类都满了就停在原地（给 TA 自己切）
      if (c.pos < goalOf("pos")) showStage("pos");
      else if (c.neg < goalOf("neg")) {
        if (notify) toast("换一种：说句普通的话，别喊唤醒词", "");
        showStage("neg");
      } else showStage(state.stage);
    }

    function showStage(stage) {
      if (stage !== "pos" && stage !== "neg") return;
      state.stage = stage;
      $("#record-stage").dataset.stage = stage;
      $$(".stage-tab").forEach((b) => b.classList.toggle("is-active", b.dataset.stage === stage));
    }

    function updateRecordUi() {
      const c = progressOf();
      const ready = c.pos >= goalOf("pos") && c.neg >= goalOf("neg");
      const note = $("#stage-note");
      if (ready) {
        note.textContent = `够了：${c.pos} 条「在叫它」+ ${c.neg} 条「没在叫它」`;
        note.classList.add("is-ok");
      } else {
        note.classList.remove("is-ok");
        const left = state.stage === "pos"
          ? { n: goalOf("pos") - c.pos, what: "「在叫它」" }
          : { n: goalOf("neg") - c.neg, what: "「没在叫它」" };
        note.textContent = `还差 ${Math.max(0, left.n)} 条${left.what}`;
      }
      const next = $("#record-next");
      next.disabled = !ready;
      next.textContent = "去教它 →";
    }

    function renderClips(label, clips) {
      const list = $(`#clips-${label}`);
      if (!list) return;
      list.innerHTML = "";
      if (!clips.length) {
        const li = document.createElement("li");
        li.className = "clip clip--empty";
        li.textContent = "还没有录音";
        list.appendChild(li);
        return;
      }
      clips.forEach((clip) => {
        const li = document.createElement("li");
        li.className = "clip";
        // 预置语料保留原名（含 `@`）：界面上只显示 `@` 之后那段，别把机器前缀露给孩子
        const shown = clip.name.includes("@")
          ? clip.name.slice(clip.name.indexOf("@") + 1) : clip.name;
        li.innerHTML = `
          <button class="clip__play" type="button" aria-label="试听">
            <svg viewBox="0 0 24 24" width="13" height="13" fill="currentColor"><path d="M8 5v14l11-7L8 5Z"/></svg>
          </button>
          <span class="clip__name">${clip.source === "synth" ? '<i class="clip__tag">合成</i>' : ""}${shown}</span>
          <span class="clip__dur">${(clip.duration || 0).toFixed(2)}s</span>
          ${delBtn}`;
        li.querySelector(".clip__play").addEventListener("click", () => {
          const audio = new Audio(`/api/corpus/${label}/${encodeURIComponent(clip.name)}?t=${Date.now()}`);
          audio.play().catch(() => toast("试听失败", "error"));
        });
        const del = li.querySelector(".clip__del");
        if (del) del.addEventListener("click", async () => {
          try {
            renderCorpus(await api(`/api/corpus/${label}/${encodeURIComponent(clip.name)}`,
              { method: "DELETE" }));
            toast("已删除", "ok");
          } catch (err) { toast(err.message, "error"); }
        });
        list.appendChild(li);
      });
    }

    async function loadCorpus() {
      try {
        renderCorpus(await api("/api/corpus"));
      } catch (err) { toast(err.message, "error"); }
    }

    /* -------------------------------------------------- 录制 */
    function recordStatus(label, text, kind = "") {
      const el = $(`[data-status="${label}"]`);
      if (!el) return;
      el.textContent = text;
      el.className = `record-status${kind ? ` is-${kind}` : ""}`;
    }

    async function startRecording(label) {
      const card = $(`.record-card[data-label="${label}"]`);
      const canvas = $(`[data-wave="${label}"]`);
      const button = $(`[data-record="${label}"]`);
      if (!window.AudioKit) { toast("录音脚本没加载，刷新试试", "error"); return; }
      if (state.recordPending || state.recordingLabel) return;
      // getUserMedia 可能要等权限弹窗，这期间必须挡住第二次点击，
      // 否则会同时开两路麦克风（旧实现的实际 bug：点两下变成两条并行录音）。
      state.recordPending = true;
      const recorder = new window.AudioKit.Recorder();
      state.recordSamples = [];
      recordStatus(label, "正在请求麦克风…");
      try {
        await recorder.start((chunk) => {
          state.recordSamples.push(chunk);
          if (state.recordSamples.length > 6) state.recordSamples.shift();
          const merged = state.recordSamples.reduce((acc, c) => {
            const out = new Float32Array(acc.length + c.length);
            out.set(acc); out.set(c, acc.length);
            return out;
          }, new Float32Array(0));
          window.AudioKit.drawWaveform(canvas, merged, {
            color: label === "pos" ? "#22d3ee" : "#818cf8", bars: 70,
          });
        });
      } catch (err) {
        state.recordPending = false;
        const info = window.AudioKit.micErrorMessage
          ? window.AudioKit.micErrorMessage(err) : { title: "打不开麦克风", message: err.message };
        recordStatus(label, info.title, "error");
        alertMicFailure(err);
        return;
      }
      state.recordPending = false;
      if (!state.recordWanted) {                 // 权限弹窗期间用户取消了
        recorder.cancel();
        recordStatus(label, "已取消");
        return;
      }
      state.recorder = recorder;
      state.recordingLabel = label;
      if (button) button.classList.add("is-live");
      if (card) card.classList.add("is-recording");
      recordStatus(label, "在听… 说完再点一下（最长 5 秒）");
      const btnText = $(`[data-record="${label}"] span`);
      if (btnText) btnText.textContent = "点击结束";
      // 4.5s 就收，给浏览器 buffer 一点余量，避免刚好越过服务端 5s 上限被拒
      state.recordTimer = setTimeout(() => stopRecording(), 4500);
    }

    async function stopRecording(cancel = false) {
      const label = state.recordingLabel;
      if (!label) return;
      state.recordWanted = false;
      clearTimeout(state.recordTimer);
      state.recordTimer = null;
      state.recordingLabel = "";
      const card = $(`.record-card[data-label="${label}"]`);
      if (card) card.classList.remove("is-recording");
      const btn = $(`[data-record="${label}"]`);
      if (btn) btn.classList.remove("is-live");
      const btnText = $(`[data-record="${label}"] span`);
      if (btnText) btnText.textContent = "按下开始";

      const recorder = state.recorder;
      state.recorder = null;
      if (!recorder) return;
      if (cancel) { recorder.cancel(); return; }

      recordStatus(label, "处理中…");
      let out = null;
      try { out = await recorder.stop(); } catch (err) {
        recordStatus(label, err.message || "录音失败", "error");
        showAlert("录音失败了", err.message || "浏览器在处理这段录音时出错了", {
          hint: "再按一次「按下开始」试试；一直不行就刷新页面。",
        });
        return;
      }
      // 围栏：先用与训练同口径的统计拦一道，别让空录 / 爆音浪费一次上传。
      if (!out) {
        recordStatus(label, "没录到声音", "error");
        showAlert("没录到声音", "这次没抓到麦克风数据。", {
          hint: "看看麦克风开没开，再按一次试试。", kind: "warn",
        });
        return;
      }
      const rejection = (window.AudioKit.auditClip && out.stats)
        ? window.AudioKit.auditClip(out.stats) : null;
      if (rejection) {
        recordStatus(label, rejection.title, "error");
        showAlert(rejection.title, rejection.message, { hint: rejection.hint, kind: "warn" });
        return;
      }
      try {
        const data = await upload(`/api/corpus/${label}`, out.wav, `${label}-${Date.now()}.wav`);
        renderCorpus(data, { notify: true });
        const total = (data.counts && data.counts[label]) || 0;
        recordStatus(label, `已保存 ${out.duration.toFixed(2)}s · 共 ${total} 条`, "ok");
      } catch (err) {
        recordStatus(label, err.message, "error");
        showAlert("这条没存上", err.message, {
          hint: "再录一次；如果反复失败，把这句话告诉老师。",
        });
      }
      window.AudioKit.drawWaveform($(`[data-wave="${label}"]`), new Float32Array(0));
    }

    $$("[data-record]").forEach((btn) => {
      btn.addEventListener("click", () => {
        const label = btn.dataset.record;
        if (state.recordPending) {                 // 等权限时再点 = 取消
          state.recordWanted = false;
          recordStatus(label, "已取消");
          return;
        }
        state.recordWanted = true;
        if (state.recordingLabel === label) stopRecording();
        else if (state.recordingLabel) toast("先把当前这条录完", "error");
        else startRecording(label);
      });
    });

    $$(".stage-tab").forEach((b) => b.addEventListener("click", () => {
      if (state.recordingLabel) { toast("先把当前这条录完", "error"); return; }
      showStage(b.dataset.stage);
      updateRecordUi();
    }));

    $("#record-next").addEventListener("click", () => setStep("train"));

    /* -------------------------------------------------- 合成语料 */
    const synthVoice = $("#synth-voice");
    const synthStatus = $("#synth-status");
    let synthVoices = null;

    function syncSynthVoice() {
      if (!synthVoice || !synthVoices) return;
      const voice = synthVoices.find((v) => v.id === synthVoice.value);
      const pos = $("#synth-pos");
      if (!pos) return;
      const canPos = !voice || voice.can_pos;
      pos.disabled = !canPos;
      if (!canPos) pos.value = "0";
    }

    async function loadVoices() {
      if (!synthVoice || state.voiceLoaded) return;
      state.voiceLoaded = true;
      try {
        const data = await api("/api/voices");
        synthVoices = data.voices || [];
        synthVoice.innerHTML = "";
        synthVoices.forEach((v) => {
          const opt = document.createElement("option");
          opt.value = v.id;
          opt.textContent = `${v.label}（${v.lang === "en" ? "英文" : "中文"}）`
            + (v.can_pos ? "" : " · 只能造负样本");
          synthVoice.appendChild(opt);
        });
        const pos = $("#synth-pos");
        const neg = $("#synth-neg");
        if (pos) { pos.max = String(data.max_pos || 6); pos.value = String(Math.min(3, data.max_pos || 6)); }
        if (neg) { neg.max = String(data.max_neg || 6); neg.value = String(Math.min(3, data.max_neg || 6)); }
        if (!data.configured) {
          $("#synth-go").disabled = true;
          synthStatus.textContent = "服务器没配豆包 TTS 凭证，这里先用不了，改麦克风录制。";
        }
        syncSynthVoice();
      } catch (err) {
        state.voiceLoaded = false;
        synthStatus.textContent = err.message;
      }
    }

    if (synthVoice) synthVoice.addEventListener("change", syncSynthVoice);

    const synthGo = $("#synth-go");
    if (synthGo) synthGo.addEventListener("click", async () => {
      if (!synthVoice.value) { toast("先选一个音色", "error"); return; }
      const pos = Math.max(0, Number($("#synth-pos").value) || 0);
      const neg = Math.max(0, Number($("#synth-neg").value) || 0);
      if (pos + neg === 0) { toast("至少选一条要合成", "error"); return; }
      setBusy(synthGo, true);
      synthStatus.textContent = "正在合成…（每条 0.5~2 秒）";
      try {
        const data = await api("/api/corpus/synth", {
          method: "POST", body: { voice: synthVoice.value, pos, neg },
        });
        renderCorpus(data);
        const created = data.created || { pos: 0, neg: 0 };
        const failed = data.failed ? `，${data.failed} 条没通过校验` : "";
        synthStatus.textContent = `合成完成：正 ${created.pos} 条 / 负 ${created.neg} 条${failed}。`;
        toast(`已合成 ${created.pos + created.neg} 条声音`, "ok");
      } catch (err) {
        synthStatus.textContent = err.message;
        toast(err.message, "error");
      } finally {
        setBusy(synthGo, false);
      }
    });

    /* -------------------------------------------------- 自定义唤醒词（第 4 个房间） */
    // 数据集列表：默认词 + 当前自定义词（旧词留盘但服务端不再返回）。
    function renderDatasets() {
      const box = $("#dataset-tabs");
      if (!box) return;
      const rows = [{ id: "", text: `默认（${state.wakeWord}）`, counts: null }]
        .concat(state.words);
      box.innerHTML = "";
      rows.forEach((row) => {
        const btn = document.createElement("button");
        btn.type = "button";
        btn.dataset.dataset = row.id;
        btn.classList.toggle("is-active", row.id === state.dataset);
        const c = row.counts;
        btn.textContent = c
          ? `${row.text}（${c.mic}${c.ready ? "" : `/${c.min_pos}`}）`
          : row.text;
        btn.title = c
          ? `真人 ${c.mic} 条 · 合成 ${c.synth} 条`
          : "本地录音 + 预置语料";
        btn.addEventListener("click", () => selectDataset(row.id));
        box.appendChild(btn);
      });
    }

    function updateDatasetNote() {
      const note = $("#dataset-note");
      if (!note) return;
      const w = state.dataset ? state.words.find((x) => x.id === state.dataset) : null;
      note.textContent = w
        ? `「${w.text}」：正例 ${w.counts.mic} 条真人 + ${w.counts.synth} 条合成；`
          + "负例 = 本地录音 + 预置语料的全部片段（都当「不是这个词」）。"
        : `默认数据集：你录的「${state.wakeWord}」+ 预置语料。`;
    }

    function selectDataset(id) {
      state.dataset = state.words.some((w) => w.id === id) ? id : "";
      renderDatasets();
      updateDatasetNote();
      updateTrainIdle();
      paintDataChips();
    }

    const CJK_RE = /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/;

    // 一键合成的规模：中文词 15 个音色、纯英文词 17 个，每个音色 2 条。
    function synthPlanOf(word) {
      const voices = CJK_RE.test((word && word.text) || "") ? 15 : 17;
      return { voices, total: voices * 2 };
    }

    function renderWordClips() {
      const list = $("#word-clips");
      if (!list) return;
      list.innerHTML = "";
      if (!state.wordClips.length) {
        list.innerHTML = '<li class="clip clip--empty">还没有录音</li>';
        return;
      }
      state.wordClips.forEach((clip) => {
        const li = document.createElement("li");
        li.className = "clip";
        const name = document.createElement("span");
        name.className = "clip__name";
        if (clip.source === "synth") {
          const tag = document.createElement("i");
          tag.className = "clip__tag";
          tag.textContent = "合成";
          name.appendChild(tag);
        }
        name.appendChild(document.createTextNode(clip.name.split("@").pop()));
        const play = document.createElement("button");
        play.type = "button";
        play.className = "clip__play";
        play.textContent = "▶";
        play.title = "试听";
        play.addEventListener("click", () => {
          const url = `/api/words/${state.word.id}/clips/${encodeURIComponent(clip.name)}`;
          new Audio(`${url}?t=${Date.now()}`).play().catch(() => toast("试听失败", "error"));
        });
        const del = document.createElement("button");
        del.type = "button";
        del.className = "clip__del";
        del.textContent = "×";
        del.title = "删除这条";
        del.addEventListener("click", async () => {
          try {
            await api(`/api/words/${state.word.id}/clips/${encodeURIComponent(clip.name)}`,
              { method: "DELETE" });
            toast("已删除", "ok");
            await loadWords();
          } catch (err) { toast(err.message, "error"); }
        });
        const dur = document.createElement("span");
        dur.className = "clip__dur";
        dur.textContent = `${Number(clip.duration || 0).toFixed(2)}s`;
        li.append(play, name, dur, del);
        list.appendChild(li);
      });
    }

    async function refreshWordClips() {
      const list = $("#word-clips");
      if (!state.word) {
        state.wordClips = [];
        if (list) list.innerHTML = '<li class="clip clip--empty">还没有录音</li>';
        return;
      }
      try {
        const data = await api(`/api/words/${state.word.id}`);
        state.wordClips = data.clips || [];
        if (data.counts && state.word) state.word.counts = data.counts;
        renderWordClips();
      } catch (err) {
        if (list) list.innerHTML = '<li class="clip clip--empty">读取录音失败</li>';
      }
    }

    function renderWordView() {
      const w = state.word;
      const counts = (w && w.counts) || { mic: 0, synth: 0, min_pos: state.minWordPos, ready: false };
      const active = !!w && !state.wordDraft;      // 草稿态只留步骤 1，方便重新起名
      const current = $("#word-current");
      if (current) current.textContent = w ? w.text : "还没有";
      const sw = $("#word-switch");
      if (sw) {
        sw.hidden = !w;
        sw.textContent = state.wordDraft ? "取消" : "换一个";
      }
      const note = $("#word-switch-note");
      if (note) note.hidden = !w || state.wordDraft;

      $$(".wstep").forEach((el) => {
        const n = Number(el.dataset.wstep);
        el.hidden = !active ? n !== 1 : (n === 4 ? !counts.ready : false);
      });

      if ($("#word-count")) $("#word-count").textContent = counts.mic;
      if ($("#word-goal")) $("#word-goal").textContent = counts.min_pos;
      if ($("#word-goal-title")) $("#word-goal-title").textContent = counts.min_pos;

      const dots = $("#word-dots");
      if (dots) {
        dots.innerHTML = "";
        for (let i = 0; i < counts.min_pos; i += 1) {
          const li = document.createElement("li");
          if (i < counts.mic) li.className = "is-on";
          dots.appendChild(li);
        }
      }

      const status = $("#word-record-status");
      if (status && !state.wordRecording) {
        status.textContent = !active
          ? "先在上面写一个唤醒词"
          : counts.mic
            ? `已录 ${counts.mic} 条真人（合成 ${counts.synth} 条）`
            : `点一下，然后喊「${w.text}」`;
      }
      const hint = $("#word-hint");
      if (hint) {
        hint.textContent = state.wordDraft
          ? "写一个新词，点「用它」后旧词就不再显示（录音仍留在服务器上）。"
          : active ? `正在做「${w.text}」` : "1~4 个字最好念；中英混排也行。";
      }

      const plan = synthPlanOf(w);
      if ($("#word-synth-meta")) {
        $("#word-synth-meta").textContent = active
          ? `约 ${plan.total} 条 · ${plan.voices} 个音色` : "";
      }
      const synthBtn = $("#word-synth");
      if (synthBtn) {
        const running = !!(state.wordJob && state.wordJob.state === "running");
        synthBtn.disabled = !active || running || state.wordSynthBusy;
      }

      const goto = $("#word-goto-train");
      if (goto) goto.disabled = !counts.ready;
      const trainHint = $("#word-train-hint");
      if (trainHint) {
        trainHint.textContent = !active ? ""
          : counts.ready
            ? `用「${w.text}」的 ${counts.mic} 条真人考卷 + ${counts.synth} 条合成训练；`
              + "点按钮会把数据集切到这个词。"
            : `还差 ${counts.min_pos - counts.mic} 条真人录音才能训练。`;
      }
      renderDatasets();
      updateDatasetNote();
      paintDataChips();
    }

    async function loadWords() {
      try {
        const data = await api("/api/words");
        state.minWordPos = data.min_pos || state.minWordPos;
        state.words = data.words || [];
        // 服务端只返回"当前词"，所以这里直接以它为准；换词后旧词自然消失
        state.word = data.word || state.words[0] || null;
        if (state.dataset && !state.words.some((w) => w.id === state.dataset)) {
          state.dataset = "";
        }
        renderWordView();
        updateTrainIdle();
        await refreshWordClips();
      } catch (err) {
        toast(err.message, "error");
      }
    }

    async function enterWordRoom() {
      await loadWords();
      await checkWordProgress();
      if (!state.word || state.wordDraft) $("#word-text").focus();
    }

    $("#word-save").addEventListener("click", async () => {
      const text = $("#word-text").value.trim();
      if (!text) { $("#word-text").focus(); toast("先写一个唤醒词", "error"); return; }
      const btn = $("#word-save");
      setBusy(btn, true);
      try {
        const row = await api("/api/words", { method: "POST", body: { text } });
        state.dataset = row.id;
        state.wordDraft = false;
        $("#word-text").value = "";
        await loadWords();
        await checkWordProgress();
        toast(row.created ? `新唤醒词：「${row.text}」` : `接着录「${row.text}」`, "ok");
      } catch (err) {
        toast(err.message, "error");
      } finally {
        setBusy(btn, false);
      }
    });

    // 「换一个」：页面内二次确认（第一次点变「取消」，再点回到草稿态），不删旧录音
    $("#word-switch").addEventListener("click", () => {
      if (!state.word) return;
      if (state.wordDraft) {
        state.wordDraft = false;
        renderWordView();
        return;
      }
      state.wordDraft = true;
      $("#word-text").value = "";
      renderWordView();
      $("#word-text").focus();
    });

    /* 合成：一键多音色，后台跑，400ms 轮询进度 */
    function applyWordProgress(job) {
      state.wordJob = job || null;
      const total = job && job.total ? job.total : 0;
      const done = job && job.done ? job.done : 0;
      const box = $("#word-synth-box");
      const bar = $("#word-synth-bar");
      if (box) box.hidden = !total;
      if (bar) bar.style.width = total ? `${Math.round((done / total) * 100)}%` : "0%";
      const btn = $("#word-synth");
      if (btn) {
        const running = !!(job && job.state === "running");
        btn.disabled = !state.word || state.wordDraft || running || state.wordSynthBusy;
      }
      const status = $("#word-synth-status");
      if (!status) return;
      if (job && job.state === "running") {
        status.classList.remove("is-warn");
        status.textContent = `正在合成… 已完成 ${done}/${total}`
          + `（成功 ${job.created}，失败 ${job.failed}）`;
      } else if (job && job.state === "done") {
        status.classList.remove("is-warn");
        status.textContent = `合成完成：成功 ${job.created} 条、失败 ${job.failed} 条`
          + `（共 ${total} 条 · ${(job.voices || []).length} 个音色）。`;
      } else if (job && job.state === "failed") {
        status.classList.add("is-warn");
        status.textContent = job.error || "合成失败，再点一次试试。";
      }
    }

    function stopWordPoll() {
      if (state.wordJobTimer) {
        clearInterval(state.wordJobTimer);
        state.wordJobTimer = 0;
      }
    }

    function stopWordProgress() {
      stopWordPoll();
    }

    function startWordPoll() {
      if (state.wordJobTimer) return;
      state.wordJobTimer = setInterval(async () => {
        const wid = state.word && state.word.id;
        if (!wid || (state.wordJob && state.wordJob.state !== "running")) {
          stopWordPoll();
          return;
        }
        try {
          const job = await api(`/api/words/${wid}/synth/progress`);
          applyWordProgress(job);
          if (job.state !== "running") {
            stopWordPoll();
            if (job.state === "done") {
              await loadWords();
              toast(`合成完成：${job.created} 条`, "ok");
            }
          }
        } catch (err) {
          stopWordPoll();
        }
      }, 400);
    }

    async function checkWordProgress() {
      if (!state.word) {
        stopWordPoll();
        return;
      }
      try {
        const job = await api(`/api/words/${state.word.id}/synth/progress`);
        applyWordProgress(job);
        if (job.state === "running") startWordPoll();
        else stopWordPoll();
      } catch (err) {
        stopWordPoll();
      }
    }

    $("#word-synth").addEventListener("click", async () => {
      if (!state.word || state.wordDraft) { toast("先写一个唤醒词", "error"); return; }
      if (state.wordJob && state.wordJob.state === "running") return;
      state.wordSynthBusy = true;
      const btn = $("#word-synth");
      setBusy(btn, true);
      $("#word-synth-status").classList.remove("is-warn");
      $("#word-synth-status").textContent = "正在准备合成…";
      try {
        const job = await api(`/api/words/${state.word.id}/synth/auto`, { method: "POST" });
        applyWordProgress({ ...job, state: "running" });
        startWordPoll();
      } catch (err) {
        $("#word-synth-status").classList.add("is-warn");
        $("#word-synth-status").textContent = err.message;
        toast(err.message, "error");
      } finally {
        state.wordSynthBusy = false;
        setBusy(btn, false);
        applyWordProgress(state.wordJob);
      }
    });

    /* 真人录音：麦克风授权 / 停止 / 上传，全程只允许一路 */
    function wordRecordIdle() {
      $("#word-record").classList.remove("is-live");
      const label = $("#word-record span");
      if (label) label.textContent = "按下开始";
      const canvas = $("#word-wave");
      if (canvas && window.AudioKit && window.AudioKit.drawWaveform) {
        window.AudioKit.drawWaveform(canvas, null);
      }
    }

    async function startWordRecording() {
      if (!state.word || state.wordDraft) { toast("先写一个唤醒词", "error"); return; }
      if (!window.AudioKit) { toast("录音脚本没加载，刷新试试", "error"); return; }
      if (state.wordRecording || state.wordPending) return;
      const token = state.wordToken + 1;
      state.wordToken = token;
      state.wordPending = true;
      $("#word-record-status").textContent = "正在请求麦克风…";
      const rec = new window.AudioKit.Recorder();
      // 边录边画波形：只留最近 2 秒，够看出"有没有声"
      const canvas = $("#word-wave");
      const frames = [];
      let total = 0;
      const onFrame = (frame) => {
        frames.push(frame);
        total += frame.length;
        while (total > 32000 && frames.length > 1) total -= frames.shift().length;
        if (!canvas || !window.AudioKit.drawWaveform) return;
        const merged = new Float32Array(total);
        let off = 0;
        frames.forEach((f) => { merged.set(f, off); off += f.length; });
        window.AudioKit.drawWaveform(canvas, merged, { color: "#fb7185", bars: 80 });
      };
      try {
        await rec.start(onFrame);
      } catch (err) {
        state.wordPending = false;
        if (state.wordToken !== token) return;      // 授权期间学生点了取消
        state.wordRecording = false;
        const info = window.AudioKit.micErrorMessage
          ? window.AudioKit.micErrorMessage(err) : { title: "打不开麦克风", message: err.message };
        $("#word-record-status").textContent = info.title;
        alertMicFailure(err);
        return;
      }
      if (state.wordToken !== token) {               // 回来晚了：这次启动作废
        rec.cancel();
        wordRecordIdle();
        return;
      }
      state.wordPending = false;
      state.wordRecording = true;
      state.wordRecorder = rec;
      $("#word-record").classList.add("is-live");
      $("#word-record span").textContent = "点击结束";
      $("#word-record-status").textContent = `在听… 说完再点一下（最长 5 秒）`;
      state.wordTimer = setTimeout(() => stopWordRecording(), 4500);
    }

    async function stopWordRecording() {
      // 授权还没回来：把 token 移走，等 start 的 await 回来后自动 cancel
      if (state.wordPending) {
        state.wordToken += 1;
        state.wordPending = false;
        wordRecordIdle();
        $("#word-record-status").textContent = "已取消，再按一次重来";
        return;
      }
      const rec = state.wordRecorder;
      if (!rec) return;
      state.wordRecorder = null;
      clearTimeout(state.wordTimer);
      state.wordTimer = null;
      state.wordRecording = false;
      wordRecordIdle();
      let out = null;
      try { out = await rec.stop(); } catch (err) {
        $("#word-record-status").textContent = "录音失败";
        showAlert("录音失败了", err.message || "浏览器在处理这段录音时出错了", {
          hint: "再按一次重录；一直不行就刷新页面。",
        });
        return;
      }
      if (!out) {
        $("#word-record-status").textContent = "没录到声音";
        showAlert("没录到声音", "这次没抓到麦克风数据。", {
          hint: "看看麦克风开没开，再喊一次。", kind: "warn",
        });
        return;
      }
      const rejection = (window.AudioKit.auditClip && out.stats)
        ? window.AudioKit.auditClip(out.stats) : null;
      if (rejection) {
        $("#word-record-status").textContent = rejection.title;
        showAlert(rejection.title, rejection.message, { hint: rejection.hint, kind: "warn" });
        return;
      }
      try {
        const data = await upload(`/api/words/${state.word.id}/clips`,
          out.wav, `word-${Date.now()}.wav`);
        $("#word-record-status").textContent =
          `已保存 ${out.duration.toFixed(2)}s · 真人 ${data.counts.mic}/${data.counts.min_pos}`;
        await loadWords();
      } catch (err) {
        $("#word-record-status").textContent = err.message;
        showAlert("这条没存上", err.message, { hint: "再喊一次试试。" });
      }
    }

    $("#word-record").addEventListener("click", () => {
      if (state.wordRecording || state.wordPending) stopWordRecording();
      else startWordRecording();
    });

    $("#word-goto-train").addEventListener("click", () => {
      if (!state.word || !state.word.counts.ready) return;
      selectDataset(state.word.id);
      setStep("train");
      toast(`数据集已切到「${state.word.text}」，可以开始训练了`, "ok");
    });

    /* -------------------------------------------------- 训练表单 */
    $$("#model-grid input").forEach((input) => {
      input.addEventListener("change", () => {
        $$(".model-card").forEach((c) => c.classList.toggle("is-active", c.contains(input) && input.checked));
      });
    });

    // 内置语料勾选会影响自定义词数据集的负例口径 → 顺手刷新右上角统计
    const builtinToggle = $("#use-builtin");
    if (builtinToggle) builtinToggle.addEventListener("change", paintDataChips);

    // 「训练数据集大小」两个滑杆：拖动时实时回显数值，提交时按上限传服务端
    [["max-pos", "max-pos-val"], ["max-neg", "max-neg-val"]].forEach(([id, out]) => {
      const input = $(`#${id}`);
      if (!input) return;
      const paint = () => { if ($(`#${out}`)) $(`#${out}`).textContent = input.value; };
      input.addEventListener("input", paint);
      paint();
    });

    function readForm() {
      const model = ($("#model-grid input:checked") || {}).value || "cnn";
      const num = (id) => Number($(`#${id}`).value);
      const body = {
        model,
        epochs: num("epochs"),
        batch: num("batch"),
        lr: num("lr"),
        // 考卷比例/种子页面不再暴露，交给服务端默认值（0.12 / 42）
        max_pos: num("max-pos"),
        max_neg: num("max-neg"),
        use_builtin: $("#use-builtin") ? $("#use-builtin").checked : true,
      };
      if (state.dataset) body.wake_word_id = state.dataset;
      for (const [key, value] of Object.entries(body)) {
        if (key === "model" || key === "use_builtin" || key === "wake_word_id") continue;
        if (!Number.isFinite(value)) {
          toast("进阶里的数字要填对", "error");
          return null;
        }
      }
      return body;
    }

    function updateTrainIdle() {
      const hint = $("#train-idle-hint");
      if (!hint) return;
      // 选了自定义唤醒词：正例门槛是"这个词的真人录音"，跟共享池的正例无关
      if (state.dataset) {
        const w = state.words.find((x) => x.id === state.dataset);
        const c = (w && w.counts) || { mic: 0, synth: 0, min_pos: state.minWordPos, ready: false };
        hint.textContent = c.ready
          ? `用「${w ? w.text : "这个词"}」的 ${c.mic + c.synth} 条正例 + 全班负例来教。`
          : `「${w ? w.text : "这个词"}」还差 ${c.min_pos - c.mic} 条真人录音。`;
        hint.classList.toggle("is-warn", !c.ready);
        return;
      }
      const c = progressOf();
      const ready = c.pos >= goalOf("pos") && c.neg >= goalOf("neg");
      const builtin = !!(state.corpus && state.corpus.builtin && state.corpus.builtin.available);
      if (ready) {
        hint.textContent = `用 ${c.pos + c.neg} 条声音来教。`;
      } else if (builtin) {
        hint.textContent = "你的声音还差一点，先用内置合成语料练一次也行。";
      } else {
        hint.textContent = `还差声音：${state.minPos} 条「在叫它」+ ${state.minNeg} 条「没在叫它」。`;
      }
      hint.classList.toggle("is-warn", !ready && !builtin);
    }

    $("#start-train").addEventListener("click", startTrain);

    async function startTrain() {
      if (!state.corpus) await loadCorpus();
      const body = readForm();
      if (!body) return;
      const c = progressOf();
      const ready = c.pos >= goalOf("pos") && c.neg >= goalOf("neg");
      const builtin = !!(state.corpus && state.corpus.builtin && state.corpus.builtin.available);
      // 自定义唤醒词的门槛在服务端（真人 15 条），这里只做一个更早的提示
      if (body.wake_word_id) {
        const word = state.words.find((w) => w.id === body.wake_word_id);
        if (word && word.counts && !word.counts.ready) {
          toast(`「${word.text}」还差 ${word.counts.min_pos - word.counts.mic} 条真人录音`, "error");
          setStep("word");
          return;
        }
      } else if (!ready && !(builtin && body.use_builtin)) {
        toast(`还差声音：${goalOf("pos")} 条「在叫它」+ ${goalOf("neg")} 条「没在叫它」`, "error");
        setStep("record");
        return;
      }
      const btn = $("#start-train");
      setBusy(btn, true);
      // 子进程起 torch 要几秒，这期间界面上必须"在动"，否则孩子以为卡死会点第二下
      showTrainPane("running");
      setPrep(true);
      if (state.stageChart) state.stageChart.clear();     // 曲线每次从空开始长：证明是这一次训的
      $("#result-model").innerHTML = modelBadge({
        model: body.model,
        wake_word: body.wake_word_id ? { text: wordTextOf(body.wake_word_id) } : null,
      });
      $("#result-meta").textContent = `学了 ${body.epochs} 遍`;
      $("#result-when").textContent = "正在准备，马上开始第一遍";
      try {
        const res = await api("/api/runs", { method: "POST", body });
        toast("开始教它了", "ok");
        await loadRuns();
        await selectRun(res.id);
      } catch (err) {
        showTrainPane("idle");
        toast(err.message, "error");
      } finally {
        setBusy(btn, false);
      }
    }

    /* -------------------------------------------------- 训练运行 */
    function showTrainPane(name) {
      ["idle", "running", "done", "failed"].forEach((n) => {
        const pane = $(`#train-${n}`);
        if (pane) pane.hidden = n !== name;
      });
      if (name !== "running") setPrep(false);
      syncResultCard();
    }

    // 结果卡：模型是谁、学了几遍、多久、第几次训的
    function renderStageCard(detail, { running = false } = {}) {
      const cfg = (detail && detail.config) || {};
      const result = (detail && detail.result) || {};
      $("#result-model").innerHTML = modelBadge(cfg);

      const meta = [];
      const epochs = Number(cfg.epochs) || 30;
      if (running) {
        meta.push(`学了 ${epochs} 遍`);
      } else {
        meta.push(`学了 ${(detail && (detail.history || []).length) || epochs} 遍`);
        const secs = Number(result.elapsed_s);
        if (Number.isFinite(secs)) meta.push(`${secs.toFixed(1)} 秒`);
        const params = Number(result.params_total);
        if (Number.isFinite(params) && params > 0) {
          meta.push(`模型里 ${(params / 1000).toFixed(1)}k 个数字`);
        }
      }
      $("#result-meta").textContent = meta.join(" · ");

      const idx = state.runs.findIndex((r) => r.id === (detail && detail.id));
      const run = idx >= 0 ? state.runs[idx] : null;
      const when = run ? `第 ${state.runs.length - idx} 次训练 · ${runLabel(run)}` : "这一次训练";
      $("#result-when").textContent = running ? "正在一遍一遍地学，曲线会自己长出来" : when;

      if (state.stageChart) {
        state.stageChart.setData((detail && detail.history) || []);
        state.stageChart.resize();
      }
      syncResultCard();
    }

    // 曲线卡：正在学（且已经有第一个点）/ 学完了才出现；准备阶段只有进度条，别摆一张空图
    function syncResultCard() {
      const card = $("#result-card");
      if (!card) return;
      const done = !$("#train-done").hidden;
      const running = !$("#train-running").hidden;
      const prep = running && $("#train-running").classList.contains("is-prep");
      const hasPoints = !!(state.stageChart && state.stageChart.history.length);
      const show = done || (running && !prep && hasPoints);
      const wasHidden = card.hidden;
      card.hidden = !show;
      // 只有从隐藏变可见时才重量尺寸：每来一个 epoch 都 resize 会闪
      if (show && wasHidden && state.stageChart) {
        requestAnimationFrame(() => state.stageChart.resize());
      }
    }

    // "准备中"：从点下按钮到第一个 epoch 之间（起进程 + import torch），
    // 进度条走不确定动画、文案换成"正在准备"，不然界面看着像卡死。
    function setPrep(on) {
      const pane = $("#train-running");
      if (!pane) return;
      pane.classList.toggle("is-prep", !!on);
      if (on) {
        $("#progress-title").textContent = "正在准备…";
        $("#progress-eta").textContent = "正在把声音装进模型，第一次要几秒钟";
        $("#progress-bar").style.width = "0%";
        state.epochStart = 0;
      }
      syncResultCard();
    }

    function runLabel(run) {
      const m = /^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})/.exec(run.id || "");
      if (!m) return run.id || "—";
      const [, , mo, d, h, mi] = m;
      const day = `${m[1]}-${mo}-${d}`;
      return day === todayKey() ? `今天 ${h}:${mi}` : `${Number(mo)}/${Number(d)} ${h}:${mi}`;
    }

    function renderRuns() {
      const list = $("#run-list");
      if (!list) return;
      list.innerHTML = "";
      if (!state.runs.length) {
        list.innerHTML = '<li class="run-list__empty">还没有运行记录</li>';
        return;
      }
      state.runs.forEach((run) => {
        const li = document.createElement("li");
        li.className = "run-row" + (run.id === state.selectedRun ? " is-active" : "");
        li.dataset.id = run.id;
        li.title = run.id;
        li.innerHTML = `
          <span class="run-row__model">${modelBadge(run.config)}</span>
          <span class="run-row__id">${runLabel(run)}</span>
          <span class="run-row__acc">${run.final ? pct(run.final.val_acc) : "--"}</span>
          <span class="run-row__state">${STATE_LABEL[run.state] || run.state}${run.active ? " · 在用" : ""}</span>`;
        li.addEventListener("click", () => selectRun(run.id));
        list.appendChild(li);
      });
    }

    function renderModelSwitch() {
      const sel = $("#model-switch");
      if (!sel) return;
      const usable = state.runs.filter((r) => r.state === "succeeded" && r.final);
      if (!usable.length) {
        sel.innerHTML = '<option value="">还没有可用的模型</option>';
        sel.disabled = true;
        return;
      }
      sel.disabled = false;
      sel.innerHTML = "";
      usable.forEach((run) => {
        const opt = document.createElement("option");
        opt.value = run.id;
        const word = wordOf(run.config);
        opt.textContent = `${word ? `${word} · ` : ""}${(run.config.model || "?").toUpperCase()}`
          + ` · ${pct(run.final.val_acc)} · ${runLabel(run)}`;
        sel.appendChild(opt);
      });
      sel.value = state.activeRun || usable[0].id;
      if (!usable.some((r) => r.id === state.activeRun)) sel.value = "";
    }

    $("#model-switch").addEventListener("change", async (event) => {
      const id = event.target.value;
      if (!id || id === state.activeRun) return;
      try {
        await api(`/api/runs/${id}/apply`, { method: "POST" });
        await refreshActive();
        toast("换好了，喊一声试试", "ok");
      } catch (err) { toast(err.message, "error"); }
    });

    /* -------------------------------------------------- 我的模型（大卡片） */
    // 只列「训练成功 + 真有权重」的运行；单数据域，所以这份清单就是这台机器的全部。
    function usableModels() {
      return state.runs.filter((r) => r.state === "succeeded" && r.has_ckpt);
    }

    function renderModels() {
      const deck = $("#model-deck");
      if (!deck) return;
      const rows = usableModels();
      if (!rows.length) {
        deck.innerHTML = '<p class="model-deck__empty">还没有训好的模型，先去「教它」训一版</p>';
        return;
      }
      deck.innerHTML = "";
      rows.forEach((run) => {
        const cfg = run.config || {};
        const model = modelLabel(cfg.model);
        const val = run.final ? run.final.val_acc : undefined;
        const deployable = EXPORTABLE.includes(String(cfg.model || ""));
        const tile = document.createElement("article");
        tile.className = "model-tile" + (run.active ? " is-active" : "");
        tile.dataset.id = run.id;
        tile.title = run.id;
        tile.innerHTML = `
          <div class="model-tile__head">
            <b class="model-tile__name">${esc(wordOf(cfg))} · ${model}
              <small>${runLabel(run)} 训好</small></b>
            ${run.active ? '<span class="pill pill--lime">在用</span>' : ""}
          </div>
          <div class="model-tile__score">
            <b>${pct(val)}</b><span>它认对了<br>训练时没见过的题</span>
          </div>
          <div class="model-tile__meta">
            <span>耗时 ${fmtTime(run.elapsed_s)}</span>
            ${run.params_total ? `<span>${(run.params_total / 1000).toFixed(1)}k 参数</span>` : ""}
          </div>
          <div class="model-tile__actions">
            <button class="btn btn--primary btn--sm" type="button" data-act="apply"
                    ${run.active ? "disabled" : ""}>${run.active ? "正在使用" : "使用"}</button>
            <button class="btn btn--outline btn--sm" type="button" data-act="export"
                    ${deployable ? "" : 'disabled title="只有简单 / 复杂 CNN 能导出：head 依赖预训练底座，包会大 40 倍"'}>为桌宠下载</button>
            <button class="btn btn--ghost btn--sm model-tile__del" type="button" data-act="delete">删除</button>
          </div>`;
        deck.appendChild(tile);
      });
    }

    async function applyModel(run, btn) {
      setBusy(btn, true);
      try {
        await api(`/api/runs/${run.id}/apply`, { method: "POST" });
        await refreshActive();          // 顶部 chip、测一测的下拉都跟着换
        toast("换好了，去「喊一声试试」", "ok");
      } catch (err) {
        toast(err.message, "error");
        setBusy(btn, false);
      }
    }

    function downloadModel(run) {
      const link = document.createElement("a");
      link.href = `/api/runs/${run.id}/export`;
      link.download = "";
      document.body.appendChild(link);
      link.click();
      link.remove();
      toast("包已开始下载 · 桌宠里执行 ./run.sh pull 就能装上", "ok");
    }

    async function deleteModel(run, btn) {
      const cfg = run.config || {};
      const ok = await askConfirm({
        title: "删掉这个模型？",
        message: `${wordOf(cfg)} · ${String(cfg.model || "").toUpperCase()}`
          + `（${runLabel(run)} 训好，认对 ${pct(run.final && run.final.val_acc)}）`,
        hint: "权重、曲线和日志会一起删掉，不能恢复；想换回来只能重训一遍。",
        okText: "删掉它",
      });
      if (!ok) return;                  // 取消：什么都不做
      setBusy(btn, true);
      try {
        const data = await api(`/api/runs/${run.id}`, { method: "DELETE" });
        state.runs = data.runs || [];
        state.activeRun = data.active || "";
        if (state.selectedRun === run.id) {   // 训练台选中的正是它 → 收回待机
          state.selectedRun = "";
          state.detail = null;
          closeStream();
          showTrainPane("idle");
        }
        await refreshActive();
        toast("删掉了", "ok");
      } catch (err) {
        toast(err.message, "error");
        setBusy(btn, false);
      }
    }

    $("#model-deck").addEventListener("click", (event) => {
      const btn = event.target.closest("button[data-act]");
      if (!btn || btn.disabled) return;
      const tile = btn.closest(".model-tile");
      const run = tile && state.runs.find((r) => r.id === tile.dataset.id);
      if (!run) return;
      if (btn.dataset.act === "apply") applyModel(run, btn);
      if (btn.dataset.act === "export") downloadModel(run);
      if (btn.dataset.act === "delete") deleteModel(run, btn);
    });

    function renderAdvanced(detail) {
      const result = detail.result || {};
      const status = detail.status || {};
      const final = result.final || status.final || {};
      const history = detail.history || [];
      $("#metric-val").textContent = pct(final.val_acc);
      $("#metric-train").textContent = pct(final.train_acc);
      $("#metric-params").textContent = result.params_total
        ? (result.params_total / 1000).toFixed(1) + "k" : "--";
      $("#metric-time").textContent = fmtTime(result.elapsed_s || status.elapsed_s);
      if (state.chart) state.chart.setData(history);
      $("#chart-empty").hidden = history.length > 0;
      $("#run-log").textContent = detail.log || "（暂无日志）";
      const running = status.state === "running" || status.state === "queued";
      $("#btn-stop").disabled = !running;
      $("#btn-rerun").disabled = running;
      $("#btn-apply").disabled = !(status.state === "succeeded" && detail.has_ckpt) || detail.active;
      $("#btn-apply").textContent = detail.active ? "已是用着的模型" : "用这个模型";

      // 部署包支持两种 CNN（桌宠侧是纯 numpy 前向，head 要带 2.4 MB 底座）
      const deploy = $("#btn-deploy");
      if (deploy) {
        const deployable = EXPORTABLE.includes(String((detail.config || {}).model || ""));
        deploy.disabled = !(status.state === "succeeded" && detail.has_ckpt && deployable);
        deploy.textContent = deployable ? "部署到桌宠" : "只有 CNN 能导出";
        deploy.title = deployable ? "下载 ~60 KB 的模型包，桌宠一条命令装上"
          : "head 依赖预训练底座，包会大 40 倍；先用简单 / 复杂 CNN 训一版";
      }
      // 提示只在刚点过的那次运行上显示，切换运行就收起来
      const hint = $("#deploy-hint");
      if (hint) {
        hint.hidden = hint.dataset.run !== state.selectedRun;
      }
    }

    function showDone(detail) {
      const result = detail.result || {};
      const final = result.final || (detail.status || {}).final || {};
      const val = final.val_acc;
      $("#score").textContent = pct(val);
      let tip = "这是它训练时没见过的题";
      if (Number.isFinite(val)) {
        if (val >= 0.9) tip = "很棒，它基本都认对了 · 这些题它训练时没见过";
        else if (val >= 0.75) tip = "还不错，再多录几条会更稳 · 这些题它训练时没见过";
        else tip = "有点低，多录几条或者换个模型试试 · 这些题它训练时没见过";
      }
      $("#score-note").textContent = tip;
      showTrainPane("done");
      renderStageCard(detail);
    }

    function showFailed(detail) {
      $("#fail-log").textContent = (detail && detail.log) || "（没有日志）";
      showTrainPane("failed");
    }

    function closeStream() {
      if (state.es) {
        state.es.close();
        state.es = null;
      }
    }

    function setProgressText(row, epochs) {
      const done = Math.max(1, Math.min(epochs, row.epoch || 1));
      $("#progress-bar").style.width = `${Math.round((done / epochs) * 100)}%`;
      $("#progress-title").textContent = `正在学第 ${done} / ${epochs} 遍`;
      const now = performance.now();
      if (!state.epochStart) state.epochStart = now;
      const per = (now - state.epochStart) / done;
      const left = per * (epochs - done);
      $("#progress-eta").textContent = done >= epochs
        ? "快好了…"
        : left < 1200 ? "马上就好" : `大约还要 ${Math.round(left / 1000)} 秒`;
    }

    function subscribe(runId) {
      closeStream();
      state.epochStart = 0;
      const es = new EventSource(`/api/runs/${runId}/stream`);
      state.es = es;
      es.addEventListener("epoch", (event) => {
        let row = null;
        try { row = JSON.parse(event.data); } catch { return; }
        const epochs = Number((state.detail && state.detail.config && state.detail.config.epochs) || 30);
        showTrainPane("running");
        setPrep(false);
        setProgressText(row, epochs);
        if (state.stageChart) state.stageChart.push(row);
        $("#result-model").innerHTML = modelBadge((state.detail && state.detail.config) || {});
        $("#result-meta").textContent = `学了 ${row.epoch} / ${epochs} 遍`;
        $("#result-when").textContent = "正在一遍一遍地学，曲线会自己长出来";
        syncResultCard();
        if (state.chart) state.chart.push(row);
        $("#chart-empty").hidden = true;
        $("#metric-val").textContent = pct(row.val_acc);
        $("#metric-train").textContent = pct(row.train_acc);
      });
      es.addEventListener("status", async (event) => {
        let status = {};
        try { status = JSON.parse(event.data); } catch { /* noop */ }
        closeStream();
        await loadRuns();
        let detail = null;
        try { detail = await api(`/api/runs/${runId}`); } catch { /* noop */ }
        if (detail) {
          state.detail = detail;
          renderAdvanced(detail);
        }
        if (status.state === "succeeded" && detail && detail.has_ckpt) {
          // 训完就自动「用这个模型」：孩子不需要理解"应用"这一步
          try {
            await api(`/api/runs/${runId}/apply`, { method: "POST" });
            await refreshActive();
            toast("训好了，已经换成这个模型", "ok");
          } catch (err) { toast(err.message, "error"); }
        }
        if (status.state === "succeeded") showDone(detail || { result: status });
        else if (status.state === "failed") { showFailed(detail); toast("这次没学会，日志展开看看", "error"); }
        else if (status.state === "stopped") { showTrainPane("idle"); toast("已经停下", ""); }
        else if (status.state === "interrupted") { showFailed(detail); }
      });
      es.onerror = () => { /* 断线由浏览器重连；结束后服务端关闭、我们在 status 里主动 close */ };
    }

    async function selectRun(runId) {
      if (!runId) return;
      closeStream();
      state.selectedRun = runId;
      try {
        const detail = await api(`/api/runs/${runId}`);
        state.detail = detail;
        renderAdvanced(detail);
        renderRuns();
        const st = (detail.status || {}).state;
        if (st === "running" || st === "queued") {
          showTrainPane("running");
          const hist = detail.history || [];
          setPrep(hist.length === 0);
          if (hist.length) setProgressText(hist[hist.length - 1],
            Number((detail.config || {}).epochs) || 30);
          renderStageCard(detail, { running: true });
          subscribe(runId);
        } else if (st === "succeeded") {
          showDone(detail);
        } else if (st === "failed" || st === "interrupted") {
          showFailed(detail);
        } else {
          showTrainPane("idle");
        }
      } catch (err) {
        toast(err.message, "error");
      }
    }

    async function loadRuns() {
      try {
        const data = await api("/api/runs");
        state.runs = data.runs || [];
        state.activeRun = data.active || state.activeRun;
        renderRuns();
        renderModelSwitch();
        renderModels();
        updateSuggestion();
      } catch (err) { toast(err.message, "error"); }
    }

    $("#refresh-runs").addEventListener("click", async () => {
      await loadRuns();
      if (state.selectedRun) await selectRun(state.selectedRun);
      toast("已刷新", "ok");
    });

    $("#btn-stop").addEventListener("click", async () => {
      if (!state.selectedRun) return;
      try {
        await api(`/api/runs/${state.selectedRun}/stop`, { method: "POST" });
        toast("让它先停下", "");
      } catch (err) { toast(err.message, "error"); }
    });

    async function rerun() {
      if (!state.selectedRun) return;
      // 模型卡、滑杆、数据集都按页面当前选择走：否则切到 CNN/切到自定义词再点
      // "再学一遍"，训的还是旧 run 的 HEAD + 默认词
      const model = ($("#model-grid input:checked") || {}).value || "cnn";
      const body = {
        model,
        max_pos: Number($("#max-pos").value),
        max_neg: Number($("#max-neg").value),
        wake_word_id: state.dataset || null,
        default_word: !state.dataset,
      };
      const word = state.dataset ? wordTextOf(state.dataset) : state.wakeWord;
      setBusy($("#btn-rerun"), true);
      try {
        const res = await api(`/api/runs/${state.selectedRun}/rerun`,
          { method: "POST", body });
        toast(`用「${word}」的 ${model.toUpperCase()} 再教一遍`, "ok");
        await loadRuns();
        await selectRun(res.id);
      } catch (err) {
        toast(err.message, "error");
      } finally {
        setBusy($("#btn-rerun"), false);
      }
    }
    $("#btn-rerun").addEventListener("click", rerun);
    $("#btn-again").addEventListener("click", rerun);
    $("#btn-again-fail").addEventListener("click", startTrain);

    $("#btn-apply").addEventListener("click", async () => {
      if (!state.selectedRun) return;
      const btn = $("#btn-apply");
      setBusy(btn, true);
      try {
        await api(`/api/runs/${state.selectedRun}/apply`, { method: "POST" });
        await refreshActive();
        if (state.detail) renderAdvanced({ ...state.detail, active: true });
        toast("换好了，喊一声试试", "ok");
      } catch (err) {
        toast(err.message, "error");
      } finally {
        setBusy(btn, false);
      }
    });

    $("#btn-deploy").addEventListener("click", () => {
      if (!state.selectedRun) return;
      const link = document.createElement("a");
      link.href = `/api/runs/${state.selectedRun}/export`;
      link.download = "";
      document.body.appendChild(link);
      link.click();
      link.remove();
      const hint = $("#deploy-hint");
      hint.hidden = false;
      hint.dataset.run = state.selectedRun;
      hint.textContent = "包已开始下载。桌宠那边一条命令就能装：./run.sh pull";
    });

    $("#chart-mode").addEventListener("click", (event) => {
      const btn = event.target.closest("button[data-mode]");
      if (!btn) return;
      $$("#chart-mode button").forEach((b) => b.classList.toggle("is-active", b === btn));
      state.mode_ = btn.dataset.mode;
      if (state.chart) state.chart.setMode(state.mode_);
    });

    $("#go-check").addEventListener("click", () => setStep("check"));
    $("#train-back").addEventListener("click", () => setStep("record"));

    /* -------------------------------------------------- 喊一声试试 */
    function renderHits() {
      const box = $("#hits");
      if (!box) return;
      if (!state.hits.length) {
        box.innerHTML = '<span class="hits__empty">最近命中会显示在这里</span>';
        return;
      }
      box.innerHTML = "";
      state.hits.forEach((hit) => {
        const row = document.createElement("div");
        row.className = "hit";
        row.innerHTML = `
          <span class="hit__time">${hit.time}</span>
          <span class="hit__bar"><i style="width:${Math.round(hit.p * 100)}%"></i></span>
          <span class="hit__p">${Math.round(hit.p * 100)}%</span>`;
        box.appendChild(row);
      });
    }

    function setListeningUI(on) {
      const stage = $("#mic-stage");
      stage.classList.toggle("is-listening", on);
      if (!on) stage.classList.remove("is-hit");
      $("#mic-btn").setAttribute("aria-label", on ? "停止聆听" : "开始聆听");
      if (!on) $("#mic-status").textContent = "待机 · 点一下开始";
    }

    function stopListening() {
      state.micToken += 1;              // 取消还在等权限的启动
      state.listeningPending = false;
      clearTimeout(state.noHitTimer);
      state.noHitTimer = 0;
      if (state.hitTimer) {
        clearInterval(state.hitTimer);
        state.hitTimer = 0;
      }
      cancelAnimationFrame(state.raf);
      state.raf = 0;
      if (state.segmenter) {
        state.segmenter.stop();
        state.segmenter = null;
      }
      state.listening = false;
      setListeningUI(false);
      const canvas = $("#test-wave");
      if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }

    function onHit(p) {
      if (!state.listening) return;
      state.hitOnce = true;
      clearTimeout(state.noHitTimer);
      state.noHitTimer = 0;
      state.hits.unshift({ p, time: new Date().toLocaleTimeString("zh-CN", { hour12: false }) });
      state.hits = state.hits.slice(0, 5);
      renderHits();

      const stage = $("#mic-stage");
      stage.classList.add("is-hit");
      const status = $("#mic-status");
      status.innerHTML = `听到啦！<b>${currentWordText()}</b> · ${Math.round(p * 100)}%`;
      if (state.segmenter) state.segmenter.paused = true;

      let left = 2;
      if (state.hitTimer) clearInterval(state.hitTimer);
      state.hitTimer = setInterval(() => {
        if (!state.listening) {
          clearInterval(state.hitTimer);
          state.hitTimer = 0;
          return;
        }
        left -= 1;
        if (left <= 0) {
          clearInterval(state.hitTimer);
          state.hitTimer = 0;
          stage.classList.remove("is-hit");
          status.textContent = "待机 · 继续聆听";
          state.hitOnce = false;
          if (state.segmenter) state.segmenter.paused = false;
        } else {
          status.innerHTML = `听到啦！<b>${currentWordText()}</b> · ${left}s 后继续`;
        }
      }, 1000);
    }

    async function startListening() {
      if (!window.AudioKit) { toast("录音脚本没加载，刷新试试", "error"); return; }
      if (state.listening || state.listeningPending) return;
      if (!state.activeRun) {
        toast("还没有模型，先去「教它」一次", "error");
        setStep("train");
        return;
      }
      state.hitOnce = false;
      const token = ++state.micToken;
      state.listeningPending = true;
      const status = $("#mic-status");
      status.textContent = "正在请求麦克风权限…";
      const seg = new window.AudioKit.Segmenter({
        windowSeconds: 1.0,
        hopMs: 300,
        onSegment: async (wav) => {
          try {
            const res = await upload("/api/predict", wav, "probe.wav");
            if (res.p_wake >= state.threshold) onHit(res.p_wake);
          } catch (err) {
            const message = String(err.message);
            if (message.includes("还没有应用")) {
              stopListening();
              $("#mic-status").textContent = "还没有模型 · 先去「教它」一次";
            } else if (message.includes("不兼容") || message.includes("底座")) {
              stopListening();
              $("#mic-status").textContent = message;
              toast(message, "error");
            }
          }
        },
      });
      try {
        await seg.start();
      } catch (err) {
        if (token !== state.micToken) return;
        state.listeningPending = false;
        const raw = err && err.message ? err.message : "";
        const message = /permission|denied|notallowed/i.test(raw)
          ? "浏览器没给麦克风权限：点地址栏的小锁，允许麦克风再试"
          : (raw || "无法打开麦克风");
        status.textContent = message;
        toast(message, "error");
        return;
      }
      if (token !== state.micToken) {       // 等权限期间用户已点击停止
        seg.stop();
        return;
      }
      state.listeningPending = false;
      state.segmenter = seg;
      state.listening = true;
      setListeningUI(true);
      $("#mic-status").textContent = `在听… 说「${currentWordText()}」`;

      const canvas = $("#test-wave");
      const draw = () => {
        if (!state.listening || !state.segmenter) return;
        window.AudioKit.drawWaveform(canvas, state.segmenter.samples(1400), {
          color: $("#mic-stage").classList.contains("is-hit") ? "#a3e635" : "#22d3ee",
          bars: 110,
        });
        state.raf = requestAnimationFrame(draw);
      };
      draw();
    }

    $("#mic-btn").addEventListener("click", () => {
      if (state.listening || state.listeningPending) stopListening();
      else startListening();
    });

    $("#check-retry").addEventListener("click", () => {
      if (state.listening) stopListening();
      startListening();
    });
    $("#check-retrain").addEventListener("click", () => { stopListening(); setStep("train"); });

    const threshold = $("#threshold");
    threshold.value = String(state.threshold);
    $("#threshold-val").textContent = state.threshold.toFixed(2);
    threshold.addEventListener("input", () => {
      state.threshold = Number(threshold.value);
      $("#threshold-val").textContent = state.threshold.toFixed(2);
      localStorage.setItem("nova.threshold", String(state.threshold));
    });

    /* -------------------------------------------------- 建议条 */
    function computeSuggestion() {
      const c = state.counts || { pos: 0, neg: 0 };
      if (!state.activeRun) return { text: "先教它一次，它才会听你的", step: "train" };
      if (c.pos < state.minPos * 2 || c.neg < state.minNeg * 2) {
        return { text: "多录几条，它会听得更准", step: "record" };
      }
      const last = state.runs.find((r) => r.state === "succeeded");
      const day = (last && last.created_at ? last.created_at : "").slice(0, 10);
      if (day !== todayKey()) return { text: "教它认识今天的声音", step: "train" };
      return { text: "喊一声试试，看它认不认得你", step: "check" };
    }

    function updateSuggestion() {
      const bar = $("#tipbar");
      if (!bar) return;
      if (localStorage.getItem("nova.tip.off") === todayKey()) { bar.hidden = true; return; }
      const tip = computeSuggestion();
      const go = $("#tip-go");
      go.dataset.step = tip.step;
      // 建议指向的就是当前这一间 → 页面自己已经说清楚了，不再加一条
      go.hidden = false;
      bar.hidden = tip.step === state.step;
      $("#tip-text").textContent = tip.text;
    }

    $("#tip-go").addEventListener("click", (e) => setStep(e.currentTarget.dataset.step));
    $("#tip-close").addEventListener("click", () => {
      localStorage.setItem("nova.tip.off", todayKey());
      $("#tipbar").hidden = true;
    });

    /* -------------------------------------------------- 进阶折叠 */
    $$("details.adv").forEach((d) => d.addEventListener("toggle", () => {
      if (!d.open) return;
      if (d.querySelector("#chart") && state.chart) setTimeout(() => state.chart.resize(), 0);
      if (d.querySelector("#synth-voice")) loadVoices();
    }));

    // 负样本提示词里如果正好是当前唤醒词（自定义词叫"我在举手"这种），先藏起来
    function filterNegWords() {
      const word = currentWordText();
      $$("#neg-words li").forEach((li) => {
        li.hidden = li.textContent.trim() === word;
      });
    }

    /* -------------------------------------------------- 首屏 */
    (async function startup() {
      paintActive();
      await refreshActive();                 // 拿到 active 模型的唤醒词（自定义词要显示它）
      filterNegWords();
      await loadCorpus();
      await loadWords();
      await loadRuns();
      if (state.activeRun) await selectRun(state.activeRun);
      else if (state.runs.length) await selectRun(state.runs[0].id);

      const fromUrl = new URLSearchParams(window.location.search).get("step");
      let step = fromUrl;
      if (!STEPS.includes(step)) {
        step = localStorage.getItem("nova.step") || boot.dataset.step;
      }
      setStep(STEPS.includes(step) ? step : "train", { remember: !!fromUrl });
    })();
  }

  /* ------------------------------------------------------------ 启动 */
  const bootEl = $("#boot");
  if (bootEl) initStudio(bootEl);
})();
