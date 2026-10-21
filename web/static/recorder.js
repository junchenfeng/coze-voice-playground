/* 浏览器录音 → 16k / 16bit / 单声道 WAV
 *
 * 容器里没有 ffmpeg，MediaRecorder 的 webm/opus 服务端解不了，所以这里直接抓
 * PCM：ScriptProcessor 拿原生采样率的 Float32，再用 OfflineAudioContext 重采样
 * 到 16k，最后手写 44 字节 WAV 头。44.1k / 48k 设备都覆盖。
 */
(() => {
  const TARGET_SR = 16000;
  const ENERGY_GATE = 0.008;        // RMS 低于这个当静音，不上传
  // 与 web/settings.py 的 MIN_CLIP_S / MIN_CLIP_RMS_DB / MIN_VOICED_S / MAX_CLIP_PCT
  // 一一对应：前端先拦一道（省一次上传、错因说得更具体），服务端仍是最终裁判。
  const GATES = {
    minSeconds: 0.3,
    minRmsDb: -45.0,
    minVoicedS: 0.25,
    maxClipPct: 1.0,
    voiceGateDb: -38.0,             // trainer/data.py 的 VOICE_GATE_DB
  };

  async function getStream() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      throw new Error("浏览器不支持麦克风采集（需要 https 或 localhost）");
    }
    const tuned = {
      channelCount: 1,
      echoCancellation: false,
      noiseSuppression: false,
      autoGainControl: false,
    };
    try {
      return await navigator.mediaDevices.getUserMedia({ audio: tuned });
    } catch (err) {
      // 个别浏览器不认 channelCount 之类的约束，退回默认麦克风（仍是取第一声道）
      if (err && (err.name === "OverconstrainedError" || err.name === "TypeError")) {
        return navigator.mediaDevices.getUserMedia({ audio: true });
      }
      throw err;
    }
  }

  function concatFloat(chunks, total) {
    const out = new Float32Array(total);
    let offset = 0;
    for (const c of chunks) {
      out.set(c, offset);
      offset += c.length;
    }
    return out;
  }

  async function resampleTo16k(input, fromRate) {
    if (fromRate === TARGET_SR) return input;
    const Offline = window.OfflineAudioContext || window.webkitOfflineAudioContext;
    if (!Offline) throw new Error("浏览器不支持 OfflineAudioContext，无法重采样");
    const len = Math.max(1, Math.round((input.length * TARGET_SR) / fromRate));
    const ctx = new Offline(1, len, TARGET_SR);
    const buf = ctx.createBuffer(1, input.length, fromRate);
    buf.copyToChannel(input, 0);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start();
    const rendered = await ctx.startRendering();
    return rendered.getChannelData(0);
  }

  function encodeWav(samples, sampleRate = TARGET_SR) {
    const bytes = samples.length * 2;
    const buffer = new ArrayBuffer(44 + bytes);
    const view = new DataView(buffer);
    const writeStr = (offset, str) => {
      for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
    };
    writeStr(0, "RIFF");
    view.setUint32(4, 36 + bytes, true);
    writeStr(8, "WAVE");
    writeStr(12, "fmt ");
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);            // PCM
    view.setUint16(22, 1, true);            // mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeStr(36, "data");
    view.setUint32(40, bytes, true);
    let off = 44;
    for (let i = 0; i < samples.length; i++, off += 2) {
      const s = Math.max(-1, Math.min(1, samples[i]));
      view.setInt16(off, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    }
    return new Blob([buffer], { type: "audio/wav" });
  }

  function rms(samples) {
    if (!samples || !samples.length) return 0;
    let sum = 0;
    for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / samples.length);
  }

  /* 与 trainer.data.speech_stats 同口径：25 ms 窗 / 10 ms 跳，帧 RMS 超过
   * -38 dBFS 算"在说话"。前端复算一遍是为了在录音结束的瞬间就能告诉孩子
   * 哪里不对，而不是上传绕一圈再被服务端退回。 */
  function analyze(samples, sampleRate = TARGET_SR) {
    const empty = { seconds: 0, rms: 0, rms_db: -120, voiced_s: 0, clip_pct: 0 };
    if (!samples || !samples.length) return empty;
    const n = samples.length;
    const seconds = n / sampleRate;
    const rmsValue = rms(samples);
    const rmsDb = 20 * Math.log10(Math.max(rmsValue, 1e-9));

    let clipped = 0;
    for (let i = 0; i < n; i++) if (Math.abs(samples[i]) >= 0.999) clipped += 1;

    const win = Math.round(sampleRate * 0.025);
    const hop = Math.max(1, Math.round(sampleRate * 0.01));
    const gate = Math.pow(10, GATES.voiceGateDb / 20);
    let frames = 0;
    let voicedFrames = 0;
    if (n >= win) {
      for (let start = 0; start + win <= n; start += hop) {
        let sum = 0;
        for (let i = start; i < start + win; i++) sum += samples[i] * samples[i];
        frames += 1;
        if (Math.sqrt(sum / win) > gate) voicedFrames += 1;
      }
    }
    return {
      seconds,
      rms: rmsValue,
      rms_db: rmsDb,
      voiced_s: frames ? (voicedFrames / frames) * (hop / sampleRate) * frames : 0,
      clip_pct: (clipped / n) * 100,
    };
  }

  /* 前端围栏：返回 null 表示可以上传；否则给出给孩子看的标题 + 一句怎么做。
   * 判据与 web/audio.py 的 check_speech 一致，只是多一条"太短"。 */
  function auditClip(stats) {
    if (!stats) {
      return { title: "没录到声音", message: "这次没抓到麦克风数据，再按一次试试", hint: "" };
    }
    if (stats.seconds < GATES.minSeconds) {
      return {
        title: "录得太短了",
        message: `这次只有 ${stats.seconds.toFixed(2)} 秒，至少要说满 ${GATES.minSeconds} 秒。`,
        hint: "按住按钮，说完一整句再松手。",
      };
    }
    if (stats.rms_db < GATES.minRmsDb) {
      return {
        title: "没听到声音",
        message: "麦克风好像没收到声音，或者你离得太远了。",
        hint: "看看浏览器地址栏的麦克风权限有没有打开，再靠近一点重录。",
      };
    }
    if (stats.voiced_s < GATES.minVoicedS) {
      return {
        title: "只听到一点点音",
        message: `有效声音只有 ${stats.voiced_s.toFixed(2)} 秒，太短了听不清。`,
        hint: "按下去之后先等一下，完整说一句再松手。",
      };
    }
    if (stats.clip_pct > GATES.maxClipPct) {
      return {
        title: "声音爆掉了",
        message: `有 ${stats.clip_pct.toFixed(1)}% 的声音贴到了上限，听起来会破。`,
        hint: "离麦克风太近了，退后半臂再录一遍。",
      };
    }
    return null;
  }

  /* getUserMedia 的报错是英文 DOMException，直接弹给孩子看不懂。 */
  function micErrorMessage(err) {
    const name = (err && err.name) || "";
    if (name === "NotAllowedError" || name === "PermissionDeniedError" || name === "SecurityError") {
      return {
        title: "麦克风被拦住了",
        message: "浏览器没有允许这个网页使用麦克风。",
        hint: "点地址栏左边的锁 / 麦克风图标，把权限改成「允许」，然后刷新页面。",
      };
    }
    if (name === "NotFoundError" || name === "DevicesNotFoundError") {
      return {
        title: "没找到麦克风",
        message: "这台电脑 / 平板上没有可用的录音设备。",
        hint: "插上耳机麦克风，或换一台有麦克风的设备再试。",
      };
    }
    if (name === "NotReadableError" || name === "TrackStartError") {
      return {
        title: "麦克风被占用了",
        message: "另一个程序正在用麦克风（会议软件、录音机之类）。",
        hint: "关掉那个程序，回来再按一次。",
      };
    }
    if (name === "OverconstrainedError" || name === "ConstraintNotSatisfiedError") {
      return {
        title: "麦克风不支持这个设置",
        message: "设备拒绝了 16 kHz 单声道的录音参数。",
        hint: "换一个麦克风或浏览器再试。",
      };
    }
    if (/https|localhost/i.test((err && err.message) || "")) {
      return {
        title: "需要安全连接",
        message: "浏览器只在 https 页面里给麦克风权限。",
        hint: "把网址开头的 http 换成 https，或者从学校给的正式链接进入。",
      };
    }
    return {
      title: "打不开麦克风",
      message: (err && err.message) || "浏览器没有给出原因。",
      hint: "刷新页面再试；还不行就换 Chrome / Safari 最新版。",
    };
  }

  function drawWaveform(canvas, samples, { color = "#22d3ee", bars = 96 } = {}) {
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = Math.max(60, rect.width || canvas.clientWidth || 240);
    const h = Math.max(40, rect.height || canvas.clientHeight || 64);
    if (canvas.width !== Math.round(w * dpr)) canvas.width = Math.round(w * dpr);
    if (canvas.height !== Math.round(h * dpr)) canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(148,163,184,0.16)";
    ctx.beginPath();
    ctx.moveTo(0, h / 2 + 0.5);
    ctx.lineTo(w, h / 2 + 0.5);
    ctx.stroke();

    if (!samples || !samples.length) return;
    const step = Math.max(1, Math.floor(samples.length / bars));
    const barW = Math.max(1.5, (w / bars) * 0.55);
    ctx.fillStyle = color;
    ctx.shadowColor = color;
    ctx.shadowBlur = 8;
    for (let i = 0; i < bars; i++) {
      let peak = 0;
      const start = i * step;
      for (let j = 0; j < step; j++) {
        const v = Math.abs(samples[start + j] || 0);
        if (v > peak) peak = v;
      }
      const bh = Math.max(1.5, Math.min(1, peak * 3.2) * (h - 8));
      const x = (i + 0.5) * (w / bars) - barW / 2;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(x, h / 2 - bh / 2, barW, bh, barW / 2)
                    : ctx.rect(x, h / 2 - bh / 2, barW, bh);
      ctx.fill();
    }
    ctx.shadowBlur = 0;
  }

  class Recorder {
    constructor() {
      this.ctx = null;
      this.stream = null;
      this.node = null;
      this.chunks = [];
      this.count = 0;
      this.onFrame = null;
    }

    get recording() { return !!this.ctx; }
    get seconds() { return this.ctx ? this.count / this.ctx.sampleRate : 0; }

    async start(onFrame) {
      if (this.ctx) return;
      this.stream = await getStream();
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx();
      if (this.ctx.state === "suspended") await this.ctx.resume();
      this.onFrame = onFrame || null;
      this.chunks = [];
      this.count = 0;

      const src = this.ctx.createMediaStreamSource(this.stream);
      this.node = this.ctx.createScriptProcessor(4096, 1, 1);
      this.node.onaudioprocess = (event) => {
        if (!this.ctx) return;
        const copy = new Float32Array(event.inputBuffer.getChannelData(0));
        this.chunks.push(copy);
        this.count += copy.length;
        if (this.onFrame) this.onFrame(copy);
      };
      // 必须接到 destination 才会持续触发 onaudioprocess；用 0 增益避免回声
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      src.connect(this.node);
      this.node.connect(mute);
      mute.connect(this.ctx.destination);
      this._mute = mute;
      this._src = src;
    }

    _teardown() {
      try { if (this.node) this.node.onaudioprocess = null; } catch (e) { /* noop */ }
      try { this._src && this._src.disconnect(); } catch (e) { /* noop */ }
      try { this.node && this.node.disconnect(); } catch (e) { /* noop */ }
      try { this._mute && this._mute.disconnect(); } catch (e) { /* noop */ }
      try {
        if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      } catch (e) { /* noop */ }
      try { this.ctx && this.ctx.close(); } catch (e) { /* noop */ }
      this.ctx = null;
      this.stream = null;
      this.node = null;
      this._src = null;
      this._mute = null;
    }

    async stop() {
      if (!this.ctx) return null;
      const rate = this.ctx.sampleRate;
      const data = concatFloat(this.chunks, this.count);
      this._teardown();
      if (!data.length) return null;
      const pcm = await resampleTo16k(data, rate);
      return { wav: encodeWav(pcm), pcm, duration: pcm.length / TARGET_SR,
               stats: analyze(pcm) };
    }

    cancel() {
      this.chunks = [];
      this.count = 0;
      this._teardown();
    }
  }

  class Segmenter {
    constructor({ onSegment, onLevel, windowSeconds = 1.0, hopMs = 300 } = {}) {
      this.onSegment = onSegment || null;
      this.onLevel = onLevel || null;
      this.windowSeconds = windowSeconds;
      this.hopMs = hopMs;
      this.paused = false;
      this.busy = false;
      this.ctx = null;
      this.timer = null;
      this.filled = 0;
      this.write = 0;
    }

    get running() { return !!this.ctx; }

    async start() {
      if (this.ctx) return;
      this.stream = await getStream();
      const Ctx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new Ctx();
      if (this.ctx.state === "suspended") await this.ctx.resume();
      this.rate = this.ctx.sampleRate;
      this.ring = new Float32Array(Math.round(this.rate * Math.max(1.5, this.windowSeconds + 0.5)));
      this.write = 0;
      this.filled = 0;

      const src = this.ctx.createMediaStreamSource(this.stream);
      const node = this.ctx.createScriptProcessor(4096, 1, 1);
      node.onaudioprocess = (event) => {
        if (!this.ctx) return;
        const chunk = event.inputBuffer.getChannelData(0);
        for (let i = 0; i < chunk.length; i++) {
          this.ring[this.write] = chunk[i];
          this.write = (this.write + 1) % this.ring.length;
        }
        this.filled = Math.min(this.ring.length, this.filled + chunk.length);
      };
      const mute = this.ctx.createGain();
      mute.gain.value = 0;
      src.connect(node);
      node.connect(mute);
      mute.connect(this.ctx.destination);

      this._src = src;
      this._node = node;
      this._mute = mute;
      this.timer = setInterval(() => this._tick(), this.hopMs);
    }

    samples(n = 1200) {
      if (!this.ring) return new Float32Array(0);
      const count = Math.min(n, this.filled);
      const out = new Float32Array(count);
      let start = this.write - count;
      if (start < 0) start += this.ring.length;
      for (let i = 0; i < count; i++) out[i] = this.ring[(start + i) % this.ring.length];
      return out;
    }

    async _tick() {
      if (!this.ctx || this.paused || this.busy) return;
      const want = Math.round(this.rate * this.windowSeconds);
      if (this.filled < want) return;
      const slice = this.samples(want);
      const level = rms(slice);
      if (this.onLevel) this.onLevel(level);
      if (level < ENERGY_GATE) return;

      this.busy = true;
      try {
        const pcm = await resampleTo16k(slice, this.rate);
        if (this.onSegment) await this.onSegment(encodeWav(pcm), pcm);
      } catch (err) {
        console.warn("segment failed", err);
      } finally {
        this.busy = false;
      }
    }

    stop() {
      if (this.timer) clearInterval(this.timer);
      this.timer = null;
      try { this._src && this._src.disconnect(); } catch (e) { /* noop */ }
      try { this._node && this._node.disconnect(); } catch (e) { /* noop */ }
      try { this._mute && this._mute.disconnect(); } catch (e) { /* noop */ }
      try {
        if (this.stream) this.stream.getTracks().forEach((t) => t.stop());
      } catch (e) { /* noop */ }
      try { this.ctx && this.ctx.close(); } catch (e) { /* noop */ }
      this.ctx = null;
      this.stream = null;
      this.ring = null;
      this.filled = 0;
    }
  }

  window.AudioKit = { TARGET_SR, ENERGY_GATE, GATES, encodeWav, resampleTo16k, rms,
                      analyze, auditClip, micErrorMessage,
                      drawWaveform, Recorder, Segmenter };
})();
