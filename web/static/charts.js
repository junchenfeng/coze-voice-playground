/* 训练曲线：纯 canvas 手绘，无外部图表库（ECS 上不依赖 CDN） */
(() => {
  const PAD = { l: 46, r: 16, t: 18, b: 28 };

  class MetricChart {
    constructor(canvas, { mode = "loss" } = {}) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      this.history = [];
      this.mode = mode;
      this.dpr = Math.min(window.devicePixelRatio || 1, 2);
      this._onResize = () => this.draw();
      window.addEventListener("resize", this._onResize);
      this.resize();
    }

    resize() {
      const rect = this.canvas.getBoundingClientRect();
      const w = Math.max(240, rect.width || this.canvas.clientWidth || 640);
      const h = Math.max(180, rect.height || this.canvas.clientHeight || 300);
      this.canvas.width = Math.round(w * this.dpr);
      this.canvas.height = Math.round(h * this.dpr);
      this.w = w;
      this.h = h;
      this.draw();
    }

    setMode(mode) {
      this.mode = mode;
      this.draw();
    }

    setData(history) {
      this.history = Array.isArray(history) ? history.slice() : [];
      this.draw();
    }

    push(row) {
      if (!row || typeof row.epoch !== "number") return;
      const idx = this.history.findIndex((r) => r.epoch === row.epoch);
      if (idx >= 0) this.history[idx] = row;
      else this.history.push(row);
      this.draw();
    }

    clear() {
      this.history = [];
      this.draw();
    }

    _series() {
      if (this.mode === "acc") {
        return { a: "train_acc", b: "val_acc", fmt: (v) => `${Math.round(v * 100)}%`, min: 0, max: 1 };
      }
      return { a: "train_loss", b: "val_loss", fmt: (v) => v.toFixed(2), min: null, max: null };
    }

    draw() {
      const { ctx, w, h } = this;
      if (!ctx) return;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      const plotW = w - PAD.l - PAD.r;
      const plotH = h - PAD.t - PAD.b;
      const rows = this.history;
      const cfg = this._series();

      let lo = cfg.min;
      let hi = cfg.max;
      if (lo === null || hi === null) {
        const vals = [];
        rows.forEach((r) => {
          if (Number.isFinite(r[cfg.a])) vals.push(r[cfg.a]);
          if (Number.isFinite(r[cfg.b])) vals.push(r[cfg.b]);
        });
        if (!vals.length) { lo = 0; hi = 1; }
        else {
          lo = Math.min(...vals);
          hi = Math.max(...vals);
          const pad = Math.max((hi - lo) * 0.18, this.mode === "acc" ? 0.03 : 0.02);
          lo = Math.max(this.mode === "acc" ? 0 : 0, lo - pad);
          hi = hi + pad;
        }
      }
      if (hi - lo < 1e-6) hi = lo + 1;

      const x = (i) => PAD.l + (rows.length <= 1 ? plotW / 2 : (plotW * i) / (rows.length - 1));
      const y = (v) => PAD.t + plotH - ((v - lo) / (hi - lo)) * plotH;

      // 网格 + y 轴刻度
      ctx.strokeStyle = "rgba(148,163,184,0.12)";
      ctx.fillStyle = "rgba(100,116,139,0.9)";
      ctx.lineWidth = 1;
      ctx.font = "11px ui-monospace, Menlo, monospace";
      ctx.textAlign = "right";
      const ticks = 4;
      for (let i = 0; i <= ticks; i++) {
        const v = lo + ((hi - lo) * i) / ticks;
        const py = Math.round(y(v)) + 0.5;
        ctx.beginPath();
        ctx.moveTo(PAD.l, py);
        ctx.lineTo(w - PAD.r, py);
        ctx.stroke();
        ctx.fillText(cfg.fmt(v), PAD.l - 8, py + 3.5);
      }

      if (!rows.length) return;

      // x 轴 epoch 标签
      ctx.textAlign = "center";
      const step = Math.max(1, Math.ceil(rows.length / 6));
      for (let i = 0; i < rows.length; i += step) {
        ctx.fillText(String(rows[i].epoch), x(i), h - PAD.b + 16);
      }

      const line = (key, color, dashed) => {
        ctx.save();
        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        if (dashed) ctx.setLineDash([5, 5]);
        ctx.lineJoin = "round";
        let started = false;
        rows.forEach((r, i) => {
          const v = r[key];
          if (!Number.isFinite(v)) return;
          const px = x(i);
          const py = y(v);
          if (!started) { ctx.moveTo(px, py); started = true; }
          else ctx.lineTo(px, py);
        });
        ctx.stroke();
        ctx.restore();

        const last = [...rows].reverse().find((r) => Number.isFinite(r[key]));
        if (last) {
          const i = rows.lastIndexOf(last);
          ctx.beginPath();
          ctx.fillStyle = color;
          ctx.arc(x(i), y(last[key]), 3, 0, Math.PI * 2);
          ctx.fill();
        }
      };

      if (this.mode === "loss") {
        // train 曲线下面垫一层渐变
        const grad = ctx.createLinearGradient(0, PAD.t, 0, PAD.t + plotH);
        grad.addColorStop(0, "rgba(34,211,238,0.22)");
        grad.addColorStop(1, "rgba(34,211,238,0)");
        ctx.beginPath();
        let started = false;
        rows.forEach((r, i) => {
          if (!Number.isFinite(r.train_loss)) return;
          if (!started) { ctx.moveTo(x(i), y(r.train_loss)); started = true; }
          else ctx.lineTo(x(i), y(r.train_loss));
        });
        if (started) {
          const lastIdx = rows.length - 1;
          ctx.lineTo(x(lastIdx), PAD.t + plotH);
          ctx.lineTo(x(0), PAD.t + plotH);
          ctx.closePath();
          ctx.fillStyle = grad;
          ctx.fill();
        }
        line("train_loss", "#22d3ee", false);
        line("val_loss", "#a3e635", true);
      } else {
        line("train_acc", "#22d3ee", false);
        line("val_acc", "#a3e635", true);
      }
    }

    destroy() {
      window.removeEventListener("resize", this._onResize);
    }
  }

  window.MetricChart = MetricChart;
})();
