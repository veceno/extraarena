// === ExtraOrchestra arena-bridge (path B) ===
// Загружается ПОСЛЕ arena.js (как classic-script), разделяет глобальное
// lexical-окружение: top-level `let` (userId, prebattleComplete, …) и function
// handleStateChanged из arena.js доступны по bare-имени.
//
// Обязанности:
//  1. window.__orchestraInit() — baked hook → prebattle-гейт снят (RISK C);
//  2. stub window.io (Socket.IO не нужен — мы гоним handleStateChanged сами);
//  3. fetch /api/orchestra/frames/<run_id> → userId=<viewer_uid> (RISK E) →
//     итерация кадров через handleStateChanged({state, sound_events}) с
//     await sleep(display_ms);
//  4. controls (play/pause/step/speed) через window.__orchestraController;
//     по завершении window.__orchestraDone=true (ждёт recorder).
(function () {
  "use strict";

  // 1. prebattle bypass
  try { if (typeof window.__orchestraInit === "function") window.__orchestraInit(); }
  catch (e) { console.warn("[ORCH] __orchestraInit failed", e); }

  // 1b. Снять prebattle-оверлей. __orchestraInit ставит prebattleComplete=true,
  // но НЕ скрывает #prebattle-screen (это делает startPrebattleSequence в конце
  // анимации, которую мы обходим). Без этого дефолтно-видимый оверлей висит
  // поверх поля — mp4 получался «5с preBattleScreen и ничего больше».
  function hidePrebattleOverlay() {
    try { if (typeof hidePrebattleScreen === "function") hidePrebattleScreen(); } catch (e) {}
    var el = document.getElementById("prebattle-screen");
    if (el) {
      el.setAttribute("aria-hidden", "true");
      el.classList.add("is-hidden");
    }
  }
  hidePrebattleOverlay();
  // повтор — после DOMContentLoaded (bridge может грузиться до построения DOM)
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", hidePrebattleOverlay, { once: true });
  }
  // и на всякий случай — гарантированный CSS-фолбэк
  try {
    var s = document.createElement("style");
    s.id = "orch-prebattle-hide";
    s.textContent = "#prebattle-screen{display:none !important;}";
    (document.head || document.documentElement).appendChild(s);
  } catch (e) {}

  // 1c. Мобильное соотношение сторон в предпросмотре. /player открывается как
  // popup (window.opener есть) в реальном браузере (editor Preview / MCP-хост).
  // Подгоняем окно под мобильный портрет (414×896), чтобы сработал
  // @media (max-width:420px) из arena-styles.css → мобильный лейаут арены.
  // В headless-рекордере window.opener===null (Playwright goto без opener),
  // плюс INIT_SCRIPT ставит ExtraArenaApp=true → здесь no-op (viewport уже
  // мобильный). resizeTo может блокироваться браузером — это best-effort.
  try {
    if (window.opener && !window.ExtraArenaApp && typeof window.resizeTo === "function") {
      window.resizeTo(414, 896);
    }
  } catch (e) {}

  // 2. socket.io stub (на случай, если add_init_script ещё не сработал)
  if (typeof window.io !== "function") {
    window.io = function () {
      var s = {
        on: function () { return this; }, off: function () { return this; },
        once: function () { return this; }, emit: function () { return this; },
        close: function () { return this; }, connect: function () { return this; },
        disconnect: function () { return this; },
        connected: true, id: "orch-stub",
        io: { on: function () { return this; }, off: function () { return this; } }
      };
      return s;
    };
  }

  window.__orchestraDone = false;
  window.__orchestraError = null;
  window.__orchestraController = null;
  window.__orchestraPlay = null;       // (speed?) => Promise — manual start
  window.__orchestraLoad = null;       // () => {frames, viewer_uid, ...}
  // Диагностика (top-level lets доступны по bare-имени — общее lexical-окружение).
  window.__orch_diag = function () {
    try {
      return {
        present: window.__orchestraPresent, init: typeof window.__orchestraInit,
        done: window.__orchestraDone, err: window.__orchestraError,
        prebattleComplete: prebattleComplete, prebattleRendered: prebattleRendered,
        prebattleSequenceStarted: prebattleSequenceStarted,
        userId: userId, matchId: matchId, authToken: authToken ? "set" : "null",
        ctl: window.__orchestraController && {
          frame: window.__orchestraController.frame, total: window.__orchestraController.total,
          playing: window.__orchestraController.playing
        }
      };
    } catch (e) { return { diagErr: String(e) }; }
  };

  var params = new URLSearchParams(location.search);
  var runId = params.get("id");
  var autoplay = params.get("autoplay") === "1";
  var speedParam = parseFloat(params.get("speed") || "1") || 1;

  var sleep = function (ms) { return new Promise(function (r) { setTimeout(r, ms); }); };

  function fetchFrames() {
    return fetch("/api/orchestra/frames/" + encodeURIComponent(runId))
      .then(function (r) {
        if (!r.ok) throw new Error("frames HTTP " + r.status);
        return r.json();
      });
  }

  function play(frames, viewerUid, speed) {
    // userId — top-level let из arena.js (общее lexical-окружение).
    try { userId = viewerUid; } catch (e) { /* arena.js let absent? */ }
    var ctl = {
      playing: true, speed: speed, frame: 0, total: frames.length,
      pause: function () { this.playing = false; },
      resume: function () { if (this.playing) return; this.playing = true; this._tick && this._tick(); },
      step: function () { this._step && this._step(); },
      seek: function (n) { this._seek && this._seek(n); },
      _tick: null, _step: null, _seek: null
    };
    window.__orchestraController = ctl;

    return new Promise(function (resolve) {
      var i = 0;
      var paused = false;

      function sendFrame(idx) {
        var f = frames[idx];
        if (!f) return;
        try {
          handleStateChanged({
            state: f.snapshot,
            sound_events: f.sound_events || [],
            data: { actor_user_id: viewerUid, sound_events: f.sound_events || [] }
          });
        } catch (e) { console.warn("[ORCH] handleStateChanged threw at frame", idx, e); }
      }

      function loop() {
        if (i >= frames.length) { window.__orchestraDone = true; resolve(); return; }
        if (!ctl.playing) { paused = true; return; }
        ctl.frame = i;
        sendFrame(i);
        var wait = Math.max(20, (frames[i].display_ms || 0) / (speed || 1));
        i++;
        ctl._tick = function () { if (ctl.playing && paused) { paused = false; loop(); } };
        ctl._step = function () { if (i < frames.length) { sendFrame(i); i++; } };
        ctl._seek = function (n) { i = Math.max(0, Math.min(frames.length - 1, n | 0)); };
        setTimeout(loop, wait);
      }
      loop();
    });
  }

  window.__orchestraLoad = function () { return fetchFrames(); };
  window.__orchestraPlay = function (s) { return fetchFrames().then(function (d) { return play(d.frames, d.viewer_uid, s || speedParam); }); };

  if (!runId) {
    window.__orchestraError = "no run id in URL (?id=...)";
    console.error("[ORCH]", window.__orchestraError);
    return;
  }

  if (autoplay) {
    fetchFrames().then(function (data) {
      if (data.error) { window.__orchestraError = data.error; console.error("[ORCH] run error:", data.error); return; }
      return play(data.frames, data.viewer_uid, speedParam);
    }).catch(function (e) {
      window.__orchestraError = String(e && e.message || e);
      console.error("[ORCH] play failed:", e);
      // даже при ошибке ставим done, чтобы recorder не завис
      window.__orchestraDone = true;
    });
  } else {
    // ручной режим: expose __orchestraPlay; пометим done-готовность после старта
    console.log("[ORCH] ready (autoplay=0). Call window.__orchestraPlay() to start.");
  }
})();