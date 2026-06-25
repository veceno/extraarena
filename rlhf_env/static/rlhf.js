// RLHF Arena — клиентский JS для index.html
// Грузит модели в p2_select (человек = P1, бот = P2).
// POST /api/groups (create_series) → редирект на redirect_url (/arena?...&_auth=...&ea_platform=android_app),
// где отдаётся настоящая 1:1 арена (arena.html + arena.js из webapp_borrow/).

const API = (path) => `/api${path}`;

const els = {
  p2Select: document.getElementById("p2_model"),
  form: document.getElementById("new-group-form"),
  error: document.getElementById("form-error"),
  versionSpan: document.getElementById("version"),
  // RLHF-логин
  identifier: document.getElementById("rlhf-identifier"),
  code: document.getElementById("rlhf-code"),
  sendCodeBtn: document.getElementById("rlhf-send-code"),
  loginBtn: document.getElementById("rlhf-login-btn"),
  logoutBtn: document.getElementById("rlhf-logout-btn"),
  loginMsg: document.getElementById("rlhf-login-msg"),
  userInfo: document.getElementById("rlhf-user-info"),
  importedDecks: document.getElementById("p1-imported-decks"),
  customBlock: document.getElementById("p1-custom-block"),
  // Звук (музыка / SFX) — применяется ко всем боям серии через redirect_url.
  musicChk: document.getElementById("rlhf-music"),
  sfxChk: document.getElementById("rlhf-sfx"),
};

// Состояние RLHF-логина (сессия хранится серверно в cookie rlhf_sid).
const rlhfState = { loggedIn: false, decks: [], maxDecks: 3, extraPass: false };

async function loadCardsCatalog() {
  // Кешируем каталог карт (имя/статы/картинка) для превью колод в UI.
  try {
    const r = await fetch(API("/cards"));
    const d = await r.json();
    const map = new Map();
    for (const c of (d.cards || [])) map.set(Number(c.id), c);
    rlhfState.cards = map;
  } catch (e) {
    rlhfState.cards = new Map();
  }
}
function cardById(id) { return rlhfState.cards && rlhfState.cards.get(Number(id)); }
function cardImg(id) {
  const n = Number(id);
  return `/DesignAssets/CardsPreview/w384/${n}.webp`;
}

function rlhfSetMsg(msg, isErr) {
  els.loginMsg.textContent = msg || "";
  els.loginMsg.style.color = isErr ? "#a33" : "#566";
}

async function loadModels() {
  try {
    const r = await fetch(API("/registry/models"));
    const data = await r.json();
    const models = data.models || [];
    const groups = { action_onnx: "── ONNX action-conditioned (V4) ──",
                     legacy_onnx: "── ONNX legacy (V2/V3) ──" };
    let lastGroup = null;
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      if (groups[m.kind] && groups[m.kind] !== lastGroup) {
        const sep = document.createElement("option");
        sep.disabled = true;
        sep.textContent = groups[m.kind];
        els.p2Select.appendChild(sep);
        lastGroup = groups[m.kind];
      } else if (!groups[m.kind] && (m.kind === "random" || m.kind === "greedy_face" || m.kind === "end_turn")) {
        if (lastGroup !== "baselines") {
          const sep = document.createElement("option");
          sep.disabled = true;
          sep.textContent = "── Baselines ──";
          els.p2Select.appendChild(sep);
          lastGroup = "baselines";
        }
      }
      opt.textContent = `${m.name} [${m.kind}]`;
      els.p2Select.appendChild(opt);
    }
    // Выбор по умолчанию для человека — Extra-LR-V4-Max (модель max-возможностей).
    // Раньше дефолтом был end_turn (baseline «сдаться ходом»); теперь — сильнейшая
    // V4. Если её нет в реестре — оставляем fallback end_turn.
    const DEFAULT_MODEL = "extra-lr-v4-max";
    if ([...els.p2Select.options].some((o) => o.value === DEFAULT_MODEL)) {
      els.p2Select.value = DEFAULT_MODEL;
    } else {
      els.p2Select.value = "end_turn";
    }
  } catch (e) {
    console.warn("loadModels failed:", e);
  }
}

async function loadVersion() {
  try {
    const r = await fetch("/health");
    const d = await r.json();
    els.versionSpan.textContent = d.version || "0.1.0";
  } catch (e) {
    els.versionSpan.textContent = "0.1.0";
  }
}

// ---- Звук: музыка / SFX (выбор в главном меню, persist в localStorage) --------
// Арена (arena.js initArenaMusic) читает ?music=0/?sfx=0 из redirect_url; сервер
// подставляет эти параметры в каждый бой серии из spec.audio. localStorage лишь
// хранит выбор человека между заходами в меню.
function restoreAudioPrefs() {
  const m = localStorage.getItem("rlhf_music");
  const s = localStorage.getItem("rlhf_sfx");
  if (els.musicChk && m !== null) els.musicChk.checked = m === "1";
  if (els.sfxChk && s !== null) els.sfxChk.checked = s === "1";
}
function saveAudioPrefs() {
  if (els.musicChk) localStorage.setItem("rlhf_music", els.musicChk.checked ? "1" : "0");
  if (els.sfxChk) localStorage.setItem("rlhf_sfx", els.sfxChk.checked ? "1" : "0");
}
if (els.musicChk) els.musicChk.addEventListener("change", saveAudioPrefs);
if (els.sfxChk) els.sfxChk.addEventListener("change", saveAudioPrefs);

function showError(msg) {
  els.error.textContent = msg;
  els.error.style.display = "block";
}
function clearError() {
  els.error.textContent = "";
  els.error.style.display = "none";
}

// ---- RLHF: вход по коду + импорт колод -----------------------------------
function rlhfRenderImportedDecks(decks) {
  els.importedDecks.innerHTML = "";
  if (!decks || !decks.length) {
    const note = document.createElement("div");
    note.className = "rlhf-hint";
    note.textContent = "У вас нет сохранённых колод в игре.";
    els.importedDecks.appendChild(note);
    return;
  }
  decks.forEach((d) => {
    const row = document.createElement("div");
    row.className = "p1-imported-opt" + (d.is_playable ? "" : " unplayable");

    const radio = document.createElement("input");
    radio.type = "radio";
    radio.name = "p1_deck_source";
    radio.value = `imported:${d.preset_number}`;
    radio.id = `p1-imported-${d.preset_number}`;
    if (!d.is_playable) radio.disabled = true;

    const label = document.createElement("label");
    label.htmlFor = radio.id;
    const name = document.createElement("span");
    name.textContent = d.preset_name || `Колода ${d.preset_number}`;
    label.appendChild(name);
    if (d.is_primary) {
      const b = document.createElement("span"); b.className = "badge primary"; b.textContent = "★ основная";
      label.appendChild(b);
    }
    if (!d.is_playable) {
      const b = document.createElement("span"); b.className = "badge unplayable"; b.textContent = "недоступна";
      label.appendChild(b);
    }

    // Превью карт колоды (имя/статы/картинка) — read-only, не отправляется.
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "preview-toggle";
    toggle.textContent = "показать карты";
    const preview = document.createElement("div");
    preview.style.cssText = "display:none;flex-basis:100%;flex-wrap:wrap;gap:6px;margin-top:6px";
    toggle.addEventListener("click", () => {
      const open = preview.style.display === "flex";
      preview.style.display = open ? "none" : "flex";
      toggle.textContent = open ? "показать карты" : "скрыть карты";
      if (!preview.childElementCount) rlhfFillDeckPreview(preview, d.card_ids);
    });

    row.appendChild(radio);
    row.appendChild(label);
    row.appendChild(toggle);
    row.appendChild(preview);
    els.importedDecks.appendChild(row);
  });
}

function rlhfFillDeckPreview(container, cardIds) {
  (cardIds || []).forEach((id) => {
    const c = cardById(id) || {};
    const cell = document.createElement("div");
    cell.style.cssText = "width:64px;text-align:center;font-size:.75em";
    const img = document.createElement("img");
    img.src = cardImg(id);
    img.alt = c.name || `card ${id}`;
    img.style.cssText = "width:60px;height:80px;object-fit:cover;border-radius:6px";
    img.onerror = () => { img.src = `/DesignAssets/Cards/${Number(id)}.png`; img.onerror = () => { img.src = "/DesignAssets/Cards/9.png"; }; };
    const nm = document.createElement("div"); nm.textContent = c.name || `#${id}`;
    const st = document.createElement("div"); st.style.color = "#678";
    st.textContent = `⭐${c.mana_cost ?? 0} ⚔️${c.base_attack ?? 0} ❤️${c.base_hp ?? 0}`;
    cell.appendChild(img); cell.appendChild(nm); cell.appendChild(st);
    container.appendChild(cell);
  });
}

function rlhfShowLoggedIn(payload) {
  rlhfState.loggedIn = true;
  rlhfState.decks = payload.decks || [];
  rlhfState.maxDecks = payload.max_decks || 3;
  rlhfState.extraPass = !!payload.extra_pass_active;
  els.userInfo.style.display = "block";
  els.userInfo.textContent = `Войдено. Колод доступно: ${rlhfState.decks.length} (из ${rlhfState.maxDecks}${rlhfState.extraPass ? ", ExtraPass" : ""}).`;
  els.logoutBtn.style.display = "inline-block";
  rlhfRenderImportedDecks(rlhfState.decks);
  rlhfSetMsg("");
}

function rlhfShowLoggedOut() {
  rlhfState.loggedIn = false;
  rlhfState.decks = [];
  els.userInfo.style.display = "none";
  els.logoutBtn.style.display = "none";
  els.importedDecks.innerHTML = "";
  // если был выбран imported — откатимся на случайную
  const checked = document.querySelector('input[name="p1_deck_source"]:checked');
  if (checked && checked.value.startsWith("imported:")) {
    const rnd = document.querySelector('input[name="p1_deck_source"][value="random"]');
    if (rnd) rnd.checked = true;
    toggleP1CustomBlock();
  }
}

els.sendCodeBtn.addEventListener("click", async () => {
  const identifier = (els.identifier.value || "").trim();
  if (!identifier) { rlhfSetMsg("Введите Telegram ID или ExtraID.", true); return; }
  els.sendCodeBtn.disabled = true;
  rlhfSetMsg("Отправка кода…");
  try {
    const r = await fetch(API("/rlhf/request-code"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      rlhfSetMsg("Не удалось отправить код: " + (j.error || `HTTP ${r.status}`), true);
    } else {
      const hint = j.hint === "mail"
        ? "Код отправлен во внутриигровую почту («Меню» → «Почта»)."
        : "Код отправлен в Telegram-бота.";
      rlhfSetMsg(hint + " Действует 5 минут.");
    }
  } catch (e) {
    rlhfSetMsg("Сеть/прод недоступен: " + e.message, true);
  } finally {
    els.sendCodeBtn.disabled = false;
  }
});

els.loginBtn.addEventListener("click", async () => {
  const identifier = (els.identifier.value || "").trim();
  const code = (els.code.value || "").trim();
  if (!identifier || code.length !== 6) {
    rlhfSetMsg("Введите идентификатор и 6-значный код.", true); return;
  }
  els.loginBtn.disabled = true;
  rlhfSetMsg("Проверка кода…");
  try {
    const r = await fetch(API("/rlhf/verify"), {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ identifier, code }),
    });
    const j = await r.json().catch(() => ({}));
    if (!r.ok) {
      rlhfSetMsg("Вход не удался: " + (j.error || `HTTP ${r.status}`), true);
    } else {
      rlhfShowLoggedIn(j);
      els.code.value = "";
    }
  } catch (e) {
    rlhfSetMsg("Сеть/прод недоступен: " + e.message, true);
  } finally {
    els.loginBtn.disabled = false;
  }
});

els.logoutBtn.addEventListener("click", async () => {
  try { await fetch(API("/rlhf/logout"), { method: "POST" }); } catch (e) {}
  rlhfShowLoggedOut();
  rlhfSetMsg("Вы вышли.");
});

// Восстановление сессии после перезагрузки страницы.
async function rlhfRestoreSession() {
  try {
    const r = await fetch(API("/rlhf/me"));
    const j = await r.json();
    if (j && j.authenticated) rlhfShowLoggedIn(j);
  } catch (e) {}
}

// Переключение блока JSON-колоды P1 по radio.
function toggleP1CustomBlock() {
  const v = document.querySelector('input[name="p1_deck_source"]:checked');
  els.customBlock.style.display = (v && v.value === "custom") ? "block" : "none";
}
document.querySelectorAll('input[name="p1_deck_source"]').forEach((r) => {
  r.addEventListener("change", toggleP1CustomBlock);
});

// Переключение блоков custom-колоды для P1/P2.
document.querySelectorAll(".deck-strategy").forEach((sel) => {
  sel.addEventListener("change", () => {
    const side = sel.id.endsWith("p1") ? "p1" : "p2";
    const block = document.querySelector(`.custom-deck-block[data-side="${side}"]`);
    if (block) block.style.display = sel.value === "custom" ? "block" : "none";
  });
});

// Кнопки «Сгенерировать пример» для P1/P2.
document.querySelectorAll(".btn-sample-deck").forEach((btn) => {
  btn.addEventListener("click", async () => {
    const side = btn.dataset.side;
    const out = document.querySelector(`.sample-deck-out[data-side="${side}"]`);
    const ta = document.getElementById(`custom_deck_${side}`);
    try {
      const r = await fetch(API("/registry/sample-deck"));
      const data = await r.json();
      out.textContent = JSON.stringify(data.deck, null, 2);
      ta.value = JSON.stringify(data.deck);
    } catch (e) {
      out.textContent = "error: " + e;
    }
  });
});

function parseCustomDeck(side) {
  const ta = document.getElementById(`custom_deck_${side}`);
  const txt = (ta.value || "").trim();
  if (!txt) {
    return { ok: false, err: `Загрузите JSON-колоду для P${side === "p1" ? 1 : 2} или выберите «Случайная ArenaENV колода»` };
  }
  try {
    return { ok: true, deck: JSON.parse(txt) };
  } catch (err) {
    return { ok: false, err: `Невалидный JSON (P${side === "p1" ? 1 : 2}): ${err.message}` };
  }
}

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  const fd = new FormData(els.form);
  const spec = {
    p1_model: "human",          // sentinel — человек играет за P1
    p2_model: fd.get("p2_model"),
    // «Система сложностей» удалена: модель всегда играет на максимум (argmax).
    // Поле difficulty больше не отправляется — сервер фиксирует "max".
    deck_strategy_p2: fd.get("deck_strategy_p2"),
    battles_planned: parseInt(fd.get("battles_planned") || "3", 10),
    seed: parseInt(fd.get("seed") || "0", 10),
    starting_player: fd.get("starting_player"),
    max_turns: parseInt(fd.get("max_turns") || "60", 10),
    interactive: true,
    human_player: 1000,
    // Звук арены: музыка/SFX включаются/выключаются в меню (применяется ко всем
    // боям серии — сервер подставит &music=/&sfx= в каждый redirect_url).
    audio: {
      music: !!(els.musicChk && els.musicChk.checked),
      sfx: !!(els.sfxChk && els.sfxChk.checked),
    },
  };
  // P1: источник колоды — radio p1_deck_source (random | custom | imported:N).
  const p1src = document.querySelector('input[name="p1_deck_source"]:checked');
  const p1val = p1src ? p1src.value : "random";
  if (p1val === "custom") {
    spec.deck_strategy_p1 = "custom";
    spec.p1_deck_source = { type: "custom" };
    const r = parseCustomDeck("p1");
    if (!r.ok) { showError(r.err); return; }
    spec.custom_deck_p1 = r.deck;
  } else if (p1val.startsWith("imported:")) {
    if (!rlhfState.loggedIn) { showError("Сначала войдите, чтобы играть своей колодой."); return; }
    const presetNumber = parseInt(p1val.split(":")[1], 10);
    spec.deck_strategy_p1 = "imported";
    spec.p1_deck_source = { type: "imported", preset_number: presetNumber };
    // card_ids НЕ отправляем — сервер резолвит из сессии/прода и валидирует.
  } else {
    spec.deck_strategy_p1 = "random_arenaenv";
    spec.p1_deck_source = { type: "random" };
  }
  if (spec.deck_strategy_p2 === "custom") {
    const r = parseCustomDeck("p2");
    if (!r.ok) { showError(r.err); return; }
    spec.custom_deck_p2 = r.deck;
  }
  try {
    const r = await fetch(API("/groups"), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(spec),
    });
    if (!r.ok) {
      const j = await r.json().catch(() => ({}));
      throw new Error(j.error || `HTTP ${r.status}`);
    }
    const data = await r.json();
    if (data.redirect_url) {
      window.location.href = data.redirect_url;
    } else {
      throw new Error("API не вернул redirect_url");
    }
  } catch (err) {
    showError("Ошибка: " + err.message);
  }
});

loadModels();
loadVersion();
loadCardsCatalog();
rlhfRestoreSession();
restoreAudioPrefs();