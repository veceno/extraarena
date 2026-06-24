// RLHF Arena — клиентский JS для index.html
// Загружает реестр моделей, отправляет форму на /api/groups, редиректит на /battle.

const API = (path) => `/api${path}`;

const els = {
  p1Select: document.getElementById("p1_model"),
  p2Select: document.getElementById("p2_model"),
  deckStrategy: document.getElementById("deck_strategy"),
  customBlock: document.getElementById("custom-deck-block"),
  customDeckTA: document.getElementById("custom_deck_p1"),
  sampleBtn: document.getElementById("btn-sample-deck"),
  sampleOut: document.getElementById("sample-deck-out"),
  form: document.getElementById("new-group-form"),
  error: document.getElementById("form-error"),
  versionSpan: document.getElementById("version"),
};

async function loadModels() {
  try {
    const r = await fetch(API("/registry/models"));
    const data = await r.json();
    const models = data.models || [];
    const groups = { action_onnx: "ONNX action-conditioned (V4)", legacy_onnx: "ONNX legacy (V2/V3)" };
    let lastGroup = null;
    for (const m of models) {
      const opt = document.createElement("option");
      opt.value = m.name;
      const group = groups[m.kind] || "Baseline";
      if (group !== lastGroup) {
        if (lastGroup !== null) {
          const sep = document.createElement("option");
          sep.disabled = true;
          sep.textContent = `── ${group} ──`;
          els.p1Select.appendChild(sep.cloneNode(true));
          els.p2Select.appendChild(sep);
        }
        lastGroup = group;
      }
      opt.textContent = `${m.name} [${m.kind}]`;
      els.p1Select.appendChild(opt);
      els.p2Select.appendChild(opt.cloneNode(true));
    }
  } catch (e) {
    console.warn("loadModels failed:", e);
  }
}

async function loadVersion() {
  try {
    const r = await fetch(API("/registry/models"));
    // нет /api/version — оставим как 0.1.0
    els.versionSpan.textContent = "0.1.0";
  } catch (e) {
    els.versionSpan.textContent = "?";
  }
}

function showError(msg) {
  els.error.textContent = msg;
  els.error.style.display = "block";
}
function clearError() {
  els.error.textContent = "";
  els.error.style.display = "none";
}

els.deckStrategy.addEventListener("change", () => {
  els.customBlock.style.display = els.deckStrategy.value === "custom" ? "block" : "none";
});

els.sampleBtn.addEventListener("click", async () => {
  try {
    const r = await fetch(API("/registry/sample-deck"));
    const data = await r.json();
    els.sampleOut.textContent = JSON.stringify(data.deck, null, 2);
    els.customDeckTA.value = JSON.stringify(data.deck);
  } catch (e) {
    els.sampleOut.textContent = "error: " + e;
  }
});

els.form.addEventListener("submit", async (e) => {
  e.preventDefault();
  clearError();
  const fd = new FormData(els.form);
  const spec = {
    p1_model: fd.get("p1_model"),
    p2_model: fd.get("p2_model"),
    deck_strategy: fd.get("deck_strategy"),
    battles_planned: parseInt(fd.get("battles_planned") || "1", 10),
    seed: parseInt(fd.get("seed") || "0", 10),
    starting_player: fd.get("starting_player"),
    max_turns: parseInt(fd.get("max_turns") || "60", 10),
  };
  if (spec.deck_strategy === "custom") {
    const txt = (els.customDeckTA.value || "").trim();
    if (!txt) {
      showError("Загрузите JSON-колоду или выберите «Случайные ArenaENV колоды»");
      return;
    }
    try {
      const parsed = JSON.parse(txt);
      spec.custom_deck_p1 = parsed;
      spec.custom_deck_p2 = parsed;
    } catch (err) {
      showError("Невалидный JSON: " + err.message);
      return;
    }
  }
  // Режим: human-vs-model → p1 должен играть человек → бой интерактивный
  // Сейчас мы НЕ запускаем интерактивно (нет WS-страницы в этой версии);
  // p1 = random baseline, бой = model vs model. В будущем — добавим checkbox.
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
    window.location.href = `/groups/${data.group_id}`;
  } catch (err) {
    showError("Ошибка: " + err.message);
  }
});

loadModels();
loadVersion();
