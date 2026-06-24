// RLHF Arena — клиентский JS для index.html
// Грузит модели в p2_select (человек = P1, бот = P2).
// POST /api/groups → редирект на /battle?group_id=...&battle_id=...

const API = (path) => `/api${path}`;

const els = {
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
      } else if (!groups[m.kind] && m.kind === "random") {
        const sep = document.createElement("option");
        sep.disabled = true;
        sep.textContent = "── Baselines ──";
        els.p2Select.appendChild(sep);
        lastGroup = "baselines";
      }
      opt.textContent = `${m.name} [${m.kind}]`;
      els.p2Select.appendChild(opt);
    }
  } catch (e) {
    console.warn("loadModels failed:", e);
  }
}

async function loadVersion() {
  try {
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
    p1_model: "human",  // sentinel — человек играет за P1
    p2_model: fd.get("p2_model"),
    deck_strategy: fd.get("deck_strategy"),
    battles_planned: parseInt(fd.get("battles_planned") || "3", 10),
    seed: parseInt(fd.get("seed") || "0", 10),
    starting_player: fd.get("starting_player"),
    max_turns: parseInt(fd.get("max_turns") || "60", 10),
    interactive: true,
    human_player: 1000,
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
    if (data.battle_id) {
      window.location.href = `/battle?group_id=${encodeURIComponent(data.group_id)}&battle_id=${encodeURIComponent(data.battle_id)}`;
    } else {
      throw new Error("API не вернул battle_id");
    }
  } catch (err) {
    showError("Ошибка: " + err.message);
  }
});

loadModels();
loadVersion();
