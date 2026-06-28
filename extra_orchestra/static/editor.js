"use strict";
// ExtraOrchestra — визуальный graph-редактор (v2).
// Один SVG-канвас, pan/zoom через viewBox, ноды со свободным позиционированием,
// рёбра рисует пользователь (port-drag). Один путь (max-1 in/out, reachability,
// no cycles) → детерминированный replay. Сырой JSON внизу — двусторонний sync.
// v1-сценарии при загрузке авто-мигрируются в v2 через /api/orchestra/migrate-v1.

var $ = function (id) { return document.getElementById(id); };
var NS = "http://www.w3.org/2000/svg";
var statusEl = $("status");
var editor = $("json-editor");
var svg = $("graph-svg");

var NODE_W = 180, NODE_H = 84, PORT_R = 6, HIT_R = 12, CONNECT_R = 20;
var VB = { x: 0, y: 0, w: 1200, h: 520 };

var cards = [], cardsById = {};
var scenario = null;
var selected = null;          // {type:'node'|'edge', id}
var nodeSeq = 0, edgeSeq = 0;
var initOverlayOpen = false;

function setStatus(msg, kind) {
  statusEl.textContent = msg;
  statusEl.className = "orch-status" + (kind ? " " + kind : "");
}
function esc(s) { return String(s == null ? "" : s).replace(/[&<>"]/g, function (c) {
  return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
function nextId(prefix) {
  if (prefix === "n") { nodeSeq += 1; return "n" + nodeSeq; }
  edgeSeq += 1; return "e" + edgeSeq;
}

// ----------------------------- модель ------------------------------------
function blankV2() {
  return {
    schema: "extra_orchestra.scenario.v2",
    name: "Новый сценарий", seed: 42, viewer_side: "p1", match_id: "new-scenario",
    classic_params: { sudden_death_enabled: false, mana_per_turn: 1, turn_duration_seconds: 25 },
    graph: {
      start: "s0",
      nodes: [
        { id: "s0", kind: "scene", scene: {
            type: "init", turn_number: 1, starting_side: "p1", display_ms: 2000,
            p1: { user_id: 1001, nickname: "Демо", title: "", rarity: "common", mana: 6, max_mana: 6,
                  avatar_url: "/DesignAssets/PlayerCosmetics/Avatars/1.png",
                  background_url: "/DesignAssets/PlayerCosmetics/Background/7.png",
                  hero: { card_id: 1, level: 1 }, hand: [], board: [], deck: [] },
            p2: { user_id: 2002, is_bot: true, nickname: "Оппонент", title: "", rarity: "epic", mana: 6, max_mana: 6,
                  avatar_url: "/DesignAssets/PlayerCosmetics/Avatars/2.png",
                  background_url: "/DesignAssets/PlayerCosmetics/Background/3.png",
                  hero: { card_id: 3, level: 1 }, hand: [], board: [], deck: [] } } }
      ],
      edges: []
    },
    layout: { s0: { x: 60, y: 200 } },
    editor: { zoom: 1 }
  };
}

function g() { return scenario.graph; }
function nodeById(id) { return g().nodes.find(function (n) { return n.id === id; }); }
function outgoingOf(id) { var e = g().edges.find(function (x) { return x.from === id; }); return e ? e : null; }
function incomingOf(id) { var e = g().edges.find(function (x) { return x.to === id; }); return e ? e : null; }
function initNode() { return g().nodes.find(function (n) { return n.kind === "scene" && n.scene.type === "init"; }); }
function pos(id) { return scenario.layout[id] || { x: 60, y: 200 }; }

function newActionNode(type) {
  var a = { type: type, delay_ms: 800 };
  if (type === "play_card") { a.hand_index = 0; a.target_id = null; a.target_index = null; a.target_is_hero = false; a.position = null; }
  else if (type === "attack") { a.attacker_id = null; a.attacker_index = 0; a.target_id = null; a.target_index = 0; a.target_is_hero = false; }
  return { id: nextId("n"), kind: "action", side: "p1", action: a };
}
function newHoldNode() { return { id: nextId("n"), kind: "scene", scene: { type: "hold", display_ms: 1000 } }; }
function newTurnNode() { return { id: nextId("n"), kind: "turn", turn: { side: "p1", intro_ms: 0 } }; }

// computed-side walk: id → "p1"|"p2" (чей ход по структуре графа) + warnings
function computedSides() {
  var sides = {}, warns = [];
  var init = initNode();
  if (!init) return { sides: sides, warns: warns };
  var cur = init.scene.starting_side || "p1";
  var id = g().start;
  var visited = {};
  while (id) {
    if (visited[id]) break;
    visited[id] = true;
    var n = nodeById(id);
    if (!n) break;
    if (n.kind === "action") {
      sides[id] = cur;
      if (n.side && n.side !== cur) warns.push({ id: id, msg: "side '" + n.side + "' ≠ позиции в пути '" + cur + "' (забыт end_turn?)" });
      if ((n.action || {}).type === "end_turn") cur = cur === "p1" ? "p2" : "p1";
    } else if (n.kind === "turn") {
      if (n.turn && n.turn.side && n.turn.side !== cur) warns.push({ id: id, msg: "turn.side '" + n.turn.side + "' ≠ current '" + cur + "'" });
    }
    var e = outgoingOf(id);
    id = e ? e.to : null;
  }
  return { sides: sides, warns: warns };
}

// ----------------------------- sync --------------------------------------
var syncTimer = null;
function syncToJSON(immediate) {
  if (!scenario) return;
  scenario.editor = scenario.editor || {};
  scenario.editor.zoom = VB.w / 1200;
  var txt = JSON.stringify(scenario, null, 2);
  // не перезаписывать textarea, пока пользователь в нём печатает (иначе debounced
  // sync молча затрёт только что введённые символы); отложим до потери фокуса
  function write() {
    if (document.activeElement === editor) { syncTimer = setTimeout(write, 200); return; }
    if (editor.value !== txt) editor.value = txt;
  }
  if (immediate) { if (syncTimer) clearTimeout(syncTimer); write(); }
  else { if (syncTimer) clearTimeout(syncTimer); syncTimer = setTimeout(write, 150); }
}
function loadFromJSON(silent) {
  var raw = editor.value.trim();
  if (!raw) return false;
  try { scenario = JSON.parse(raw); }
  catch (e) { if (!silent) setStatus("JSON parse error: " + e.message, "err"); return false; }
  if (!scenario.graph) { if (!silent) setStatus("не v2-сценарий (нет graph)", "err"); return false; }
  selected = null;
  renderAll();
  return true;
}
function readMetaInto(sc) {
  sc.name = $("f-name").value.trim() || sc.name || "scenario";
  sc.viewer_side = $("f-viewer").value;
  sc.seed = parseInt($("f-seed").value, 10) || 0;
  sc.match_id = $("f-match").value.trim() || sc.match_id || "orchestra";
  return sc;
}
function writeMetaFromScenario() {
  if (!scenario) return;
  $("f-name").value = scenario.name || "";
  $("f-viewer").value = scenario.viewer_side || "p1";
  $("f-seed").value = scenario.seed != null ? scenario.seed : 42;
  $("f-match").value = scenario.match_id || "";
}

// ----------------------------- render ------------------------------------
function renderAll() {
  writeMetaFromScenario();
  renderGraph();
  renderProps();
  syncToJSON(true);
  $("zone-pick").hidden = !(selected && selected.type === "node" && nodeById(selected.id) === initNode());
}

function applyViewBox() {
  svg.setAttribute("viewBox", VB.x + " " + VB.y + " " + VB.w + " " + VB.h);
}

function edgePath(fromId, toId) {
  var a = pos(fromId), b = pos(toId);
  var x1 = a.x + NODE_W, y1 = a.y + NODE_H / 2;
  var x2 = b.x, y2 = b.y + NODE_H / 2;
  var dx = Math.max(40, (x2 - x1) / 2);
  return "M " + x1 + " " + y1 + " C " + (x1 + dx) + " " + y1 + ", " + (x2 - dx) + " " + y2 + ", " + x2 + " " + y2;
}

function renderGraph() {
  applyViewBox();
  var nodes = g().nodes, edges = g().edges;
  // sync seq counters above existing ids
  nodes.forEach(function (n) { var m = /^n(\d+)$/.exec(n.id); if (m) nodeSeq = Math.max(nodeSeq, +m[1]); });
  edges.forEach(function (e) { var m = /^e(\d+)$/.exec(e.id); if (m) edgeSeq = Math.max(edgeSeq, +m[1]); });

  var cs = computedSides();
  var reachable = reachability();
  $("empty-hint").hidden = nodes.length > 0;

  // clear
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  var edgesLayer = document.createElementNS(NS, "g"); svg.appendChild(edgesLayer);
  var nodesLayer = document.createElementNS(NS, "g"); svg.appendChild(nodesLayer);

  // edges
  edges.forEach(function (e) {
    var d = edgePath(e.from, e.to);
    var hit = document.createElementNS(NS, "path");
    hit.setAttribute("class", "ogedge-hit" + (selected && selected.type === "edge" && selected.id === e.id ? " selected" : ""));
    hit.setAttribute("d", d);
    hit.dataset.edgeId = e.id;
    hit.addEventListener("click", onEdgeClick);
    edgesLayer.appendChild(hit);
    var path = document.createElementNS(NS, "path");
    path.setAttribute("class", "ogedge" + (selected && selected.type === "edge" && selected.id === e.id ? " selected" : ""));
    path.setAttribute("d", d);
    path.style.pointerEvents = "none";
    edgesLayer.appendChild(path);
  });

  // nodes
  nodes.forEach(function (n) {
    var p = pos(n.id);
    var gr = document.createElementNS(NS, "g");
    var cls = "ognode";
    if (selected && selected.type === "node" && selected.id === n.id) cls += " selected";
    if (!reachable[n.id]) cls += " unreachable";
    if (cs.warns.find(function (w) { return w.id === n.id; })) cls += " warn";
    gr.setAttribute("class", cls);
    gr.setAttribute("transform", "translate(" + p.x + "," + p.y + ")");
    gr.dataset.nodeId = n.id;

    var rect = document.createElementNS(NS, "rect");
    rect.setAttribute("class", "box");
    rect.setAttribute("width", NODE_W); rect.setAttribute("height", NODE_H); rect.setAttribute("rx", 10);
    rect.addEventListener("mousedown", onNodeDown);
    gr.appendChild(rect);

    var kind = document.createElementNS(NS, "text");
    kind.setAttribute("class", "kind"); kind.setAttribute("x", 10); kind.setAttribute("y", 18);
    kind.textContent = nodeKindLabel(n);
    gr.appendChild(kind);

    var title = document.createElementNS(NS, "text");
    title.setAttribute("class", "title"); title.setAttribute("x", 10); title.setAttribute("y", 40);
    title.textContent = nodeTitle(n);
    gr.appendChild(title);

    var sub = document.createElementNS(NS, "text");
    sub.setAttribute("class", "sub"); sub.setAttribute("x", 10); sub.setAttribute("y", 58);
    sub.textContent = nodeSub(n);
    gr.appendChild(sub);

    // side chip (action)
    if (n.kind === "action" && cs.sides[n.id]) {
      var chipG = document.createElementNS(NS, "g"); chipG.setAttribute("class", "sidechip");
      chipG.setAttribute("transform", "translate(" + (NODE_W - 34) + ",10)");
      var chipR = document.createElementNS(NS, "rect");
      chipR.setAttribute("width", 26); chipR.setAttribute("height", 14); chipR.setAttribute("rx", 7);
      chipG.appendChild(chipR);
      var chipT = document.createElementNS(NS, "text");
      chipT.setAttribute("x", 13); chipT.setAttribute("y", 11); chipT.setAttribute("text-anchor", "middle");
      chipT.textContent = cs.sides[n.id];
      chipG.appendChild(chipT);
      gr.appendChild(chipG);
    }

    var idx = document.createElementNS(NS, "text");
    idx.setAttribute("class", "idx"); idx.setAttribute("x", 10); idx.setAttribute("y", 76);
    idx.textContent = n.id;
    gr.appendChild(idx);

    // input port (кроме init)
    if (!(n.kind === "scene" && n.scene.type === "init")) {
      var inP = document.createElementNS(NS, "circle");
      inP.setAttribute("class", "og-port in");
      inP.setAttribute("cx", 0); inP.setAttribute("cy", NODE_H / 2); inP.setAttribute("r", PORT_R);
      inP.style.pointerEvents = "none";
      gr.appendChild(inP);
    }
    // output port + hit
    var outHit = document.createElementNS(NS, "circle");
    outHit.setAttribute("class", "og-port-hit");
    outHit.setAttribute("cx", NODE_W); outHit.setAttribute("cy", NODE_H / 2); outHit.setAttribute("r", HIT_R);
    outHit.dataset.nodeId = n.id;
    outHit.addEventListener("mousedown", onPortDown);
    gr.appendChild(outHit);
    var outP = document.createElementNS(NS, "circle");
    outP.setAttribute("class", "og-port");
    outP.setAttribute("cx", NODE_W); outP.setAttribute("cy", NODE_H / 2); outP.setAttribute("r", PORT_R);
    outP.style.pointerEvents = "none";
    gr.appendChild(outP);

    nodesLayer.appendChild(gr);
  });
}

function reachability() {
  var vis = {}; var id = g().start;
  while (id) { if (vis[id]) break; vis[id] = true; var e = outgoingOf(id); id = e ? e.to : null; }
  return vis;
}

function nodeKindLabel(n) {
  if (n.kind === "scene") return n.scene.type === "init" ? "init-сцена" : "сцена (hold)";
  if (n.kind === "turn") return "ход";
  if (n.kind === "action") return (n.action || {}).type || "action";
  return n.kind;
}
function nodeTitle(n) {
  if (n.kind === "scene" && n.scene.type === "init") return "ход " + (n.scene.turn_number || 1) + " · start " + (n.scene.starting_side || "p1");
  if (n.kind === "scene") return "удержание кадра";
  if (n.kind === "turn") return "сторона " + ((n.turn || {}).side || "—");
  if (n.kind === "action") {
    var a = n.action || {}, t = a.type;
    if (t === "play_card") return "карта из руки #" + (a.hand_index || 0);
    if (t === "attack") return "атака #" + (a.attacker_index || 0) + " → " + (a.target_is_hero ? "герой" : "#" + (a.target_index || 0));
    if (t === "mana_draw") return "добор по мане";
    if (t === "end_turn") return "завершить ход";
  }
  return "";
}
function nodeSub(n) {
  if (n.kind === "scene") return "display " + (n.scene.display_ms || 0) + " мс";
  if (n.kind === "turn") return "intro " + ((n.turn || {}).intro_ms || 0) + " мс";
  if (n.kind === "action") return "delay " + ((n.action || {}).delay_ms || 0) + " мс";
  return "";
}

// ----------------------------- properties --------------------------------
function renderProps() {
  var panel = $("props-panel"), title = $("props-title");
  $("btn-del-sel").disabled = !selected;
  if (!selected) {
    title.textContent = "—";
    panel.innerHTML = '<div class="hint">кликни узел или ребро на канвасе.<br>Тяни от ●порта узла к другому узлу, чтобы соединить.<br>Граф = один путь (один исходящий/входящий на узел).</div>';
    return;
  }
  if (selected.type === "edge") {
    var e = g().edges.find(function (x) { return x.id === selected.id; });
    if (!e) { selected = null; renderProps(); return; }
    title.textContent = "ребро " + e.id;
    panel.innerHTML = '<div class="hint">' + esc(e.from) + " → " + esc(e.to) + '</div>' +
      '<button id="btn-del-edge" class="danger" type="button">Удалить ребро</button>';
    $("btn-del-edge").onclick = deleteEdge;
    return;
  }
  var n = nodeById(selected.id);
  if (!n) { selected = null; renderProps(); return; }
  title.textContent = nodeKindLabel(n);
  var body = "";
  if (n.kind === "scene" && n.scene.type === "init") {
    var s = n.scene;
    body = pField("scene.turn_number", "ход (turn_number)", s.turn_number || 1, "number") +
      pSelect("scene.starting_side", "стартовая сторона", ["p1", "p2"], s.starting_side || "p1") +
      pField("scene.display_ms", "показ init (мс)", s.display_ms || 1200, "number") +
      '<button id="btn-init-advanced" type="button">расширенные настройки init →</button>';
    panel.innerHTML = body;
    panel.querySelectorAll("[data-prop]").forEach(function (el) { el.addEventListener("change", onPropChange); });
    $("btn-init-advanced").onclick = openInitOverlay;
    return;
  }
  if (n.kind === "scene") {
    body = pField("scene.display_ms", "display_ms (мс)", n.scene.display_ms || 0, "number");
  } else if (n.kind === "turn") {
    var t = n.turn || {};
    body = pSelect("turn.side", "side (опц. sanity-маркер)", ["", "p1", "p2"], t.side || "") +
      pField("turn.intro_ms", "intro_ms (0 = нет кадра)", t.intro_ms || 0, "number");
  } else if (n.kind === "action") {
    var a = n.action || {};
    body = pSelect("side", "side (чей ход)", ["p1", "p2"], n.side || "p1") +
      pField("action.delay_ms", "delay_ms (мс)", a.delay_ms || 0, "number");
    if (a.type === "play_card") {
      body += pField("action.hand_index", "hand_index", a.hand_index || 0, "number") +
        pField("action.target_id", "target_id (instance_id, опц.)", a.target_id == null ? "" : a.target_id, "text") +
        pField("action.target_index", "target_index (опц.)", a.target_index == null ? "" : a.target_index, "number") +
        pCheck("action.target_is_hero", "цель — герой", !!a.target_is_hero) +
        pField("action.position", "position (опц.)", a.position == null ? "" : a.position, "number");
    } else if (a.type === "attack") {
      body += pField("action.attacker_id", "attacker_id (instance_id, опц.)", a.attacker_id == null ? "" : a.attacker_id, "text") +
        pField("action.attacker_index", "attacker_index", a.attacker_index || 0, "number") +
        pField("action.target_id", "target_id (опц.)", a.target_id == null ? "" : a.target_id, "text") +
        pField("action.target_index", "target_index (опц.)", a.target_index == null ? "" : a.target_index, "number") +
        pCheck("action.target_is_hero", "цель — герой", !!a.target_is_hero);
    }
  }
  panel.innerHTML = body;
  panel.querySelectorAll("[data-prop]").forEach(function (el) { el.addEventListener("change", onPropChange); });
}

function onPropChange(e) {
  var n = nodeById(selected.id); if (!n) return;
  var p = e.target.dataset.prop, v;
  if (e.target.type === "checkbox") v = e.target.checked;
  else if (e.target.type === "number") v = e.target.value === "" ? null : (parseInt(e.target.value, 10) || 0);
  else v = e.target.value === "" ? null : e.target.value;
  if (p === "side") n.side = v;
  else if (p === "scene.display_ms") n.scene.display_ms = v;
  else if (p === "scene.turn_number") n.scene.turn_number = v;
  else if (p === "scene.starting_side") { n.scene.starting_side = v; }
  else if (p === "turn.side") n.turn.side = v;
  else if (p === "turn.intro_ms") n.turn.intro_ms = v;
  else if (p === "action.delay_ms") n.action.delay_ms = v;
  else if (p === "action.hand_index") n.action.hand_index = v;
  else if (p === "action.target_id") n.action.target_id = v;
  else if (p === "action.target_index") n.action.target_index = v;
  else if (p === "action.target_is_hero") n.action.target_is_hero = v;
  else if (p === "action.position") n.action.position = v;
  else if (p === "action.attacker_id") n.action.attacker_id = v;
  else if (p === "action.attacker_index") n.action.attacker_index = v;
  renderGraph(); renderProps(); syncToJSON();
}
function pField(k, label, val, type) {
  return '<div class="prop-row"><label>' + esc(label) +
    '<input data-prop="' + esc(k) + '" type="' + type + '" value="' + esc(val == null ? "" : val) + '"></label></div>';
}
function pSelect(k, label, opts, val) {
  var o = opts.map(function (x) { return '<option value="' + esc(x) + '"' + (x === val ? " selected" : "") + ">" + esc(x === "" ? "—" : x) + "</option>"; }).join("");
  return '<div class="prop-row"><label>' + esc(label) + '<select data-prop="' + esc(k) + '">' + o + '</select></label></div>';
}
function pCheck(k, label, val) {
  return '<div class="prop-row"><label><input type="checkbox" data-prop="' + esc(k) + '"' + (val ? " checked" : "") + "> " + esc(label) + "</label></div>";
}

// ----------------------------- init overlay ------------------------------
function openInitOverlay() {
  initOverlayOpen = true;
  $("init-overlay").hidden = false;
  renderInitForm();
}
function closeInitOverlay() { initOverlayOpen = false; $("init-overlay").hidden = true; }
function renderInitForm() {
  var f = $("init-form");
  var s = initNode().scene;
  var html = "";
  ["p1", "p2"].forEach(function (side) {
    var p = s[side] || {};
    var heroId = (p.hero && p.hero.card_id) || 1;
    html += '<div class="ifs-section"><h4>сторона ' + side + '</h4>' +
      iField(side + ":user_id", "user_id", p.user_id || (side === "p1" ? 1001 : 2002), "number") +
      iField(side + ":nickname", "никнейм", p.nickname || "", "text") +
      iField(side + ":title", "титул", p.title || "", "text") +
      iSelect(side + ":rarity", "редкость", ["common", "rare", "epic", "legendary"], p.rarity || "common") +
      iField(side + ":avatar_url", "аватар url", p.avatar_url || "", "text") +
      iField(side + ":background_url", "фон url", p.background_url || "", "text") +
      iField(side + ":mana", "mana", p.mana != null ? p.mana : 6, "number") +
      iField(side + ":max_mana", "max_mana", p.max_mana != null ? p.max_mana : 6, "number") +
      iHeroSelect(side, heroId) +
      iZoneChips(side, "hand", p.hand) + iZoneChips(side, "board", p.board) + iZoneChips(side, "deck", p.deck) +
      '</div>';
  });
  f.innerHTML = html;
  f.querySelectorAll("[data-ifield]").forEach(function (el) { el.addEventListener("change", onInitFieldChange); });
  f.querySelectorAll(".zchip .x").forEach(function (x) {
    x.onclick = function () {
      var parts = x.getAttribute("data-zone").split(".");
      var arr = initNode().scene[parts[0]][parts[1]];
      arr.splice(parseInt(x.getAttribute("data-zi"), 10), 1);
      renderInitForm(); syncToJSON();
    };
  });
}
function iField(k, label, val, type) {
  return '<label>' + esc(label) + '<input data-ifield="' + esc(k) + '" type="' + type + '" value="' + esc(val) + '"></label>';
}
function iSelect(k, label, opts, val) {
  var o = opts.map(function (x) { return '<option value="' + esc(x) + '"' + (x === val ? " selected" : "") + ">" + esc(x) + "</option>"; }).join("");
  return '<label>' + esc(label) + '<select data-ifield="' + esc(k) + '">' + o + '</select></label>';
}
function iHeroSelect(side, val) {
  var o = cards.filter(function (c) { return c.card_type === "hero"; })
    .map(function (c) { return '<option value="' + c.id + '"' + (Number(c.id) === Number(val) ? " selected" : "") + ">" + esc(c.name) + " (#" + c.id + ")</option>"; }).join("");
  return '<label>герой<select data-ifield="' + side + ':hero">' + o + '</select></label>';
}
function iZoneChips(side, zone, arr) {
  arr = arr || [];
  var chips = arr.map(function (c, i) {
    var name = (cardsById[c.card_id] && cardsById[c.card_id].name) || ("#" + c.card_id);
    return '<span class="zchip">' + esc(name) + ' L' + (c.level || 1) +
      ' <span class="x" data-zone="' + side + "." + zone + '" data-zi="' + i + '">×</span></span>';
  }).join("");
  return '<div class="zone"><div>' + zone + ' (' + arr.length + ')</div><div class="zchips">' + chips +
    '</div><div class="zhint">клик по карте каталога → добавится в выбранную зону (справа вверху)</div></div>';
}
function onInitFieldChange(e) {
  var key = e.target.dataset.ifield;
  var v = e.target.type === "number" ? (parseInt(e.target.value, 10) || 0) : e.target.value;
  var s = initNode().scene;
  if (key.indexOf(":") < 0) return;
  var parts = key.split(":"), side = parts[0], fld = parts[1];
  if (fld === "hero") s[side].hero = { card_id: parseInt(v, 10), level: 1 };
  else s[side][fld] = v;
  syncToJSON();
}

// ----------------------------- pan / zoom --------------------------------
function svgPoint(e) {
  var p = svg.createSVGPoint(); p.x = e.clientX; p.y = e.clientY;
  return p.matrixTransform(svg.getScreenCTM().inverse());
}
svg.addEventListener("wheel", function (e) {
  e.preventDefault();
  var rect = svg.getBoundingClientRect();
  var scale = VB.w / rect.width;
  var cuX = VB.x + (e.clientX - rect.left) * scale;
  var cuY = VB.y + (e.clientY - rect.top) * scale;
  var factor = e.deltaY < 0 ? 0.9 : 1.1;
  var newScale = scale * factor;
  var newW = rect.width * newScale, newH = rect.height * newScale;
  if (newW < 200 || newW > 6000) return;
  VB.x = cuX - (e.clientX - rect.left) * newScale;
  VB.y = cuY - (e.clientY - rect.top) * newScale;
  VB.w = newW; VB.h = newH;
  applyViewBox();
}, { passive: false });

// pan on empty canvas
var pan = null;
svg.addEventListener("mousedown", function (e) {
  if (e.target.tagName === "svg") {
    var u = svgPoint(e);
    pan = { sx: u.x, sy: u.y, vx: VB.x, vy: VB.y, moved: false };
    $("canvas-wrap").classList.add("panning");
  }
});
document.addEventListener("mousemove", function (e) {
  if (pan) {
    var u = svgPoint(e);
    VB.x = pan.vx - (u.x - pan.sx);
    VB.y = pan.vy - (u.y - pan.sy);
    if (Math.abs(u.x - pan.sx) > 3 || Math.abs(u.y - pan.sy) > 3) pan.moved = true;
    applyViewBox();
  }
});
document.addEventListener("mouseup", function () {
  if (pan) {
    $("canvas-wrap").classList.remove("panning");
    if (!pan.moved) { selected = null; renderProps(); renderGraph(); }
    pan = null;
  }
});

// ----------------------------- node drag ---------------------------------
var nodeDrag = null;
function onNodeDown(e) {
  if (e.button !== 0) return;
  if (e.target.classList.contains("og-port-hit")) return; // port handled separately
  var gNode = e.currentTarget.parentNode;
  var id = gNode.dataset.nodeId;
  var u = svgPoint(e);
  var p = pos(id);
  nodeDrag = { id: id, sx: u.x, sy: u.y, ox: p.x, oy: p.y, moved: false, gNode: gNode };
  e.stopPropagation();
  e.preventDefault();
}
document.addEventListener("mousemove", function (e) {
  if (!nodeDrag) return;
  var u = svgPoint(e);
  if (Math.abs(u.x - nodeDrag.sx) > 3 || Math.abs(u.y - nodeDrag.sy) > 3) nodeDrag.moved = true;
  var nx = nodeDrag.ox + (u.x - nodeDrag.sx);
  var ny = nodeDrag.oy + (u.y - nodeDrag.sy);
  scenario.layout[nodeDrag.id] = { x: nx, y: ny };
  nodeDrag.gNode.setAttribute("transform", "translate(" + nx + "," + ny + ")");
  // update incident edges
  g().edges.forEach(function (ed) {
    if (ed.from === nodeDrag.id || ed.to === nodeDrag.id) {
      var hit = svg.querySelector('.ogedge-hit[data-edge-id="' + ed.id + '"]');
      var path = hit && hit.nextElementSibling;
      var d = edgePath(ed.from, ed.to);
      if (hit) hit.setAttribute("d", d);
      if (path) path.setAttribute("d", d);
    }
  });
});
document.addEventListener("mouseup", function () {
  if (nodeDrag) {
    if (!nodeDrag.moved) { // click → select
      selected = { type: "node", id: nodeDrag.id };
      renderProps(); renderGraph();
    } else { syncToJSON(); }
    nodeDrag = null;
  }
});

// ----------------------------- edge drag (port) --------------------------
var edgeDrag = null;
function onPortDown(e) {
  if (e.button !== 0) return;
  e.stopPropagation(); e.preventDefault();
  var fromId = e.currentTarget.dataset.nodeId;
  var p = pos(fromId);
  edgeDrag = { from: fromId, sx: p.x + NODE_W, sy: p.y + NODE_H / 2 };
  var temp = document.createElementNS(NS, "path");
  temp.setAttribute("class", "ogedge-temp"); temp.setAttribute("id", "og-temp");
  temp.setAttribute("d", edgePathTemp(edgeDrag.sx, edgeDrag.sy, edgeDrag.sx, edgeDrag.sy));
  svg.appendChild(temp);
}
function edgePathTemp(x1, y1, x2, y2) {
  var dx = Math.max(40, Math.abs(x2 - x1) / 2);
  return "M " + x1 + " " + y1 + " C " + (x1 + dx) + " " + y1 + ", " + (x2 - dx) + " " + y2 + ", " + x2 + " " + y2;
}
document.addEventListener("mousemove", function (e) {
  if (!edgeDrag) return;
  var u = svgPoint(e);
  var temp = $("og-temp");
  if (temp) temp.setAttribute("d", edgePathTemp(edgeDrag.sx, edgeDrag.sy, u.x, u.y));
  // highlight drop target
  var tgt = findDropTarget(u);
  svg.querySelectorAll(".ogdrop-target").forEach(function (g) { g.classList.remove("ogdrop-target"); });
  if (tgt) {
    var gn = svg.querySelector('.ognode[data-node-id="' + tgt + '"]');
    if (gn) gn.classList.add("ogdrop-target");
  }
});
function findDropTarget(u) {
  var best = null, bestD = CONNECT_R;
  g().nodes.forEach(function (n) {
    if (n.id === edgeDrag.from) return;
    if (n.kind === "scene" && n.scene.type === "init") return; // no input port
    var p = pos(n.id);
    var cx = p.x, cy = p.y + NODE_H / 2;
    var dPort = Math.hypot(cx - u.x, cy - u.y);
    // курсор внутри коробки узла — тоже валидный drop (более прощающий UX)
    var inside = u.x >= p.x && u.x <= p.x + NODE_W && u.y >= p.y && u.y <= p.y + NODE_H;
    var d = inside ? 0 : dPort;
    if (d < bestD) { bestD = d; best = n.id; }
  });
  return best;
}
document.addEventListener("mouseup", function (e) {
  if (!edgeDrag) return;
  var temp = $("og-temp"); if (temp) temp.remove();
  svg.querySelectorAll(".ogdrop-target").forEach(function (g) { g.classList.remove("ogdrop-target"); });
  var u = svgPoint(e);
  var tgt = findDropTarget(u);
  var from = edgeDrag.from;
  edgeDrag = null;
  if (!tgt) return;
  var existingOut = outgoingOf(from);
  var existingIn = incomingOf(tgt);
  if (existingOut && existingOut.to === tgt) { setStatus("ребро " + from + " → " + tgt + " уже есть", "ok"); return; }
  if (existingOut || existingIn) {
    var msg = "Узел уже имеет " + (existingOut ? "исходящее" : "входящее") + " ребро. Заменить?";
    if (!confirm(msg)) return;
    if (existingOut) removeEdge(existingOut.id);
    if (existingIn) removeEdge(existingIn.id);
  }
  var eid = nextId("e");
  g().edges.push({ id: eid, from: from, to: tgt });
  selected = { type: "edge", id: eid };
  renderGraph(); renderProps(); syncToJSON();
  setStatus("ребро " + from + " → " + tgt + " создано", "ok");
});

function onEdgeClick(e) {
  e.stopPropagation();
  selected = { type: "edge", id: e.currentTarget.dataset.edgeId };
  renderProps(); renderGraph();
}
function removeEdge(id) {
  g().edges = g().edges.filter(function (e) { return e.id !== id; });
}
function deleteEdge() {
  if (!selected || selected.type !== "edge") return;
  removeEdge(selected.id); selected = null;
  renderGraph(); renderProps(); syncToJSON();
}

// ----------------------------- palette / delete --------------------------
function addNode(kind) {
  var n;
  if (kind === "hold") n = newHoldNode();
  else if (kind === "turn") n = newTurnNode();
  else n = newActionNode(kind);
  // цепочка вправо от выбранного узла (или от конца пути), чтобы не накладывались
  var anchor = selected && selected.type === "node" ? pos(selected.id) : null;
  if (!anchor) {
    var last = g().start, id = g().start;
    while (id) { last = id; var e = outgoingOf(id); id = e ? e.to : null; }
    anchor = pos(last);
  }
  var cx = anchor.x + 230;
  var cy = anchor.y;
  g().nodes.push(n);
  scenario.layout[n.id] = { x: cx, y: cy };
  selected = { type: "node", id: n.id };
  renderGraph(); renderProps(); syncToJSON();
}
function deleteSelected() {
  if (!selected) return;
  if (selected.type === "edge") { deleteEdge(); return; }
  var n = nodeById(selected.id);
  if (!n) return;
  if (n.kind === "scene" && n.scene.type === "init") { setStatus("init-узел нельзя удалить (создай Новый сценарий)", "err"); return; }
  g().nodes = g().nodes.filter(function (x) { return x.id !== selected.id; });
  g().edges = g().edges.filter(function (e) { return e.from !== selected.id && e.to !== selected.id; });
  delete scenario.layout[selected.id];
  selected = null;
  renderGraph(); renderProps(); syncToJSON();
}

// авто-раскладка: init → цепочка вправо по пути
function relayout() {
  var id = g().start, x = 60, y = 200;
  var vis = {};
  while (id) {
    if (vis[id]) break; vis[id] = true;
    scenario.layout[id] = { x: x, y: y };
    x += 230;
    var e = outgoingOf(id);
    id = e ? e.to : null;
  }
  // orphan nodes — отдельной колонкой ниже цепочки (не поверх start)
  var oy = 360;
  g().nodes.forEach(function (n) {
    if (!vis[n.id]) { scenario.layout[n.id] = { x: 60, y: oy }; oy += 110; }
  });
  renderGraph(); syncToJSON();
}

// ----------------------------- scenarios ---------------------------------
function newScenario() {
  scenario = blankV2();
  nodeSeq = 0; edgeSeq = 0; selected = null;
  VB = { x: 0, y: 0, w: 1200, h: 520 };
  renderAll();
  setStatus("новый v2-сценарий", "ok");
}
function openScenario(name) {
  fetch("/api/orchestra/scenarios/" + encodeURIComponent(name)).then(function (r) { return r.json(); }).then(function (sc) {
    if (sc.error) { setStatus("load error: " + sc.error, "err"); return; }
    if (sc.schema !== "extra_orchestra.scenario.v2") {
      // авто-миграция v1 → v2
      fetch("/api/orchestra/migrate-v1", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(sc) })
        .then(function (r) { return r.json(); }).then(function (v2) {
          if (v2.error) { setStatus("migrate error: " + v2.error, "err"); return; }
          scenario = v2; selected = null;
          renderAll(); setStatus("загружен (v1→v2 миграция): " + (sc.name || name), "ok");
          refreshScenarios();
        });
      return;
    }
    scenario = sc; selected = null;
    renderAll(); setStatus("загружен: " + (sc.name || name), "ok"); refreshScenarios();
  });
}
function refreshScenarios() {
  fetch("/api/orchestra/scenarios").then(function (r) { return r.json(); }).then(function (d) {
    var ul = $("scenario-list"); ul.innerHTML = "";
    (d.scenarios || []).forEach(function (s) {
      var li = document.createElement("li");
      if (s.file === currentName) li.className = "active";
      var left = document.createElement("div");
      var cnt = s.nodes != null ? s.nodes + " узлов" : (s.turns || 0) + " turns";
      left.innerHTML = '<div class="sc-name">' + esc(s.name) + ' <span style="color:var(--muted);font-size:10px">' + esc((s.schema || "").replace("extra_orchestra.scenario.", "")) + '</span></div>' +
        '<div class="sc-meta">' + cnt + " · " + esc(s.file || "") + "</div>";
      left.style.cursor = "pointer";
      left.onclick = function () { currentName = s.file; openScenario(s.file || s.name); };
      var del = document.createElement("button");
      del.className = "del"; del.textContent = "✕"; del.title = "удалить";
      del.onclick = function (e) { e.stopPropagation();
        if (confirm("Удалить сценарий " + s.name + "?")) {
          fetch("/api/orchestra/scenarios/" + encodeURIComponent(s.file || s.name), { method: "DELETE" }).then(refreshScenarios);
        }
      };
      li.appendChild(left); li.appendChild(del); ul.appendChild(li);
    });
  });
}
var currentName = null;

// ----------------------------- actions -----------------------------------
function doValidate() {
  if (!scenario) return; readMetaInto(scenario); syncToJSON(true);
  fetch("/api/orchestra/validate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) })
    .then(function (r) { return r.json(); })
    .then(function (d) { if (d.ok) setStatus("OK: " + d.frame_count + " кадров, " + d.total_ms + " мс", "ok");
      else setStatus("ошибка: " + (d.error || "?"), "err"); });
}
function doSave() {
  if (!scenario) return; readMetaInto(scenario); syncToJSON(true);
  fetch("/api/orchestra/scenarios", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) })
    .then(function (r) { return r.json(); })
    .then(function (d) { if (d.ok) { currentName = d.file; setStatus("сохранён: " + d.file, "ok"); refreshScenarios(); }
      else setStatus("save error: " + (d.error || "?"), "err"); });
}
function doPreview() {
  if (!scenario) return; readMetaInto(scenario); syncToJSON(true);
  fetch("/api/orchestra/compute-frames", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) })
    .then(function (r) { return r.json(); })
    .then(function (d) {
      if (d.error) { setStatus("frames error: " + d.error, "err"); return; }
      setStatus("предпросмотр: " + d.frame_count + " кадров → открываю арену", "ok");
      var auth = d.auth || "";
      // Предпросмотр через /preview — обёртка грузит арену в <iframe> мобильного
      // портретного размера (414×896) → срабатывает @media (max-width:420px)
      // из arena-styles.css → мобильный лейаут (область сужена до моб. аспекта).
      // preview.html сам подставляет music=1/sfx=1/ea_platform, если их нет.
      window.open("/preview?id=" + encodeURIComponent(d.run_id) + "&autoplay=1&_auth=" + encodeURIComponent(auth),
        "orch_preview");
    });
}
var recordJobId = null, recordPollTimer = null;
function doExport(fmt) {
  if (!scenario) return; readMetaInto(scenario); syncToJSON(true);
  fmt = fmt || "mp4";
  setStatus("запуск записи (" + fmt + ")…", "ok"); $("record-box").hidden = false; $("record-state").textContent = "pending…";
  fetch("/api/orchestra/record?format=" + fmt, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(scenario) })
    .then(function (r) { return r.json(); })
    .then(function (d) { if (d.error) { setStatus("record error: " + d.error, "err"); return; }
      recordJobId = d.job_id; var nm = d.file_name || d.mp4_name || ""; $("record-name").textContent = nm; pollRecord(); });
}
function doExportGif() { doExport("gif"); }
function pollRecord() {
  if (!recordJobId) return;
  fetch("/api/orchestra/record/" + encodeURIComponent(recordJobId)).then(function (r) { return r.json(); }).then(function (d) {
    $("record-state").textContent = d.status + (d.error ? " — " + d.error : "");
    var nm = d.file_name || d.mp4_name;
    if (d.status === "done" && nm) {
      var a = $("download-link");
      a.href = "/api/orchestra/record/" + encodeURIComponent(recordJobId) + "/download";
      a.download = nm; a.textContent = "Скачать " + nm; a.hidden = false;
      setStatus("готово: " + nm, "ok");
    } else if (d.status === "failed") { setStatus("запись не удалась: " + (d.error || "?"), "err"); }
    else { if (recordPollTimer) clearTimeout(recordPollTimer); recordPollTimer = setTimeout(pollRecord, 2000); }
  });
}

// ----------------------------- cards / cosmetics -------------------------
function renderCards(filter) {
  var ul = $("card-list"); ul.innerHTML = "";
  var f = (filter || "").toLowerCase();
  cards.filter(function (c) { return !f || (c.name || "").toLowerCase().indexOf(f) >= 0 || String(c.id) === f; })
    .forEach(function (c) {
      var li = document.createElement("li");
      li.title = "клик — добавить в зону init (если открыт init-overlay) / копировать spec";
      li.innerHTML = '<img src="' + c.image + '" alt=""><span class="ci-name">' + esc(c.name) +
        '</span><span class="ci-stats">' + c.mana_cost + "/" + c.base_attack + "/" + c.base_hp +
        '</span><span class="ci-id">#' + c.id + "</span>";
      li.onclick = function () { onCardClick(c); };
      ul.appendChild(li);
    });
}
function onCardClick(c) {
  if (initOverlayOpen) {
    var z = $("zone-target").value.split(".");
    var arr = initNode().scene[z[0]][z[1]];
    arr.push({ card_id: c.id, level: 1 });
    renderInitForm(); syncToJSON();
    setStatus("добавлено в " + z.join(".") + ": " + c.name, "ok");
  } else {
    var snip = '{ "card_id": ' + c.id + ', "level": 1 }';
    var ta = editor, s = ta.selectionStart, e = ta.selectionEnd;
    ta.value = ta.value.slice(0, s) + snip + ta.value.slice(e);
    ta.selectionStart = ta.selectionEnd = s + snip.length; ta.focus();
    setStatus("spec вставлен в JSON: " + c.name, "ok");
  }
}
function renderCosmetics(d) {
  var av = d.avatars || [], bg = d.backgrounds || [];
  $("av-count").textContent = "(" + av.length + ")";
  $("bg-count").textContent = "(" + bg.length + ")";
  function fill(el, items, kind) {
    el.innerHTML = "";
    items.forEach(function (it) {
      var img = document.createElement("img");
      img.src = it.url; img.title = kind + ": " + it.name;
      img.onclick = function () { navigator.clipboard && navigator.clipboard.writeText(it.url);
        setStatus(kind + " url скопирован: " + it.url, "ok"); };
      el.appendChild(img);
    });
  }
  fill($("avatars"), av, "avatar"); fill($("backgrounds"), bg, "background");
}

// ----------------------------- wire up -----------------------------------
$("btn-new").onclick = newScenario;
$("btn-load-demo").onclick = function () { openScenario("soldatik-demo.json"); };
$("btn-validate").onclick = doValidate;
$("btn-save").onclick = doSave;
$("btn-preview").onclick = doPreview;
$("btn-export").onclick = function () { doExport("mp4"); };
$("btn-export-gif").onclick = doExportGif;
$("btn-record-poll").onclick = pollRecord;
$("btn-del-sel").onclick = deleteSelected;
$("btn-relayout").onclick = relayout;
$("init-overlay-close").onclick = closeInitOverlay;
$("card-filter").oninput = function () { renderCards($("card-filter").value); };
document.querySelectorAll("#palette [data-add]").forEach(function (b) {
  b.onclick = function () { addNode(b.dataset.add); };
});
["f-name", "f-viewer", "f-seed", "f-match"].forEach(function (id) {
  $(id).addEventListener("change", function () { if (!scenario) return; readMetaInto(scenario); syncToJSON(); });
});
editor.addEventListener("change", function () { loadFromJSON(false); });
document.addEventListener("keydown", function (e) {
  if ((e.key === "Delete" || e.key === "Backspace") && selected && document.activeElement.tagName !== "INPUT"
      && document.activeElement.tagName !== "TEXTAREA" && document.activeElement.tagName !== "SELECT") {
    e.preventDefault(); deleteSelected();
  }
  if (e.key === "Escape") closeInitOverlay();
});

// init
fetch("/api/orchestra/cards").then(function (r) { return r.json(); }).then(function (d) {
  cards = d.cards || []; cardsById = {};
  cards.forEach(function (c) { cardsById[c.id] = c; });
  $("card-count").textContent = "(" + cards.length + ")";
  renderCards("");
  newScenario();
});
fetch("/api/orchestra/cosmetics").then(function (r) { return r.json(); }).then(renderCosmetics);
refreshScenarios();
setStatus("готов. Тяни от ●порта узла к другому — соединишь ребром. Drag пустого = pan, wheel = zoom.", "ok");