// RLHF Arena — упрощённый рендер арены поверх arena_styles.css
// Получает state через WebSocket (/ws/groups/{gid}/battles/{bid}),
// отправляет {type:"action", index:N} при клике на карту/цель/конец хода.

(function () {
  "use strict";

  const params = new URLSearchParams(window.location.search);
  const gid = params.get("group_id");
  const bid = params.get("battle_id") || ("b_" + Date.now().toString(36));
  const WS_URL = `/ws/groups/${gid}/battles/${bid}`;

  const els = {
    loading: document.getElementById("loading-arena"),
    container: document.getElementById("arena-container"),
    opponentPanel: document.getElementById("opponent-panel"),
    opponentBoard: document.getElementById("opponent-board"),
    playerPanel: document.getElementById("player-panel"),
    playerHand: document.getElementById("player-hand"),
    playerBoard: document.getElementById("player-board"),
    history: document.getElementById("arena-action-history"),
    endTurn: document.getElementById("end-turn-btn"),
    status: document.getElementById("battle-status"),
  };

  let ws = null;
  let currentLegal = []; // [{index, action}]
  let currentState = null;
  let selectedHandIndex = null; // hand_index выбранной карты
  let awaitingTarget = false;
  let seriesIndex = 1;   // 1-based — номер текущего боя в серии
  let seriesTotal = 1;   // сколько боёв в серии всего

  // ----------------------------------------------------------------------
  // Утилиты
  // ----------------------------------------------------------------------
  function el(tag, cls, text) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }

  function hpBlock(side, player) {
    const wrap = el("div", side === "player" ? "player-hp-block" : "opponent-hp-block");
    const name = el("div", side + "-name-block");
    const avatar = el("div", side + "-avatar", (player.name || "P").charAt(0));
    name.appendChild(avatar);
    name.appendChild(el("div", null, (player.name || (side === "player" ? "Игрок" : "Противник"))));
    const hp = el("div", "hp-value-large", `${player.hp}/${player.max_hp}`);
    wrap.appendChild(name);
    wrap.appendChild(hp);
    if (player.mana != null) {
      const mana = el("div", "mana-counter", `${player.mana}/${player.max_mana}`);
      wrap.appendChild(mana);
    }
    return wrap;
  }

  function cardEl(card, side, opts = {}) {
    const c = el("div", "board-slot");
    if (opts.targetable) c.classList.add("targetable-friendly");
    if (opts.attackTarget) c.classList.add("attack-target");
    c.appendChild(el("div", "attack", String(card.attack)));
    c.appendChild(el("div", null, card.name || "—"));
    c.appendChild(el("div", "hp", `${card.hp}/${card.max_hp}`));
    if (opts.onClick) c.addEventListener("click", opts.onClick);
    if (opts.cost != null) {
      const mc = el("div", "mana-circle", String(opts.cost));
      mc.style.position = "absolute";
      mc.style.top = "4px";
      mc.style.right = "4px";
      c.appendChild(mc);
    }
    return c;
  }

  function handCardEl(card, idx, playable) {
    const c = el("div", "hand-card" + (playable ? " card-playable" : " card-disabled"));
    c.appendChild(el("div", null, card.name || "—"));
    c.appendChild(el("div", null, `A${card.attack} / H${card.hp}`));
    const mc = el("div", "mana-circle", String(card.mana_cost));
    c.appendChild(mc);
    if (playable) {
      c.addEventListener("click", () => onHandClick(idx));
    }
    return c;
  }

  // ----------------------------------------------------------------------
  // Клики
  // ----------------------------------------------------------------------
  function onHandClick(handIdx) {
    // Найти первое legal действие с этим hand_index
    const candidates = currentLegal.filter(la =>
      la.action.type === "play_card" && la.action.hand_index === handIdx
    );
    if (!candidates.length) return;
    if (candidates.length === 1) {
      sendAction(candidates[0].index);
      return;
    }
    // Есть выбор цели → нужно кликнуть по цели
    selectedHandIndex = handIdx;
    awaitingTarget = true;
    els.status.textContent = "Выберите цель для карты";
  }

  function onBoardTargetClick(targetId) {
    if (!awaitingTarget) return;
    const action = currentLegal.find(la =>
      la.action.type === "play_card" &&
      la.action.hand_index === selectedHandIndex &&
      la.action.target_id === targetId
    );
    if (action) {
      sendAction(action.index);
    }
    selectedHandIndex = null;
    awaitingTarget = false;
  }

  function onEndTurnClick() {
    const endTurn = currentLegal.find(la => la.action.type === "end_turn");
    if (endTurn) sendAction(endTurn.index);
  }

  function sendAction(index) {
    if (!ws || ws.readyState !== WebSocket.OPEN) return;
    ws.send(JSON.stringify({ type: "action", index }));
  }

  // ----------------------------------------------------------------------
  // Рендер state
  // ----------------------------------------------------------------------
  function render(state, legalActions, yourTurn) {
    currentLegal = legalActions || [];
    currentState = state;

    // P2 (opponent) — наверху
    els.opponentPanel.innerHTML = "";
    els.opponentPanel.appendChild(hpBlock("opponent", state.p2));
    els.opponentBoard.innerHTML = "";
    (state.p2.board || []).forEach(c => {
      const isTarget = awaitingTarget && currentLegal.some(la =>
        la.action.type === "play_card" && la.action.target_id === c.instance_id
      );
      els.opponentBoard.appendChild(cardEl(c, "opponent", {
        targetable: isTarget,
        onClick: isTarget ? () => onBoardTargetClick(c.instance_id) : null,
      }));
    });

    // Action history
    els.history.innerHTML = "";
    if (state.action_history) {
      for (const line of state.action_history.slice(-20)) {
        els.history.appendChild(el("div", null, line));
      }
    }

    // P1 (player) — снизу
    els.playerBoard.innerHTML = "";
    (state.p1.board || []).forEach(c => {
      els.playerBoard.appendChild(cardEl(c, "player", {}));
    });
    els.playerHand.innerHTML = "";
    const handActions = new Set(
      currentLegal.filter(la => la.action.type === "play_card").map(la => la.action.hand_index)
    );
    (state.p1.hand || []).forEach((c, i) => {
      els.playerHand.appendChild(handCardEl(c, i, handActions.has(i)));
    });
    els.playerPanel.innerHTML = "";
    els.playerPanel.appendChild(hpBlock("player", state.p1));

    // End-turn кнопка
    const hasEndTurn = currentLegal.some(la => la.action.type === "end_turn");
    els.endTurn.disabled = !hasEndTurn;
    els.endTurn.classList.toggle("card-disabled", !hasEndTurn);

    // Статус
    const series = (seriesTotal > 1) ? ` [Бой ${seriesIndex}/${seriesTotal}]` : "";
    if (state.is_over) {
      els.status.textContent = `Бой окончен: ${state.status || "?"}${series}`;
    } else if (yourTurn) {
      els.status.textContent = `Ваш ход (P1)${series}`;
    } else {
      els.status.textContent = `Ход противника…${series}`;
    }
  }

  // ----------------------------------------------------------------------
  // WS lifecycle
  // ----------------------------------------------------------------------
  function connect() {
    if (!gid) {
      els.loading.textContent = "Не указан group_id";
      return;
    }
    const proto = location.protocol === "https:" ? "wss:" : "ws:";
    ws = new WebSocket(`${proto}//${location.host}${WS_URL}`);
    ws.onopen = () => {
      els.loading.style.display = "none";
      els.container.style.display = "block";
    };
    ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data);
        if (data.type === "state") {
          if (typeof data.series_index === "number") seriesIndex = data.series_index;
          if (typeof data.series_total === "number") seriesTotal = data.series_total;
          render(data.state, data.legal_actions, data.your_turn);
        } else if (data.type === "result") {
          const winner = data.battle_log?.result?.winner_user_id;
          const seriesIndex = data.series_index;
          const seriesTotal = data.series_total;
          const nextBid = data.next_battle_id;
          let msg = `Бой ${seriesIndex}/${seriesTotal} завершён. `;
          msg += winner === 1000 ? "Победил: ВЫ" : winner === 2000 ? "Победил: БОТ" : "Ничья";
          if (nextBid) {
            msg += ` — следующий через 3 сек…`;
            els.status.textContent = msg;
            // Перезагружаем на следующий бой серии (battle_id меняется)
            setTimeout(() => {
              window.location.href = `/battle?group_id=${encodeURIComponent(gid)}&battle_id=${encodeURIComponent(nextBid)}`;
            }, 3000);
          } else {
            msg += ` — СЕРИЯ ЗАВЕРШЕНА!`;
            els.status.textContent = msg;
            // Через 5 сек — на страницу группы
            setTimeout(() => {
              window.location.href = `/groups/${encodeURIComponent(gid)}`;
            }, 5000);
          }
        } else if (data.type === "error") {
          els.status.textContent = "Ошибка: " + data.message;
        } else if (data.type === "pong") {
          // ignore
        }
      } catch (e) {
        console.error("bad message", e);
      }
    };
    ws.onclose = () => {
      els.status.textContent = "Соединение закрыто";
    };
    ws.onerror = () => {
      els.status.textContent = "Ошибка WebSocket";
    };
  }

  els.endTurn.addEventListener("click", onEndTurnClick);
  connect();
})();
