# Мини-гайд: игра по MCP в мультипотоке (через ultracode Workflow)

Цель: играть бои ExtraArena против бот-моделей **по-настоящему** (LLM-агент решает каждый ход) через rlhf MCP-инструменты, **многопоточно** — N боёв параллельно одним Workflow, один общий bot-brain на все.

Проверено на серии 10 боёв vs `extra-lr-v4-max` (см. `docs/ArenaBalanceAudits/2026-06-25_vs-extra-lr-v4-max_10battles.md`).

---

## Почему так, а не «просто дёргать MCP из каждого агента»

`rlhf_env/mcp_server.py` — это **in-process** `HeadlessHub` + `MCPServer` с методом `_tool(name, args)`. Проблема: каждый bash-вызов — отдельный Python-процесс, in-process движок умирает по выходу. Нельзя «держать одну партию» между вызовами, и нельзя переиспользовать один ONNX bot-brain.

**Решение — персистентный bridge-демон:** один долгоживущий процесс поднимает `HeadlessHub`+`MCPServer` (один bot-brain) и слушает unix-сокет. Короткие CLI-клиенты (по одному на ход) подключаются, дёргают `_tool`, отключаются. Так 10 параллельных боёв едут на **одном** движке и **одном** ONNX-сеансе, а инференс бота параллелится (проверено: 3 конкурентных bot-turn'а за 3.2с, не 15с).

---

## Шаг 1. Bridge-демон (`rlhf_mcp_bridge.py`)

Ставит `HeadlessHub`+`MCPServer` и слушает `/tmp/rlhf_mcp.sock`. Протокол: клиент шлёт одну JSON-строку `{tool, args, id}`, сервер отвечает одной строкой `{id, result}` (или `{id, error}`) и закрывает соединение.

```python
#!/usr/bin/env python3
"""Persistent unix-socket bridge to rlhf MCPServer._tool (one bot-brain, shared)."""
from __future__ import annotations
import asyncio, json, os, socket, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
logging.basicConfig(level=logging.WARNING)
from rlhf_env.mcp_server import HeadlessHub, MCPServer
from rlhf_env.components.policy_registry import PolicyRegistry

SOCK = "/tmp/rlhf_mcp.sock"

async def handle(conn, srv):
    req = None
    try:
        buf = b""
        conn.setblocking(False)
        while True:
            try:
                chunk = await asyncio.to_thread(lambda: conn.recv(65536))
            except BlockingIOError:
                await asyncio.sleep(0.005); continue
            if not chunk: break
            buf += chunk
            if b"\n" in buf or len(buf) > 1_000_000: break
        req = json.loads(buf.decode("utf-8").strip() or "{}")
        if "tool" in req:
            result = await srv._tool(req["tool"], req.get("args", {}) or {})
        else:
            result = await srv.dispatch(req.get("method", "tools/call"), req.get("params", {}))
        conn.setblocking(True)
        conn.sendall((json.dumps({"id": req.get("id"), "result": result}, ensure_ascii=False) + "\n").encode())
    except Exception as e:
        try:
            conn.setblocking(True)
            conn.sendall((json.dumps({"id": (req or {}).get("id"), "error": str(e)}, ensure_ascii=False) + "\n").encode())
        except Exception: pass
    finally:
        try: conn.close()
        except Exception: pass

async def main():
    reg = PolicyRegistry.scan("ai/models")
    hub = HeadlessHub(sessions_dir="rlhf_env/sessions", models_dir="ai/models", cards_path="ai/cards.json")
    srv = MCPServer(hub, reg)
    if os.path.exists(SOCK): os.remove(SOCK)
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.bind(SOCK); s.listen(64)        # НЕ setblocking(False) — accept в to_thread должен блокироваться
    print(f"[bridge] listening on {SOCK}", flush=True)
    while True:
        conn, _ = await asyncio.to_thread(s.accept)   # блокирующий accept в потоке
        asyncio.create_task(handle(conn, srv))         # async handle в event-loop (т.к. _tool async)

if __name__ == "__main__":
    asyncio.run(main())
```

Запуск (из корня репо, чтобы `ai/models`/`ai/cards.json` резолвились):
```bash
nohup python3 /tmp/rlhf_mcp_bridge.py > /tmp/rlhf_mcp_bridge.log 2>&1 &
# дождаться сокета
for i in $(seq 1 50); do [ -S /tmp/rlhf_mcp.sock ] && break; sleep 0.1; done
# sanity-check:
python3 -c "import socket,json; s=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM); s.connect('/tmp/rlhf_mcp.sock'); s.sendall((json.dumps({'tool':'list_models','args':{},'id':1})+'\n').encode()); b=b''; 
import sys
exec('while True:\n c=s.recv(65536)\n if not c: break\n b+=c')
print(b.decode()[:200])"
```

**Гоча:** сокет НЕ делать non-blocking — `asyncio.to_thread(s.accept)` в потоке сразу кидает `BlockingIOError`. `handle` остаётся async (т.к. `srv._tool` — coroutine); `accept` блокируется в потоке, потом `handle` планируется в event-loop.

---

## Шаг 2. CLI-клиент (`mcp_cli.py`)

Короткие подкоманды, по одной на ход. Агент выбирает ход **по индексу** легального действия (не конструирует JSON — надёжнее). Компактный state держит контекст агента маленьким.

```
python3 /tmp/mcp_cli.py start [--seed N] [--first p1|p2|random]   # start_series + state + numbered legal
python3 /tmp/mcp_cli.py state <mid>                                # state + legal (auto-advance бота)
python3 /tmp/mcp_cli.py act <mid> <index>                          # legal[index] → submit → next state + legal
python3 /tmp/mcp_cli.py next <gid>                                 # next_battle (для серии >1 боя)
python3 /tmp/mcp_cli.py manifest <gid>                             # итог группы (p1_wins/p2_wins)
```

Компактный вывод:
```
--- turn=7 me_p1 hp=55 mana=4/4 | bot hp=39 | is_my=True over=False
    match=m_XXX group=YYY
    my_board:  b0:Стив a8 h8 ready
    bot_board: B0:Альфонс a2 h5
    hand:      [0]Сакура c2 a4 h4 [1]Леви c3 a6 h2 charge
    legal:     [0] play h0 Сакура(c2,a4,h4) pos0  [1] atk b0:Стив->B0:Альфонс(a2,h5)  [2] atk b0:Стив->HERO  [3] end_turn
```

Ключевые моменты реализации (см. полный файл ниже):
- **`act` всегда показывает мой следующий ход или game_over:** после submit, если не мой ход и не конец → `_to_my_turn()` вызывает `advance_bot` пока бот не доиграет. `submit_action` с `end_turn` **НЕ прокручивает бот-ход автоматически** — без этого агент увидит «is_my=False, legal=(none)» и запутается.
- **Терминальный winner берётся из свежего `get_state`:** ответ `submit_action` несёт **stripped** state (без `winner_id`/hp). После каждого `act` — `call("get_state")` заново. `winner_id`: 1000 = я (p1), 2000 = бот.
- `build_action(legal)`: `play_card` → `{type, hand_index, target_position, target_id}`; `attack` → `{type, attacker_id, target_id, target_is_hero}`; `end_turn` → `{type:"end_turn"}`.

Полный клиент (self-contained per вызов — `act <mid> <index>` сам рефетчит state, берёт legal[index], сабмитит):
```python
#!/usr/bin/env python3
import json, socket, sys
SOCK = "/tmp/rlhf_mcp.sock"

def call(tool, args=None):
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect(SOCK); s.setblocking(True)
    s.sendall((json.dumps({"tool": tool, "args": args or {}, "id": 1}) + "\n").encode())
    buf = b""
    while True:
        c = s.recv(65536)
        if not c: break
        buf += c
    s.close()
    resp = json.loads(buf.decode().strip())
    return {"__error": resp["error"]} if "error" in resp else resp.get("result", {})

def _atk(o): return o.get("attack") if o.get("attack") is not None else o.get("atk", 0)
def _hp(o):  return o.get("hp_current") if o.get("hp_current") is not None else o.get("hp", 0)
def _short(u): return str(u)[-6:] if u else "?"

def build_action(legal):
    t = legal.get("type")
    if t == "play_card": return {"type":"play_card","hand_index":legal.get("hand_index"),
                                 "target_position":legal.get("position") or 0,"target_id":legal.get("target_id")}
    if t == "attack":    return {"type":"attack","attacker_id":legal.get("attacker_id"),
                                 "target_id":legal.get("target_id"),"target_is_hero":legal.get("target_is_hero", False)}
    return {"type":"end_turn"}

def _to_my_turn(mid, st):
    g = 0
    while not (st.get("is_ended") or st.get("game_over")) and not st.get("is_my_turn") and g < 40:
        r = call("advance_bot", {"match_id": mid})
        st = r if isinstance(r, dict) and ("is_my_turn" in r or "is_ended" in r) else call("get_state", {"match_id": mid})
        g += 1
    return st

# ... fmt_board/fmt_hand/fmt_legal/show — форматируют compact state (см. репо-копию) ...

def cmd_start(args):
    spec = {"p2_model":"extra-lr-v4-max","battles_planned":1,
            "starting_player": args.get("--first","random"), "seed": int(args.get("--seed","0") or 0),
            "deck_strategy_p1":"random_arenaenv","deck_strategy_p2":"random_arenaenv",
            "p1_name":"Claude","p2_name":"extra-lr-v4-max"}
    r = call("start_series", {"spec": spec}); mid=r["match_id"]; gid=r["group_id"]
    st = _to_my_turn(mid, call("get_state", {"match_id": mid}))
    print(f"START group={gid} match={mid} opponent={r.get('opponent')}"); show(st, mid, gid)

def cmd_act(mid, idx):
    st = call("get_state", {"match_id": mid})
    if st.get("is_ended") or st.get("game_over"): print("ALREADY OVER"); show(st, mid); return
    st = _to_my_turn(mid, st)
    legal = st.get("legal_actions") or []
    if idx < 0 or idx >= len(legal): print(f"BAD index {idx}; legal len={len(legal)}"); show(st, mid); return
    resp = call("submit_action", {"match_id": mid, "action": build_action(legal[idx])})
    print(f"ACT[{idx}] {json.dumps(build_action(legal[idx]), ensure_ascii=False)}")
    ns = _to_my_turn(mid, call("get_state", {"match_id": mid}))   # ВСЕГДА свежий полный get_state
    show(ns, mid)
# main: dispatch start/state/act/next/manifest
```

Полный файл с форматтерами — в репо: `rlhf_env/tools/mcp_cli.py` (или `/tmp/mcp_cli.py`).

---

## Шаг 3. Workflow (ultracode) — N параллельных боёв + аналитик

ultracode ON → используй `Workflow`. Паттерн: **N player-агентов параллельно (по бою на агента), потом барьер, потом 1 аналитик**, который читает все N нарративов.

```js
export const meta = {
  name: 'play-N-vs-model',
  description: 'Play N real LLM-decided battles vs model via rlhf MCP bridge, narrate, assess',
  phases: [{ title: 'Play' }, { title: 'Assess' }],
}
const CLI = 'python3 /tmp/mcp_cli.py'   // или путь в репо
const MODEL = 'extra-lr-v4-max'

const PLAYER_SCHEMA = { type:'object', properties:{
  idx:{type:'integer'}, seed:{type:'integer'},
  result:{type:'string', enum:['win','lose','draw','error']},
  winner:{type:'string'}, my_hp_final:{type:'integer'}, bot_hp_final:{type:'integer'},
  turns:{type:'integer'}, first_player:{type:'string'}, started_ok:{type:'boolean'},
  key_decisions:{type:'array', items:{type:'string'}},
  narrative:{type:'string', description:'Detailed turn-by-turn narrative'},
}, required:['idx','seed','result','started_ok','narrative'] }

function playerPrompt(seed, idx) {
  return `Ты играешь КАК ОПЫТНЫЙ ИГРОК (Hearthstone-подобная) против бота "${MODEL}". Бой №${idx}.
Решай КАЖДЫЙ ход сам, стратегически, без рандома. Цель — победить.

ИНТЕРФЕЙС (bash):
1) ${CLI} start --first random --seed ${seed}      → "START ... match=m_XXX group=YYY" + state + numbered legal
2) ${CLI} act <MID> <INDEX>                          → следующий state + legal (выбери лучший индекс)
3) ${CLI} state <MID>                                → обновить (если действие отклонено)
4) ${CLI} manifest <GID>                             → подтвердить итог

СТОП когда увидишь "over=True" или "WINNER=". Если >60 действий без конца — manifest и фиксируй.

СТРАТЕГИЯ: lethal (сумма face ≥ bot hp) → только лицо; иначе выгодные трейды (мой atk ≥ его hp И мой hp > его atk), убирать угрозы с большой атакой, качать ману по кривой, бить лицом если стол мой. "ready"=может атаковать, "zzz"=только что разыгран.

ВЕРНИ структурированный результат: idx, seed, result (win=winner_id 1000), winner, my_hp_final, bot_hp_final, turns, first_player, started_ok, key_decisions, narrative (подробный рассказ боя, ≥250 слов).`
}

phase('Play')
const seeds = [201,202,203,204,205,206,207,208,209,210]
const battles = (await parallel(seeds.map((s,i) => () =>
  agent(playerPrompt(s, i+1), {label:`battle#${i+1}`, phase:'Play', effort:'high', schema:PLAYER_SCHEMA})
))).filter(Boolean)
log(`played ${battles.length}; wins=${battles.filter(b=>b.result==='win').length}`)

phase('Assess')
const ANALYST_SCHEMA = { type:'object', properties:{
  balance_assessment:{type:'string'}, winrate_summary:{type:'string'},
  bot_strength:{type:'string'}, player_strategy_observations:{type:'string'},
  notable_battles:{type:'array', items:{type:'string'}}, recommendations:{type:'string'},
}, required:['balance_assessment'] }
const assessment = await agent(
  `Ты — аналитик баланса ExtraArena. LLM-игрок сыграл ${battles.length} боёв vs ${MODEL}. Данные (JSON):\n${JSON.stringify(battles,null,2)}\n\nОцени баланс: винрейт, сила бота, рабочие стратегии, дисбаланс карт/кривой/первого хода, длину боёв, баги механик. balance_assessment — развёрнутый текст на русском.`,
  {label:'balance-analyst', phase:'Assess', effort:'high', schema:ANALYST_SCHEMA})
return { battles, assessment }
```

---

## Чек-лист перед запуском Workflow

1. **Bridge жив.** Перед Workflow (из main-loop, не из JS — JS не держит процесс):
   ```bash
   pkill -f rlhf_mcp_bridge.py; rm -f /tmp/rlhf_mcp.sock
   nohup python3 /tmp/rlhf_mcp_bridge.py > /tmp/rlhf_mcp_bridge.log 2>&1 &
   # проверка: list_models вернул модели
   ```
   Bridge должен пережить весь workflow. Поставь Monitor на лог (`grep Traceback|Error|Killed`) для раннего пойма краша.
2. **Рабочая директория агентов** = корень репо (чтобы `ai/models` резолвились и CLI-пути работали). Абсолютные пути в CLI — надёжнее.
3. **Каждый агент — свой seed** (бои разные) + `starting_player: random` (или фиксируй для изоляции эффектов).
4. **Серия = 1 бой** (`battles_planned:1`) на агента — параллелизм на уровне агентов, а не `next_battle`. Для серии из N боёв одним «игроком» — один агент + `next`, но это последовательный бот-инференс.
5. **Не перезапускай прод (8081).** Bridge крутит in-process движок rlhf (8090-стек), прод не трогает.

---

## Гоча-список (наступал)

- `submit_action` с `end_turn` **не прокручивает бот-ход**. Клиент обязан `advance_bot` до `is_my_turn`. Иначе агент видит «is_my=False, legal=(none)».
- Терминальный `get_state` несёт `winner_id` (1000=я / 2000=бот) и `is_ended/game_over`; но **ответ `submit_action` — stripped** (без winner/hp). После `act` всегда свежий `get_state`.
- Сокет bridge'а **НЕ non-blocking** (`to_thread(accept)` → `BlockingIOError`). `handle` — async, `accept` — блокирующий в потоке.
- ONNX-инференс **потокобезопасен** и параллелится по матчам; MatchRunner'ы независимы по match_id — конкурентные бои безопасны (проверено на 3 одновременных). Замок НЕ нужен.
- Бот всегда **argmax** (`BOT_MAX_DIFFICULTY="max"`); `spec.difficulty` игнорируется. Ходит автоматически после моего `end_turn` (через `advance_bot`).
- Если будешь читать коды из БД в f-string: `regexp_match(text,'([0-9]{{6}})')` — **двойные фигурные** в f-string, иначе `{6}` интерполируется в `6`.
- Сессии пишутся в `rlhf_env/sessions/<group_id>/` (gitignored) — NDJSON V5-трейсинг; итоги также в `get_battle_group_manifest`.
- **Без image-input** (текущая модель не поддерживает): верификация только через DOM/ARIA/console/network/HTTP и терминальный `get_state`/`manifest`. Никаких `page.screenshot()` / Read PNG.

---

## Файлы

- `rlhf_env/tools/rlhf_mcp_bridge.py` (или `/tmp/rlhf_mcp_bridge.py`) — bridge-демон
- `rlhf_env/tools/mcp_cli.py` (или `/tmp/mcp_cli.py`) — компактный CLI-клиент
- `rlhf_env/mcp_server.py` — источник `HeadlessHub`/`MCPServer._tool` (in-process); инструменты: `start_series`/`next_battle`/`get_state`/`get_legal_actions`/`submit_action`/`advance_bot`/`surrender`/`list_models`/`get_battle_group_manifest`/`get_dataset`
- `rlhf_env/audit/audit_mcp_step1.py` — образец in-process play-loop + `build_action`

## Связанные документы
- `docs/ArenaBalanceAudits/2026-06-25_vs-extra-lr-v4-max_10battles.md` — первый аудит, сделанный по этой методике (3/10, разрыв по первому ходу)
- Memory: `rlhf-env-reference`, `rlhf-difficulty-removed`, `rlhf-components-untracked-no-git`