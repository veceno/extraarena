# ExtraArena RLHF environment

Автономная Arena-среда для human/LLM/RL-vs-model боёв, V5-трейсов и
training-ready управления приватными датасетами. Полная документация:
[`DOCS.md`](./DOCS.md). MCP skills:
[`../.codex/skills/extra-rlhf/`](../.codex/skills/extra-rlhf/).

## Быстрый старт

```bash
# Web Arena @ 127.0.0.1:8090
./rlhf_env/start_rlhf_env.sh

# MCP stdio с checkout-local окружением:
./rlhf_env/start_rlhf_env.sh setup --python /path/to/python3.13
./rlhf_env/.venv/bin/python \
  -m rlhf_env.mcp_server \
  --models-dir ai/models \
  --sessions-dir rlhf_env/sessions \
  --datasets-dir datasets \
  --cards-path ai/cards.json
```

Не используйте случайный bare `python3`: версия интерпретатора не гарантирует
наличие NumPy/ONNX Runtime. Зависимости, включая `asyncpg` и `python-dotenv`,
находятся в `rlhf_env/requirements.txt`.

## Два контура

### Headless Arena

- тот же `core.engine.ArenaEnvironment`, каталог из `ai/cards.json` и
  `mana_draw`;
- адаптеры legacy/V4/V5 и baselines;
- browser human-vs-model, LLM player и auto-play RL-vs-RL;
- manifest + omniscient V5
  `battles/<battle_id>/v5/{meta,turns,actions}.jsonl`;
- поля для Metronome/TimeStamp в общем V5-контракте, которые считаются
  обучающими labels только после поступления реальных production-наблюдений;
- глубокая проверка state/action continuity, terminal outcome,
  catalog/card-count и `degraded`.

Headless-контур пишет только в `RLHF_SESSIONS_DIR` и не подключается к
production. Поэтому его `training_ready` — только backward-compatible alias
для `v5_policy_training_ready` (`training_ready_scope="v5_policy_only"`).
Headless policy traces пригодны для V5 policy и, после собственных eligibility
checks, Nemesis Lite, но не доказывают готовность Metronome/TimeStamp. CPU/
wall-clock длительность, LLM latency и синтетические задержки не являются
human timing labels.

TimeStamp loader обязан использовать явный prebattle allowlist: только колода
или пара колод, `starting_player` и заранее одобренные признаки, доступные до
старта боя. `duration_seconds`, `turns`, `finished_at` и производные — только
labels/audit. Передавать целиком `timestamp_features` или `meta` как model input
запрещено из-за target leakage.

### Private dataset toolbox

MCP умеет:

- показать readiness и inventory;
- inspect/validate V5, Nemesis и ReturnClock artifacts;
- read-only экспортировать terminal production V5 bundles;
- materialize V5 transport в canonical `rlhf_v5_storage_v1`;
- извлечь единый Nemesis dataset (Lite base + optional standard extension);
- всегда материализовать Nemesis Lite deck-grouped split и, когда проходят
  Standard gates, добавить player-disjoint/chronological/deck-grouped views;
- экспортировать cutoff-safe ReturnClock survival dataset;
- разбить ReturnClock на organic-only grouped-by-user temporal
  train/validation/test с leakage gate.

Все пути ограничены `RLHF_DATASETS_DIR` (`datasets/`), symlink/traversal
запрещены, файлы создаются с mode `0600`. Новый путь собирается во временной
директории и публикуется same-filesystem rename. Overwrite откатывает обычные
перехваченные ошибки, но не crash-atomic при `SIGKILL`/power loss; используйте
versioned destination с `overwrite=false`. Raw player IDs и raw privacy salt не
являются MCP-опциями; production V5/Nemesis также заменяет battle/match IDs
экспорт-локальными `record_<hex>` alias, чтобы opaque ID не мог содержать raw
user ID.

## Production opt-in

Локальные inspect/validate/materialize/split доступны по умолчанию. Production
reads fail-closed до явного запуска:

```bash
export RLHF_ENABLE_PRODUCTION_DATASETS=1
export RETURNCLOCK_DATASET_SALT='<export-specific secret, at least 32 bytes>'
export RETURNCLOCK_DATASET_SALT_KEY_ID='<non-secret rotation id>'
```

ReturnClock output псевдонимизирован, но не анонимен. Salt остаётся только в
environment; key id нужен для аудита ротаций. Никогда не помещайте DSN или salt
value в MCP config/arguments.

## MCP dataset workflow

1. `get_training_data_status`.
2. Export в новый путь (`overwrite=false`).
3. `inspect_training_export`.
4. `validate_training_export` → требовать `ok=true` и readiness именно
   обучаемого контура. Для headless V5 это
   `v5_policy_training_ready=true`; timing readiness проверяется отдельно.
5. V5: `materialize_v5_training_dataset`, затем повторная validation.
   Nemesis: `split_nemesis_training_dataset`; Lite-only handoff допустим,
   Standard требует `training_ready_standard=true`, минимум шесть игроков,
   три pairwise-disjoint human-human боя, три matchup group и три cutoff cohort.
   Player-disjoint view исключает и учитывает cross-partition battles,
   остальные Standard views проверяют deck/time drift.
   ReturnClock: `split_returnclock_training_dataset`.
6. Зафиксировать SHA-256, format/version, validation summary,
   split/materialization manifest, catalog/weights provenance и privacy key id.

Для action training допустимы только строки `accepted is True`; rejected rows
остаются audit evidence. ReturnClock estimator читает только
`header.feature_columns`; `post_cutoff`, `user_id_hash` и cutoff timestamp в
features не входят. Raw ReturnClock export может хранить treated intervals
только для аудита; natural-return train/eval читает исключительно organic-only
split files, где каждая строка имеет
`post_cutoff.organic_candidate=true`, а manifest учитывает исключённые treated
rows. Causal notification policy нельзя обучать до рандомизированного
no-send/control пилота.

ReturnClock production snapshot читается keyset-страницами до 50,000 строк,
максимум 1,000,000 строк на raw stream, внутри одной repeatable-read
transaction. Exclusive `end_at` ограничивает event time/censoring, а отдельный
более поздний `ingested_before` — создание session/decision/delivery rows.
Так позднее status/update не удаляет старый assignment. Safety lag применяется
при отсутствующем явном `end`; исторический explicit `end` используется как
есть.
Достижение лимита означает неполный export — независимо цензурированные
выгрузки нельзя склеивать молча. Текущая реализация держит выбранные raw
потоки и split window в памяти, поэтому окно до потолка нужно подбирать с
учётом RAM, а не считать путь полностью streaming.

## Player loop

LLM worker обязан владеть полным lifecycle в одном persistent MCP process:
`start_series → compact get_state → indexed submit_action → finish/next`.
`match_id` нельзя переносить в другой MCP процесс.

Перед игрой LLM читает обязательный гайд:
[`../.codex/skills/extrarlhf-player/references/arena-strategy-guide.md`](../.codex/skills/extrarlhf-player/references/arena-strategy-guide.md).
Используйте `compact=true`, `history_limit=8`, `legal_action_index` и
`compact_response=true`.

## MCP wire

MCP — stdio JSON-RPC 2.0. `tools/call` возвращает JSON text в
`content[0].text`, тот же объект в `structuredContent` и `isError`. Клиентам
следует читать `structuredContent`, если он доступен.

## Проверка

```bash
PY=./rlhf_env/.venv/bin/python

echo '{"jsonrpc":"2.0","id":1,"method":"tools/list"}' \
  | "$PY" -m rlhf_env.mcp_server

"$PY" -m pytest rlhf_env/tests -q
"$PY" rlhf_env/tests/smoke_e2e.py --port 8096 --models random
```

Детальные tool schemas и data formats:

- [`../.codex/skills/extra-rlhf/references/mcp-tools.md`](../.codex/skills/extra-rlhf/references/mcp-tools.md)
- [`../.codex/skills/extra-rlhf/references/data-format.md`](../.codex/skills/extra-rlhf/references/data-format.md)
- [`../docs/returnclock-dataset-contract.md`](../docs/returnclock-dataset-contract.md)
