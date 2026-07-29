# ReturnClock: контракт сбора и экспорта датасета

## Граница V1

Контур собирает данные для прогноза **естественного времени следующего
содержательного возвращения**. Он не меняет частоту или расписание
уведомлений, не обучает модель и не доказывает causal uplift уведомлений.

Для будущего uplift-планировщика каждая попытка должна иметь сохранённые
`experiment_id`, `treatment_arm`, `assignment_probability` и control/no-send
ветку. Статус `provider_accepted` означает только принятие провайдером, а не
доставку или прочтение.

## Потоки данных

1. Главный WebApp и отдельная Arena-страница создают analytics-v2 сессию с
   новым UUID на каждый документ: канонический `source`, IANA timezone, UTC
   offset, entrypoint и однократная attribution уведомления. Бой считается
   только на терминальном результате и дедуплицируется по `matchId`.
2. Короткий background отправляет heartbeat. После 30 минут inactivity старая
   сессия закрывается временем ухода, а возврат начинает новую.
3. Reload, BFCache и пересекающиеся вкладки объединяются при офлайн-
   sessionization, если разрыв не превышает 30 минут.
4. Решение о discretionary-уведомлении сохраняется до постановки в outbox.
   Обычные shop/generator/social уведомления автоматически получают
   observational envelope: иначе они были бы невидимыми confounders.
5. Dispatcher и прямые административные Telegram/Android broadcast'ы пишут
   per-recipient observational assignment и идемпотентные provider events.
   First-party deep-link связывает первую запущенную сессию с
   decision/delivery/outbox; сама такая attribution считается подтверждённым
   open даже если best-effort callback записался на миллисекунды позже старта.
   Внешние URL никогда не получают внутренние ID.
6. При естественном возврате ещё не отправленные discretionary reminders
   отменяются и получают событие `cancelled`.
7. Outbox возвращает в `pending` зависшие `sending` lease старше 10 минут,
   а исчерпавшие лимит попыток — в `failed`, поэтому crash dispatcher не
   теряет доставку навсегда.

## Хранилище

### `user_sessions`

Помимо базовой активности хранятся:

- `analytics_version`, `source`, `timezone`, `utc_offset_minutes`;
- `entrypoint`, `returnclock_decision_id`, `returnclock_delivery_id`;
- `last_heartbeat_at`, `last_resumed_at`, `resume_count`;
- упорядоченные `screens_visited`, дедуплицированные terminal `battle_ids`,
  счётчики боёв и кейсов.

При coalesce нескольких heartbeat/update payload `battle_ids` объединяются
как дедуплицированный список с сохранением порядка первого появления.

Update/end всегда ограничены парой `(user_id, session_id)`. Публичный API не
может переоткрыть завершённую сессию.

### `returnclock_decisions`

Одна строка — immutable assignment envelope плюс изменяемый execution status:

- eligible actions, решение и причина;
- policy/model version;
- experiment, arm и propensity;
- eligible/planned/expiry timestamps;
- prediction/context;
- связанный outbox и cancellation.

### `returnclock_delivery_events`

Append-only события с уникальным `event_id`: канал, provider message,
delivery/outbox, тип события, server/client time и связанная сессия. Повторная
доставка того же события безопасна.

## Training JSONL

Формат `extraarena_returnclock_dataset_v1`:

- первая строка — header с точным allowlist `feature_columns`, правилами
  meaningful-session и сводкой качества;
- каждая следующая строка содержит отдельно:
  - `features` — только значения, известные на prediction cutoff;
  - `label` — survival event/censoring и observation window;
  - `post_cutoff` — delivery/treatment/assignment metadata, запрещённая как
    вход модели;
  - `user_id_hash` — псевдоним только для grouped split;
  - `prediction_cutoff_at` — только для временного split и аудита.

В `post_cutoff` приём запроса провайдером хранится отдельно как
`provider_accepted_count`. Он не увеличивает `notification_sent_count`:
последний учитывает только отдельные `sent`/`delivered`/`shown` события.
Оба сигнала запрещены как model features и делают интервал неорганическим,
но только delivery/open/deep-link события являются свидетельством фактической
экспозиции или взаимодействия.

Сессия считается содержательной строго по формуле:
`(duration_seconds >= 120 AND screen_count >= 2) OR battles_completed > 0 OR
cases_opened > 0`. То есть один только долгий экран не проходит gate, а
завершённый бой или открытие кейса проходят независимо от длительности.
Незавершённая строка старше 30 минут может быть закрыта по последнему heartbeat
и помечается `last_session_end_inferred=true`.

Интервалы с send assignment, provider exposure, open или deep-link attribution
не считаются чистыми organic examples. Временной split обязателен; строки
одного `user_id_hash` нельзя распределять между train и validation случайно.

## Экспорт

```bash
RETURNCLOCK_DATASET_SALT='<at-least-32-random-secret-bytes>' \
RETURNCLOCK_DATASET_SALT_KEY_ID='returnclock-prod-2026-08' \
python3 scripts/export_returnclock_dataset.py \
  --start 2026-08-01T00:00:00Z \
  --end 2026-09-01T00:00:00Z \
  --output datasets/returnclock/2026-08.jsonl
```

Экспорт:

- читает только людей (`users.is_bot=false`);
- читает три сырых потока в одной read-only `REPEATABLE READ` транзакции;
- при пропущенном `--end` использует десятиминутный ingestion safety lag
  (переопределяется `--safety-lag-minutes`);
- добавляет `max(28 дней, label horizon)` pre-cutoff history, то есть до
  31 дня при максимальном horizon;
- читает каждый поток keyset-страницами не более 50 000 строк в одной
  `REPEATABLE READ` транзакции; общий fail-closed ceiling — 1 000 000 строк
  на поток;
- разделяет две границы: эксклюзивный `dataset_end` ограничивает event time и
  censoring, а более поздний `ingested_before` ограничивает
  `user_sessions.created_at`, `returnclock_decisions.created_at` и
  `returnclock_delivery_events.created_at`; поэтому позднее обновление статуса
  не выбрасывает существующий assignment и не превращает treated interval в
  organic;
- не мутирует production;
- атомарно пишет файл с правами `0600`;
- fail-closed, если любой поток достиг `--limit`, чтобы усечение не породило
  ложные no-return/censoring labels.

Текущая реализация exporter/splitter материализует выбранное ограниченное
окно в памяти. Потолок защищает от silent truncation, но не является
рекомендованным размером одного запуска: крупные окна нужно подбирать с учётом
доступной RAM.

Salt должен содержать не менее 32 байт случайного секретного материала.
Версия ключа записывается в header как non-secret
`pseudonymization_key_id`; сам salt никогда не попадает в JSONL или MCP.
JSONL остаётся
**псевдонимизированным, не анонимным**: его нельзя публиковать или переносить из
закрытого training storage.

Перед обучением `split_returnclock_training_dataset` создаёт
`train/validation/test` bundle только из
`post_cutoff.organic_candidate=true`. Treated-строки остаются в исходном
audit-экспорте и учитываются в manifest как исключённые, но не попадают в
natural-return train/eval файлы. Все строки одного `user_id_hash` остаются в
одном split, а группы пользователей упорядочиваются по первому
`prediction_cutoff_at`; строки, пересекающие более позднюю временную границу
своего cohort, исключаются. Manifest фиксирует source SHA-256, key ID,
training filter, границы и число исходных/исключённых/обучающих строк.
Raw readiness считается истинным только если уже существуют минимум три
organic user groups с тремя строго упорядоченными first-cutoff cohorts — те же
минимальные условия, которые нужны обязательному train/validation/test split.
`inspect_training_export` возвращает non-secret `ingested_before`, aggregate
bundle checksum/mode и все exclusion counters без строк датасета.

Новый путь публикуется через временный каталог и same-filesystem rename.
`overwrite=true` имеет rollback при штатной ошибке, но не является
crash-atomic при `SIGKILL`; для обучающих handoff следует использовать новый
versioned путь с `overwrite=false`, а внешний указатель переключать только
после повторной валидации.

## Минимальные quality gates перед обучением

- `min_analytics_version=2`;
- минимум три organic user groups и три first-cutoff cohorts;
- ноль silent truncation;
- отдельно посчитать inferred, unfinished, legacy и non-meaningful exclusions;
- проверить долю right-censored и treated intervals;
- обучать estimator только на `header.feature_columns`;
- natural-return baseline оценивать отдельно на `organic_candidate=true`;
- causal send-time политику не обучать до рандомизированного no-send/control
  пилота.
