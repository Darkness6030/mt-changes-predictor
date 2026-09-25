# Research: Backend, NDTP, поток и поставка

Дата: 25 сентября 2026. Статус: исследование и предлагаемые решения; API, сервисы и
измерения ниже пока **не реализованы**. Основание: [PLAN](PLAN.md), [RULES](RULES.md),
[ТЗ](description.md), [датасет](../dataset/README.md),
[NDTP-спецификация](../dataset/docs/Emulator-and-Telematic-Packets-Specification.md).
Официальные документации инструментов проверены по вебу 25.09.2026.

## 1. Решение и связь с баллами

Рекомендуется три контейнера: **Backend FastAPI + отдельный ML inference FastAPI + UI**.
Backend принимает NDTP, хранит ограниченную историю, вызывает единую библиотеку признаков,
запрашивает ML, формирует инциденты и обслуживает UI. ML загружает готовую модель и отвечает
на запросы признаков; обучение запускается отдельно. UI собирается в статические файлы,
отдаётся собственным контейнером с проксированием `/api` в Backend.
Это решение следует критерию трёх модулей, а не добавляет необязательные микросервисы.

| Критерий | Вклад Backend | Доказательство на защите |
|---|---|---|
| Система + Docker, до 6 | Три отдельных сервиса, NDTP → features → model → UI | Чистый Compose, реальный пакет, новый прогноз в Swagger и UI |
| Горизонт и предупреждение, до 4 | Выбор первого посещения `(T+600,T+900]`, фиксация времени алерта | Replay с исходными часами, алерт до события, журнал |
| Производительность/надёжность, до 4 | Ограниченные очереди, reconnect, stale, health, замеры | p50/p95/p99, throughput, график очереди, сценарий обрыва |
| BI, до 6 | Согласованный snapshot, фильтры, обновления, provenance | Быстрый обзор + карточка ТС, возраст данных и основания риска |
| ML, до 6 | Causal-признаки и воспроизводимость модели | Равенство batch/replay; никаких фактов расписания в inference |

Баллы относятся к общим критериям и не суммируются как гарантированная прибавка за
отдельную функцию. Источник шкалы — PDF, сведённый в [RULES, §1 и §5](RULES.md).

## 2. Стек и альтернативы

| Область | P0-выбор | Почему / граница применения |
|---|---|---|
| HTTP | Python 3.12+, FastAPI, Pydantic, Uvicorn | Общий язык с ML, схемы и OpenAPI. Точные версии фиксировать lock-файлом |
| TCP | `asyncio.start_server`, `struct`, собственный небольшой NDTP codec | Протокол подробно задан; не нужен отдельный брокер или Java-сервис |
| State | Ограниченные in-memory истории по ТС | Достаточно масштаба данной раздачи; рестарт честно теряет прогретую историю |
| Inference | HTTP JSON batch → отдельный ML service | Проверяемый независимый контракт; gRPC пока усложняет отладку |
| Обновления UI | Polling snapshot раз в 1 с; SSE — P1 | Polling проще и восстанавливается после обрыва; факт задержки обновлений измерять |
| История прогнозов | Ограниченный журнал JSONL / SQLite при необходимости | Для аудита алертов; не делать PostgreSQL обязательным P0 |
| Оркестрация | Docker Compose | Три сервисных имени, readiness, воспроизводимая конфигурация |
| Наблюдаемость | JSON logs + `/metrics` и `/api/v1/status` | Prometheus/Grafana серверы необязательны, но данные для измерений обязательны |

FastAPI умеет задавать OpenAPI и Swagger URL; сделать `/openapi.json`, `/docs`, примеры
запросов и ошибки частью поставки. Для офлайн-защиты локально отдавать Swagger assets,
а не полагаться на CDN. [FastAPI: metadata/docs](https://fastapi.tiangolo.com/tutorial/metadata/).

Модель загружать один раз через `lifespan`, readiness включать после проверки её manifest
и контрольного inference. Не смешивать `lifespan` и устаревшие startup/shutdown callbacks.
[FastAPI: lifespan](https://fastapi.tiangolo.com/advanced/events/).

Backend P0 — **один процесс Uvicorn**, иначе in-memory состояние и слушатель NDTP
размножатся и расходятся. ML тоже начать с одного процесса и ограниченного thread count
CatBoost; расширять по измерениям. Несколько workers не разделяют обычную память модели.
[FastAPI: deployment concepts](https://fastapi.tiangolo.com/deployment/concepts/).

Redis/Kafka/Kubernetes/Celery — P2: их появление оправдано только доказанным ограничением.
Масштабируемость в MVP показать stateless ML, batch endpoint и явным владением состоянием.

## 3. Владение контрактами и общие признаки

**Предлагаемый однозначный контракт:** ML принимает **готовые features**, а не сырую
телеметрию. Единственная реализация feature builder находится в общем Python-пакете
`transport_predictor.features`; владельцем математической схемы является ML-разработчик.
Обучающий CLI и Backend импортируют этот же пакет из одного commit/сборки.

| Компонент | Владеет | Не делает |
|---|---|---|
| `data/contracts` | Typed Event, PlanVisit, PredictionContext, FeatureSchema | Не хранит target в inference-схемах |
| `stream` / Backend | NDTP/CSV адаптеры, mapping, часы, дедупликация, ограниченная история | Не рассчитывает вторую независимую версию признаков |
| `features` / ML owner | Очистка, target selection, causal window, агрегации, online deviation | Не читает файлы фактического расписания |
| ML service | Проверка версии/schema, model transform, predict, calibration | Не получает labels и не обучается в запросе |
| Backend API | Оркестрация, инциденты, storage, UI snapshots | Не интерпретирует SHAP как доказанную причину ДТП |
| UI | Визуализация серверных чисел, фильтры, выбранное ТС | Не считает собственную классификацию риска |

Функция-концепт:
`build_features(history_prefix, plan_only, context, feature_config) -> FeatureRow`.
`context` содержит `tr_id`, T, заданное целевое посещение либо запрос выбора, доступный
`cur_dev_s` и его источник. Истории соседних ТС, если нужны модели, обрезаются тем же T.
Builder не имеет скрытого доступа к текущим часам, labels или полному дню телеметрии.
Все imputer/scaler/encoder, обучаемые на данных, сохраняются в ML artifact, а не fit в Backend.

Model manifest: model hash/version, commit, `feature_schema_version`, упорядоченный список
features/types/units, missing policy, окна истории, time basis, cleaning/config hashes,
training split/seed, valid input regimes. При несовпадении схемы ML возвращает явную ошибку,
а Backend degraded; нельзя молча переставлять или дозаполнять новые колонки.
Pandas batch-оптимизация допустима только при тесте численного равенства эталонной функции.

## 4. API и события: минимальный согласованный дизайн

Имена ниже — план контракта, не действующие URL проекта.

### Нормализованное событие

```text
TelemetryEvent:
  schema_version, event_id, source = csv_replay | ndtp_live | ndtp_replay
  run_id, unit_id, tr_id?                         # mapping может отсутствовать
  event_time_ns, source_time_basis               # в Python int64, в JSON строка
  received_at_utc, received_monotonic_ns          # реальные часы приёмника
  lon?, lat?, speed_kmh?, heading_deg?, altitude_m?
  gps_valid, quality_flags[], source_packet_id?
```

`event_time_ns` не объявляется Unix epoch для CSV до решения timezone; basis может быть
`dataset_naive`. Внешний JSON хранит наносекундный integer строкой, поскольку JS Number
не сохраняет такие целые точно. Live NDTP Unix time известен и нормализуется в UTC.
Человеко-читаемое время CSV передавать строкой с явным `time_basis`, без ложного `Z`.

Дедупликация использует source event identity при наличии, иначе fingerprint полного
нормализованного события. Один `(tr_id,event_time)` не достаточен: разные GPS/speed поля
в один момент нельзя автоматически считать дублями. Для конфликтов фиксировать стабильный
tie-break и флаг качества; исходные события сохранять для воспроизведения.

### ML API

`POST /v1/predict` — пакет объектов с `request_id`, `feature_schema_version`, `tr_id`, T,
`target_stop_id`, `target_time_begin`, `features`, quality metadata. Target keys — только
metadata, не признаки. Ограничить batch size и body size; запретить неожиданные поля.

Ответ: тот же `request_id`, конечный signed `delay_s`, `model_version`, `schema_version`,
`inference_ms`; `late_probability` для полной поставки, с method/version и отчётом проверки калибровки;
интервал — только при отдельной проверке покрытия. В раннем MVP `late_probability=null` допустим
как явно неполный этап, но severity регрессии не закрывает требование вероятности из ТЗ.
Нет модели → 503; неверная схема → 422/409 с диагностикой; request timeout — ограничен.
Устаревший ответ не перезаписывает более новый: сравнивать `(run_id,T,prediction_seq)`.

### Backend API

| Endpoint | Назначение |
|---|---|
| `GET /health/live`, `/health/ready` | Процесс жив / конфигурация и зависимости готовы |
| `GET /api/v1/status` | Режим, replay clock, модель, состояние ML/NDTP, возраст данных |
| `GET /api/v1/snapshot` | Согласованный snapshot ТС, alerts, summary, run_id/revision/clock; фильтры риск/stale |
| `GET /api/v1/vehicles/{tr_id}` | Позиция, короткая история, прогноз, план и evidence |
| `GET /api/v1/predictions` | Последние прогнозы с ключами и временем |
| `GET /api/v1/alerts` | Активные/закрытые инциденты, сортировка по важности |
| `GET /api/v1/metrics/quality` | Реально измеренные offline/replay MAE и контекст эксперимента |
| `GET /metrics` | Технические counters/gauges/histograms |
| `POST /api/v1/replay/control` | start/pause/reset; только локальный demo mode, фиксированный dataset allowlist |
| `POST /api/v1/alerts/{id}/ack` | P1 отметка ознакомления диспетчером; не команда управления ТС |
| `GET /api/v1/events` | P1 SSE; snapshot остаётся средством восстановления |

Общий `PredictionView` (семантические поля, именование согласовать со snapshot UI): `prediction_id`, `run_id`, `mode`, `tr_id`, `unit_id`, `as_of`,
`time_basis`, `target_stop_id`, `target_time_begin`, `planned_horizon_s`, `delay_s`,
`risk_level`, `risk_basis`, `late_probability`, `calibration_status`, `model_version`, `data_age_s`, `prediction_age_s`, `stale`,
`quality_flags`, `cur_dev_source`, `status`, `evidence[]`.
Статусы различают `ok`, `warming_up`, `no_mapping`, `no_schedule`, `no_target_in_horizon`,
`stale`, `ml_unavailable`. Отсутствие прогноза не равно нулевой задержке.

`risk_level` сначала детерминированно привязан к signed прогнозу и утверждённым порогам,
а не выдуманной вероятности. В полной P0 отдельно нужен минимальный classifier опоздания
и его проверка на holdout; при недостаточной калибровке это ограничение явно показывается
и не маскируется уверенным процентом. Инцидент привязан к
`(run_id,tr_id,target_stop_id)`; обновление не создаёт новые дубли. Хранить first_alert_at,
latest_prediction_at, evidence и transitions. Hysteresis/cooldown P1 должны измеряться:
они уменьшают дребезг, но могут ухудшить раннее предупреждение.

## 5. Часы, replay и онлайн `cur_dev_s`

Источник задачи задаёт `event_time <= T`, известную подсказку `cur_dev_s` и окно
`600 < target_plan - T <= 900`. Полученный пакет не содержит `tr_id`, расписание или
`cur_dev_s`; это отдельные ответственности приложения.
[README, §1/3/8](../dataset/README.md), [NDTP, §5/6](../dataset/docs/Emulator-and-Telematic-Packets-Specification.md).

Нужны три независимые временные величины:

1. **Source/event clock** — исходная ось данных. По ней выбираются окна и план.
2. **Replay clock** — виртуальное T, движется с выбранной скоростью по source clock;
   pause не старит данные относительно виртуального мира.
3. **Monotonic wall clock** — технические latency/timeout; UTC wall clock — журналы запуска.

P0 replay сортирует по event_time и стабильному tie-break. До запроса в T пропускает
только префикс `<=T`. Это воспроизводит официальный event-time контракт. Отдельный
robustness replay симулирует доставка/поздние события; `receive_time` CSV имеет аномалии,
поэтому такой опыт нельзя объявлять автоматически тождественным официальной оценке.

В live T берётся из согласованной source clock, а не всегда из последнего пакета:
при обрыве T продолжает идти, иначе stale никогда не наступит. Позднее уже полученное
событие можно вставить в ограниченное окно для **следующего** прогноза; опубликованный
прогноз на старое T не переписывать, будто событие было тогда доступно. События старше
retention не восстанавливают уже закрытую историю; future-skew quarantine с диагностикой.
Восстановление после рестарта: warm-up и явное отсутствие истории, либо P1 snapshot +
проверяемый replay. Нельзя после disconnect незаметно смешивать январскую CSV-историю с
сентябрьским live потоком.

**Онлайн deviation:** causal matching валидных GPS к последовательности плановых посещений
конкретного ТС, с радиусом/порядком/окном времени и подтверждением достигнутой остановки.
Детектированное прибытие минус её план даёт оценку `cur_dev_s`. Дверей в CSV нет — их не
использовать как обязательный сигнал. Радиусы, dwell и неоднозначные повторные посещения
подбираются на train, не на скрытых ответах. Хранить confidence/source/age такого deviation.
Подсказку из points использовать только в официальном offline-контуре на указанный T.

Для живого режима сравнить: supplied hint / estimated hint / hint missing. P0 минимум —
модель, проверенная на пропусках hint, missing indicator и честный статус fallback;
не подставлять 0 с видимостью точного знания. P1 — причинная оценка deviation и отдельная
абляция её ошибки и end-to-end MAE. Baseline fallback не называть основным ML-решением.

**Горизонт и раннее предупреждение — разные проверки.** `planned_horizon_s = target_plan-T`
всегда должен быть в заданном окне. `actual_lead_s = observed_arrival-first_alert_at`
можно посчитать после события в evaluation sidecar. Подставлять `time_fact_begin` в live
не нужно. Показать coverage допустимых целей, долю алертов до события, lead-time distribution,
precision/recall опозданий и MAE отдельно. Нет подходящей остановки → `no_target_in_horizon`.

## 6. NDTP: точная реализация и ловушки

Источник всех бинарных констант — [локальная спецификация, §4–6](../dataset/docs/Emulator-and-Telematic-Packets-Specification.md).

- Backend — **TCP server**; эмулятор сам подключается клиентом отдельно на каждый unit.
  Handshake, пауза 200 мс, realtime; reconnect повторяет handshake. Не перепутать API
  эмулятора на 18080 с портом нашего приёмника, например 9201.
- Frame: 15 байт NPL + `dataSize` байт; `dataSize` включает NPH 10 байт и body.
  Размер body не равен `dataSize`. Все структуры little-endian без padding.
- Читать точные 15 байт заголовка, проверить signature/type/размер, затем тело.
  `readexactly` обрабатывает split/coalesced TCP, EOF даёт IncompleteReadError.
  Добавить timeout, cap соединений и собственную проверку максимального кадра:
  `StreamReader.limit` сам по себе не заменяет protocol length validation.
  [Python 3.12 Streams](https://docs.python.org/3.12/library/asyncio-stream.html).
- `signature=0x7E7E`, NPL type=2; `10 <= dataSize <= 65535` до чтения body.
  Handshake NPH `(serviceId,type)=(0,100)` и body 18 байт; realtime `(1,101)`.
  Проверить совпадение unit в handshake и NPL; новое соединение не наследует старый parser buffer.
- CRC16/Modbus: polynomial `0xA001`, init `0xFFFF`, считается **только NPH+body**;
  в NPL CRC расположен со swap bytes. Флаги `crc=0` в этом эмуляторе не отменяют
  заполненное поле CRC. Нужны независимые golden bytes и тест неверного порядка байтов.
- Realtime — последовательность `[type:u8, number:u8, payload]`. Размер Nav00 payload 26,
  полная ячейка 28 байт. Из явно заданных payload lengths: type 2 → 26, 8 → 6,
  10 → 37, 15 → 50, 16 → 8. Number различает экземпляры одного типа.
- Неизвестные type не имеют универсального поля длины. P0 распознаёт документированные
  layouts; неизвестную/неполную ячейку отклонять с диагностикой всего кадра. Не сканировать
  байты в поисках следующего type и не придумывать skip length. Эмулятор с default
  autoGenerate использует известные 0/8/16/2/10; для остальных типов нужен layout или fixture.
- Nav00 timestamp u32 seconds; longitude/latitude **unsigned magnitudes /1e7**, знаки
  берутся из bit6 E/W и bit5 N/S. bit7 — valid. Нулевая synthetic координата может иметь
  valid=true; валидность местоположения проверяется дополнительно. Speed км/ч; курс градусы.
- NPH request counter может wrap; один request_id не вечный уникальный ID.
  Не придумывать ACK-пакет: текущий эмулятор ответы не анализирует. Совместимость с ним
  не доказывает поддержку полного внешнего NDTP-протокола за пределами спецификации.

При ошибке signature/size предпочтительнее закрыть конкретное соединение и ждать reconnect,
чем неограниченно resync сканировать поток. Ошибка одного клиента не останавливает listener.
Это явно документированная политика P0; P1 может добавить bounded resync с тестами.

## 7. Как доказать NDTP → честный прогноз

**Два обязательных сценария и один усиливающий:**

1. Официальный emulator smoke: configure, handshake, valid packets, новый timestamp и
   движение UI, disconnect/reconnect, invalid/malformed fixtures отдельно. Доказывает
   бинарный приём и live цепочку, но случайное движение не подтверждает MAE по маршруту.
2. Исторический CSV replay с точными исходными временами, plan-only, causal признаками,
   моделью, UI и evaluation sidecar. Доказывает совпадение модели/горизонта/ранних алертов.
3. Усиливающий CSV → наш NDTP encoder → TCP receiver → ML → UI, с виртуальными часами
   и заранее выбранным честным фрагментом. Этот клиент явно называется собственным
   historical packet replayer, не официальным эмулятором.

Официальный эмулятор **всегда** ставит Nav00 timestamp сам; конфигом январь задать нельзя,
а autoGenerate не следует расписанию. Поэтому нельзя просто соединить его сентябрьский
поток с январским планом и обещать корректный 10–15-минутный прогноз.
[NDTP, §3/7](../dataset/docs/Emulator-and-Telematic-Packets-Specification.md).

Для NDTP исторического replayer остаётся timezone CSV: до подтверждения выбрать и
объявить обратимое соответствие source naive ↔ wire Unix в manifest demo run, сохранять
оригинальный timestamp и не объявлять эту гипотезу установленной реальной timezone.
Лучше использовать уже проверенный фрагмент с секундами и диапазонами полей NDTP.
Nav00 имеет секунды и integer speed: наносекунды и дробные скорости исходного CSV могут
теряться. **Не обещать точное равенство полного CSV и бинарного replay после такого
квантования.** Сравнить decoder с каноническими quantized input и отдельно измерить
отличие от исходного CSV; официальный submission продолжает использовать точный CSV.

Начать replay минимум на максимальное окно features раньше первого оцениваемого T,
либо показывать warm-up. Future targets читаются только из plan-only. Исторические labels
получает лишь evaluation sidecar после фактического времени события; inference контейнеру
их не монтировать. При reset меняется run_id и сбрасываются state, clocks и incident cache.

## 8. Ограничение памяти, сбои и backpressure

| Риск | Предлагаемая политика P0 | Проверка |
|---|---|---|
| Быстрый источник | Ограниченная ingestion queue; TCP чтение замедляется при заполнении | Очередь выходит на плато, memory bounded |
| Медленный ML | Ограничить concurrency; coalesce pending prediction jobs по ТС | Не накапливаются устаревшие запросы |
| Потеря телеметрии | Не coalesce raw events, нужные для окон; если потеря неизбежна — counter и degraded | Пропуски не выглядят нормой |
| Зависший ML | Timeout, bounded retry/backoff, last prediction с возрастом | NDTP/API работают при падении ML |
| Давно нет GPS | Stale data, последняя точка обозначена; новый прогноз лишь по проверенной политике | UI не красит отсутствие данных зелёным |
| Новый unit | `no_mapping`, счётчик, без случайного tr_id | Чужой план не назначается |
| Слишком много unit | Максимум активных connections/units; idle TTL | Ограниченная память, понятный reject |
| Медленный SSE клиент | Bounded outgoing queue, disconnect/resnapshot | Один браузер не тормозит ingestion |
| Новый run / старый ответ | run_id + seq validation | Старый inference не попадает в новый replay |

Retention выбрать из **максимального окна модели + допустимого опоздания**, например
кандидат 30 минут; дополнительно cap событий на ТС. Это настройка для проверки, не
доказанный оптимум. Экстремальная частота при cap может урезать окно — выставлять quality
flag. Для stale начать с объяснимого конфигурационного порога относительно штатного
интервала источника и проверить на train gaps; не выдавать 30/60 секунд за регламент.

Не блокировать event loop CPU-агрегациями и model.predict. Ограниченный executor для
тяжёлой работы плюс batch inference; порядок snapshots фиксировать до отправки в executor.
Состояние одного ТС обновлять последовательно. HTTP polling возвращает уже сохранённый
snapshot, а не запускает inference на каждом обновлении страницы.

## 9. Compose и локальный OCI: что установлено

**Прочитано read-only:** `dataset/ndtp-telemetry-emulator/manifest.json`, `index.json`,
вложенный index digest `0a218bbfb38cbf797c47cfddcc2ba52c2143332cad16e2a01e8f1d811293f5f4`
и config `4f0d3396b72b9d7259edee2253ff2fee715b4db66999e474325401a31c4affc0`.

- В раздаче каталог OCI/Docker archive layout, **готового tar нет**.
- RepoTag: `ndtp-telemetry-emulator:1.0`.
- Единственная заявленная исполнимая platform — **linux/amd64**; второй manifest
  `unknown/unknown` — attestation, не ARM image.
- Entrypoint `java -jar /app/app.jar`, exposed port `18080/tcp`, working dir `/app`.
- Импорт, запуск и исполнение на Apple Silicon **не проверялись**. Проверить до демо;
  на ARM может требоваться amd64 эмуляция или отдельный x86 host. Не считать её performance
  сопоставимой с native Linux без отметки.

Следующий шаг реализации — упаковать содержимое каталога с сохранением `manifest.json`,
`index.json`, `oci-layout`, `blobs` в tar **вне Git**, выполнить `docker image load`, проверить
tag/platform, затем `/api/cells`. `docker image load` принимает tar и восстанавливает
образы/теги; совместимость конкретного перепакованного архива ещё требует smoke test.
[Docker image load](https://docs.docker.com/reference/cli/docker/image/load/).

Compose core: `ml`, `backend`, `ui`; emulator — optional profile/внешний инструмент жюри,
training/replay CLI — отдельные команды/профили, не постоянно работающие новые сервисы.
Backend слушает NDTP на 0.0.0.0:9201; внутри Compose emulator targetHost=`backend`.
Backend обращается к `http://ml:8001`, UI proxy — `http://backend:8000`; не использовать
`localhost` для другого контейнера. Имена services устойчивы к смене container IP.
[Docker Compose networking](https://docs.docker.com/compose/how-tos/networking/).

`depends_on: condition: service_healthy` помогает дождаться готового ML при старте,
но не заменяет runtime timeout/reconnect. ML ready только после загрузки артефакта;
Backend live может оставаться true при ML degraded. Для чистого старта предпочтительно
проверять ready отдельно и возвращать объяснимую диагностику, а не зависать бессрочно.
[Compose startup order](https://docs.docker.com/compose/how-tos/startup-order/).

Поставка: зафиксированные dependencies/base image versions, non-root процессы,
read-only модель/plan/mapping mounts, отдельный writable reports volume, `.env.example`
без токенов, ограничение логов, ресурсы в README, graceful shutdown. Исходные labels и
факты расписания не нужны inference контейнерам. Проверить, что UI и Swagger работают
без CDN; карта должна иметь fallback без внешних тайлов.

## 10. Измерения для жюри: протокол, а не выдуманные результаты

Пока latency/throughput/build time **не измерены**. Число «<1–2 с» в критериях — ориентир
из PDF, а не достигнутая метрика. [RULES, §5](RULES.md).

1. Зафиксировать commit, model hash, dataset slice/hash, CPU/arch/RAM, OS/Docker,
   resource limits, число активных ТС, interval, batch size, concurrency, replay speed.
2. Разделить model.predict CPU time, ML HTTP round-trip, feature build, queue wait,
   NDTP accepted → persisted prediction и prediction → displayed UI. Базовая шкала
   runtime — monotonic clock; event_time при accelerated replay не latency.
3. После warm-up измерять фиксированный интервал, например 10 минут, в 3 повторах.
   Пример матрицы: фактические 30/56 ТС раздачи, затем 100 и 500 синтетических отправителей
   только как нагрузочный тест (не измерение ML-качества). Числа нагрузки — предлагаемые.
4. Сохранять p50/p95/p99/max, events/s, predictions/s, error/drop counts, memory/CPU,
   queue length/oldest job age и активные соединения. Throughput успешен, только если
   очередь не растёт со временем; показывать принятые и завершённые задачи отдельно.
5. Измерять холодный старт отдельно: image build, image pull/load, Compose start → ready,
   start → first usable prediction. Указывать наличие кэша и прогрев истории.
6. Fault run: разорвать NDTP, перезапустить ML, отправить CRC error, задержать события,
   выключить источник. Измерить detection/recovery time, отсутствие crash, отсутствие
   дублирующих инцидентов, корректность stale. Restart Backend отдельно показывает warm-up.
7. После теста — таблица результатов и сырые machine-readable logs/CSV в reports;
   скриншот UI со stale и восстановлением плюс краткая запись сквозного демо.

Метрики приложения: `ndtp_frames_total{status}`, `active_connections`, `unmapped_units`,
`invalid_gps_total`, `late_events_total`, `dropped_events_total`, `queue_depth`,
`prediction_latency_seconds`, `ml_errors_total`, `data_age_seconds`, `replay_speed`.
Не делать tr_id/request_id labels каждой временной серии без необходимости; конкретные
ТС связывать через JSON logs: run_id, request_id, unit_id, tr_id, model_version.

## 11. Backlog с зависимостями и оценкой времени

Оценки — человеко-часы опытного Python-разработчика, не обещание сроков; можно распараллелить
UI/ML после фиксации контрактов. Главные риски интеграции — timezone/hint/OCI.

| Приоритет | Работа | Оценка | Готовность и зависимость |
|---|---|---:|---|
| P0 | Schemas, model manifest, plan-only + mapping, time contract | 2–3 ч | ML/Backend/UI согласовали поля и статусы |
| P0 | NDTP codec/server + unit fixtures | 4–6 ч | Golden/split/coalesced/CRC/unknown-cell/reconnect тесты |
| P0 | State, virtual clock, exact CSV replay, cutoff tests | 3–5 ч | Нет будущего; одинаковые prefix features |
| P0 | ML HTTP boundary + batch, errors/timeouts, контракт late_probability | 2–3 ч | Версия/схема, graceful unavailable, Swagger; classifier готовит ML-ветка |
| P0 | Backend snapshots/incidents + polling contract | 3–4 ч | UI получает одинаковую картину, возраст и горизонт |
| P0 | Compose, OCI import proof, README, code/API docs | 3–5 ч | Чистый запуск и понятная инструкция жюри |
| P0 | Сквозное демо, замеры, faults, reports | 3–4 ч | Фактические показатели и запись восстановления |
| P1 | CSV→NDTP causal replay с тестами квантования | 2–4 ч | Полный TCP→ML→UI на историческом сценарии |
| P1 | Online stop passage → deviation + абляция | 4–8 ч | Causal estimated hint действительно полезен |
| P1 | SSE, incident hysteresis, restart state snapshot | 2–4 ч | Есть время после исправления P0 |
| P2 | Broker, distributed state, autoscaling, gRPC | Не оценивать до нужды | Только при измеренном bottleneck |

P0 суммарно примерно **20–30 человеко-часов**, включая Backend-интеграцию, без обучения
модели и создания UI. Если остаётся один разработчик, снимать P1 и сложное хранение,
сохраняя NDTP, точный CSV replay, три модуля, API, понятную деградацию и измерения.

## 12. Приёмка и открытые вопросы

Обязательные содержательные тесты:

- Golden NDTP packet независим от encoder реализации; fragmentation по каждому месту,
  несколько кадров за read, truncation/oversize/CRC, signed coordinates, invalid GPS,
  unknown cell, два reconnect, handshake identity mismatch.
- Изменение будущих событий/фактов расписания не меняет feature row; target boundary
  +600 исключена, +900 включена; нет цели — явный статус.
- Один event prefix в batch/replay даёт одинаковые признаки и predictions; NDTP
  quantization сравнивается с отдельным каноническим входом.
- Empty history, unknown unit, out-of-order, duplicated timestamps, stale, model outage,
  schema mismatch, late old response после нового run не приводят к ложным свежим ответам.
- Clean Compose + Swagger example + официальный NDTP smoke + CSV replay до UI;
  перезапуск ML не ломает listener; после disconnect UI показывает stale.

До реализации остаются неизвестными: timezone CSV; доступность hint/plan/mapping в живой
проверке; правила питания real NDTP за пределами эмулятора; среда жюри/архитектура машины;
максимальная нагрузка и фактические ресурсы. Можно продолжать локальные контракты и replay,
явно записав допущения. Не обращаться к организаторам и не отправлять решение от имени
пользователя без отдельной инструкции.

Что дать жюри: README с одной проверенной последовательностью запуска, конкретные
конфиги emulator/replay, доступ к трём модулям, OpenAPI/Swagger, HTML PyDoc/Sphinx,
model/data manifests, report performance/faults, replay сценарий раннего алерта, CSV и
описание известных ограничений. Частный GitHub требует отдельно проверить доступ жюри;
не менять публичность автоматически. Эти артефакты повышают проверяемость по формальным
критериям; они не гарантируют победу или определённый балл.
