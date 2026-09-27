# Предиктор изменений в графике городского транспорта

Прогноз знаковой задержки за **10–15 минут** до планового прибытия, приём телеметрии
**NDTP** по TCP, отдельный ML-сервис и диспетчерский дашборд. Три модуля запускаются
одной командой Docker Compose.

| Что | Значение | Где проверить |
|---|---|---|
| Модель по умолчанию | **v5**: регрессия v4 (рецепт v3 на train + test, 1494 точки) + новая вероятность | [отчёт ML v4/v5](ml/reports/ml-v5.md) |
| Score платформы | **0,86126** (27.09, после публикации v4; прежний лучший 0,70106) | сообщено пользователем |
| MAE на test вне фолдов: v3-рецепт только на train → v4 | **70,67 → 67,87 с** (без подсказки 83,27 → 80,12 с) | [проверка](ml/experiments/improve18-test-refit.json) |
| MAE v3 на размеченном test (test не в обучении) | **71,7707 с** | [исследование ML v3](ml/reports/ml-v3.md) |
| Исторический periodic replay v3 (до исправления ties) | **91,1775 с** вместо 100,4563 с у прежней логики | [протокол](ml/reports/ml-v3.md) |
| Baseline `cur_dev_s` / нулевой прогноз | 93,3598 с / 103,3371 с | там же |
| Вероятность `P(задержка > 120 с)`, вне фолдов на test | Brier 0,1065 (без подсказки 0,1162) против 0,1843 у базовой частоты; ROC-AUC 0,895 / 0,878 | [отчёт ML v4/v5](ml/reports/ml-v5.md) |
| Движок инференса в Docker | **ONNX Runtime** с проверкой задержки и возвратом к CatBoost | [режимы и замеры](docs/PERFORMANCE.md) |
| Объяснение прогноза | SHAP-вклады групп признаков в секундах, точная сумма, в API и карточке ТС | [отчёт](ml/reports/ml-v5.md) |
| CSV для Data Science (151 прогноз) | [`ml/pretrained/v5/submission.csv`](ml/pretrained/v5/submission.csv) (= v4) | [проверка формата](ml/src/transport_ml/submission.py) |
| Тесты | 195 (ML, признаки, NDTP, состояние, движок, API, демо, история, GPS-фильтр) | `python -m pytest -q` |

Возможности и зачем они диспетчеру — [FEATURES.md](docs/FEATURES.md); инструкция для жюри —
[DEMO.md](docs/DEMO.md); тексты формы сдачи — [SUBMISSION.md](docs/SUBMISSION.md).

## Запуск за одну команду

Для карты укажите `VITE_YANDEX_MAPS_API_KEY` в корневом `.env` (см. `.env.example`).
Ключ JavaScript API используется браузером; после его изменения пересоберите UI.
Без ключа очередь и управление временем работают, карта показывает сообщение настройки.

Требуются Git и Docker с Compose v2. Из новой рабочей директории:

```bash
git clone https://github.com/Darkness6030/mt-changes-predictor.git
cd mt-changes-predictor
# При необходимости создайте .env по .env.example и укажите ключ карты.
docker compose up -d --build --wait  # ML + Backend + UI
```

Для уже клонированного проекта достаточно последней команды из его корня.
Пока репозиторий приватный, клонирование требует доступа к нему.

| Сервис | Адрес по умолчанию | Назначение |
|---|---|---|
| UI (дашборд) | http://localhost:8080 | Очередь ТС, карта, карточка, панели системы и демо |
| Backend Swagger | http://localhost:8080/docs | Пробный запрос к API, схемы OpenAPI |
| Backend напрямую | http://localhost:8010 | `/api/v1/snapshot`, `/api/v1/status`, `/api/v1/metrics/quality` |
| ML-сервис | http://localhost:8011/docs | `/v1/predict`, `/v1/model`, `/health/ready` |
| Приём NDTP | `tcp://localhost:9201` | Сервер для эмулятора и нашего replayer |

Порты хоста и параметры демо меняются через `.env` (см. [`.env.example`](.env.example));
8000/8001 специально не заняты, чтобы не конфликтовать с другими проектами.
По умолчанию запускается **исторический replay размеченного дня** со скоростью 60×
с 08:00 — на нём сразу видны алерты, и по факту события считается MAE потока.

В Compose по умолчанию `ML_RUNTIME=onnx`; фактически выбранный движок и причина
возврата к CatBoost видны в http://localhost:8011/v1/model (`runtime`, `runtime_note`).
Принудительный CatBoost: `ML_RUNTIME=catboost docker compose up -d --wait ml`.
Локальный запуск сервиса без этой переменной использует CatBoost. Модель остаётся v5;
деревья экспортируются при старте во временный каталог, комплект модели не меняется.

**Пошаговая инструкция для проверяющего — [`docs/DEMO.md`](docs/DEMO.md)**: подача
живого NDTP, официальный эмулятор, просмотр алертов и метрик, проверка деградации.

## Архитектура: три модуля, не монолит

```text
                официальный эмулятор NDTP ─┐
наш NDTP-replayer (CSV → NDTP-кадры) ──────┤ TCP 9201
                                           v
CSV replay (event-time) ──> [ Backend: framing/CRC → нормализованное событие →
                              состояние ТС (ограниченное) → общий FeatureBuilder →
                              GPS-оценки / политика hint → инциденты → snapshot API ]
                                           │ HTTP /v1/predict (готовые признаки)
                                           v
                              [ ML-сервис: CatBoost + калибровка ]
                                           │
                                           v
                              [ UI: очередь → карта → карточка ]
```

- `ml/` — библиотека признаков и моделей `transport_ml`, обучение/оценка/submission CLI и
  HTTP-сервис инференса. Обучение — отдельная команда, не часть запроса.
- `backend/` — `transport_backend`: TCP-сервер NDTP, виртуальные часы, ограниченное
  состояние, инциденты, API диспетчера и OpenAPI.
- `frontend/` — React + Vite + TypeScript + Яндекс Карты, статика в nginx, прокси `/api`.

Признаки существуют **в одной реализации** (`transport_ml.features.FeatureBuilder`):
Backend импортирует её, а не пишет вторую версию. Равенство batch и потока проверяется
тестом на реальных данных.

## Документация

| Документ | Содержание |
|---|---|
| [`docs/DEMO.md`](docs/DEMO.md) | Инструкция жюри: сценарии, команды, что смотреть |
| [`docs/API_CONTRACT.md`](docs/API_CONTRACT.md) | Контракт snapshot/prediction/alert, статусы, время |
| [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md) | Короткий замер latency и границы проверки |
| [`ml/README.md`](ml/README.md) | Признаки, обучение, модели, вероятность, CLI, сервис |
| [`backend/README.md`](backend/README.md) | NDTP, часы, состояние, инциденты, endpoints |
| [`frontend/README.md`](frontend/README.md) | Экраны, состояния, пороги, сборка |
| [`ml/reports/ml-v3.md`](ml/reports/ml-v3.md) | 109 конфигураций, holdout, автономная оценка и ограничения |
| [`docs/DOCUMENTATION.md`](docs/DOCUMENTATION.md) | Готовый HTML Sphinx, OpenAPI JSON и Swagger обоих сервисов, воспроизведение |
| [`docs/RULES.md`](docs/RULES.md), [`docs/PLAN.md`](docs/PLAN.md), [`docs/RESEARCH.md`](docs/RESEARCH.md) | Требования, план, исследование до реализации |
| [`docs/PROGRESS.md`](docs/PROGRESS.md) | Журнал фактически выполненного |
| [`docs/HACKATHON_STRATEGY.md`](docs/HACKATHON_STRATEGY.md) | Приоритеты по критериям, вау-фичи и план до дедлайна |
| [`AGENTS.md`](AGENTS.md) | Правила работы над проектом |

Swagger Backend — `/docs`, схема — `/openapi.json`; ML-сервис имеет собственный Swagger.

## Локальная разработка без Docker

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml -e backend
.venv/bin/python -m pytest -q                                  # 195 тестов
.venv/bin/ruff check ml backend && .venv/bin/ruff format --check ml backend

# ML-сервис и Backend в двух терминалах
ML_MODEL_DIR=ml/pretrained/v5 ML_PORT=8011 .venv/bin/transport-ml-serve
BACKEND_ML_URL=http://127.0.0.1:8011 BACKEND_LABELS=dataset/labels/labels_test.csv \
  .venv/bin/transport-backend serve --port 8010

# UI в режиме разработки (прокси на 8010)
# Сначала задать VITE_YANDEX_MAPS_API_KEY в frontend/.env.local или окружении.
npm --prefix frontend ci
npm --prefix frontend run dev
```

Готовый CSV для платформы воспроизводится без обучения:

```bash
.venv/bin/python -m transport_ml predict --model ml/pretrained/v5 \
  --output artifacts/check/submission.csv   # побайтово равен ml/pretrained/v5/submission.csv
```

## Что честно, а что ограничено

- Для точки `T` используются только события с `event_time <= T`; будущее **плановое**
  расписание допустимо, фактическое время прибытия в признаки не попадает (allowlist +
  тесты). Цель — первое плановое посещение в `(T+600, T+900]`.
- Отсутствие прогноза публикуется статусом (`no_target_in_horizon`, `stale`,
  `warming_up`, `ml_unavailable`, …), а не нулевой задержкой.
- Обучение только на реальных размеченных точках train и test. Синтетические ТС train —
  сдвинутые во времени копии реальных с почти теми же задержками, в том числе на
  validate-посещениях; факты `time_fact_begin` train/test покрывают все validate-цели.
  Поэтому ни то, ни другое не используется ни в признаках, ни в обучении (ML-IMPROVE-18).
  Так как test вошёл в обучение v4, для неё публикуется только оценка вне фолдов, а
  MAE replay на test в дашборде помечается как in-sample.
- У v5 вероятность калибрована Platt по out-of-fold оценкам и проверена вне фолдов
  одного дня; статус `validated` виден в API/UI. Перенос на независимый день не доказан.
  У исторических v2–v4 сохранена development-калибровка.
- V3 отделяет supplied `cur_dev_s` от шумных оценок GPS/плана. В потоке без supplied
  подсказки работает обученная автономная модель с отдельными признаками matching.
  Старые v1/v2 сохраняют GPS-estimated hint; сравнение — в отчёте v3.
- Официальный эмулятор двигает ТС случайно и ставит свои timestamp: он доказывает приём
  NDTP и живую цепочку, но не качество прогноза по маршруту. Для качества используется
  размеченный replay; [автономный event-time отчёт](ml/reports/ml-v3-autonomous.json)
  отделён от проверки TCP/HTTP и реального времени публикации. NDTP по умолчанию
  не читает forecast points.
- Линии на карте — схема плановых посещений и пройденный трек, не дорожный маршрут.
  Маршрутного графа, дверей, ДТП и пассажиропотока в раздаче нет.
- `test`/`validate` делят телеметрию и один день: локальные метрики — benchmark, а не
  доказанный перенос на другой день и не score платформы.

## Исходные данные

CSV раздачи сохранены в репозитории неизменными; `dataset/README.md` и
[спецификация NDTP](dataset/docs/Emulator-and-Telematic-Packets-Specification.md) —
первичные источники. Локальный OCI-каталог эмулятора в Git не входит; как его
импортировать, описано в [`docs/DEMO.md`](docs/DEMO.md).
Исходная ссылка на раздачу из PDF задания: https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw.

Панель «Демо / отладка» переключает CSV replay / наш NDTP-replayer / официальный
эмулятор без перезапуска Backend. Последний требует импорт образа и
`docker compose --profile emulator up -d emulator`; источник архива и полный сценарий —
[DEMO, раздел 4](docs/DEMO.md#4-сценарий-c-официальный-эмулятор-организаторов).

Оба NDTP-источника записывают историю для шкалы времени: можно посмотреть прошлый
снимок и вернуться «К потоку», пока приём и прогнозы продолжаются. Это журнал реально
выданных состояний, без пересчёта прошлого по будущим данным. По умолчанию — до 2 часов
времени источника / 7200 снимков / 64 МиБ сжатых записей, только текущий прогон.

Красную метку времени можно плавно перетаскивать отдельно; вне метки drag сдвигает
диапазон. В replay отпускание делает один seek, в NDTP выбирает снимок истории.
