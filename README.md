# Предиктор изменений в графике городского транспорта

Прогноз знаковой задержки за **10–15 минут** до планового прибытия, приём телеметрии
**NDTP** по TCP, отдельный ML-сервис и диспетчерский дашборд. Три модуля запускаются
одной командой Docker Compose.

| Что | Значение | Где проверить |
|---|---|---|
| MAE основной модели на размеченном test | **79,0874 с** | [отчёт ML v2](ml/reports/ml-v2.md) |
| Baseline `cur_dev_s` / нулевой прогноз | 93,3598 с / 103,3371 с | там же |
| Вероятность `P(задержка > 120 с)` | Brier 0,1395 против 0,1843 у базовой частоты, ROC-AUC 0,8177 | [отчёт ML v2](ml/reports/ml-v2.md) |
| CSV для Data Science (151 прогноз) | [`ml/pretrained/v2/submission.csv`](ml/pretrained/v2/submission.csv) | [проверка формата](ml/src/transport_ml/submission.py) |
| Тесты | 80 (ML, признаки, NDTP, состояние, движок, API) | `python -m pytest -q` |

## Запуск за одну команду

```bash
docker compose up --build            # ML + Backend + UI
```

| Сервис | Адрес по умолчанию | Назначение |
|---|---|---|
| UI (дашборд) | http://localhost:8080 | Очередь ТС, карта, карточка прогноза, панель системы |
| Backend Swagger | http://localhost:8080/docs | Пробный запрос к API, схемы OpenAPI |
| Backend напрямую | http://localhost:8010 | `/api/v1/snapshot`, `/api/v1/status`, `/api/v1/metrics/quality` |
| ML-сервис | http://localhost:8011/docs | `/v1/predict`, `/v1/model`, `/health/ready` |
| Приём NDTP | `tcp://localhost:9201` | Сервер для эмулятора и нашего replayer |

Порты хоста и параметры демо меняются через `.env` (см. [`.env.example`](.env.example));
8000/8001 специально не заняты, чтобы не конфликтовать с другими проектами.
По умолчанию запускается **исторический replay размеченного дня** со скоростью 60×
с 08:00 — на нём сразу видны алерты, и по факту события считается MAE потока.

**Пошаговая инструкция для проверяющего — [`docs/DEMO.md`](docs/DEMO.md)**: подача
живого NDTP, официальный эмулятор, просмотр алертов и метрик, проверка деградации.

## Архитектура: три модуля, не монолит

```text
                официальный эмулятор NDTP ─┐
наш NDTP-replayer (CSV → NDTP-кадры) ──────┤ TCP 9201
                                           v
CSV replay (event-time) ──> [ Backend: framing/CRC → нормализованное событие →
                              состояние ТС (ограниченное) → общий FeatureBuilder →
                              оценка cur_dev → инциденты → snapshot API ]
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
- `frontend/` — React + Vite + TypeScript + Leaflet, статика в nginx, прокси `/api`.

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
| [`ml/reports/ml-v2.md`](ml/reports/ml-v2.md) | Протокол обучения, метрики, калибровка, ограничения |
| `docs/sphinx/` | PyDoc/Sphinx по коду: `python -m sphinx -b html docs/sphinx docs/sphinx/_build/html` |
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
.venv/bin/python -m pytest -q                                  # 80 тестов
.venv/bin/ruff check ml backend && .venv/bin/ruff format --check ml backend

# ML-сервис и Backend в двух терминалах
ML_MODEL_DIR=ml/pretrained/v2 ML_PORT=8011 .venv/bin/transport-ml-serve
BACKEND_ML_URL=http://127.0.0.1:8011 BACKEND_LABELS=dataset/labels/labels_test.csv \
  .venv/bin/transport-backend serve --port 8010

# UI в режиме разработки (прокси на 8010)
cd frontend && npm install && npm run dev
```

Готовый CSV для платформы воспроизводится без обучения:

```bash
.venv/bin/python -m transport_ml predict --model ml/pretrained/v2 \
  --output artifacts/check/submission.csv   # побайтово равен ml/pretrained/v2/submission.csv
```

## Что честно, а что ограничено

- Для точки `T` используются только события с `event_time <= T`; будущее **плановое**
  расписание допустимо, фактическое время прибытия в признаки не попадает (allowlist +
  тесты). Цель — первое плановое посещение в `(T+600, T+900]`.
- Отсутствие прогноза публикуется статусом (`no_target_in_horizon`, `stale`,
  `warming_up`, `ml_unavailable`, …), а не нулевой задержкой.
- Вероятность опоздания калибрована на development-фолде того же дня; статус
  `fitted_on_development` виден в API и в UI. Это не проверка на независимом дне.
- Онлайн-подсказка `cur_dev_s` в потоке **оценивается** по GPS и плану: MAE к выданной
  подсказке 56,3 с, покрытие 86,4 % на test (параметры подобраны на train,
  `transport-backend check-hint`). Если оценки нет — модель без подсказки и явный статус.
- Официальный эмулятор двигает ТС случайно и ставит свои timestamp: он доказывает приём
  NDTP и живую цепочку, но не качество прогноза по маршруту. Для качества используется
  размеченный replay; автономный NDTP-replayer проверяет сквозную цепочку,
  но ещё не имеет отдельного отчёта качества. NDTP по умолчанию не читает forecast points.
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
