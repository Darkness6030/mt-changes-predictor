# Прогресс

Исторические записи ниже описывают состояние на момент записи. Начинать с текущей
таблицы; строки «ещё не реализовано» в старом журнале не заменяют текущий статус.


## Точка продолжения для команды

**Сейчас:** ML v1 обучен, MAE test 79,0874 с, no-hint 87,3271 с, baseline 93,3598 с.
18 тестов. Готовые модели и CSV доступны после clone в `ml/pretrained/v1/`; локальные
`artifacts/ml-v1/` не нужны для первого запуска. Backend/frontend — заглушки,
HTTP/NDTP/Compose отсутствуют. Платформенный score неизвестен, CSV не отправлялся.

**Первый шаг нового агента:** прочитать корневой AGENTS, установить зависимости по
`ml/README.md`, запустить тесты и inference готовой модели. Затем взять одну задачу ниже
в соответствии со своей ролью. Повторное обучение — только для ML-эксперимента.

| ID | Статус / владелец | Сделать | Готово, когда | Зависимость |
|---|---|---|---|---|
| CONTRACT-01 | Очередь / не назначен | Версионированный HTTP/snapshot контракт и fixtures в `docs/API_CONTRACT.md` | Согласованы время, IDs, null/stale/no-target, hint source, schema version, ошибки | Первым для интеграции; примеры в RESEARCH |
| ML-02 | Очередь / не назначен | Group holdout по реальным ТС до нового tuning | Сохранён split, baseline/main/fallback MAE, размеры групп и ограничения | Готовая v1; не использовать test для подбора |
| ML-03 | Очередь / не назначен | Минимальный late classifier и калибровка | Вероятности проверены temporal/group данными, reliability bins/N, artifact/schema | ML-02 для протокола оценки |
| ML-API-01 | Очередь / не назначен | FastAPI inference над DelayModel, модель загружается один раз | Ready/health, version/schema checks, batch predict, ошибки, Docker | CONTRACT-01; не ждать ML-03, пока probability=null с явным статусом |
| BE-01 | Очередь / не назначен | State, virtual clock, plan/mapping, CSV replay, snapshot | Общий FeatureBuilder, cutoff/prefix parity, stale/run_id, API/Swagger | CONTRACT-01; ML API подключить после ML-API-01 |
| BE-02 | Очередь / не назначен | NDTP TCP framing/CRC/Nav00/reconnect | Packet fixtures и split/coalesced/invalid тесты, официальный emulator smoke | Нормализованное событие из CONTRACT-01/BE-01 |
| FE-01 | Очередь / не назначен | React/Vite/TS/Leaflet: очередь, карта, карточка | Выбор ТС, план/ETA/горизонт/evidence, error/stale, live/replay; fixtures явно dev | CONTRACT-01; реальное API подключить после BE-01 |
| INT-01 | Очередь / не назначен | Compose UI/Backend/ML, интеграция и замеры | Чистый запуск, NDTP→ML→UI, исторический replay, reconnect, измерения | ML-API-01, BE-01/02, FE-01 |

После ML-02: synthetic families/weights и estimated cur_dev — отдельные измеряемые
эксперименты. Не блокировать Backend/UI ожиданием лучшего MAE. Вероятность нужна для полной
поставки, но HTTP и UI можно интегрировать с явным временным отсутствием этого поля.

**Общие неизвестные:** timezone CSV, граф маршрутов, cur_dev/план/mapping в live-проверке,
MAE_TARGET. Не компенсировать их утечками или выдуманными данными. Инструкции эмулятора
ссылаются на tar, локально был OCI linux/amd64; самого образа в Git нет.

## Журнал выполненного

## 25.09.2026 — исследование перед реализацией

- Три субагента исследовали ML, Backend/NDTP и Frontend/UX; основной агент — критерии оценки, поставку и защиту.
- Созданы четыре `docs/research-*.md` и объединённый `docs/RESEARCH.md` (после переноса пользователем).
- Согласованы стек, границы модулей, API, обязательная вероятность задержки, clocks/replay и проверка квантования CSV→NDTP.
- Повторно проверены baseline и граничные labels на открытой разметке; read-only изучен OCI manifest эмулятора (linux/amd64). Официальные технические источники приведены в отчётах.
- Проверены локальные ссылки, якоря и структура документов. Исходные данные не изменены. Обучение, runtime, Docker и frontend пока не реализованы и не проверялись.
- Следующий шаг: схемы/FeatureBuilder и split, затем минимальная сквозная система и первая ML-модель.


## 25.09.2026 — ML v1: структура, обучение и проверка

- Созданы `ml/`, `backend/`, `frontend/`; у Backend/frontend только README и пустые Dockerfile.
- Реализован устанавливаемый пакет `transport_ml`: allowlist CSV, 44 causal features,
  temporal selection, main/no-hint CatBoost, native artifacts, evaluation и submission CLI.
- Train real-only: 1 141 строка; fit/development/purged = 844/287/10, boundary 16:35:00.
  Test не использовался для early stopping/подбора; synthetic rows пока исключены.
- Обучены main residual depth=4/368 trees и fallback direct depth=6/293 trees, seed=42.
  Test MAE: **79,0874 с** main, **87,3271 с** fallback, **93,3598 с** baseline cur_dev;
  улучшение main **15,29%**. Отчёт: `ml/reports/ml-v1.md` и JSON-метрики.
- Модели и 151-row submission сохранены в `artifacts/ml-v1/` (вне Git). CSV на платформу
  не отправлялся. Model/data/source hashes и параметры зафиксированы в manifest.
- Найдены равные earliest plan times: заданный target ID теперь сохраняется. Граница
  событий и horizon проверяются с наносекундами; future/fact данные не влияют на признаки.
- Проверки: **18 passed**, Ruff lint/format, pip check; parity на 5 реальных test-точках.
  Docker image `mt-changes-ml:0.1` собран; help/inference на Linux arm64 прошли.
  Контейнерный submission побайтово совпал с локальным macOS arm64.
- После исправления точности horizon обучение повторено с тем же split/кандидатами;
  MAE и submission не изменились. Исходная раздача осталась неизменной.
- Повторить: `python -m transport_ml train --model artifacts/new-run`, затем `predict`
  по инструкции `ml/README.md`. Существующий каталог не перезаписывается.
- Дальше: group holdout, synthetic-family анализ/абляции, estimated cur_dev, late classifier
  и калибровка, ML HTTP-service. Backend/NDTP/UI пока отсутствуют; current test не является
  независимым днём, платформенный score неизвестен.


## 25.09.2026 — подготовка передачи команде

- В AGENTS добавлены порядок чтения, старт после clone, границы частей, правила совместной
  работы и ссылки на конкретные следующие задачи. В PROGRESS вынесена текущая точка
  продолжения; старые записи остаются историей, владельцы новых задач не выдуманы.
- Зафиксированный комплект `ml/pretrained/v1/` содержит обе модели, manifest, метрики,
  split, важности, test predictions, submission и его checksum. Он небольшой и включён
  в Git; прочие эксперименты/окружение/Docker-слои остаются исключёнными.
- Research помечен как исследование до реализации; актуальность определяется кодом,
  тестами и текущей таблицей прогресса. Быстрый inference не требует переобучения.
- Проверена чистая выгрузка staged Git-дерева без `.venv`/`artifacts`: Docker build,
  18 тестов, Ruff lint/format и inference из `ml/pretrained/v1` прошли. Сгенерированный
  CSV побайтово совпал с эталонным; ссылки/якоря, checksum моделей и исходников проверены.
