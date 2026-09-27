# ML: прогноз задержки и оценка риска

Текущий комплект — [`pretrained/v6`](pretrained/v6/README.md) (schema 3, ансамбль v4 +
trip-модели, [отчёт](reports/ml-v6.md)); v1–v5 сохранены. Общий FeatureBuilder
используется в batch, CSV replay и NDTP; FastAPI принимает готовые признаки.
Обучение выполняется отдельной командой, вне HTTP.

## Быстрый запуск

Из корня проекта после установки зависимостей по основному README:

```bash
.venv/bin/python -m transport_ml predict --model ml/pretrained/v6 \
  --output artifacts/check-v6/submission.csv
ML_MODEL_DIR=ml/pretrained/v6 ML_PORT=8011 .venv/bin/transport-ml-serve
```

Выходной CSV должен отсутствовать: команда отказывается перезаписывать его. Результат
побайтово совпадает с `ml/pretrained/v6/submission.csv`; все 151 ID проверяются повторным
чтением. Комплекты v1–v5 сохранены и поддерживаются; default Compose — v6.

## Архитектура и контракт

- `data.py` — allowlist входных полей; labels загружаются отдельно.
- `features.py` — один причинный FeatureBuilder, точность времени до наносекунд.
- `schedule_context.py` — GPS относительно плановых посещений, ограниченное окно 30 минут.
- `trip_context.py` — schema 3: отстои на конечных и положение в рейсе, только по плану.
- `synthetic.py` — семьи синтетических копий и удаление копий моментов validate.
- `model.py` — native CatBoost, контроль schema/checksum, main/fallback и взвешенные ансамбли.
- `research.py` — фиксированные group/forward splits, подбор кандидатов, refit рецепта.
- `training.py`, `group_validation.py` — исходный протокол v1/v2 и прежний group holdout.
- `service.py` — `/health/live`, `/health/ready`, `/v1/model`, `/v1/predict`.
- `submission.py` — signed прогноз и строгий двухколоночный CSV.

`FeatureConfig()` сохраняет schema 1 и прежние 44 признака. Опция
`schedule_context=True` включает schema 2: ещё 29 признаков расстояния до планового
положения, наблюдаемых прохождений остановок, геометрического прогресса, направления,
взвешенной по времени скорости и качества наблюдений. Геометрические отрезки — схема,
не дорожный маршрут. Расстояние до матчинга и пропуски сохраняются явно.

`DelayModel(...).predict(features)` выбирает main при известном `cur_dev_s`, fallback
при NaN; `no_hint=True` принудительно выбирает fallback. Порядок всех признаков должен
совпадать с manifest. Отрицательные задержки сохраняются.

У v3 `hint_policy=supplied_only`: backend не записывает GPS-оценку в поле CSV-подсказки;
GPS/плановый контекст рассчитывается общим builder и используется автономной моделью.
Backend получает config и schema через `/v1/model`. Для schema 2 нужно хранить минимум
1800 секунд истории; caps памяти сохраняются. Snapshot API остаётся версии 1.

## Результаты trip-модели (ML-IMPROVE-13, половина v6)

| Проверка | v3, MAE с | trip, MAE с |
|---|---:|---:|
| Same-day block CV, main | 68,59 | **62,97** |
| Аудит без test labels → test, main | 71,77 | **66,68** |
| Тот же аудит, без hint | 83,80 | **76,32** |

Trip-модель обучена на train+test labels и очищенной синтетике; platform score 0,85649
(v4: 0,86126). V6 = их среднее; аудит v6 без test labels на test — 67,82 с (v3 71,74).
Подробно — [отчёт v6](reports/ml-v6.md).

## Результаты v3

| Проверка | v2 / прежняя логика | v3, MAE с |
|---|---:|---:|
| Test, 353 точки, supplied hint | 79,0874 | **71,7707** |
| Test, 353 точки, без hint | 87,3271 | **83,7950** |
| Отложенные 3 ТС train, 206 точек, supplied hint | 76,9720 | **74,3725** |
| Те же отложенные ТС, без hint | 88,3703 | **73,5623** |
| Периодический replay, 1311 оценённых прогнозов, без supplied hint | 100,4563 (GPS-estimated + fallback) | **91,1775** |

Для holdout обе версии заново обучены без отложенных ТС. Для остальных строк — полные
зафиксированные комплекты. Периодические прогнозы коррелируют внутри посещения;
MAE с равным весом посещений: 96,4139 → 90,2839 с, покрыто 322/353 известных посещений.
Test уже был исследован до этой работы, но не использовался для нового подбора.
Один день и 13 ТС не доказывают перенос на новый день; срезы и ограничения —
в [отчёте](reports/ml-v3.md). Вероятность и её development-калибровка сохранены из v2.

## Воспроизведение

V6 (default): trip-рецепт, аудит, block CV и сборка ансамбля.

```bash
.venv/bin/python -m transport_ml train-recipe \
  --recipe ml/experiments/improve13-recipe.json --model artifacts/trip-repeat
.venv/bin/python ml/experiments/improve13_audit.py --recipe ml/experiments/improve13-recipe.json \
  --bundle artifacts/trip-repeat --output artifacts/trip-audit/metrics.json
.venv/bin/python ml/experiments/improve13_block_cv.py --output artifacts/trip-block-cv.json
.venv/bin/python ml/experiments/improve13_build_v6.py --trip artifacts/trip-repeat \
  --output artifacts/v6-repeat
.venv/bin/python -m transport_ml predict --model artifacts/v6-repeat \
  --output artifacts/v6-repeat/submission.csv
```

V3 и ниже. Все каталоги результата должны быть новыми. Обучение v3 использует только
1141 real train точку; синтетика там исключена. Исходные CSV неизменны.

```bash
# Полный поиск 109 конфигураций: пять vehicle folds + два forward folds; без test.
.venv/bin/python -m transport_ml research \
  --protocol ml/experiments/improve12-protocol.json \
  --candidates ml/experiments/improve12-candidates.json \
  --output artifacts/research-repeat

# Аудит уже замороженного рецепта. Не использовать для дальнейшего подбора.
.venv/bin/python -m transport_ml validate-recipe \
  --protocol ml/experiments/improve12-protocol.json \
  --recipe ml/experiments/improve12-recipe.json \
  --output artifacts/holdout-repeat

# Итоговый refit без чтения test/validate; далее отдельная оценка и submission.
.venv/bin/python -m transport_ml train-recipe \
  --recipe ml/experiments/improve12-recipe.json --model artifacts/v3-repeat
.venv/bin/python -m transport_ml evaluate --model artifacts/v3-repeat
.venv/bin/python -m transport_ml predict --model artifacts/v3-repeat \
  --output artifacts/v3-repeat/submission.csv

# Автономный event-time replay всего дня, periodic 30 с, без points и supplied hints.
.venv/bin/python -m transport_backend.live_evaluation \
  --model ml/pretrained/v3 --baseline ml/pretrained/v2 \
  --output artifacts/autonomous-repeat
```

`evaluate` записывает `test_predictions.csv` в каталог модели. Для оценки готового bundle
сначала скопировать его в новый `artifacts/` каталог. Refit-recipe сохраняет неизменённые
классификаторы v2; это явно зафиксировано в `classifier_provenance` manifest.

Полный поиск сохраняет кандидатов, split, predictions и source/data hashes. В проверке
воспроизведения совпали предикты main, fallback и экспериментального augmentation-кандидата;
отдельно воспроизведены все 206 holdout-прогнозов. Дополнительные моменты перед train-target
тестировались только внутри родительского fold, без новых labels, и не включены в v3.

Исходные команды `train`, `validate-groups` остаются для воспроизведения старой процедуры.
Они не являются способом обучения v3; используйте `train-recipe`.

## Проверки

```bash
.venv/bin/ruff check ml backend
.venv/bin/ruff format --check ml backend
.venv/bin/python -m pytest -q
docker compose up -d --build --wait
```

Проверяются causal cutoff (включая +1 нс), inert факты/labels, signed delay, горизонт,
равенство batch/ограниченного streaming prefix, режимы подсказок, ансамбли/checksum,
совместимость schema 1/2, NDTP, API и submission. Event-time evaluator не измеряет
доставку TCP, очереди и wall-clock время публикации — это отдельная интеграционная проверка.
