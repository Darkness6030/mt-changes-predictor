# ML: первая обученная модель

Реализованы причинные признаки, обучение CatBoost, выбор на temporal development,
повторная загрузка модели, оценка test и генерация submission. Это offline ML-пакет
с общей функцией признаков для будущего Backend; HTTP/NDTP-сервис ещё не реализован.

## Быстрая проверка готовой модели после clone

Сначала установить окружение командами ниже, затем без обучения:

```bash
.venv/bin/python -m transport_ml predict --model ml/pretrained/v1 --output artifacts/onboarding/submission.csv
```

`ml/pretrained/v1/` включён в Git: модели, manifest, метрики, split и готовый
`submission.csv`. Это неизменяемая исходная версия для команды; свои эксперименты писать
в `artifacts/<new-run>/`. Команда `evaluate` записывает test_predictions в каталог модели:
для экспериментов сначала скопировать bundle в новый локальный каталог или обучить новый run.

## Запуск из корня проекта

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml
.venv/bin/python -m transport_ml train --model artifacts/my-run
.venv/bin/python -m transport_ml evaluate --model artifacts/my-run
.venv/bin/python -m transport_ml predict --model artifacts/my-run --output artifacts/my-run/submission.csv
```

У автора также есть локальный `artifacts/ml-v1/`; после clone его не будет.
Передаваемая команде готовая версия — `ml/pretrained/v1/`. Для нового обучения выбирать новый каталог: CLI не перезаписывает
существующую модель или submission. Все пути задаются относительно текущей директории;
`--data` по умолчанию `dataset`. Дополнительные параметры train: `--iterations`,
`--threads`, `--seed`. Полный вывод — `python -m transport_ml --help`.

## Файлы и интерфейсы

```text
ml/
  src/transport_ml/
    data.py          # allowlist CSV, схемы и время без придуманной timezone
    features.py      # один FeatureBuilder для полного CSV и causal prefix
    model.py         # загрузка, checksum/schema, основной и no-hint inference
    training.py      # temporal selection, refit, отчёт и manifests
    submission.py    # validate prediction, строгая проверка готового CSV
    cli.py           # train / evaluate / predict
  tests/             # существенные свойства данных, модели и файлов
  reports/           # небольшие результаты первого запуска для Git
  requirements.lock  # точные версии проверенного окружения
  Dockerfile         # рабочий CLI-контейнер ML
```

`FeatureBuilder(traffic, plan).transform(points)` возвращает dataframe признаков.
`FeatureBuilder.one(point)` выполняет ту же логику для одной точки.
`DelayModel(model_directory).predict(features)` автоматически выбирает fallback для строк
с отсутствующим `cur_dev_s`; `no_hint=True` принудительно использует fallback.
Порядок/имена признаков должны совпадать с manifest, иначе возникает понятная ошибка.

Builder принимает нормализованные табличные входы в схеме датасета; будущий NDTP-адаптер
должен явно согласовать time basis и unit mapping. В нём пока нет изменяемого streaming
state/ограничения памяти. Это проверенный причинный интерфейс, а не уже готовый TCP-server.

## Данные и эксперимент

- Обучение и selection — только 1 141 real-ID точка train (`tr_id < 9000000`).
  3 293 синтетические точки исключены до выяснения семейств.
- Development начинается 06.01.2026 в 16:35:00: fit 844, development 287, purged 10.
  Fit использует только исходы, наступившие до границы; окна истории разделены 10 минутами.
- Выбор depth=4/6, direct/residual и числа деревьев — только на development, seed=42.
  После выбора две модели дообучаются на всех 1 141 train-точках. Test не используется
  для early stopping, выбора параметров или fit.
- Основная модель: residual к `cur_dev_s`, depth=4, 368 деревьев, MAE loss.
  Fallback: direct target без `cur_dev_s`, depth=6, 293 дерева.
- 44 числовых признака: подсказка, горизонт, время суток, целевая геометрия, расстояние,
  возраст/качество данных, наблюдаемый простой и окна 60/180/300/600 с.
- Окна `(T−window, T]`; точная граница T в наносекундах. В фактическое расписание builder
  не заглядывает. IDs/targets/classes не входят в model features.
- При нескольких посещениях с одинаковым earliest plan time сохраняется target ID из
  points. При автоматическом выборе без точки — детерминированный порядок по ID.

Начальная политика очистки: скорость 0…130 км/ч только при valid GPS; (0,0) не считается
валидной позицией; допустимые координаты проверяются; одинаковые нормализованные события
удаляются, конфликтующие события одного времени сохраняются в стабильном порядке.
Средняя скорость/доля стоянок в окнах пока считаются по наблюдениям, не по длительности.
Простой не продолжается через gap >60 с; stale GPS — >120 с. Все эти пороги —
конфигурация модели, не норматив перевозчика. NaN остаётся отдельным отсутствующим значением.

## Результат v1 на test

| Предиктор | MAE, с |
|---|---:|
| Нулевой прогноз | 103,3371 |
| Медиана использованного real train | 101,2011 |
| Подсказка `cur_dev_s` | 93,3598 |
| Обученный fallback без подсказки | 87,3271 |
| Основная обученная модель | **79,0874** |

Основная модель снижает MAE на 14,2724 с, или 15,29% относительно `cur_dev_s`.
Test содержит 353 точки / 13 ТС. Это локальный benchmark одного периода с пересечениями
выборок, а не доказанный перенос на новый день и не score платформы. Development-MAE
использовался для выбора модели и также не является независимой оценкой.
[Полный отчёт](reports/ml-v1.md), [метрики](reports/ml-v1-metrics.json).

## Сохранённые артефакты

В `artifacts/ml-v1/`: `main.cbm`, `fallback.cbm`, `manifest.json`, `metrics.json`, `split.csv`,
важности признаков, `test_predictions.csv`, `submission.csv` и checksum-manifest submission.
Manifest хранит версии библиотек, feature schema/config, seed, SHA-256 данных/кода/моделей
и базовый git commit; code hashes отражают в том числе незакоммиченный код запуска.

`artifacts/` исключён из Git. Небольшие копии метрик и manifest сохранены в `ml/reports/`.
Комплект v1 дополнительно опубликован в `ml/pretrained/v1/`, включая native модели и CSV.
Новые эксперименты не должны перезаписывать этот зафиксированный комплект.

## Проверки и Docker

```bash
.venv/bin/ruff check ml
.venv/bin/ruff format --check ml
.venv/bin/python -m pytest -q
docker build -t mt-changes-ml:0.1 ml
docker run --rm mt-changes-ml:0.1 --help
```

Прогноз готовой версии из Git в контейнере (выходной файл должен быть новым):

```bash
mkdir -p artifacts
docker run --rm --user "$(id -u):$(id -g)" \
  -v "$PWD/dataset:/app/dataset:ro" \
  -v "$PWD/ml/pretrained/v1:/app/model:ro" \
  -v "$PWD/artifacts:/app/artifacts" \
  mt-changes-ml:0.1 predict --model model \
  --output artifacts/submission-docker.csv
```

Для обучения в Docker вместо `predict ...` передать `train --model artifacts/docker-run`.
Том artifacts должен существовать и быть доступным указанному UID. Образ не содержит
датасет и обученную модель. Backend/frontend Dockerfile пока намеренно пусты.

## Следующая итерация

Групповой holdout по ТС; абляции и веса синтетики после определения семейств; причинный
estimated cur_dev; classifier вероятности опоздания и калибровка; затем HTTP inference
и NDTP интеграция. Нынешний fallback проверен без hint, но reconstructed hint ещё не
проверен. Пакет не выдаёт вероятности или вымышленные причины инцидента.

Основание выбора MAE/residual и native CatBoost artifact:
[официальные objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression),
[CatBoostRegressor](https://catboost.ai/docs/en/concepts/python-reference_catboostregressor).
