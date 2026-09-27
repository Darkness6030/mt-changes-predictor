# Документация кода и API

## Sphinx: код ML и Backend

**[Скачать готовую HTML-документацию](code-reference.zip)** — нажмите
**Download raw file**, распакуйте архив и откройте `index.html`.
Python и запущенный стенд для чтения не нужны.

В справочнике описаны признаки и модели, обучение, инференс, NDTP,
состояние транспорта, предупреждения и API. В архив включены исходники
с подсветкой и схемы OpenAPI. [Исходники Sphinx](sphinx/).

## Swagger: API работающей системы

После [запуска Docker](../README.md):

| Сервис | Swagger | OpenAPI без запуска |
|---|---|---|
| Backend | http://localhost:8080/docs | [backend.json](openapi/backend.json) |
| ML | http://localhost:8011/docs | [ml.json](openapi/ml.json) |

В Swagger Backend попробуйте `GET /api/v1/snapshot` — текущие ТС, прогнозы
и алерты; затем `GET /api/v1/metrics/quality` — метрики качества.
В Swagger ML запрос `GET /v1/model` покажет модель и движок инференса.
`POST /v1/predict` принимает готовые признаки, которые рассчитывает Backend.

Живые схемы доступны по `/openapi.json` на тех же адресах.
При запуске на сервере замените `localhost` на его адрес.
Единицы, статусы и поля диагностических ответов описаны в
[контракте API](API_CONTRACT.md); [интерфейс — в README Frontend](../frontend/README.md).

## Пересборка документации

Из корня проекта, Python 3.12:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml -e backend
.venv/bin/python -m pip install "sphinx==8.2.3"
.venv/bin/python docs/build_reference.py
```

Команда обновляет ZIP и обе схемы OpenAPI. Сервисы и обучение запускать не нужно.
Версия исходного кода записана в `provenance.json` внутри архива.
