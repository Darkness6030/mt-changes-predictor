# Документация кода и API для жюри

Актуальный код поставки — `6b33e7a` (27.09.2026), модель по умолчанию v5.
Точка входа для третьего поля формы сдачи. [Запуск системы](../README.md),
[инструкция жюри](DEMO.md), [контракт API](API_CONTRACT.md).

## Sphinx: готовая документация по коду

**[Скачать HTML-документацию Sphinx](code-reference.zip)** — на странице файла GitHub
нажать **Download raw file**, распаковать ZIP и открыть `index.html` в браузере.
Сборка и Python для чтения не нужны. В архиве есть исходный код с подсветкой,
справочники модулей/функций, индекс и снимки OpenAPI обоих сервисов.

Описаны общий FeatureBuilder, модели и SHAP, обучение и калибровка, NDTP/TCP,
состояние транспорта, движок предупреждений, what-if, проблемные участки,
демо-источники и история. Исходники Sphinx — [sphinx/](sphinx/).
Документация интерфейса — [frontend/README.md](../frontend/README.md).

Если браузер ограничивает поиск при открытии локальных файлов, из распакованного
каталога выполнить `python3 -m http.server 8090` и открыть http://localhost:8090.

## OpenAPI и Swagger

После `docker compose up -d --build --wait` из корня проекта:

| Сервис | Интерактивный Swagger | OpenAPI JSON | Снимок для чтения без запуска |
|---|---|---|---|
| Backend | http://localhost:8080/docs | http://localhost:8080/openapi.json | [backend.json](openapi/backend.json) |
| ML | http://localhost:8011/docs | http://localhost:8011/openapi.json | [ml.json](openapi/ml.json) |

В Swagger Backend открыть `GET /api/v1/snapshot` → **Try it out** → **Execute**:
ответ содержит ТС, прогнозы, алерты и проблемные участки. `GET /api/v1/metrics/quality`
показывает качество и паспорт предупреждений; `GET /api/v1/status` — готовность,
версию модели, приём NDTP и latency. В Swagger ML проверить `GET /v1/model`.
`POST /v1/predict` принимает готовые признаки по схеме модели; их строит Backend.

Snapshot/detail/predictions/alerts/ack/explanation и запрос ML типизированы.
Часть диагностических ответов, what-if и hotspots пока описана общими объектами:
точные поля и единицы приведены в [API_CONTRACT.md](API_CONTRACT.md).
Адреса `localhost` работают на машине, где запущен Docker; при удалённом запуске
заменить хост и учитывать настроенные порты.

## Воспроизведение документации

Из корня чистого клона (Python 3.12):

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml -e backend
.venv/bin/python -m pip install "sphinx==8.2.3"
.venv/bin/python docs/build_reference.py
```

Команда собирает Sphinx с `-W --keep-going`, обновляет `docs/code-reference.zip` и
`docs/openapi/*.json`. Внешний intersphinx отключён для сборки без сети после установки
зависимостей. Внутри ZIP `provenance.json` указывает исходный Git commit.
OpenAPI получен из приложений FastAPI без запуска фонового потока и загрузки модели.
