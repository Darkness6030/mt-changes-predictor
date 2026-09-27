# Прогноз задержек городского транспорта

Система предупреждает диспетчера об отклонении от расписания за **10–15 минут**
до плановой остановки. Принимает телеметрию NDTP или воспроизводит исторический
датасет, показывает прогнозы, алерты и положение транспорта на карте.

Три модуля работают в Docker:

| Модуль | Назначение |
|---|---|
| ML-ядро | CatBoost, вероятность опоздания и объяснение прогноза; инференс через FastAPI |
| Backend | Приём NDTP по TCP, подготовка признаков, состояние транспорта и API |
| BI-дашборд | Очередь транспорта, Яндекс Карты, карточки предупреждений и метрики |

## Запуск

Нужны Git и Docker с Compose v2. Модель **v8** и исходные CSV уже в репозитории,
обучение для запуска не требуется.

```bash
git clone https://github.com/Darkness6030/mt-changes-predictor.git
cd mt-changes-predictor
docker compose up -d --build --wait
```

Откройте **http://localhost:8080**. По умолчанию запустится исторический поток
с 08:00 на скорости 60×. Вход не требуется.

Для карты перед сборкой создайте `.env` по [`.env.example`](.env.example)
и задайте `VITE_YANDEX_MAPS_API_KEY` — ключ JavaScript API Яндекса.
После изменения ключа повторите команду запуска. Без ключа очередь, прогнозы
и метрики доступны, а карта показывает сообщение о настройке.

| Адрес | Что открыть |
|---|---|
| http://localhost:8080 | Дашборд |
| http://localhost:8080/docs | Swagger Backend |
| http://localhost:8011/docs | Swagger ML |
| `localhost:9201` | TCP-порт приёма NDTP |

Порты и режимы задаются в `.env`. Для остановки:

```bash
docker compose --profile tools --profile emulator down
```

## Проверка и документация

- [Инструкция для жюри](docs/DEMO.md) — исторический поток, NDTP, прогнозы и алерты.
- [Документация кода и API](docs/DOCUMENTATION.md) — готовый Sphinx и Swagger.
- [Производительность и дополнительные возможности](FEATURES.md).
- [Условия и результаты замеров](docs/PERFORMANCE.md).
- [Контракт API](docs/API_CONTRACT.md).

## Модель и данные

Модель v8 объединяет ансамбли CatBoost и классификатор нулевой задержки.
При отсутствии текущего отклонения использует отдельную модель без подсказки.
Backend и offline-инференс строят признаки одним `FeatureBuilder`;
учитываются только события, доступные к моменту прогноза.

Готовый файл для Data Science — [submission.csv](ml/pretrained/v8/submission.csv):
151 прогноз, формат `sample_id;prediction`. Подробности модели —
[описание v8](ml/pretrained/v8/README.md), данных — [dataset/README.md](dataset/README.md).

Итоговая модель обучена на train и test, поэтому метрика replay на test —
диагностика на знакомых данных. Перенос качества на другой день не проверен.
Линии на карте соединяют плановые остановки; это схема, а не дорожный маршрут.

## Локальная разработка

Python 3.12; команды выполняются из корня проекта:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml -e backend
.venv/bin/python -m pytest -q
```

Команды отдельных модулей: [ML](ml/README.md), [Backend](backend/README.md),
[Frontend](frontend/README.md).
