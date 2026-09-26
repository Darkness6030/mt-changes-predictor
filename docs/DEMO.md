# Инструкция проверяющему: запуск, поток, алерты, метрики

Требуется Docker с Compose v2. Ничего кроме репозитория скачивать не нужно: модель,
CSV раздачи и UI собираются локально. Все команды выполняются из корня проекта.

## 1. Запуск системы (1 команда)

```bash
docker compose up --build
```

Готовность: три контейнера `ml`, `backend`, `ui` переходят в `healthy`. Дождаться автоматически:
`docker compose up -d --build --wait`. Время первой сборки зависит от окружения.

| Открыть | Что там |
|---|---|
| http://localhost:8080 | Дашборд диспетчера |
| http://localhost:8080/docs | Swagger Backend — пробный запрос к API |
| http://localhost:8011/docs | Swagger ML-сервиса |

Порты хоста меняются в `.env` (пример — `.env.example`). Проверка вручную:

```bash
curl -s localhost:8010/health/ready
curl -s localhost:8011/health/ready
curl -s "localhost:8010/api/v1/snapshot?only_attention=true" | head -c 400
```

## 2. Сценарий A: исторический replay размеченного дня (по умолчанию)

Запускается автоматически: телеметрия `dataset/test/traffic.csv` подаётся по времени
события с 08:00 со скоростью 60×, план берётся без фактических времён.

Что смотреть на дашборде:

1. **Шапка**: режим, время источника, скорость, четыре числа (с прогнозом, требуют
   внимания, без прогноза, устаревшие).
2. **Очередь слева**: по умолчанию «Требуют внимания». Строка — ТС, знаковая задержка,
   время целевой остановки, качество данных.
3. **Клик по строке** → карточка справа: задержка крупно, плановое и ожидаемое прибытие,
   горизонт (всегда 600–900 с), момент расчёта `T`, вероятность опоздания, источник
   подсказки `cur_dev`, наблюдаемые основания, предлагаемое действие, кнопка
   «принято в работу».
4. **Карта**: цвет и знак маркера соответствуют риску; выбранное ТС подсвечено, показаны
   схема плановых посещений, цветной участок подхода к цели и пройденный трек.
   Клик по маркеру открывает ту же карточку. Легенда раскрывается кнопкой
   «Обозначения карты»; геометрия схематичная, не дорожный маршрут.
5. **Панель системы** (кнопка внизу справа): версии, задержки p50/p95, счётчики потока,
   измеренное качество — offline MAE и MAE этого прогона по разметке.
6. **Управление внизу**: пауза, скорость 1×…300×, сброс прогона (новый `run_id`).

При узком окне до 1000 px карточка располагается под очередью и картой; до 640 px
панели идут вертикально и страница прокручивается. На широком экране очередь и
карточка прокручиваются независимо. Все данные доступны и при свёрнутой легенде.

Проверка честности прогноза прямо в UI: в карточке `T` всегда раньше планового
прибытия на 10–15 минут, а «Этот прогон по разметке» в панели системы показывает
фактический lead time. Он может выходить за 10–15 минут и должен измеряться
отдельно: это не тот же показатель, что плановый горизонт.

## 3. Сценарий B: живой NDTP от нашего replayer

Backend — сервер NDTP; клиенты сами подключаются. Наш replayer кодирует телеметрию
раздачи в настоящие кадры NDTP (по соединению на каждый `unitId`).

```bash
export BACKEND_MODE=ndtp                       # Backend слушает 9201, часы следуют за потоком
docker compose up -d backend
docker compose --profile tools run --rm -d ndtp-replay   # 08:00–10:00, 60×
```

Проверка приёма:

```bash
curl -s localhost:8010/api/v1/status | python3 -m json.tool | sed -n '/"ndtp"/,/}/p'
```

Видны `connections_total`, `handshakes`, `realtime_frames`, `crc_errors: 0`,
`invalid_frames: 0`, список `units`. В UI появляются периодические прогнозы из бинарного потока. По умолчанию
forecast points отключены (`status.points = null`); подсказка `cur_dev` **оценивается**
по GPS и плану либо отсутствует (fallback). Совпадение с offline-прогнозом не
обещается. `BACKEND_USE_POINTS=true` включает points только для явной диагностики.

Обрыв и переподключение:

```bash
docker kill $(docker ps -q --filter name=ndtp-replay)     # разрыв
curl -s localhost:8010/api/v1/status | grep -o '"disconnects": [0-9]*'
docker compose --profile tools run --rm -d ndtp-replay    # reconnect: handshakes растут
```

Listener не останавливается, ТС постепенно переходят в «устаревшие данные», алерты не
дублируются после восстановления.

## 4. Сценарий C: официальный эмулятор организаторов

Образ в Git не входит. Импорт из распакованного OCI-каталога раздачи:

```bash
tar -C dataset/ndtp-telemetry-emulator -cf /tmp/ndtp-emulator.tar .
docker image load -i /tmp/ndtp-emulator.tar          # ndtp-telemetry-emulator:1.0 (linux/amd64)
docker compose --profile emulator up -d emulator
curl -s localhost:18080/api/cells | head -c 200
```

Эмулятор ставит собственные timestamp «сейчас», поэтому для получения прогнозов план
выравнивается по дате явным параметром (сдвиг показывается в UI баннером):

```bash
export BACKEND_MODE=ndtp BACKEND_PLAN_SHIFT_S=auto
docker compose up -d backend
curl -s -X POST localhost:18080/api/config -H 'Content-Type: application/json' -d '{
  "targetHost": "backend", "targetPort": 9201,
  "units": [
    {"unitId": 664030, "intervalMs": 3000, "autoGenerate": true, "cells": []},
    {"unitId": 913870, "intervalMs": 3000, "autoGenerate": true, "cells": []},
    {"unitId": 2147483000, "intervalMs": 5000, "autoGenerate": true, "cells": []}
  ]}'
```

Ожидаемо: `handshakes` и `realtime_frames` растут, `crc_errors` и `unknown_cells`
остаются нулевыми (разбираются документированные ячейки Nav00/Usi08/Termo16/
IntSensor02/Can10), незнакомое устройство `2147483000` попадает в `unmapped_units`
и **не** получает чужой `tr_id`.

Честная граница: автогенерация эмулятора двигает ТС случайно около Москвы и не следует
расписанию. Этот сценарий доказывает приём бинарного протокола и живую цепочку
«поток → признаки → ML → UI», но не качество прогноза по маршруту. Качество смотреть
в сценарии A.

Остановка: `docker compose --profile emulator stop emulator`, возврат к replay —
`unset BACKEND_MODE BACKEND_PLAN_SHIFT_S && docker compose up -d backend`.

## 5. Метрики и измерения

```bash
curl -s localhost:8010/api/v1/metrics/quality | python3 -m json.tool   # качество
curl -s localhost:8010/api/v1/metrics | python3 -m json.tool           # задержки/счётчики
```

- `offline` — измеренный benchmark комплекта модели: MAE 79,0874 с против baseline
  `cur_dev_s` 93,3598 с на 353 размеченных точках, плюс отчёт вероятности.
- `replay_sidecar` — метрика **этого** прогона: считается только после наступления
  фактического времени цели, разметка в инференс не попадает. Там же распределение
  фактического lead time.
- `performance` — p50/p95 инференса, обращения к ML, цикла Backend и полного пути
  «событие → опубликованный прогноз». Подробности и условия — [`PERFORMANCE.md`](PERFORMANCE.md).

## 6. Проверка деградации

| Проверка | Команда | Ожидаемое поведение |
|---|---|---|
| Падение ML | `docker compose stop ml` | Backend остаётся `healthy`, статус прогнозов `ml_unavailable`, в UI баннер, прошлые числа с возрастом; новых выдуманных значений нет |
| Восстановление ML | `docker compose start ml` | Прогнозы возобновляются без перезапуска Backend |
| Потеря API у UI | `docker compose stop backend` | В UI баннер «Связь потеряна», последнее состояние не перекрашивается в зелёное |
| Нет интернета/тайлов | Заблокировать доступ к tile.openstreetmap.org | Появляется сообщение о тайлах, работает кнопка «Схема без подложки», список и прогнозы не страдают |
| Пауза replay | Кнопка «Пауза» | Виртуальное время и возрасты данных останавливаются, wall-clock health продолжает считаться |
| Сброс прогона | Кнопка «Сброс прогона» | Новый `run_id`, чистое состояние, инциденты не переносятся |

## 7. CSV для Data Science

Готовый файл — `ml/pretrained/v2/submission.csv` (151 строка, `sample_id;prediction`).
Воспроизведение без обучения и без Docker:

```bash
.venv/bin/python -m transport_ml predict --model ml/pretrained/v2 \
  --output artifacts/check/submission.csv
shasum -a 256 artifacts/check/submission.csv ml/pretrained/v2/submission.csv
```

В Docker:

```bash
docker compose --profile tools run --rm trainer predict --model /app/model \
  --output artifacts/check/submission.csv
```

Повторное обучение (необязательно, ~5 с на 4 потоках):

```bash
docker compose --profile tools run --rm trainer train --model artifacts/new-run
```

## 8. Остановка

```bash
docker compose --profile tools --profile emulator down
```
