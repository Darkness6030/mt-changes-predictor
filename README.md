# Предиктор изменений в графике транспорта

Проект хакатона Московского транспорта: прогноз задержки за 10–15 минут,
потоковый приём телеметрии NDTP и диспетчерский дашборд.

Реализована и обучена первая ML-модель: MAE на test **79,09 с** против baseline
**93,36 с**. В `backend/` и `frontend/` пока заглушки с пустыми Dockerfile.
`main.py` — исходная неиспользуемая заготовка; рабочий вход — CLI `transport_ml`.

- [ML: запуск, устройство и результаты](ml/README.md)
- [Итоговое исследование: стек, фичи, ML, сдача и защита](docs/RESEARCH.md)
- [Подробный план хакатона](docs/PLAN.md)
- [Правила, форматы и ограничения](docs/RULES.md)
- **Новому участнику:** [инструкции агентам](AGENTS.md) → [текущие задачи](docs/PROGRESS.md#точка-продолжения-для-команды)
- [Прогресс](docs/PROGRESS.md) — выполненные исследования и проверки
- [Исходное задание](docs/description.md) и [сообщения организаторов](docs/messages.md)
- [Описание датасета](dataset/README.md) и [спецификация NDTP](dataset/docs/Emulator-and-Telematic-Packets-Specification.md)

В репозитории сохранены исходные CSV. `test` и `validate` имеют совпадающую телеметрию
и пересекающиеся расписания: фактическое время из других выборок нельзя использовать
для восстановления скрытых ответов. Подробности анализа — в плане и правилах.

Исходная ссылка на раздачу из PDF задания: https://disk.yandex.ru/d/CA6tsj4aJJ4Aaw
(доступность ссылки при подготовке плана не проверялась).

Локальный Docker-образ `dataset/ndtp-telemetry-emulator/` исключён из Git. Для потокового
демо потребуется получить образ из раздачи организаторов; описание поставки содержит
команды для `.tar`, а локальная копия представлена распакованным OCI-layout.
## Быстрый запуск ML

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r ml/requirements.lock
.venv/bin/python -m pip install --no-deps -e ml
.venv/bin/python -m pytest -q
.venv/bin/python -m transport_ml predict --model ml/pretrained/v1 --output artifacts/onboarding/submission.csv
```

Готовые модели и [CSV для Data Science](ml/pretrained/v1/submission.csv) включены в Git
в `ml/pretrained/v1/`. После clone можно сразу делать inference. Для нового обучения:
`.venv/bin/python -m transport_ml train --model artifacts/new-run`.
`artifacts/` остаётся локальным, повторные запуски требуют нового output/run.
Виртуальное окружение, Docker-образы и эмулятор в Git не входят.
Полная инструкция, Docker и ограничения — в [ML README](ml/README.md).
