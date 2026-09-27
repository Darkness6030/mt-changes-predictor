# ML v4: структура рейса, train+test labels, очищенная синтетика

Main и fallback — по три CatBoost MAE с равными весами. 82 признака (schema 3) =
73 признака v3 + 9 признаков структуры рейса из **планового** расписания: длинные
плановые паузы (отстой на конечной > 240 с) между T и целевой остановкой, положение
цели в рейсе, «подсказка после запаса отстоя». Факты расписания не читаются.

Обучение: 1141 real train + 353 labelled test + 2689 синтетических train-точек.
Синтетические ТС — сдвинутые копии реальных ТС того же дня; их копии моментов validate
(±1200 с) удалены (604 точки), иначе модель видела бы зашумлённые ответы validate.

Качество (подробно — [отчёт](../../reports/ml-v4.md)):

- same-day block CV (геометрия validate): main **62,97 с** против 68,59 с у рецепта v3;
- аудит без test labels (train-only refit, копии test/validate вычищены): test main
  **66,68 с** (v3: 71,77), без подсказки **76,32 с** (v3: 83,80).

`metrics.json` содержит именно аудит; собственная ошибка этого комплекта на test
не является out-of-sample числом, потому что test labels входят в обучение.
Классификаторы вероятности и Platt-калибровка сохранены из v2 (`fitted_on_development`).

- `main-*.cbm`, `fallback-*.cbm` — члены ансамблей; `late*.cbm` — классификаторы v2.
- `manifest.json`, `recipe.json`, `training.json`, `split.csv` — конфигурация, данные, hashes.
- `metrics.json`, `audit_test_predictions.csv` — аудит; `*_importance.csv` — важности.
- `submission.csv`, `submission.manifest.json` — 151 прогноз validate и контрольные суммы.
