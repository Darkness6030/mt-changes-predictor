# ML v6: ансамбль v4 и trip-модели

Main и fallback — по шесть CatBoost MAE с весом 1/6: три регрессора `a-*` из v4
(рецепт v3 на real train+test, schema 2, platform 0,86126) и три `b-*` из trip-модели
ML-IMPROVE-13 (schema 3: +9 признаков структуры рейса по плану; real train+test +
синтетика без копий моментов validate; platform 0,85649). Веса не подбирались.
Классификаторы вероятности и out-of-fold калибровка скопированы из v5.

Признаки schema 3 (82) — надмножество schema 2; каждый член читает свой список колонок.
Все признаки причинные: телеметрия `event_time <= T`, плановое расписание, `cur_dev_s`.

`metrics.json` — аудит без test labels (v3 + trip-рецепт на train): test main 67,82 с,
без подсказки 78,60 с (v3: 71,74 / 83,72). Platform score v6 не проверялся.
Сборка: `ml/experiments/improve13_build_v6.py`; отчёт — [ml-v6](../../reports/ml-v6.md).

- `a-*.cbm`, `b-*.cbm` — члены ансамблей; `late*.cbm` — классификаторы v5.
- `manifest.json` — конфигурация, источники и hashes; `metrics.json`,
  `audit_test_predictions.csv` — аудит; `submission.csv` — 151 прогноз validate.
