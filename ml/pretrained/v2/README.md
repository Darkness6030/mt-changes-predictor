# Зафиксированный ML v2 — задержка и калиброванная вероятность

Регрессия не менялась относительно [v1](../v1/README.md): тот же split, seed и признаки,
test MAE **79,0874 с** (fallback 87,3271 с), `submission.csv` побайтово совпадает с v1.
Добавлены два классификатора `P(target_delay_s > 120 с)` и Platt-калибровка.

| Артефакт | Назначение |
|---|---|
| `main.cbm`, `fallback.cbm` | Signed задержка: residual к `cur_dev_s` и модель без подсказки |
| `late.cbm`, `late_fallback.cbm` | Вероятность опоздания с подсказкой и без неё |
| `manifest.json` | Признаки, конфиг, калибровка, checksum, версии среды и данных |
| `metrics.json` | MAE по срезам, reliability вероятности на development и test |
| `split.csv`, `*_importance.csv`, `test_predictions.csv` | Роли строк, важности, предикты test |
| `submission.csv` + `.manifest.json` | 151 прогноз validate и checksum |

Классификаторы обучены **только на fit-фолде**, Platt-параметры подобраны на development
(287 строк, 76 опозданий). Поэтому в manifest статус калибровки — `fitted_on_development`,
а не «проверено на независимом дне». Измерение на test: Brier 0,1395 против 0,1843 у
константной базовой частоты, ROC-AUC 0,8177, log loss 0,4276; без подсказки Brier 0,1763
при базовой 0,1843 и AUC 0,6935. Бины надёжности — в `metrics.json` и [отчёте](../../reports/ml-v2.md).

Это единственный служебный артефакт для HTTP-сервиса: `transport-ml-serve --model ml/pretrained/v2`.
Каталог не менять на месте; новые эксперименты — в `artifacts/<run>/`.
