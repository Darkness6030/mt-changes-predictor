"""ML-IMPROVE-18: CatBoost + PyTorch sequence model ensemble (PDF extra feature), train-only.

The sequence branch reads the last 10 minutes of telemetry before T (40 steps of 15 s:
speed, GPS validity, distance to the target stop and its change, age of the fix; only
``event_time <= T``) plus the 73 tabular features; a GRU and a small MLP predict the
residual to ``cur_dev_s`` (main) or the delay itself (no hint). Selection uses the improve12
inner folds only; the blend weight is chosen there and then checked once on the reserved
outer vehicles against the stored v3 outer predictions. Test and validate are not read.

    .venv/bin/python ml/experiments/improve18_sequence.py improve18-sequence.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from transport_ml.data import load_labels
from transport_ml.features import SECOND, distance_m
from transport_ml.research import inner_folds, load_research_data

sys.path.insert(0, str(Path(__file__).parent))
from improve18_experiments import PROTOCOL, RECIPE, out_of_fold, score  # noqa: E402

STEPS, STEP_S = 40, 15
CHANNELS = 5
SEED = 20260927


def sequences(points: pd.DataFrame, builder) -> np.ndarray:
    """Causal telemetry grid per point: last fix in each 15 s step up to T."""
    result = np.zeros((len(points), STEPS, CHANNELS), dtype=np.float32)
    for row, point in enumerate(points.itertuples()):
        history = builder.history.get(str(point.tr_id))
        if history is None:
            continue
        visit = builder.visit_keys.loc[(str(point.tr_id), str(point.target_stop_id))]
        at = point.T.value
        stop = np.searchsorted(history["time"], at, side="right")
        previous_distance = None
        for step in range(STEPS):
            end = at - (STEPS - 1 - step) * STEP_S * SECOND
            index = np.searchsorted(history["time"][:stop], end, side="right") - 1
            if index < 0:
                continue
            age = (end - history["time"][index]) / SECOND
            valid = bool(history["gps_valid"][index])
            speed = history["speed"][index]
            distance = (
                float(
                    distance_m(history["lon"][index], history["lat"][index], visit.lon, visit.lat)
                )
                / 1000
                if valid
                else np.nan
            )
            delta = 0.0
            if previous_distance is not None and np.isfinite(distance):
                delta = previous_distance - distance
            if np.isfinite(distance):
                previous_distance = distance
            result[row, step] = [
                0.0 if not np.isfinite(speed) else speed / 60,
                float(valid),
                0.0 if not np.isfinite(distance) else min(distance, 20) / 10,
                delta,
                min(age, 300) / 300,
            ]
    return result


class SequenceNet(nn.Module):
    def __init__(self, tabular: int):
        super().__init__()
        self.gru = nn.GRU(CHANNELS, 32, batch_first=True)
        self.tab = nn.Sequential(nn.Linear(tabular, 64), nn.ReLU(), nn.Dropout(0.1))
        self.head = nn.Sequential(nn.Linear(96, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, seq, tab):
        _, hidden = self.gru(seq)
        return self.head(torch.cat([hidden[-1], self.tab(tab)], dim=1)).squeeze(1)


def fit_predict(seq, tab, target, offset, fit, predict, *, epochs=120, seed=SEED):
    torch.manual_seed(seed)
    center = np.nanmedian(tab[fit], axis=0)
    spread = np.nanstd(tab[fit], axis=0) + 1e-6
    x = np.nan_to_num((tab - center) / spread, nan=0.0).clip(-5, 5).astype(np.float32)
    y = ((target - offset) / 100).astype(np.float32)
    model = SequenceNet(tab.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    s, t, yt = torch.from_numpy(seq), torch.from_numpy(x), torch.from_numpy(y)
    index = np.flatnonzero(fit)
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        for batch in np.array_split(generator.permutation(index), max(1, len(index) // 64)):
            optimiser.zero_grad()
            loss = torch.nn.functional.l1_loss(model(s[batch], t[batch]), yt[batch])
            loss.backward()
            optimiser.step()
    model.eval()
    with torch.no_grad():
        out = model(s[predict], t[predict]).numpy() * 100
    return out + offset[predict]


def main(output: Path) -> None:
    points, features, builder = load_research_data(Path("dataset"))
    target_all = np.full(len(points), np.nan)
    outer = points.tr_id.isin(PROTOCOL["outer_vehicles"]).to_numpy()
    target_all[~outer] = load_labels(Path("dataset"), "train", points[~outer])
    seq_all = sequences(points, builder)
    inner_points = points[~outer].reset_index(drop=True)
    inner_features = features[~outer].reset_index(drop=True)
    target = target_all[~outer]
    seq = seq_all[~outer]
    folds = inner_folds(inner_points, target, PROTOCOL)
    results = {}
    for mode in ("main", "fallback"):
        columns = [c for c in inner_features if mode == "main" or c != "cur_dev_s"]
        tab = inner_features[columns].to_numpy(dtype=np.float64)
        offset = (
            inner_features.cur_dev_s.fillna(0).to_numpy()
            if mode == "main"
            else np.zeros(len(target))
        )
        cat = out_of_fold(inner_features, target, folds, RECIPE["models"][mode])
        net = {
            name: (np.flatnonzero(valid), fit_predict(seq, tab, target, offset, fit, valid))
            for name, fit, valid, _ in folds
        }
        results[f"{mode}:catboost"] = score(cat, target)
        results[f"{mode}:sequence_net"] = score(net, target)
        best = (1.0, results[f"{mode}:catboost"][2])
        for weight in (0.9, 0.8, 0.7, 0.6, 0.5):
            blend = {
                name: (index, weight * values + (1 - weight) * net[name][1])
                for name, (index, values) in cat.items()
            }
            results[f"{mode}:blend w_catboost={weight}"] = score(blend, target)
            if results[f"{mode}:blend w_catboost={weight}"][2] < best[1]:
                best = (weight, results[f"{mode}:blend w_catboost={weight}"][2])
        results[f"{mode}:selected_weight"] = best[0]
        # One outer check of the selected blend against the stored v3 outer predictions.
        stored = pd.read_csv("ml/reports/ml-v3-holdout-predictions.csv")
        stored = stored.set_index("sample_id").loc[points.sample_id[outer]]
        fit_all = np.ones(len(target), dtype=bool)
        outer_tab = features[outer][columns].to_numpy(dtype=np.float64)
        outer_offset = (
            features[outer].cur_dev_s.fillna(0).to_numpy()
            if mode == "main"
            else np.zeros(outer.sum())
        )
        joint_seq = np.concatenate([seq, seq_all[outer]])
        joint_tab = np.concatenate([tab, outer_tab])
        joint_target = np.concatenate([target, np.zeros(outer.sum())])
        joint_offset = np.concatenate([offset, outer_offset])
        fit_mask = np.concatenate([fit_all, np.zeros(outer.sum(), dtype=bool)])
        net_outer = fit_predict(
            joint_seq, joint_tab, joint_target, joint_offset, fit_mask, ~fit_mask
        )
        truth = stored.target.to_numpy()
        base = stored[mode].to_numpy()
        blended = best[0] * base + (1 - best[0]) * net_outer
        results[f"{mode}:outer"] = {
            "catboost_v3": float(np.abs(truth - base).mean()),
            "sequence_net": float(np.abs(truth - net_outer).mean()),
            "selected_blend": float(np.abs(truth - blended).mean()),
        }
        print(
            mode, json.dumps({k: v for k, v in results.items() if k.startswith(mode)}), flush=True
        )
    output.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
