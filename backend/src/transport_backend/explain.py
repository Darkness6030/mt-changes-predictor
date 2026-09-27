"""Evidence and dispatcher hints derived only from features that were actually computed.

Nothing here claims a cause. The dataset has no incident, door or passenger data, so the
text describes observed patterns and proposes a check, never an accident or a repair.
"""

from transport_backend.config import RiskPolicy

SPEED_DROP_RATIO = 0.6
SPEED_FLOOR_KMH = 5.0
LONG_DWELL_S = 120.0
NEAR_TARGET_M = 1500.0


def _number(value: float, digits: int = 0) -> str:
    text = f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")
    return text


def signed_seconds(delay_s: float) -> str:
    sign = "+" if delay_s >= 0 else "−"
    total = abs(int(round(delay_s)))
    return f"{sign}{total // 60} мин {total % 60:02d} с" if total >= 60 else f"{sign}{total} с"


def evidence(features: dict, *, stale_after_s: float) -> list[dict]:
    """Ordered, de-duplicated observations behind the current prediction."""
    items: list[dict] = []
    recent = features.get("speed_mean_180s")
    baseline = features.get("speed_mean_600s")
    if (
        recent is not None
        and baseline is not None
        and baseline == baseline
        and recent == recent
        and baseline > SPEED_FLOOR_KMH
        and recent < baseline * SPEED_DROP_RATIO
    ):
        items.append(
            {
                "kind": "speed_drop",
                "text": f"Средняя скорость за 3 мин {_number(recent, 1)} км/ч против "
                f"{_number(baseline, 1)} км/ч за 10 мин",
                "value": float(recent),
            }
        )
    dwell = features.get("observed_stop_s")
    if dwell is not None and dwell == dwell and dwell >= LONG_DWELL_S:
        items.append(
            {
                "kind": "long_dwell",
                "text": f"Наблюдаемая стоянка {_number(dwell)} с без движения",
                "value": float(dwell),
            }
        )
    hint = features.get("cur_dev_s")
    if hint is not None and hint == hint and abs(hint) >= 60:
        word = "опоздание" if hint > 0 else "опережение"
        items.append(
            {
                "kind": "current_deviation",
                "text": f"На последней пройденной остановке {word} {_number(abs(hint))} с",
                "value": float(hint),
            }
        )
    age = features.get("gps_age_s")
    if age is not None and age == age and age > stale_after_s:
        items.append(
            {
                "kind": "stale_position",
                "text": f"Последняя валидная позиция получена {_number(age)} с назад",
                "value": float(age),
            }
        )
    distance = features.get("target_distance_m")
    if distance is not None and distance == distance:
        items.append(
            {
                "kind": "distance_to_target",
                "text": f"До целевой остановки по прямой {_number(distance / 1000, 1)} км",
                "value": float(distance),
            }
        )
    valid = features.get("valid_fraction_600s")
    if valid is not None and valid == valid and valid < 0.8:
        items.append(
            {
                "kind": "data_quality",
                "text": f"Валидных координат за 10 мин {_number(valid * 100)}%",
                "value": float(valid),
            }
        )
    return items


def recommendation(
    delay_s: float,
    items: list[dict],
    policy: RiskPolicy,
    late_probability: float | None = None,
    trip: dict | None = None,
) -> str:
    """A check to perform, phrased as a hypothesis for the dispatcher to confirm."""
    kinds = {item["kind"] for item in items}
    if "stale_position" in kinds:
        return "Проверить связь с бортовым терминалом: прогноз опирается на устаревшую позицию"
    likely_late = delay_s > policy.red_min_delay_s or (
        late_probability is not None and late_probability >= policy.late_probability_red
    )
    if trip and likely_late and (trip.get("first") or trip.get("last")):
        edge = "первого" if trip.get("first") else "последнего"
        return (
            f"Приоритет: риск срыва {edge} рейса дня (рейс {trip['number']} из "
            f"{trip['total']}). Связаться с водителем сейчас; при необходимости подготовить "
            "резервное ТС или корректировку по правилам организатора перевозок"
        )
    if delay_s > policy.red_min_delay_s:
        if "long_dwell" in kinds:
            return (
                "Уточнить у водителя причину длительной стоянки и оценить регулирование интервала"
            )
        if "speed_drop" in kinds:
            return "Вероятно затруднённое движение на участке: уточнить обстановку и интервал"
        return "Связаться с водителем и оценить оперативное регулирование по действующим правилам"
    if delay_s < policy.early_yellow_s:
        hold_min = max(1, round(-delay_s / 60))
        return (
            f"Опережение графика: придержать ТС на ближайшей остановке примерно на {hold_min} "
            "мин, чтобы не уйти раньше расписания и не сбить интервал"
        )
    if late_probability is not None and late_probability >= policy.late_probability_red:
        percent = round(late_probability * 100)
        return (
            f"Опоздание больше {policy.red_min_delay_s:g} с вероятно ({percent}%): уточнить "
            "обстановку на участке заранее и подготовить регулирование интервала"
        )
    if delay_s > policy.green_max_delay_s:
        return "Держать ТС под наблюдением: отклонение выше допустимого, но ниже порога внимания"
    return "Действий не требуется: прогноз в пределах допустимого отклонения"
