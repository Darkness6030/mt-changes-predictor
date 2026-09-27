"""What-if: release a reserve vehicle to take over the next trips of a late vehicle's line.

A transparent schedule calculation, not a traffic simulator. Assumptions (published with
every answer):

* the predicted delay at the target persists to the end of the current trip;
* at a terminal the planned layover above ``min_layover_s`` absorbs delay; the rest carries
  to the next trip start and persists along it;
* the reserve reaches the terminal ``reserve_in_s`` after now and takes the first following
  trip it can start; from then on it runs with only its own start delay, and the late
  vehicle leaves the line (e.g. to recover at the depot).

In the dataset every vehicle serves its own line, so the line's next trips are the
vehicle's planned next trips.
"""

from dataclasses import dataclass

from transport_backend.clock import SECOND_NS, format_source, parse_source

LATE_S = 120.0


@dataclass(frozen=True)
class TripWindow:
    number: int
    start_ns: int
    end_ns: int


def trip_windows(plan_trips: dict[str, dict], visit_ids: list[str]) -> list[TripWindow]:
    """Ordered trips of one vehicle from the plan's per-visit trip index."""
    seen: dict[int, TripWindow] = {}
    for visit_id in visit_ids:
        info = plan_trips.get(visit_id)
        if info is None or info["number"] in seen:
            continue
        seen[info["number"]] = TripWindow(
            info["number"], parse_source(info["start_at"]), parse_source(info["end_at"])
        )
    return [seen[number] for number in sorted(seen)]


def reserve_whatif(
    trips: list[TripWindow],
    current_trip: int,
    delay_s: float,
    now_ns: int,
    reserve_in_s: float,
    *,
    min_layover_s: float = 180.0,
    horizon_trips: int = 4,
) -> dict:
    """Start delays of the next trips without and with a reserve vehicle."""
    following = [trip for trip in trips if trip.number > current_trip][:horizon_trips]
    by_number = {trip.number: trip for trip in trips}
    if current_trip not in by_number or not following:
        return {"available": False, "reason": "no_following_trips"}
    reserve_ready_ns = now_ns + int(reserve_in_s * SECOND_NS)
    carried = max(0.0, delay_s)  # Early running is not propagated to later trips.
    previous_end = by_number[current_trip].end_ns
    reserve_carried: float | None = None
    rows = []
    for trip in following:
        slack_s = max(0.0, (trip.start_ns - previous_end) / SECOND_NS - min_layover_s)
        without_s = max(0.0, carried - slack_s)
        if reserve_carried is None and reserve_ready_ns <= trip.start_ns + without_s * SECOND_NS:
            # The reserve takes the first trip it can start no later than the late vehicle.
            reserve_carried = max(0.0, (reserve_ready_ns - trip.start_ns) / SECOND_NS)
            with_s = reserve_carried
            served_by = "reserve"
        elif reserve_carried is not None:
            reserve_carried = max(0.0, reserve_carried - slack_s)
            with_s = reserve_carried
            served_by = "reserve"
        else:
            with_s = without_s
            served_by = "vehicle"
        rows.append(
            {
                "trip": trip.number,
                "planned_start_at": format_source(trip.start_ns),
                "delay_without_s": round(without_s, 1),
                "delay_with_s": round(with_s, 1),
                "served_by": served_by,
            }
        )
        carried = without_s
        previous_end = trip.end_ns
    late_without = sum(1 for row in rows if row["delay_without_s"] > LATE_S)
    late_with = sum(1 for row in rows if row["delay_with_s"] > LATE_S)
    first_reserve = next((row["trip"] for row in rows if row["served_by"] == "reserve"), None)
    return {
        "available": True,
        "reserve_ready_at": format_source(reserve_ready_ns),
        "reserve_takes_trip": first_reserve,
        "trips": rows,
        "late_trips_without": late_without,
        "late_trips_with": late_with,
        "delay_saved_s": round(
            sum(row["delay_without_s"] - row["delay_with_s"] for row in rows), 1
        ),
        "late_threshold_s": LATE_S,
        "min_layover_s": min_layover_s,
        "assumptions": [
            "Прогнозная задержка сохраняется до конца текущего рейса",
            f"Отстой на конечной сверх {min_layover_s / 60:g} мин гасит опоздание",
            "Резерв берёт первый рейс, к которому успевает; дальше идёт по графику",
            "Схема по плану линии, не транспортная модель города",
        ],
    }
