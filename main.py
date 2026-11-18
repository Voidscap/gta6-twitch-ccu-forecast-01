"""Forecast launch-day Twitch concurrent viewers for GTA VI, hours 2..24.

Model, in one paragraph: launch day on Twitch follows the unlock clock.
Hour 1 (00:00-01:00 UTC) only has the regions that already unlocked
(New Zealand, Australia, Asia, then Europe at 23:00Z and the UK at 00:00Z);
the big English-speaking audiences arrive with the US east coast at 05:00Z
and west coast at 08:00Z, so viewership typically climbs into the evening
and peaks around hours 18-22. The kit ships 38 past launches with full
release-day hourly curves, so this program learns the curve from them at
run time instead of assuming hour 1 persists:

  * keep only "rolling-midnight" launches: a launch whose day peak is at
    most 8x its hour-1 count. Simultaneous global unlocks (one worldwide
    unlock hour, e.g. 14:00Z) have a near-empty hour 1 and a 100-1000x
    climb, which is not how a rolling midnight launch behaves, so they
    are left out of the fit. If that leaves too few launches to fit on,
    the peak limit is relaxed step by step (8, 20, then no limit) rather
    than falling back to a flat forecast.
  * for each hour h, regress the log ratio log(v_h / v_1) on log(v_1)
    across the kept launches. Big launches already draw a large hour 1
    (night owls, APAC, hype streams) and grow less; small launches grow
    far more. The slope is about 0 at hour 2 and about -0.3 by hour 19.
  * the spread for hour h is the residual standard deviation of that
    regression (floored at 0.30 log units), and the five quantiles are
    p1 * exp(median_h + z_tau * spread_h) with the usual normal z values.
  * the log(hour-1) predictor is clamped to +-2.5 standard deviations
    of the training launches, so a record-breaking GTA VI hour 1 does
    not extrapolate the slope without limit; the median and spread are
    lightly smoothed over neighbouring hours (3-hour window).

Everything is computed from the case file itself, so the final case --
whose comparables also include the four practice launches -- is handled
by the same code. Standard library only, no network, deterministic.

Local check (leave-one-out over the 26 rolling launches in the kit):
loss about 0.20 against the house baseline's 0.30 on the same launches.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

# A launch counts as rolling-midnight when its release-day peak is at
# most this many times its hour-1 count. Simultaneous global unlocks sit
# far above it (tens to thousands of times). The limit is relaxed in
# steps if too few launches pass it (see MIN_TRAINING).
ROLLING_PEAK_LIMITS = (8.0, 20.0, float("inf"))
# A training launch needs hour 1 plus at least this many of hours 1..24.
MIN_HOURS = 20
# Minimum launches needed before the peak limit is relaxed.
MIN_TRAINING = 8
# Floor on the log spread of any hour, and the clamp on the size
# predictor, in standard deviations of the training launches.
SPREAD_FLOOR = 0.30
SIZE_CLAMP_SD = 2.5


def parse_time(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_of(day_start, stamp):
    """Which hour of the release day (1..24) a timestamp falls in."""
    delta = (parse_time(stamp) - day_start).total_seconds()
    return int(delta // 3600) + 1


def release_day_series(comparable):
    """Hourly viewers {hour: viewers} for a comparable's release day."""
    day_start = parse_time(comparable["day_start"])
    series = {}
    for row in comparable.get("hourly", []):
        viewers = row.get("viewers")
        if not viewers or viewers <= 0:
            continue
        hour = hour_of(day_start, row["from"])
        if 1 <= hour <= 24:
            series[hour] = float(viewers)
    return series


def training_launches(case):
    """Rolling-midnight comparables usable for fitting the curve.

    The peak limit is relaxed step by step when too few launches pass
    it, so a differently mixed comparable set still yields a fit.
    """
    complete = []
    for comparable in case.get("comparables", []):
        series = release_day_series(comparable)
        if 1 not in series or len(series) < MIN_HOURS:
            continue
        complete.append(series)
    for limit in ROLLING_PEAK_LIMITS:
        kept = [s for s in complete if max(s.values()) / s[1] <= limit]
        if len(kept) >= MIN_TRAINING or limit == ROLLING_PEAK_LIMITS[-1]:
            kept.sort(key=lambda s: s[1])
            return kept
    return []


def mean(values):
    return sum(values) / len(values)


def pstdev(values, centre):
    return math.sqrt(sum((v - centre) ** 2 for v in values) / len(values))


def fit_hour(launches, hour):
    """Regress log(v_h / v_1) on log(v_1) for one hour.

    Returns (centre_x, centre_y, slope, residual_sd, sd_x) of the fit,
    or None when too few launches have that hour.
    """
    xs = []
    ys = []
    for series in launches:
        if hour in series:
            xs.append(math.log(series[1]))
            ys.append(math.log(series[hour] / series[1]))
    if len(xs) < 5:
        return None
    centre_x = mean(xs)
    centre_y = mean(ys)
    var_x = sum((x - centre_x) ** 2 for x in xs)
    slope = 0.0
    if var_x > 0:
        slope = sum((x - centre_x) * (y - centre_y)
                    for x, y in zip(xs, ys)) / var_x
    residuals = [y - (centre_y + slope * (x - centre_x))
                 for x, y in zip(xs, ys)]
    return (centre_x, centre_y, slope,
            pstdev(residuals, 0.0), pstdev(xs, centre_x))


def hour_forecast(launches, p1, hour):
    """Median log ratio and spread for one hour, smoothed over neighbours."""
    log_p1 = math.log(p1)
    medians = []
    spreads = []
    for neighbour in (hour - 1, hour, hour + 1):
        if neighbour < 2 or neighbour > 24:
            continue
        fit = fit_hour(launches, neighbour)
        if fit is None:
            continue
        centre_x, centre_y, slope, resid_sd, sd_x = fit
        clamped = min(centre_x + SIZE_CLAMP_SD * sd_x,
                      max(centre_x - SIZE_CLAMP_SD * sd_x, log_p1))
        medians.append(centre_y + slope * (clamped - centre_x))
        spreads.append(max(SPREAD_FLOOR, resid_sd))
    if not medians:
        return None
    median = sum(medians) / len(medians)
    spread = sum(spreads) / len(spreads)
    return median, spread


def round_half_up_2dp(value):
    """Round to 2 decimals, halves up, returning int for whole numbers."""
    scaled = value * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    rounded = whole / 100
    return int(rounded) if float(rounded).is_integer() else rounded


def baseline_quantiles(p1, hour):
    """House-style persistence fallback, used only if no fit is possible."""
    widen = 0.15 * math.sqrt(hour - 1)
    return [round_half_up_2dp(p1 * math.exp(z * widen)) for z in Z_SCORES]


def forecast_case(case):
    """Build the {"hours": [...]} forecast for one case dict."""
    known = case.get("known", [])
    first = next((row for row in known if row.get("hour") == 1), None)
    if first is None and known:
        first = known[0]
    p1 = 1.0
    if first is not None:
        p1 = max(1.0, float(first.get("viewers") or 1))
    launches = training_launches(case)
    rows = []
    for target in case.get("forecast", []):
        hour = int(target["hour"])
        fitted = hour_forecast(launches, p1, hour) if launches else None
        if fitted is None:
            quantiles = baseline_quantiles(p1, hour)
        else:
            median, spread = fitted
            quantiles = []
            previous = 0.0
            for z in Z_SCORES:
                value = p1 * math.exp(median + z * spread)
                value = min(1e9, max(1.0, value))
                rounded = round_half_up_2dp(value)
                # Rounding must not break the non-decreasing order.
                if rounded < previous:
                    rounded = previous
                quantiles.append(rounded)
                previous = rounded
        rows.append({"hour": hour, "q": quantiles})
    rows.sort(key=lambda row: row["hour"])
    return {"hours": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
