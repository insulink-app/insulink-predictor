"""Synthetic CGM generator (ROADMAP Phase 0).

Builds the whole pipeline *before* real data exists. Produces **irregular raw**
readings (≈5-min cadence with jitter + dropouts + larger gaps) so ``align`` has
real work to do. Each user has distinct dynamics; the signal has genuine,
learnable structure (circadian drift, post-meal excursions, activity dips) so a
model can *legitimately* beat persistence later — the metric is never gamed.

Determinism: per-user ``np.random.default_rng(seed + user_idx)``. No wall-clock
calls, so ``generate`` is byte-reproducible for a given config.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import Config

# Fixed simulation start: a Monday, so is_weekend features are well-defined.
_START = pd.Timestamp("2025-01-06 00:00:00", tz="UTC")
_KERNEL_MIN = 240  # 4h support for meal / insulin / activity response kernels

# Kernel peak fractions calibrated so effects stay in a realistic mg/dL range:
# a ~80 g meal at CSF≈3.3 peaks ~+48 mg/dL; a bolus at ISF≈40 peaks ~dose·3.2.
_CARB_PEAK_FRAC = 0.18  # peak carb rise = frac · carbs · csf   (csf = isf/icr)
_INS_PEAK_FRAC = 0.08  # peak insulin drop = frac · dose · isf


def _biexp_kernel(
    rise_tau: float, decay_tau: float, length: int = _KERNEL_MIN
) -> np.ndarray:
    """Normalized difference-of-exponentials, peak scaled to 1.0."""
    t = np.arange(length)
    k = np.exp(-t / decay_tau) - np.exp(-t / rise_tau)
    return k / k.max()


def _simulate_user(user_idx: int, cfg: Config) -> pd.DataFrame:
    s = cfg.synth
    rng = np.random.default_rng(s.seed + user_idx)
    user_id = f"user_{user_idx}"

    n_min = s.days * 24 * 60
    minute_utc = _START + pd.to_timedelta(np.arange(n_min), unit="min")
    tz_off = int(rng.integers(-1, 3))  # hours; per-user local offset
    minute_local = minute_utc + pd.Timedelta(hours=tz_off)
    hour_of_day = minute_local.hour.to_numpy() + minute_local.minute.to_numpy() / 60.0

    # --- per-user dynamics --------------------------------------------------
    basal = rng.uniform(95, 130)
    circ_amp = rng.uniform(8, 18)
    # Each person's circadian/dawn rhythm peaks at a DIFFERENT hour. A shared model
    # can't fit all phases at once (they average out) — a per-user model can. This
    # is the honest source of the Phase-4 personalization moat.
    circ_phase = rng.uniform(0, 24)
    # Per-user therapy physiology — the real app stores these in user_settings:
    #   isf = correction factor    (mg/dL that 1U insulin lowers glucose)
    #   icr = insulin-to-carb ratio (g carbs per 1U)
    #   csf = carb sensitivity = isf/icr (mg/dL rise per g carbs) [clinical identity]
    # A global model can't know each user's sensitivity from raw grams/units; the
    # therapy features (cob·csf, iob·isf) hand it that scale directly.
    isf = rng.uniform(25.0, 55.0)
    icr = rng.uniform(8.0, 18.0)
    csf = isf / icr
    rise_tau = rng.uniform(15, 25)
    decay_tau = rng.uniform(80, 110)
    noise_sd = rng.uniform(2.0, 5.0)
    has_insulin = user_idx < int(round(s.n_users * s.insulin_user_fraction))

    meal_kernel = _biexp_kernel(rise_tau, decay_tau)
    insulin_kernel = _biexp_kernel(20.0, 70.0)
    activity_kernel = _biexp_kernel(15.0, 60.0)

    # Circadian / dawn phenomenon: smooth daily sinusoid with a per-user phase.
    signal = basal + circ_amp * np.sin(2 * np.pi * (hour_of_day - circ_phase) / 24.0)

    steps = np.zeros(n_min)
    hr = 65.0 + rng.normal(0, 2, n_min)
    meal_min = np.zeros(n_min)  # carbs logged at that minute
    insulin_min = np.zeros(n_min)  # units logged at that minute
    activity_min = np.zeros(n_min, dtype=bool)

    def _add_kernel(
        dst: np.ndarray, start: int, amp: float, kernel: np.ndarray, sign: float = 1.0
    ):
        end = min(n_min, start + kernel.size)
        if start < 0 or end <= start:
            return
        dst[start:end] += sign * amp * kernel[: end - start]

    for d in range(s.days):
        day0 = d * 24 * 60
        # meals clustered around local breakfast / lunch / dinner
        for _ in range(rng.poisson(s.meals_per_day)):
            local_hour = float(rng.choice([8.0, 13.0, 19.0])) + rng.normal(0, 1.0)
            m0 = day0 + int(((local_hour - tz_off) % 24) * 60)
            if not (0 <= m0 < n_min):
                continue
            carbs = float(rng.uniform(20, 80))
            meal_min[m0] = carbs
            _add_kernel(signal, m0, _CARB_PEAK_FRAC * carbs * csf, meal_kernel)
            if has_insulin:
                dose = carbs / icr * rng.uniform(0.85, 1.15)  # dosed by the user's ICR
                insulin_min[m0] = dose
                _add_kernel(
                    signal, m0, _INS_PEAK_FRAC * dose * isf, insulin_kernel, sign=-1.0
                )
        # activity bouts: steps + HR up, glucose dips
        for _ in range(rng.poisson(1.2)):
            local_hour = float(rng.uniform(7, 21))
            a0 = day0 + int(((local_hour - tz_off) % 24) * 60)
            dur = int(rng.uniform(20, 45))
            a1 = min(n_min, max(0, a0) + dur)
            a0c = max(0, a0)
            steps[a0c:a1] += rng.uniform(60, 120)
            hr[a0c:a1] += rng.uniform(30, 60)
            activity_min[a0c:a1] = True
            _add_kernel(signal, a0c, rng.uniform(10, 25), activity_kernel, sign=-1.0)

    # sporadic background steps
    steps += rng.uniform(0, 5, n_min) * (rng.random(n_min) < 0.1)
    # weather: slow daily value, deliberately NOT coupled to glucose (low importance)
    daily_temp = rng.uniform(2, 16, s.days)
    weather = np.repeat(daily_temp, 24 * 60)[:n_min] + rng.normal(0, 0.5, n_min)

    # --- sample irregular CGM readings -------------------------------------
    base_read = np.arange(0, n_min, cfg.grid_minutes)
    jitter = rng.integers(-1, 2, size=base_read.size)  # ±1 min timestamp jitter
    read = np.clip(base_read + jitter, 0, n_min - 1)
    keep = rng.random(read.size) > s.dropout_prob  # random dropouts
    # larger contiguous sensor gaps
    gap_mask = np.ones(n_min, dtype=bool)
    for _ in range(rng.poisson(s.gap_events_per_day * s.days)):
        g0 = int(rng.integers(0, n_min))
        gap_mask[g0 : g0 + int(rng.uniform(20, 60))] = False
    keep &= gap_mask[read]
    read = np.unique(read[keep])

    # window aggregation of event channels since previous reading (via cumsums)
    def _window_sum(x: np.ndarray) -> np.ndarray:
        cs = np.concatenate([[0.0], np.cumsum(x)])
        prev = np.concatenate([[-1], read[:-1]])
        return cs[read + 1] - cs[prev + 1]

    steps_read = _window_sum(steps)
    carbs_read = _window_sum(meal_min)
    insulin_read = _window_sum(insulin_min)
    act_read = _window_sum(activity_min.astype(float)) > 0
    meal_read = carbs_read > 0

    glucose_read = np.clip(signal[read] + rng.normal(0, noise_sd, read.size), 40, 400)

    ts_utc = minute_utc[read]
    ts_local = (ts_utc + pd.Timedelta(hours=tz_off)).tz_localize(None)

    return pd.DataFrame(
        {
            "user_id": user_id,
            "ts_utc": ts_utc,
            "ts_local": ts_local,
            "glucose_mgdl": glucose_read,
            "meal_flag": meal_read,
            "carbs_g": np.where(meal_read, carbs_read, np.nan),
            # insulin channel present only for insulin users (else all-NaN → degrades)
            "insulin_u": insulin_read if has_insulin else np.full(read.size, np.nan),
            "steps": steps_read,
            "activity_flag": act_read,
            "hr": hr[read],
            "weather_temp": weather[read],
            # per-user therapy settings (constant per user; from user_settings in real data)
            "isf": isf,
            "icr": icr,
        }
    )


def generate(cfg: Config) -> pd.DataFrame:
    """Generate irregular raw multi-user CGM data. Deterministic for a config."""
    frames = [_simulate_user(u, cfg) for u in range(cfg.synth.n_users)]
    return pd.concat(frames, ignore_index=True)
