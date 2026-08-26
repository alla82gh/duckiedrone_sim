#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
metrics.py — evaluation metrics per thesis Section 4.2.4, Eqs. (4.1)-(4.5).

  RMSE        (4.1)  root mean square error per channel
  IAE         (4.2)  integral of |error| over time
  settling    (4.3)  time to enter and stay within +-5% band of a step
  overshoot   (4.4)  peak excursion beyond the step target, in %
  effort      (4.5)  sum ||Delta u||^2 over the run
  t_c                per-cycle computation time (mean/max) [ms]

  AUTHOR: Abdallah GHOUL  2026
"""
import numpy as np

CHANNELS = ["x", "y", "z", "phi", "theta", "psi"]
MIN_STEP = 0.02  # [m or rad] min commanded-step amplitude for a meaningful overshoot %


def rmse(err):
    return float(np.sqrt(np.mean(err ** 2)))


def iae(err, dt):
    return float(np.sum(np.abs(err)) * dt)


def settling_time(t, y, target, band=0.05):
    """Time after which |y-target| stays within +-band*|target| (or +-band
    absolute if target ~ 0). Returns NaN if never settles."""
    tol = max(abs(target) * band, 0.01)
    outside = np.abs(y - target) > tol
    if outside.all():
        return float("nan")
    last_out = np.where(outside)[0]
    if len(last_out) == 0:
        return 0.0
    idx = last_out[-1] + 1
    return float(t[idx] - t[0]) if idx < len(t) else float("nan")


def overshoot(y, target, initial=0.0):
    """Peak overshoot in % of the step amplitude."""
    amp = target - initial
    if abs(amp) < 1e-9:
        return 0.0
    peak = (y - target) if amp > 0 else (target - y)
    return float(max(0.0, peak.max() / abs(amp) * 100.0))


def control_effort(u_series):
    """u_series: (N,4) absolute inputs -> sum ||Delta u||^2 (Eq. 4.5)."""
    du = np.diff(u_series, axis=0)
    return float(np.sum(du ** 2))


def summarize(t, x_series, xref_series, u_series, comp_ms, phases,
              gamma=None, r=None):
    """Build the per-run summary dict (one row of thesis Tables 4.2-4.4).

    Metrics are computed on the 'scenario' phase only (takeoff/landing
    excluded), matching the protocol of Section 4.2.4.
    """
    mask = phases == "scenario"
    ts, xs, xr = t[mask], x_series[mask], xref_series[mask]
    us, cm = u_series[mask], comp_ms[mask]
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    row = {}
    for i, ch in enumerate(CHANNELS):
        row["rmse_" + ch] = rmse(xs[:, i] - xr[:, i])
    row["iae_pos"] = iae(xr[:, 0:3] - xs[:, 0:3], dt)
    row["iae_att"] = iae(xr[:, 3:6] - xs[:, 3:6], dt)
    # settling: time after scenario start to stay within +-5% of final level
    # overshoot: peak beyond final level over the FULL run, in % of the
    # commanded step amplitude (Eq. 4.4); channels whose commanded
    # amplitude < MIN_STEP carry no meaningful overshoot -> skipped
    stl, ovr = [], []
    for i, ch in enumerate(CHANNELS):
        tgt = float(xr[-1, i])
        s = settling_time(ts, xs[:, i], tgt)
        if not np.isnan(s):
            stl.append(s)
        amp = tgt - float(x_series[0, i])
        if abs(amp) >= MIN_STEP:
            ovr.append(overshoot(x_series[:, i], tgt,
                                 initial=float(x_series[0, i])))
    row["settling_mean"] = float(np.mean(stl)) if stl else float("nan")
    row["overshoot_mean"] = float(np.mean(ovr)) if ovr else float("nan")
    row["effort_du"] = control_effort(us)
    row["tc_mean_ms"] = float(np.mean(cm))
    row["tc_max_ms"] = float(np.max(cm))
    if gamma is not None:
        row["gamma_mean"] = float(np.mean(gamma[mask]))
        row["r_mean"] = float(np.mean(r[mask]))
    return row
