"""
solve.py -- reference solution for the insar-volcano-inversion task (v4).

v4 change (see process.md): after the InSAR-only screening/fit (v3
pipeline, unchanged), adds a second stage that jointly refits (x0, y0,
depth, rate) against the screened/corrected InSAR residuals AND an
independent continuous GPS station's 3-component (E/N/U) record. This is
standard real-world practice (joint InSAR+GNSS inversion) and is
necessary here specifically because the InSAR spatial sampling (domain
half-width 3500 m) is comparable to the true source depth (4500 m): the
far-field part of the Mogi radial-decay curve is under-sampled, so depth
and cumulative volume change are only weakly separable from InSAR alone.
An InSAR-only fit can show good residuals while sitting on the wrong
point of that depth/volume trade-off; the GPS record, being unaffected
by atmospheric noise, pins that direction down.

Pipeline:
1. Direct LOS joint inversion (x0, y0, depth, rate) using the full LOS
   projection (including north-south sensitivity), against the RAW data
   from all 20 sequential interferograms.
2. For each interferogram, decide whether elevation correction helps
   (compare residual RMS raw vs. corrected against the fitted model);
   apply only if it reduces the residual.
3. Refit with corrected data, then iterate: exclude any interferogram
   whose residual RMS still exceeds the noise threshold, refit without
   it, repeat until the excluded set stabilizes.
4. Joint refinement: refit (x0, y0, depth, rate) against the screened
   InSAR residuals AND the GPS station's E/N/U record together, each
   weighted by its own assumed noise sigma, using the stage-3 solution
   as the starting point. This is the estimate actually reported.
5. 95% CIs via the delta method (analytic covariance from the Jacobian
   of the joint fit).
6. Secondary: decompose the fitted model's cumulative displacement over
   the record into vertical/east-west components (reduced LOS formula),
   and report the model's own predicted final-day GPS displacement for
   comparison against the observed record.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import least_squares

DATA_DIR = os.environ.get("TASK_DATA_DIR", "/workspace/data")
OUTPUT_DIR = os.environ.get("TASK_OUTPUT_DIR", "/workspace/output")

POISSON_RATIO = 0.25
RNG = np.random.default_rng(555)
NOISE_RMS_THRESHOLD_M = 0.008
ONSET_DAY = 60.0  # known/disclosed, not estimated
GPS_SIGMA_ASSUMED_M = 0.003  # typical continuous GNSS daily-solution precision

SEQUENTIAL_IDS = [f"ASC-{i:02d}" for i in range(1, 11)] + [f"DESC-{i:02d}" for i in range(1, 11)]


def load_data():
    ifg = pd.read_csv(os.path.join(DATA_DIR, "interferograms.csv"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "interferogram_metadata.csv"))
    gps = pd.read_csv(os.path.join(DATA_DIR, "gps_station.csv"))
    return ifg, meta, gps


def mogi_displacement(x, y, depth, delta_v_m3, x0=0.0, y0=0.0, nu=POISSON_RATIO):
    dx, dy = x - x0, y - y0
    r = np.sqrt(dx**2 + dy**2)
    R = np.sqrt(r**2 + depth**2)
    C = (1 - nu) * delta_v_m3 / np.pi
    u_z = C * depth / R**3
    u_r = C * r / R**3
    theta = np.arctan2(dy, dx)
    u_x = u_r * np.cos(theta)
    u_y = u_r * np.sin(theta)
    return u_x, u_y, u_z


def los_projection_full(u_e, u_n, u_u, incidence_deg, heading_deg):
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return (np.sin(theta) * np.cos(alpha) * u_e
            - np.sin(theta) * np.sin(alpha) * u_n
            - np.cos(theta) * u_u)


def los_projection_reduced(u_e, u_u, incidence_deg, heading_deg):
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return -np.cos(theta) * u_u + np.sin(theta) * np.cos(alpha) * u_e


def elevation_correct(ifg, ifg_id):
    sub = ifg[ifg.interferogram_id == ifg_id]
    elev = sub["elevation_m"].values
    los = sub["los_displacement_m"].values
    X = np.column_stack([np.ones_like(elev), elev])
    coef, _, _, _ = np.linalg.lstsq(X, los, rcond=None)
    return los - coef[1] * elev


def effective_duration(day, onset=ONSET_DAY):
    return max(0.0, day - onset)


def spatial_prediction(params, points, meta_full):
    x0, y0, depth, rate = params
    preds = []
    for ifg_id, pts in points.items():
        m = meta_full[ifg_id]
        d_days = effective_duration(m["day_end"]) - effective_duration(m["day_start"])
        dV = rate * d_days
        ux, uy, uz = mogi_displacement(pts["x"], pts["y"], depth, dV, x0, y0)
        preds.append(los_projection_full(ux, uy, uz, m["incidence_deg"], m["heading_deg"]))
    return np.concatenate(preds)


def fit_spatial(points, meta_full, observed_concat, robust=False):
    def resid(p):
        return spatial_prediction(p, points, meta_full) - observed_concat

    bounds = ([-3000, -3000, 500, 1e2], [3000, 3000, 15000, 2e7])
    loss = "soft_l1" if robust else "linear"
    best_res, best_cost = None, np.inf
    for _ in range(20):
        p0 = np.array([
            RNG.uniform(-500, 500), RNG.uniform(-500, 500),
            RNG.uniform(2000, 8000), RNG.uniform(1e4, 5e5),
        ])
        try:
            res = least_squares(resid, p0, bounds=bounds, max_nfev=8000, loss=loss)
            cost = np.sum(res.fun ** 2)
            if cost < best_cost:
                best_cost = cost
                best_res = res
        except Exception:
            continue
    return best_res


def screen_and_fit(ifg, meta_full, n_passes=4):
    """Iteratively fit x0,y0,depth,rate using all currently-included
    interferograms, decide per-interferogram whether elevation correction
    helps, exclude any whose residual RMS against the fitted model still
    exceeds the threshold, and repeat until the excluded set stabilizes.
    """
    points_all = {}
    raw_all = {}
    for ifg_id in SEQUENTIAL_IDS:
        sub = ifg[ifg.interferogram_id == ifg_id]
        points_all[ifg_id] = {"x": sub["x_m"].values, "y": sub["y_m"].values}
        raw_all[ifg_id] = sub["los_displacement_m"].values

    included = list(SEQUENTIAL_IDS)
    used_data = dict(raw_all)
    fit = None
    for pass_num in range(n_passes):
        points = {k: points_all[k] for k in included}
        observed = np.concatenate([used_data[k] for k in included])
        fit = fit_spatial(points, meta_full, observed, robust=(pass_num == 0))

        new_included = []
        new_used_data = {}
        for ifg_id in SEQUENTIAL_IDS:
            pred = spatial_prediction(fit.x, {ifg_id: points_all[ifg_id]}, {ifg_id: meta_full[ifg_id]})
            raw_rms = np.sqrt(np.mean((raw_all[ifg_id] - pred) ** 2))
            corrected = elevation_correct(ifg, ifg_id)
            corr_rms = np.sqrt(np.mean((corrected - pred) ** 2))
            best_rms, best_data = (corr_rms, corrected) if corr_rms < raw_rms else (raw_rms, raw_all[ifg_id])
            new_used_data[ifg_id] = best_data
            if best_rms <= NOISE_RMS_THRESHOLD_M:
                new_included.append(ifg_id)

        used_data = new_used_data
        if set(new_included) == set(included):
            included = new_included
            break
        included = new_included

    excluded = [i for i in SEQUENTIAL_IDS if i not in included]
    return included, excluded, used_data, points_all, fit


def build_joint_resid_fn(points, meta_full, insar_observed, gps, sigma_los):
    """Build a residual function combining InSAR residuals AND the GPS
    station's E/N/U record, each weighted by its own assumed noise sigma.
    """
    gps_x = gps["x_m"].values[0]
    gps_y = gps["y_m"].values[0]
    d_days = np.array([effective_duration(d) for d in gps["day"].values.astype(float)])
    gps_obs = np.concatenate([
        gps["east_disp_m"].values, gps["north_disp_m"].values, gps["vertical_disp_m"].values,
    ])

    def resid(params):
        x0, y0, depth, rate = params
        insar_resid = (spatial_prediction(params, points, meta_full) - insar_observed) / sigma_los
        dV = rate * d_days
        ux, uy, uz = mogi_displacement(
            np.full_like(dV, gps_x), np.full_like(dV, gps_y), depth, dV, x0, y0
        )
        gps_pred = np.concatenate([ux, uy, uz])
        gps_resid = (gps_pred - gps_obs) / GPS_SIGMA_ASSUMED_M
        return np.concatenate([insar_resid, gps_resid])

    return resid


def joint_refit_with_gps(p0, resid_fn):
    """Refit (x0,y0,depth,rate) starting from the InSAR-only solution."""
    bounds = ([-3000, -3000, 500, 1e2], [3000, 3000, 15000, 2e7])
    res = least_squares(resid_fn, p0, bounds=bounds, max_nfev=8000)
    return res


def delta_method_ci(resid_fn, best_params, n_params=4):
    """Linearized (delta-method) 95% CI: compute the Jacobian of the
    (weighted) joint residuals with respect to (x0,y0,depth,rate) at the
    best fit via finite differences, then use the standard OLS asymptotic
    covariance (J^T J)^-1 (residuals are already sigma-weighted, so the
    reduced chi-square below is dimensionless and should be near 1 for a
    correctly-specified noise model).
    """
    r0 = resid_fn(best_params)
    n_data = len(r0)
    eps = 1e-6
    J = np.zeros((n_data, n_params))
    for i in range(n_params):
        p_pert = best_params.copy()
        step = eps * max(abs(best_params[i]), 1.0)
        p_pert[i] += step
        J[:, i] = (resid_fn(p_pert) - r0) / step

    reduced_chi2 = np.sum(r0 ** 2) / max(n_data - n_params, 1)
    JTJ = J.T @ J
    cov = reduced_chi2 * np.linalg.inv(JTJ + 1e-12 * np.eye(n_params))
    se = np.sqrt(np.diag(cov))

    lo = best_params - 1.96 * se
    hi = best_params + 1.96 * se
    return lo, hi


def main():
    ifg, meta, gps = load_data()
    meta_full = {row["interferogram_id"]: row for _, row in meta.iterrows()}

    included, excluded, used_data, points_all, insar_fit = screen_and_fit(ifg, meta_full)
    points = {k: points_all[k] for k in included}
    insar_observed = np.concatenate([used_data[k] for k in included])

    r0 = spatial_prediction(insar_fit.x, points, meta_full) - insar_observed
    sigma_los = float(np.sqrt(np.mean(r0 ** 2)))

    resid_fn = build_joint_resid_fn(points, meta_full, insar_observed, gps, sigma_los)
    joint_fit = joint_refit_with_gps(insar_fit.x, resid_fn)
    x0, y0, depth, rate = joint_fit.x

    lo, hi = delta_method_ci(resid_fn, joint_fit.x)

    n_days = int(meta["day_end"].max())
    cumulative_dV = rate * effective_duration(n_days)
    lo_cum = lo[3] * effective_duration(n_days)
    hi_cum = hi[3] * effective_duration(n_days)

    sample_x = np.linspace(-3000, 3000, 21)
    sample_y = np.zeros_like(sample_x)
    ux, uy, uz = mogi_displacement(sample_x, sample_y, depth, cumulative_dV, x0, y0)

    gps_x = float(gps["x_m"].values[0])
    gps_y = float(gps["y_m"].values[0])
    gpx, gpy, gpz = mogi_displacement(
        np.array([gps_x]), np.array([gps_y]), depth, cumulative_dV, x0, y0
    )

    result = {
        "interferograms_passed_filter": included,
        "interferograms_excluded": excluded,
        "interferograms_used_in_inversion": included,
        "x0_m": x0, "x0_uncertainty_95": [lo[0], hi[0]],
        "y0_m": y0, "y0_uncertainty_95": [lo[1], hi[1]],
        "depth_m": depth, "depth_uncertainty_95": [lo[2], hi[2]],
        "volume_change_m3": cumulative_dV, "volume_change_uncertainty_95": [lo_cum, hi_cum],
        "poisson_ratio_assumed": POISSON_RATIO,
        "gps_predicted_displacement_final_m": {
            "east_m": float(gpx[0]), "north_m": float(gpy[0]), "vertical_m": float(gpz[0]),
        },
        "vertical_east_west_decomposition_sample": [
            {"x_m": float(sample_x[i]), "vertical_m": float(uz[i]), "east_west_m": float(ux[i])}
            for i in range(len(sample_x))
        ],
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "result.txt"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(json.dumps({k: v for k, v in result.items() if "decomposition" not in k}, indent=2, default=float))


if __name__ == "__main__":
    main()
