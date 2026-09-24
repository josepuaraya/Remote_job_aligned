"""
solve.py -- reference solution for the insar-volcano-inversion task (v5).

v5 change (see process.md): the InSAR atmospheric noise and the GNSS
common-mode error are now genuinely correlated (spatial for InSAR,
cross-station for GNSS), not i.i.d. A correct inversion estimates both
covariance structures from the visible data and uses them to weight a
generalized least squares (GLS) / Bayesian joint refit, rather than
treating every observation as independent and equally weighted.

Pipeline:
1. Same as v4: direct LOS joint inversion on raw data, per-interferogram
   elevation-correction decision by residual RMS comparison, iterative
   exclusion until the excluded set stabilizes.
2. Estimate the InSAR noise's spatial covariance from the screened
   residuals: an empirical semivariogram restricted to short lags (where
   the true correlated-noise signal dominates over any leftover
   large-scale structure from the elevation/turbulent correction terms),
   fit to an exponential model to recover a correlation length and sill.
3. Estimate the GNSS network's noise covariance (common-mode vs.
   per-station white variance) via a method-of-moments decomposition of
   the residuals against the stage-1 source model.
4. Joint GLS refit: whiten the InSAR residuals (per interferogram, using
   the Cholesky factor of the estimated spatial covariance) and the GNSS
   residuals (per epoch/component, using the Cholesky factor of the
   estimated compound-symmetry covariance), then refit (x0, y0, depth,
   rate) against the combined whitened residuals, starting from the
   stage-1 solution.
5. 95% CIs via the delta method from this GLS fit's Jacobian.
6. Secondary: decompose the fitted model's cumulative displacement into
   vertical/east-west components, and report the model's own predicted
   final-day displacement at the primary GNSS station.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import least_squares, curve_fit
from scipy.linalg import solve_triangular, cholesky

DATA_DIR = os.environ.get("TASK_DATA_DIR", "/workspace/data")
OUTPUT_DIR = os.environ.get("TASK_OUTPUT_DIR", "/workspace/output")

POISSON_RATIO = 0.25
RNG = np.random.default_rng(555)
NOISE_RMS_THRESHOLD_M = 0.008
ONSET_DAY = 60.0
PRIMARY_GNSS_ID = "GNSS-01"
VARIOGRAM_MAX_LAG_M = 800.0
VARIOGRAM_N_BINS = 10

SEQUENTIAL_IDS = [f"ASC-{i:02d}" for i in range(1, 11)] + [f"DESC-{i:02d}" for i in range(1, 11)]


def load_data():
    ifg = pd.read_csv(os.path.join(DATA_DIR, "interferograms.csv"))
    meta = pd.read_csv(os.path.join(DATA_DIR, "interferogram_metadata.csv"))
    gnss = pd.read_csv(os.path.join(DATA_DIR, "gps_stations.csv"))
    return ifg, meta, gnss


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


def estimate_insar_covariance(points_all, meta_full, included, used_data, params):
    """Empirical semivariogram of the screened residuals, restricted to
    short lags (where genuine spatially-correlated noise dominates over
    any leftover large-scale structure from the elevation/turbulent
    correction terms), fit to an exponential model.
    """
    x = points_all[included[0]]["x"]
    y = points_all[included[0]]["y"]
    resid_matrix = np.array([
        used_data[k] - spatial_prediction(params, {k: points_all[k]}, {k: meta_full[k]})
        for k in included
    ])

    n = len(x)
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    iu = np.triu_indices(n, k=1)
    d_pairs = dist[iu]

    bins = np.linspace(0, VARIOGRAM_MAX_LAG_M, VARIOGRAM_N_BINS + 1)
    bin_idx = np.digitize(d_pairs, bins) - 1
    sums = np.zeros(VARIOGRAM_N_BINS)
    counts = np.zeros(VARIOGRAM_N_BINS)
    for k in range(resid_matrix.shape[0]):
        r = resid_matrix[k]
        sq_diff = (r[iu[0]] - r[iu[1]]) ** 2
        for b in range(VARIOGRAM_N_BINS):
            mask = bin_idx == b
            if mask.any():
                sums[b] += sq_diff[mask].sum()
                counts[b] += mask.sum()

    valid = counts > 0
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    gamma_emp = 0.5 * sums[valid] / counts[valid]

    def model(d, sill, rng_):
        return sill * (1 - np.exp(-d / rng_))

    try:
        popt, _ = curve_fit(
            model, bin_centers[valid], gamma_emp,
            p0=[max(gamma_emp[-1], 1e-8), 300.0],
            bounds=([1e-9, 10.0], [1e-2, 3000.0]),
        )
        sill, corr_length = float(popt[0]), float(popt[1])
    except Exception:
        sill, corr_length = float(np.var(resid_matrix)), 300.0

    return sill, corr_length, x, y


def build_spatial_cholesky(x, y, sill, corr_length):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    cov = sill * np.exp(-dist / corr_length)
    cov += 1e-9 * sill * np.eye(len(x))
    return cholesky(cov, lower=True)


def estimate_gnss_covariance(gnss, station_ids, params):
    """Method-of-moments decomposition of GNSS residuals (against the
    stage-1 source model) into per-station white variance and shared
    per-epoch common-mode variance.
    """
    x0, y0, depth, rate = params
    comps = ["east_disp_m", "north_disp_m", "vertical_disp_m"]
    dev_pool, mean_pool = [], []
    for day, grp in gnss.groupby("day"):
        dV = rate * effective_duration(float(day))
        resid_by_comp = {c: [] for c in comps}
        for _, row in grp.iterrows():
            ux, uy, uz = mogi_displacement(
                np.array([row.x_m]), np.array([row.y_m]), depth, dV, x0, y0
            )
            pred = {"east_disp_m": ux[0], "north_disp_m": uy[0], "vertical_disp_m": uz[0]}
            for c in comps:
                resid_by_comp[c].append(row[c] - pred[c])
        for c in comps:
            r = np.array(resid_by_comp[c])
            m = r.mean()
            mean_pool.append(m)
            dev_pool.extend((r - m).tolist())

    dev_pool = np.array(dev_pool)
    mean_pool = np.array(mean_pool)
    n_st = len(station_ids)
    var_dev = dev_pool.var()
    var_mean = mean_pool.var()
    sigma_w2 = var_dev * n_st / max(n_st - 1, 1)
    sigma_c2 = max(var_mean - sigma_w2 / n_st, 1e-10)
    return float(np.sqrt(sigma_c2)), float(np.sqrt(sigma_w2))


def build_gnss_cholesky(n_stations, sigma_common, sigma_white):
    cov = sigma_common**2 * np.ones((n_stations, n_stations)) + sigma_white**2 * np.eye(n_stations)
    return cholesky(cov, lower=True)


def build_joint_resid_fn(points, meta_full, insar_observed, insar_L, insar_ids,
                          gnss, station_ids, gnss_L):
    """Combined GLS residual function: whitens InSAR residuals (per
    interferogram, using the estimated spatial covariance's Cholesky
    factor) and GNSS residuals (per epoch/component, using the estimated
    compound-symmetry covariance's Cholesky factor), then concatenates.
    """
    station_coords = {sid: gnss[gnss.station_id == sid].iloc[0][["x_m", "y_m"]].values for sid in station_ids}
    epochs = sorted(gnss["day"].unique())
    comps = ["east_disp_m", "north_disp_m", "vertical_disp_m"]
    gnss_obs_by_epoch_comp = {}
    for day in epochs:
        grp = gnss[gnss.day == day].set_index("station_id")
        for c in comps:
            gnss_obs_by_epoch_comp[(day, c)] = grp.loc[station_ids, c].values.astype(float)

    def resid(params):
        x0, y0, depth, rate = params

        insar_whitened = []
        for ifg_id in insar_ids:
            pred = spatial_prediction(params, {ifg_id: points[ifg_id]}, {ifg_id: meta_full[ifg_id]})
            raw_r = pred - insar_observed[ifg_id]
            insar_whitened.append(solve_triangular(insar_L, raw_r, lower=True))
        insar_whitened = np.concatenate(insar_whitened)

        gnss_whitened = []
        for day in epochs:
            dV = rate * effective_duration(float(day))
            for c in comps:
                preds = []
                for sid in station_ids:
                    sx, sy = station_coords[sid]
                    ux, uy, uz = mogi_displacement(np.array([sx]), np.array([sy]), depth, dV, x0, y0)
                    preds.append({"east_disp_m": ux[0], "north_disp_m": uy[0], "vertical_disp_m": uz[0]}[c])
                raw_r = np.array(preds) - gnss_obs_by_epoch_comp[(day, c)]
                gnss_whitened.append(solve_triangular(gnss_L, raw_r, lower=True))
        gnss_whitened = np.concatenate(gnss_whitened)

        return np.concatenate([insar_whitened, gnss_whitened])

    return resid


def joint_refit(p0, resid_fn):
    bounds = ([-3000, -3000, 500, 1e2], [3000, 3000, 15000, 2e7])
    return least_squares(resid_fn, p0, bounds=bounds, max_nfev=8000)


def delta_method_ci(resid_fn, best_params, n_params=4):
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
    ifg, meta, gnss = load_data()
    meta_full = {row["interferogram_id"]: row for _, row in meta.iterrows()}
    station_ids = sorted(gnss["station_id"].unique())

    included, excluded, used_data, points_all, insar_fit = screen_and_fit(ifg, meta_full)
    points = {k: points_all[k] for k in included}
    insar_observed = {k: used_data[k] for k in included}

    sill, corr_length, x_used, y_used = estimate_insar_covariance(
        points_all, meta_full, included, used_data, insar_fit.x
    )
    insar_L = build_spatial_cholesky(x_used, y_used, sill, corr_length)

    sigma_common, sigma_white = estimate_gnss_covariance(gnss, station_ids, insar_fit.x)
    gnss_L = build_gnss_cholesky(len(station_ids), sigma_common, sigma_white)

    resid_fn = build_joint_resid_fn(
        points, meta_full, insar_observed, insar_L, included,
        gnss, station_ids, gnss_L,
    )
    joint_fit = joint_refit(insar_fit.x, resid_fn)
    x0, y0, depth, rate = joint_fit.x

    lo, hi = delta_method_ci(resid_fn, joint_fit.x)

    n_days = int(meta["day_end"].max())
    cumulative_dV = rate * effective_duration(n_days)
    lo_cum = lo[3] * effective_duration(n_days)
    hi_cum = hi[3] * effective_duration(n_days)

    sample_x = np.linspace(-3000, 3000, 21)
    sample_y = np.zeros_like(sample_x)
    ux, uy, uz = mogi_displacement(sample_x, sample_y, depth, cumulative_dV, x0, y0)

    primary_row = gnss[gnss.station_id == PRIMARY_GNSS_ID].iloc[0]
    gpx, gpy, gpz = mogi_displacement(
        np.array([primary_row.x_m]), np.array([primary_row.y_m]), depth, cumulative_dV, x0, y0
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
        "insar_covariance_estimate": {
            "correlation_length_m": corr_length, "sill_m2": sill,
        },
        "gnss_covariance_estimate": {
            "common_mode_sigma_m": sigma_common, "white_sigma_m": sigma_white,
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
