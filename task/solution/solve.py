"""
solve.py -- reference solution for the deformation-microgravity-volcano task.

Pipeline:
1. Detect the shared inflation onset day from the GNSS time series (grid
   search over candidate days, minimizing joint SSE across stations).
2. Load the total (pre->post) GNSS displacement and microgravity change.
3. Apply the free-air correction to the gravity data using each benchmark's
   co-located GNSS vertical displacement (no circular iteration needed,
   since benchmarks are co-located with GNSS stations). Because the
   corrected gravity value is built directly from the (noisy) GNSS
   vertical displacement, its uncertainty is NOT independent of the GNSS
   uncertainty: the two datasets share the same disp_up noise realization
   at every co-located station. Treating them as independent (block-
   diagonal GNSS weights and gravity weights, as in an earlier version of
   this script) understates the true joint uncertainty and biases the
   weighting of the joint inversion.
4. Build ONE joint covariance matrix over the full stacked observation
   vector [gnss_east, gnss_north, gnss_up, corrected_gravity]:
     - each GNSS component gets an exponential spatial-covariance block
       (length scale estimated empirically from residuals);
     - gravity gets its own exponential spatial-covariance block from its
       own (raw, pre-correction) instrument uncertainty and spatial length;
     - PLUS the free-air correction propagates GNSS vertical-displacement
       uncertainty into the corrected gravity variance
       (+ FREE_AIR_GRADIENT^2 * Var(disp_up)), and creates an explicit
       cross-covariance block Cov(gravity_i, gnss_up_j) =
       -FREE_AIR_GRADIENT * Cov(disp_up_i, disp_up_j) between the two
       datasets. This is the shared-uncertainty term that must be
       included for the weighting (and the bootstrap) to be honest.
   An agent/analyst that skips this and whitens GNSS and gravity
   independently gets a different, overconfident answer.
5. Jointly invert GNSS + gravity for (x0, y0, depth, dV, dM) using GLS
   against the single full covariance matrix from step 4.
6. Flag gravity benchmarks whose residual (after the joint fit, and after
   accounting for the fitted spatial + propagated covariance) is not
   explained by the model -- these behave like unrelated local anomalies
   (e.g. a hydrological signal) rather than volcano-driven signal -- and
   refit excluding them, so the final mass/density is not biased by an
   unrelated confound riding along on one benchmark.
7. Compute density = dM / dV, classify the process.
8. Compute 95% CIs via parametric bootstrap (>=300 resamples), drawing
   correlated residual vectors from the SAME full joint covariance matrix
   used in the fit -- this preserves both the spatial correlation within
   each dataset and the cross-dataset correlation from step 4 in the
   resampling, and re-estimates the spatial correlation length within
   each replicate rather than reusing a single point estimate, so the
   reported interval also reflects not knowing the true correlation
   length exactly.

Runs end-to-end from the supplied inputs only. No pasted constants, no
reads from tests/data/.
"""

import numpy as np
import pandas as pd
import json
import os
from scipy.optimize import least_squares

DATA_DIR = os.environ.get("TASK_DATA_DIR", "/workspace/data")
OUTPUT_DIR = os.environ.get("TASK_OUTPUT_DIR", "/workspace/output")

POISSON_RATIO = 0.25
G_GRAV = 6.674e-11
FREE_AIR_GRADIENT_MGAL_M = -0.3086
RNG = np.random.default_rng(321)
N_BOOTSTRAP = 400  # >= 300 required; margin kept for replicates that fail to converge

DENSITY_RECHARGE_LOW = 1500.0
DENSITY_RECHARGE_HIGH = 3700.0

JITTER = 1e-10


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
def load_data():
    gnss_coords = pd.read_csv(os.path.join(DATA_DIR, "gnss_coords.csv"))
    gnss_ts = pd.read_csv(os.path.join(DATA_DIR, "gnss_timeseries.csv"))
    gravity = pd.read_csv(os.path.join(DATA_DIR, "microgravity_stations.csv"))
    return gnss_coords, gnss_ts, gravity


# ---------------------------------------------------------------------------
# 2. Onset detection (shared grid search, no seasonal term needed here)
# ---------------------------------------------------------------------------
def detect_onset_and_velocity(gnss_ts, station_ids):
    best_day, best_sse = None, np.inf
    fits = {}
    for day0 in range(20, int(gnss_ts["day"].max()) - 20, 5):
        total_sse = 0.0
        day_fits = {}
        for sid in station_ids:
            sub = gnss_ts[gnss_ts.station_id == sid].sort_values("day")
            t = sub["day"].values.astype(float)
            y = sub["disp_up_m"].values
            ramp = np.clip(t - day0, 0, None)
            X = np.column_stack([np.ones_like(t), ramp])
            coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            resid = y - X @ coef
            sse = np.sum(resid ** 2)
            total_sse += sse
            day_fits[sid] = {"coef": coef, "X": X, "sse": sse, "n": len(t)}
        if total_sse < best_sse:
            best_sse = total_sse
            best_day = day0
            fits = day_fits

    results = {}
    for sid, fit in fits.items():
        X, coef, sse, n = fit["X"], fit["coef"], fit["sse"], fit["n"]
        dof = max(n - X.shape[1], 1)
        sigma2 = sse / dof
        XtX_inv = np.linalg.pinv(X.T @ X)
        se = np.sqrt(sigma2 * XtX_inv[1, 1])
        results[sid] = {"velocity_m_per_day": coef[1], "se": se}
    return best_day, results


def compute_total_displacement(gnss_ts, station_ids, onset_day, baseline_window=40, final_window=40):
    """Derive each station's TOTAL inflation-period displacement (and its
    propagated uncertainty) directly from the noisy daily time series,
    using a plateau-to-plateau comparison (mean of the last `final_window`
    days minus mean of the first `baseline_window` days, i.e. before
    onset) rather than extrapolating a parametric ramp model. This is
    robust to the true signal saturating before the record ends, which a
    naive linear-ramp extrapolation would systematically overshoot.
    """
    rows = []
    for sid in station_ids:
        sub = gnss_ts[gnss_ts.station_id == sid].sort_values("day")
        t = sub["day"].values.astype(float)

        baseline_mask = t < baseline_window
        final_mask = t >= (t.max() - final_window)

        de = sub["disp_east_m"].values[final_mask].mean() - sub["disp_east_m"].values[baseline_mask].mean()
        dn = sub["disp_north_m"].values[final_mask].mean() - sub["disp_north_m"].values[baseline_mask].mean()
        du = sub["disp_up_m"].values[final_mask].mean() - sub["disp_up_m"].values[baseline_mask].mean()

        def window_se(values, mask):
            v = values[mask]
            return np.std(v) / np.sqrt(max(len(v), 1))

        se_de = np.sqrt(window_se(sub["disp_east_m"].values, final_mask) ** 2 +
                         window_se(sub["disp_east_m"].values, baseline_mask) ** 2)
        se_dn = np.sqrt(window_se(sub["disp_north_m"].values, final_mask) ** 2 +
                         window_se(sub["disp_north_m"].values, baseline_mask) ** 2)
        se_du = np.sqrt(window_se(sub["disp_up_m"].values, final_mask) ** 2 +
                         window_se(sub["disp_up_m"].values, baseline_mask) ** 2)

        sigma = float(np.max([se_de, se_dn, se_du]))
        rows.append({
            "station_id": sid,
            "disp_east_m": de, "disp_north_m": dn, "disp_up_m": du,
            "sigma_m": sigma,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3. Forward models
# ---------------------------------------------------------------------------
def mogi_forward(x0, y0, depth, dV, xy):
    x, y = xy[:, 0] - x0, xy[:, 1] - y0
    r = np.sqrt(x**2 + y**2)
    R = np.sqrt(r**2 + depth**2)
    C = (1 - POISSON_RATIO) * dV / np.pi
    uz = C * depth / R**3
    ur = C * r / R**3
    theta = np.arctan2(y, x)
    ux = ur * np.cos(theta)
    uy = ur * np.sin(theta)
    return ux, uy, uz


def gravity_forward(x0, y0, depth, dM, xy):
    x, y = xy[:, 0] - x0, xy[:, 1] - y0
    r2 = x**2 + y**2
    denom = (r2 + depth**2) ** 1.5
    dg_si = G_GRAV * dM * depth / denom
    return dg_si * 1e5  # mGal


# ---------------------------------------------------------------------------
# 4. Build observation vector, apply free-air correction
# ---------------------------------------------------------------------------
def build_observations(gnss_coords, gnss_total, gravity):
    merged = gnss_total.merge(gnss_coords, on="station_id")
    gnss_xy = merged[["x_m", "y_m"]].values
    station_order = merged["station_id"].tolist()
    gnss_obs = np.concatenate([
        merged["disp_east_m"].values,
        merged["disp_north_m"].values,
        merged["disp_up_m"].values,
    ])
    gnss_sigma_per_station = merged["sigma_m"].values

    grav_merged = gravity.merge(
        gnss_coords, left_on="co_located_gnss_station_id", right_on="station_id"
    )
    grav_merged = grav_merged.merge(
        gnss_total[["station_id", "disp_up_m"]], on="station_id"
    )
    grav_xy = grav_merged[["x_m", "y_m"]].values
    # free-air correction: dg_mass = dg_observed - FREE_AIR_GRADIENT * disp_up
    # dg_mass_obs therefore carries the GNSS disp_up noise, not just the
    # gravimeter's own instrument noise.
    dg_mass_obs = (
        grav_merged["gravity_change_mgal"].values
        - FREE_AIR_GRADIENT_MGAL_M * grav_merged["disp_up_m"].values
    )
    grav_sigma = grav_merged["sigma_mgal"].values
    grav_colocated_gnss_index = [station_order.index(sid) for sid in grav_merged["station_id"]]
    grav_benchmark_ids = grav_merged["benchmark_id"].tolist()

    return (gnss_xy, gnss_obs, gnss_sigma_per_station, station_order,
            grav_xy, dg_mass_obs, grav_sigma, grav_colocated_gnss_index, grav_benchmark_ids)


# ---------------------------------------------------------------------------
# 5. Joint covariance (GNSS spatial + gravity spatial + GNSS<->gravity
#    cross-covariance propagated through the free-air correction)
# ---------------------------------------------------------------------------
def exponential_cov(coords, sigma, length_scale):
    n = len(coords)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.linalg.norm(coords[i] - coords[j])
            cov[i, j] = sigma[i] * sigma[j] * np.exp(-d / length_scale)
    return cov


def estimate_correlation_length(coords, residuals, sigma, default=1000.0):
    """Empirical variogram-style estimate: fit exponential decay of
    normalized residual cross-products vs. inter-station distance.
    With few stations this is noisy, so we fall back to a sensible
    default when the fit is poorly conditioned.
    """
    n = len(coords)
    normalized = residuals / sigma
    dists, prods = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(coords[i] - coords[j])
            dists.append(d)
            prods.append(normalized[i] * normalized[j])
    dists, prods = np.array(dists), np.array(prods)
    positive = prods > 1e-6
    if positive.sum() < 3:
        return default
    try:
        log_prods = np.log(prods[positive])
        A = np.column_stack([np.ones_like(dists[positive]), -dists[positive]])
        coef, _, _, _ = np.linalg.lstsq(A, log_prods, rcond=None)
        L = 1.0 / coef[1] if coef[1] > 0 else default
        return float(np.clip(L, 200.0, 5000.0))
    except Exception:
        return default


def build_joint_covariance(gnss_xy, gnss_sigma_per_station, L_gnss,
                            grav_xy, grav_sigma, L_grav, grav_colocated_gnss_index):
    """Full covariance over the stacked vector
    [gnss_east, gnss_north, gnss_up, corrected_gravity], including the
    cross-covariance the free-air correction creates between corrected
    gravity and GNSS vertical displacement at co-located stations.
    """
    n_gnss = len(gnss_xy)
    n_grav = len(grav_xy)

    cov_uu = exponential_cov(gnss_xy, gnss_sigma_per_station, L_gnss)  # shared by E, N, U blocks
    cov_grav_instrument = exponential_cov(grav_xy, grav_sigma, L_grav)

    idx = np.array(grav_colocated_gnss_index)
    cov_uu_colocated = cov_uu[np.ix_(idx, idx)]
    # Var(dg_corrected) = Var(raw gravity noise) + a^2 * Var(disp_up), a = FREE_AIR_GRADIENT
    cov_grav = cov_grav_instrument + (FREE_AIR_GRADIENT_MGAL_M ** 2) * cov_uu_colocated
    # Cov(dg_corrected_i, disp_up_j) = -a * Cov(disp_up_colocated_i, disp_up_j)
    cov_grav_gnssup_cross = -FREE_AIR_GRADIENT_MGAL_M * cov_uu[idx, :]  # (n_grav, n_gnss)

    n_total = 3 * n_gnss + n_grav
    cov = np.zeros((n_total, n_total))
    e0, n0, u0, g0 = 0, n_gnss, 2 * n_gnss, 3 * n_gnss
    cov[e0:n0, e0:n0] = cov_uu
    cov[n0:u0, n0:u0] = cov_uu
    cov[u0:g0, u0:g0] = cov_uu
    cov[g0:, g0:] = cov_grav
    cov[g0:, u0:g0] = cov_grav_gnssup_cross
    cov[u0:g0, g0:] = cov_grav_gnssup_cross.T
    cov += JITTER * np.eye(n_total)
    return cov


# ---------------------------------------------------------------------------
# 6. Joint residual function / fit
# ---------------------------------------------------------------------------
def joint_residuals(params, gnss_xy, gnss_obs, grav_xy, grav_obs, W_chol_inv):
    x0, y0, depth, dV, dM = params
    ux, uy, uz = mogi_forward(x0, y0, depth, dV, gnss_xy)
    gnss_pred = np.concatenate([ux, uy, uz])
    grav_pred = gravity_forward(x0, y0, depth, dM, grav_xy)

    res = np.concatenate([gnss_pred - gnss_obs, grav_pred - grav_obs])
    return W_chol_inv @ res


def fit_joint(gnss_xy, gnss_obs, grav_xy, grav_obs, W_chol_inv, p0):
    bounds = (
        [-3000, -3000, 200, 1e4, 1e6],
        [3000, 3000, 15000, 1e8, 1e12],
    )
    best_res, best_cost = None, np.inf
    for _ in range(15):
        jitter = p0 * (1 + RNG.uniform(-0.3, 0.3, size=len(p0)))
        jitter = np.clip(jitter, bounds[0], bounds[1])
        try:
            res = least_squares(
                joint_residuals, jitter,
                args=(gnss_xy, gnss_obs, grav_xy, grav_obs, W_chol_inv),
                bounds=bounds, max_nfev=5000,
            )
            cost = np.sum(res.fun ** 2)
            if cost < best_cost:
                best_cost = cost
                best_res = res
        except Exception:
            continue
    return best_res


def diagonal_weighting(gnss_sigma_per_station, grav_sigma, grav_colocated_gnss_index):
    """Pass-1 (pre-spatial-covariance) diagonal weights. Even the diagonal
    approximation must include the free-air-propagated GNSS variance in
    the gravity term, otherwise pass 1 already understates gravity
    uncertainty before the spatial-covariance refinement even starts.
    """
    n_gnss = len(gnss_sigma_per_station)
    gnss_var_flat = np.tile(gnss_sigma_per_station ** 2, 3)
    gnss_up_var_colocated = gnss_sigma_per_station[np.array(grav_colocated_gnss_index)] ** 2
    grav_var = grav_sigma ** 2 + (FREE_AIR_GRADIENT_MGAL_M ** 2) * gnss_up_var_colocated
    full_var = np.concatenate([gnss_var_flat, grav_var])
    return np.diag(1.0 / np.sqrt(full_var))


# ---------------------------------------------------------------------------
# 7. Flagged-benchmark QC
# ---------------------------------------------------------------------------
def detect_flagged_gravity_benchmarks(grav_xy, grav_obs, x0, y0, depth, dM, cov_grav, benchmark_ids):
    """Standardize each benchmark's post-fit residual by its own marginal
    std from the fitted (spatial + propagated) covariance, and flag any
    benchmark the model still cannot explain -- i.e. behaves like an
    unrelated local anomaly (e.g. a hydrological signal) rather than
    volcano-driven signal.
    """
    flagged = []
    full_pred = gravity_forward(x0, y0, depth, dM, grav_xy)
    resid = grav_obs - full_pred
    marginal_std = np.sqrt(np.diag(cov_grav))
    z_scores = resid / marginal_std
    threshold = 2.0
    for i, bid in enumerate(benchmark_ids):
        if abs(z_scores[i]) > threshold:
            flagged.append(bid)
    return flagged


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    gnss_coords, gnss_ts, gravity = load_data()
    station_ids = gnss_coords["station_id"].tolist()

    onset_day, velocities = detect_onset_and_velocity(gnss_ts, station_ids)
    gnss_total = compute_total_displacement(gnss_ts, station_ids, onset_day)

    (gnss_xy, gnss_obs, gnss_sigma_per_station, station_order,
     grav_xy, grav_obs, grav_sigma, grav_colocated_gnss_index, benchmark_ids) = build_observations(
        gnss_coords, gnss_total, gravity
    )
    n_gnss = len(gnss_xy)

    # --- Pass 1: diagonal weights (still including the free-air-propagated
    # GNSS variance in the gravity diagonal term) ---
    W_diag = diagonal_weighting(gnss_sigma_per_station, grav_sigma, grav_colocated_gnss_index)
    p0 = np.array([np.mean(gnss_xy[:, 0]), np.mean(gnss_xy[:, 1]), 3000.0, 5e6, 1e10])
    pass1 = fit_joint(gnss_xy, gnss_obs, grav_xy, grav_obs, W_diag, p0)

    # --- Estimate spatial correlation length from pass-1 residuals ---
    x0, y0, depth, dV, dM = pass1.x
    ux, uy, uz = mogi_forward(x0, y0, depth, dV, gnss_xy)
    gnss_resid_up = uz - gnss_obs[2 * n_gnss:3 * n_gnss]
    L_gnss = estimate_correlation_length(gnss_xy, gnss_resid_up, gnss_sigma_per_station)

    grav_pred = gravity_forward(x0, y0, depth, dM, grav_xy)
    grav_resid = grav_pred - grav_obs
    L_grav = estimate_correlation_length(grav_xy, grav_resid, grav_sigma)

    # --- Pass 2: full joint covariance (spatial + cross-dataset), GLS refit ---
    cov_full = build_joint_covariance(gnss_xy, gnss_sigma_per_station, L_gnss,
                                       grav_xy, grav_sigma, L_grav, grav_colocated_gnss_index)
    L_full_chol = np.linalg.cholesky(cov_full)
    W_chol_inv = np.linalg.inv(L_full_chol)

    pass2 = fit_joint(gnss_xy, gnss_obs, grav_xy, grav_obs, W_chol_inv, pass1.x)
    x0, y0, depth, dV, dM = pass2.x
    g0 = 3 * n_gnss
    cov_grav_block = cov_full[g0:, g0:]

    # --- QC: detect gravity benchmarks inconsistent with the fitted
    # (spatial + propagated) covariance structure, and refit excluding
    # them so the final answer isn't biased by an unrelated confound. ---
    flagged_benchmarks = detect_flagged_gravity_benchmarks(
        grav_xy, grav_obs, x0, y0, depth, dM, cov_grav_block, benchmark_ids
    )
    if flagged_benchmarks:
        keep_mask = np.array([bid not in flagged_benchmarks for bid in benchmark_ids])
        grav_xy = grav_xy[keep_mask]
        grav_obs = grav_obs[keep_mask]
        grav_sigma = grav_sigma[keep_mask]
        grav_colocated_gnss_index = [idx for idx, keep in zip(grav_colocated_gnss_index, keep_mask) if keep]
        benchmark_ids = [bid for bid, keep in zip(benchmark_ids, keep_mask) if keep]

        cov_full = build_joint_covariance(gnss_xy, gnss_sigma_per_station, L_gnss,
                                           grav_xy, grav_sigma, L_grav, grav_colocated_gnss_index)
        L_full_chol = np.linalg.cholesky(cov_full)
        W_chol_inv = np.linalg.inv(L_full_chol)

        pass3 = fit_joint(gnss_xy, gnss_obs, grav_xy, grav_obs, W_chol_inv, pass2.x)
        x0, y0, depth, dV, dM = pass3.x
        g0 = 3 * n_gnss
        cov_grav_block = cov_full[g0:, g0:]

    density = dM / dV

    if DENSITY_RECHARGE_LOW <= density <= DENSITY_RECHARGE_HIGH:
        classification = "magmatic_recharge"
    elif 0 <= density < DENSITY_RECHARGE_LOW:
        classification = "non_magmatic_source"
    elif density < 0:
        classification = "magmatic_mass_loss"
    else:
        classification = "unrealistic"

    # --- Bootstrap: parametric, drawing correlated residuals from the
    # SAME full joint covariance (preserving both within-dataset spatial
    # correlation and the GNSS<->gravity cross-correlation), and
    # re-estimating the spatial correlation length within each replicate
    # rather than reusing the single point estimate from the main fit. ---
    boot_params = []
    base_gnss_pred = np.concatenate(mogi_forward(x0, y0, depth, dV, gnss_xy))
    base_grav_pred = gravity_forward(x0, y0, depth, dM, grav_xy)
    n_boot_success = 0
    for _ in range(N_BOOTSTRAP):
        synth = np.concatenate([base_gnss_pred, base_grav_pred]) + L_full_chol @ RNG.standard_normal(L_full_chol.shape[0])
        synth_gnss = synth[:3 * n_gnss]
        synth_grav = synth[3 * n_gnss:]
        try:
            synth_gnss_resid_up = synth_gnss[2 * n_gnss:3 * n_gnss] - uz
            L_gnss_b = estimate_correlation_length(gnss_xy, synth_gnss_resid_up, gnss_sigma_per_station)
            synth_grav_resid = synth_grav - base_grav_pred
            L_grav_b = estimate_correlation_length(grav_xy, synth_grav_resid, grav_sigma)

            cov_full_b = build_joint_covariance(gnss_xy, gnss_sigma_per_station, L_gnss_b,
                                                 grav_xy, grav_sigma, L_grav_b, grav_colocated_gnss_index)
            W_b = np.linalg.inv(np.linalg.cholesky(cov_full_b))

            res_b = fit_joint(gnss_xy, synth_gnss, grav_xy, synth_grav, W_b, pass2.x)
            boot_params.append(res_b.x)
            n_boot_success += 1
        except Exception:
            continue
    boot_params = np.array(boot_params)
    lo = np.percentile(boot_params, 2.5, axis=0)
    hi = np.percentile(boot_params, 97.5, axis=0)
    boot_density = boot_params[:, 4] / boot_params[:, 3]
    density_lo, density_hi = np.percentile(boot_density, [2.5, 97.5])

    result = {
        "onset_day_estimate": int(onset_day),
        "station_velocities": [
            {"station_id": sid, "velocity_m_per_day": v["velocity_m_per_day"], "uncertainty_95": 1.96 * v["se"]}
            for sid, v in velocities.items()
        ],
        "x0_m": x0, "x0_uncertainty_95": [lo[0], hi[0]],
        "y0_m": y0, "y0_uncertainty_95": [lo[1], hi[1]],
        "depth_m": depth, "depth_uncertainty_95": [lo[2], hi[2]],
        "volume_change_m3": dV, "volume_change_uncertainty_95": [lo[3], hi[3]],
        "mass_change_kg": dM, "mass_change_uncertainty_95": [lo[4], hi[4]],
        "density_kgm3": density, "density_uncertainty_95": [density_lo, density_hi],
        "process_classification": classification,
        "flagged_gravity_benchmarks": flagged_benchmarks,
        "estimated_spatial_correlation_length_gnss_m": L_gnss,
        "estimated_spatial_correlation_length_gravity_m": L_grav,
        "free_air_gradient_mgal_per_m": FREE_AIR_GRADIENT_MGAL_M,
        "poisson_ratio_assumed": POISSON_RATIO,
        "n_bootstrap_resamples": int(n_boot_success),
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "result.txt"), "w") as f:
        json.dump(result, f, indent=2, default=float)

    print(json.dumps(result, indent=2, default=float))


if __name__ == "__main__":
    main()
