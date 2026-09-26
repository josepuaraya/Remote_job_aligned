"""
generate_data.py -- AUTHOR'S PROVENANCE SCRIPT for the
insar-gnss-volcano-timeseries task.

Not executed by solve.py or task/tests/test_outputs.py at runtime. This is
run once by the task author to produce the checked-in data files under
task/environment/data/ (public, agent-visible) and task/tests/data/
(private fixture + answer key, never seen by the agent). Per the scaffold's
authoring-boundary rule, the generated files themselves are what gets
checked in -- this script does not run during grading.

SCENARIO
--------
A volcano observatory receives two independent geodetic products for the
same edifice over an ~11-month episode:

  1. Two InSAR tracks (ascending + descending, real Sentinel-1-like
     geometry) processed into small-baseline (SBAS) interferogram stacks.
     Each stack has genuine redundancy (consecutive AND skip-2 pairs,
     i.e. closed triplets), atmospheric noise of varying severity per
     interferogram, exactly one phase-unwrapping error per track (a
     discrete, spatially-coherent offset equal to an integer number of
     phase-ambiguity cycles), a per-interferogram random coherence mask
     (missing pixels), and -- for the descending track only -- a
     persistent no-data zone near the summit (a permanent layover/
     foreshortening blind spot for that look geometry).
  2. A 10-station continuous GNSS network with heterogeneous quality:
     three stations sit close to the source (one of them quietly
     drifting from an unrelated local cause, not flagged by its own
     formal uncertainty; two more sit near the source but are
     genuinely noisy -- large, honestly-reported uncertainty, no
     hidden bias -- so a station that reports itself as imprecise
     still has to be used, just down-weighted, not thrown away), two
     mid-field stations have incomplete time coverage (one deployed
     late, one that failed early), and three far-field stations sample
     ground with no meaningful volcanic signal at all (a built-in
     control region).

The true source is a single McTigue (1987) finite spherical source (a
second-order refinement of the Mogi point source that accounts for the
source's own finite radius -- non-negligible here, at radius/depth = 0.4)
whose volume-change rate is NOT constant: it inflates at one linear rate,
then -- partway through the record, at an undisclosed day -- switches to
a second, faster linear rate. This is the realistic, consequential signal
the workflow must recover: where the source is, has its inflation rate
changed, by how much, and when.

The generator's true parameters (source location/depth, both rates, the
rate-change day, which stations/interferograms are defective, ramp
coefficients) are never written to task/environment/data/ -- only to
task/tests/data/answer_key.json.
"""

from __future__ import annotations

import csv
import json
import os

import numpy as np

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
GENERATOR_SEED = int(os.environ.get("GENERATOR_SEED", "13"))
RNG = np.random.default_rng(GENERATOR_SEED)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
PUBLIC_DATA_DIR = os.environ.get(
    "GENERATOR_PUBLIC_DIR", os.path.join(REPO_ROOT, "task", "environment", "data")
)
PRIVATE_DATA_DIR = os.environ.get(
    "GENERATOR_PRIVATE_DIR", os.path.join(REPO_ROOT, "task", "tests", "data")
)

# ---------------------------------------------------------------------------
# TRUE (hidden) source parameters
# ---------------------------------------------------------------------------
TRUE_X0_M = 150.0
TRUE_Y0_M = -80.0
TRUE_DEPTH_M = 2500.0
TRUE_RADIUS_M = 1000.0  # finite spherical source radius (McTigue); a/d = 0.4
POISSON_RATIO = 0.25

N_DAYS = 331  # days 0..330 inclusive
TRUE_RATE_CHANGE_DAY = 165.0
TRUE_RATE1_DV_M3_PER_DAY = 11000.0   # slower inflation, day 0 -> rate-change day
TRUE_RATE2_DV_M3_PER_DAY = 24200.0   # faster inflation, rate-change day -> day 330

DOMAIN_HALF_WIDTH_M = 15000.0

# Real satellite geometry (ascending/descending, values from a real
# Sentinel-1 acquisition pair, reused here for realism).
ASC_INCIDENCE_DEG = 39.1054
ASC_HEADING_DEG = -11.876832
DESC_INCIDENCE_DEG = 39.1840
DESC_HEADING_DEG = -168.27934

AMBIGUITY_QUANTUM_M = 0.028  # ~ half C-band wavelength; one unwrapping cycle


def cumulative_dV(day):
    day = np.asarray(day, dtype=float)
    before = TRUE_RATE1_DV_M3_PER_DAY * day
    after = (
        TRUE_RATE1_DV_M3_PER_DAY * TRUE_RATE_CHANGE_DAY
        + TRUE_RATE2_DV_M3_PER_DAY * (day - TRUE_RATE_CHANGE_DAY)
    )
    return np.where(day <= TRUE_RATE_CHANGE_DAY, before, after)


def mctigue_displacement(x, y, depth, delta_v_m3, radius=TRUE_RADIUS_M,
                          x0=TRUE_X0_M, y0=TRUE_Y0_M, nu=POISSON_RATIO):
    """McTigue (1987) finite spherical source, 2nd-order series in a/depth.

    Formula verified against the open-source VSM toolkit (Trasatti,
    github.com/EliTras/VSM, VSM_forward.py:mctigue), which implements
    McTigue (1987) directly. As radius -> 0 this reduces exactly to the
    Mogi point-source result with dV = pi * radius**3 * (dP/mu), the
    standard pressure/volume relation for a pressurized sphere -- a
    useful internal consistency check on the formula itself.
    """
    dx, dy = x - x0, y - y0
    rho = np.sqrt(dx ** 2 + dy ** 2)
    R = np.sqrt(rho ** 2 + depth ** 2)
    dP_mu = delta_v_m3 / (np.pi * radius ** 3)
    a_d = radius / depth
    f1 = (depth ** 3) / (R ** 3)
    c1 = a_d ** 3 / (7.0 - 5.0 * nu)
    uzbar = a_d ** 3 * (1 - nu) * f1 * (1 - c1 * (0.5 * (1 + nu) - 3.75 * (2 - nu) * f1))
    u_z = uzbar * dP_mu * depth
    u_r = u_z * (rho / depth)
    theta = np.arctan2(dy, dx)
    u_x = u_r * np.cos(theta)
    u_y = u_r * np.sin(theta)
    return u_x, u_y, u_z


def los_projection_full(u_e, u_n, u_u, incidence_deg, heading_deg):
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return (
        np.sin(theta) * np.cos(alpha) * u_e
        - np.sin(theta) * np.sin(alpha) * u_n
        - np.cos(theta) * u_u
    )


def build_spatial_cholesky(x, y, sigma, length_scale):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx ** 2 + dy ** 2)
    cov = sigma ** 2 * np.exp(-dist / length_scale)
    cov += 1e-9 * (sigma ** 2) * np.eye(len(x))
    return np.linalg.cholesky(cov)


# ---------------------------------------------------------------------------
# Per-track point clouds (deliberately DIFFERENT samples -- ascending and
# descending are independent SAR products with independent pixel grids;
# combining them for vertical/east-west decomposition genuinely requires
# spatial matching/interpolation, not a shared index).
#
# Each cloud mixes a sparse domain-wide background with a denser inner
# cluster near the source -- standard practice for a real InSAR product
# (quadtree/adaptive downsampling naturally keeps more points where the
# phase gradient, i.e. the deformation signal itself, is strong), and
# necessary here: the true Mogi field's curvature near the source is far
# too sharp for a uniform few-hundred-point domain-wide sample to resolve.
# ---------------------------------------------------------------------------
N_POINTS_ASC_OUTER, N_POINTS_ASC_INNER = 460, 220
N_POINTS_DESC_OUTER, N_POINTS_DESC_INNER = 430, 200
INNER_CLUSTER_RADIUS_M = 1300.0


def sample_point_cloud(n_outer, n_inner, inner_radius):
    x_outer = RNG.uniform(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, n_outer)
    y_outer = RNG.uniform(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, n_outer)
    r = inner_radius * np.sqrt(RNG.uniform(0.0, 1.0, n_inner))
    theta = RNG.uniform(0.0, 2 * np.pi, n_inner)
    x_inner = TRUE_X0_M + r * np.cos(theta)
    y_inner = TRUE_Y0_M + r * np.sin(theta)
    return np.concatenate([x_outer, x_inner]), np.concatenate([y_outer, y_inner])


asc_x, asc_y = sample_point_cloud(N_POINTS_ASC_OUTER, N_POINTS_ASC_INNER, INNER_CLUSTER_RADIUS_M)
desc_x, desc_y = sample_point_cloud(N_POINTS_DESC_OUTER, N_POINTS_DESC_INNER, INNER_CLUSTER_RADIUS_M)
N_POINTS_ASC = len(asc_x)
N_POINTS_DESC = len(desc_x)

ELEV_PEAK_X_M, ELEV_PEAK_Y_M = TRUE_X0_M, TRUE_Y0_M
PEAK_ELEV_M, DECAY_LENGTH_M, BASE_ELEV_M = 1400.0, 1800.0, 250.0


def elevation_field(x, y):
    r = np.sqrt((x - ELEV_PEAK_X_M) ** 2 + (y - ELEV_PEAK_Y_M) ** 2)
    return PEAK_ELEV_M * np.exp(-r / DECAY_LENGTH_M) + BASE_ELEV_M


asc_elev = elevation_field(asc_x, asc_y)
desc_elev = elevation_field(desc_x, desc_y)

# Persistent no-data zone: descending-only layover/foreshortening blind
# spot near (but not exactly on) the summit. No GNSS station falls inside
# it, but it removes descending coverage from part of the near-field.
PERSISTENT_HOLE_CENTER = (700.0, 500.0)
PERSISTENT_HOLE_RADIUS_M = 300.0
_desc_hole_mask = (
    np.sqrt((desc_x - PERSISTENT_HOLE_CENTER[0]) ** 2 + (desc_y - PERSISTENT_HOLE_CENTER[1]) ** 2)
    < PERSISTENT_HOLE_RADIUS_M
)

ATMOS_CORR_LENGTH_M = 500.0
_asc_chol_unit = build_spatial_cholesky(asc_x, asc_y, 1.0, ATMOS_CORR_LENGTH_M)
_desc_chol_unit = build_spatial_cholesky(desc_x, desc_y, 1.0, ATMOS_CORR_LENGTH_M)

# Irregular, spatially-coherent unwrap-defect masks (smooth random field
# thresholded at its median -- an arbitrarily-shaped connected-ish region,
# not a clean half-plane split, matching how a real unwrapping error
# follows whatever patch of ground lost phase continuity).
UNWRAP_MASK_CORR_LENGTH_M = 5000.0
_asc_unwrap_field = build_spatial_cholesky(asc_x, asc_y, 1.0, UNWRAP_MASK_CORR_LENGTH_M) @ RNG.standard_normal(N_POINTS_ASC)
_asc_unwrap_mask = _asc_unwrap_field > np.median(_asc_unwrap_field)
_desc_unwrap_field = build_spatial_cholesky(desc_x, desc_y, 1.0, UNWRAP_MASK_CORR_LENGTH_M) @ RNG.standard_normal(N_POINTS_DESC)
_desc_unwrap_mask = _desc_unwrap_field > np.median(_desc_unwrap_field)

# ---------------------------------------------------------------------------
# SBAS acquisition networks (independent per track). Ascending carries a
# genuine ~70-day data gap; descending is evenly sampled. Both include
# BOTH consecutive AND skip-2 pairs, so every interior epoch sits inside
# at least one closed triplet (i, i+1, i+2) -- the redundancy a real SBAS
# QC step (triplet/loop closure) needs to isolate an unwrapping error from
# ordinary atmospheric noise.
# ---------------------------------------------------------------------------
ASC_EPOCH_DAYS = [0, 24, 48, 72, 96, 120, 190, 214, 238, 262, 286, 320]
DESC_EPOCH_DAYS = [10, 40, 70, 100, 130, 160, 190, 220, 250, 280, 310]

# Unwrap-defect edges, deliberately placed (not left to chance) on
# ordinary, single-purpose windows: ascending's is pre-rate-change and
# does not span the acquisition gap; descending's is post-rate-change.
# Each is chosen so a skip-2 (triplet-closing) edge exists alongside it.
ASC_UNWRAP_EDGE = (1, 2)     # day 24 -> 48
DESC_UNWRAP_EDGE = (6, 7)    # day 190 -> 220

# Noisy vs. clean interferograms (varying atmospheric severity); indices
# into the edge list built below, chosen to include both tracks and not
# coincide with the unwrap-defect edges' consecutive pair (kept clean at
# the "raw" atmospheric level so the defect itself is the only anomaly
# riding on it).
ASC_NOISY_CONSECUTIVE_IDX = {0, 3, 6, 9}
DESC_NOISY_CONSECUTIVE_IDX = {1, 4, 7}

CLEAN_SIGMA_M = 0.0018
NOISY_SIGMA_M = 0.006
DECORRELATION_RETENTION_RANGE = (0.50, 0.75)


def draw_decorrelation_mask(n_points, exclude_mask=None):
    eligible = np.arange(n_points)
    if exclude_mask is not None:
        eligible = eligible[~exclude_mask]
    retention_frac = RNG.uniform(*DECORRELATION_RETENTION_RANGE)
    n_keep = int(round(len(eligible) * retention_frac))
    return np.sort(RNG.choice(eligible, size=n_keep, replace=False))


def build_edges(epoch_days):
    """Consecutive + skip-2 pairs -> list of (i, j, day_i, day_j)."""
    edges = []
    k = len(epoch_days)
    for i in range(k - 1):
        edges.append((i, i + 1, epoch_days[i], epoch_days[i + 1]))
    for i in range(k - 2):
        edges.append((i, i + 2, epoch_days[i], epoch_days[i + 2]))
    return edges


def make_ifg_los(x, y, day0, day1, incidence_deg, heading_deg):
    dV0, dV1 = cumulative_dV(day0), cumulative_dV(day1)
    ux0, uy0, uz0 = mctigue_displacement(x, y, TRUE_DEPTH_M, dV0)
    ux1, uy1, uz1 = mctigue_displacement(x, y, TRUE_DEPTH_M, dV1)
    return los_projection_full(ux1 - ux0, uy1 - uy0, uz1 - uz0, incidence_deg, heading_deg)


def build_track(track_name, x, y, epoch_days, unwrap_edge, noisy_idx, chol_unit,
                 unwrap_mask, incidence_deg, heading_deg, hole_mask=None):
    edges = build_edges(epoch_days)
    n_consecutive = len(epoch_days) - 1
    records = []
    meta = []
    answer_edges = []
    for e_idx, (i, j, day0, day1) in enumerate(edges):
        is_consecutive = e_idx < n_consecutive
        ifg_id = f"{track_name}-{e_idx + 1:02d}"
        los_true = make_ifg_los(x, y, day0, day1, incidence_deg, heading_deg)

        sigma = NOISY_SIGMA_M if (is_consecutive and e_idx in noisy_idx) else CLEAN_SIGMA_M
        noise = sigma * (chol_unit @ RNG.standard_normal(len(x)))

        has_defect = (i, j) == unwrap_edge
        if has_defect:
            jump_sign = RNG.choice([-1.0, 1.0])
            noise = noise + np.where(unwrap_mask, AMBIGUITY_QUANTUM_M * jump_sign, 0.0)

        los_observed = los_true + noise

        keep_idx = draw_decorrelation_mask(len(x), exclude_mask=hole_mask)
        for p in keep_idx:
            records.append({
                "interferogram_id": ifg_id,
                "point_id": int(p),
                "x_m": round(float(x[p]), 2),
                "y_m": round(float(y[p]), 2),
                "elevation_m": round(float(elevation_field(x[p:p+1], y[p:p+1])[0]), 2),
                "los_displacement_m": round(float(los_observed[p]), 6),
            })
        orbit_name = "ascending" if track_name == "ASC" else "descending"
        meta.append({
            "interferogram_id": ifg_id, "orbit": orbit_name,
            "day_start": int(day0), "day_end": int(day1),
            "incidence_deg": incidence_deg, "heading_deg": heading_deg,
        })
        answer_edges.append({
            "interferogram_id": ifg_id, "epoch_i": i, "epoch_j": j,
            "day_start": int(day0), "day_end": int(day1),
            "has_unwrap_defect": bool(has_defect),
            "noise_sigma_m": sigma,
        })
    return records, meta, answer_edges


records_asc, meta_asc, answer_asc = build_track(
    "ASC", asc_x, asc_y, ASC_EPOCH_DAYS, ASC_UNWRAP_EDGE, ASC_NOISY_CONSECUTIVE_IDX,
    _asc_chol_unit, _asc_unwrap_mask, ASC_INCIDENCE_DEG, ASC_HEADING_DEG, hole_mask=None,
)
records_desc, meta_desc, answer_desc = build_track(
    "DESC", desc_x, desc_y, DESC_EPOCH_DAYS, DESC_UNWRAP_EDGE, DESC_NOISY_CONSECUTIVE_IDX,
    _desc_chol_unit, _desc_unwrap_mask, DESC_INCIDENCE_DEG, DESC_HEADING_DEG, hole_mask=_desc_hole_mask,
)

all_records = records_asc + records_desc
all_meta = meta_asc + meta_desc

# ---------------------------------------------------------------------------
# GNSS network: 8 stations, heterogeneous distance from the source,
# quality, and time coverage.
# ---------------------------------------------------------------------------
GNSS_STATIONS = [
    # id,        x,      y,     day_start, day_end, white_en, white_up, local_drift_up_m_per_day
    ("GNSS-01",   300.0,  -150.0,   0, 330, 0.0025, 0.006, 0.0),
    ("GNSS-02",  -250.0,   400.0,   0, 330, 0.0025, 0.006, 0.0),
    ("GNSS-03",   100.0,   350.0,   0, 330, 0.0040, 0.009, -0.00050),  # bad: quiet local drift
    ("GNSS-04",  1400.0,  -900.0, 140, 330, 0.0025, 0.006, 0.0),        # late start
    ("GNSS-05", -1500.0,  -700.0,   0, 250, 0.0025, 0.006, 0.0),        # early stop
    ("GNSS-06", 11000.0,  5000.0,   0, 330, 0.0025, 0.006, 0.0),        # far control
    ("GNSS-07",-12000.0, -4000.0,   0, 330, 0.0025, 0.006, 0.0),        # far control (primary)
    ("GNSS-08", 10000.0, -8000.0,   0, 330, 0.0050, 0.011, 0.0),        # far control, noisier
    ("GNSS-09",  -400.0,   700.0,   0, 330, 0.0120, 0.0250, 0.0),       # near-field, genuinely noisy (unbiased)
    ("GNSS-10",   600.0,  -500.0,   0, 330, 0.0120, 0.0250, 0.0),       # near-field, genuinely noisy (unbiased)
]

GNSS_COMMON_MODE_SIGMA_EN_M = 0.0018
GNSS_COMMON_MODE_SIGMA_UP_M = 0.003

common_mode = {
    day: GNSS_COMMON_MODE_SIGMA_EN_M * RNG.standard_normal(2)
    for day in range(N_DAYS)
}
common_mode_up = {day: GNSS_COMMON_MODE_SIGMA_UP_M * RNG.standard_normal() for day in range(N_DAYS)}

gnss_records = []
for sid, sx, sy, d0, d1, sig_en, sig_up, drift_up in GNSS_STATIONS:
    for day in range(d0, d1 + 1):
        dV = cumulative_dV(day)
        ux, uy, uz = mctigue_displacement(np.array([sx]), np.array([sy]), TRUE_DEPTH_M, dV)
        wn = RNG.standard_normal(3)
        cm_e, cm_n = common_mode[day]
        cm_u = common_mode_up[day]
        gnss_records.append({
            "station_id": sid, "day": day,
            "east_disp_m": round(float(ux[0]) + cm_e + sig_en * wn[0], 6),
            "north_disp_m": round(float(uy[0]) + cm_n + sig_en * wn[1], 6),
            "vertical_disp_m": round(float(uz[0]) + drift_up * day + cm_u + sig_up * wn[2], 6),
            "east_sigma_m": sig_en, "north_sigma_m": sig_en, "vertical_sigma_m": sig_up,
        })

gnss_stations_rows = [{"station_id": sid, "x_m": sx, "y_m": sy} for sid, sx, sy, *_ in GNSS_STATIONS]

# ---------------------------------------------------------------------------
# Answer key (private only)
# ---------------------------------------------------------------------------
primary_x, primary_y = 300.0, -150.0  # GNSS-01
control_x, control_y = -12000.0, -4000.0  # GNSS-07

def true_cumulative_vertical(x, y, day_start, day_end):
    dV0, dV1 = cumulative_dV(day_start), cumulative_dV(day_end)
    _, _, uz0 = mctigue_displacement(np.array([x]), np.array([y]), TRUE_DEPTH_M, dV0)
    _, _, uz1 = mctigue_displacement(np.array([x]), np.array([y]), TRUE_DEPTH_M, dV1)
    return float(uz1[0] - uz0[0])


true_rate_before_primary = mctigue_displacement(
    np.array([primary_x]), np.array([primary_y]), TRUE_DEPTH_M, np.array([TRUE_RATE1_DV_M3_PER_DAY])
)[2][0]
true_rate_after_primary = mctigue_displacement(
    np.array([primary_x]), np.array([primary_y]), TRUE_DEPTH_M, np.array([TRUE_RATE2_DV_M3_PER_DAY])
)[2][0]

answer_key = {
    "generator_seed": GENERATOR_SEED,
    "true_x0_m": TRUE_X0_M,
    "true_y0_m": TRUE_Y0_M,
    "true_depth_m": TRUE_DEPTH_M,
    "true_radius_m": TRUE_RADIUS_M,
    "poisson_ratio": POISSON_RATIO,
    "n_days": N_DAYS,
    "true_rate_change_day": TRUE_RATE_CHANGE_DAY,
    "true_rate1_dv_m3_per_day": TRUE_RATE1_DV_M3_PER_DAY,
    "true_rate2_dv_m3_per_day": TRUE_RATE2_DV_M3_PER_DAY,
    "ambiguity_quantum_m": AMBIGUITY_QUANTUM_M,
    "primary_station_id": "GNSS-01",
    "control_station_id": "GNSS-07",
    "true_vertical_rate_before_m_per_day_primary": float(true_rate_before_primary),
    "true_vertical_rate_after_m_per_day_primary": float(true_rate_after_primary),
    "true_cumulative_vertical_displacement_m_primary": true_cumulative_vertical(primary_x, primary_y, 0, N_DAYS - 1),
    "true_cumulative_vertical_displacement_m_control": true_cumulative_vertical(control_x, control_y, 0, N_DAYS - 1),
    "bad_gnss_station_id": "GNSS-03",
    "bad_gnss_local_drift_m_per_day": -0.00050,
    "unwrap_defect_interferograms": [
        rec["interferogram_id"] for rec in (answer_asc + answer_desc) if rec["has_unwrap_defect"]
    ],
    "all_interferogram_ids": [rec["interferogram_id"] for rec in (answer_asc + answer_desc)],
    "all_gnss_station_ids": [row["station_id"] for row in gnss_stations_rows],
    "persistent_hole_track": "descending",
    "persistent_hole_center_m": list(PERSISTENT_HOLE_CENTER),
    "persistent_hole_radius_m": PERSISTENT_HOLE_RADIUS_M,
    "generator_notes": (
        "Synthetic scenario for reviewer traceability. Never disclosed to the "
        "agent. See generate_data.py for the full generative model."
    ),
}

# ---------------------------------------------------------------------------
# Write files
# ---------------------------------------------------------------------------
os.makedirs(PUBLIC_DATA_DIR, exist_ok=True)
os.makedirs(PRIVATE_DATA_DIR, exist_ok=True)


def write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


ifg_fields = ["interferogram_id", "point_id", "x_m", "y_m", "elevation_m", "los_displacement_m"]
meta_fields = ["interferogram_id", "orbit", "day_start", "day_end", "incidence_deg", "heading_deg"]
gnss_ts_fields = [
    "station_id", "day", "east_disp_m", "north_disp_m", "vertical_disp_m",
    "east_sigma_m", "north_sigma_m", "vertical_sigma_m",
]
gnss_station_fields = ["station_id", "x_m", "y_m"]

for target_dir in (PUBLIC_DATA_DIR, PRIVATE_DATA_DIR):
    write_csv(os.path.join(target_dir, "interferograms.csv"), all_records, ifg_fields)
    write_csv(os.path.join(target_dir, "interferogram_metadata.csv"), all_meta, meta_fields)
    write_csv(os.path.join(target_dir, "gnss_stations.csv"), gnss_stations_rows, gnss_station_fields)
    write_csv(os.path.join(target_dir, "gnss_timeseries.csv"), gnss_records, gnss_ts_fields)

with open(os.path.join(PRIVATE_DATA_DIR, "answer_key.json"), "w") as fh:
    json.dump(answer_key, fh, indent=2, sort_keys=True)

print(f"Generated with seed {GENERATOR_SEED}")
print(f"  ascending interferograms: {len(meta_asc)}, rows: {len(records_asc)}")
print(f"  descending interferograms: {len(meta_desc)}, rows: {len(records_desc)}")
print(f"  gnss rows: {len(gnss_records)}")
print(f"  true rate1 vertical rate @ GNSS-01: {true_rate_before_primary:.6e} m/day")
print(f"  true rate2 vertical rate @ GNSS-01: {true_rate_after_primary:.6e} m/day")
print(f"  true cumulative vertical @ GNSS-01: {answer_key['true_cumulative_vertical_displacement_m_primary']:.4f} m")
print(f"  true cumulative vertical @ GNSS-07 (control): {answer_key['true_cumulative_vertical_displacement_m_control']:.6f} m")
print(f"  unwrap defect interferograms: {answer_key['unwrap_defect_interferograms']}")
