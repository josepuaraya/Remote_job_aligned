"""
generate_insar_data.py -- ground-truth data generator for the
insar-volcano-inversion task (v7).

v7 change: the "unrepairable" category's noise is now a genuinely rough,
short-correlation-length random field instead of a smooth deterministic
trend. Real diagnostic evidence (an actual agent run) showed the previous
smooth-trend version was, in fact, repairable: a general trend-surface
correction (elevation term plus a smooth 2D surface per interferogram)
brought its residuals below threshold just as reliably as the genuinely
repairable category, defeating the intended exclusion requirement with a
legitimate, defensible method. A smooth field is exactly what a
trend-surface fit is designed to remove -- it does not represent noise
that resists repair. Genuine turbulent atmospheric noise has real power
at short spatial scales that a smooth correction cannot capture without
also removing real deformation signal; that is what actually makes it
unrepairable, and what this version now models.

AUTHOR'S PROVENANCE SCRIPT. Not executed by solve.py/test_outputs.py at
runtime; included for reviewer traceability.

True source: a Mogi (1958) point source at x0=0, y0=0, depth=4500 m, with
onset at day 60 and a LINEAR (constant-rate) post-onset inflation profile.

v5 change: the InSAR atmospheric noise and the GNSS
common-mode error are now genuinely spatially/cross-station correlated
(drawn from an actual covariance model), not just independent per-point
draws. A correct inversion should estimate that covariance structure from
the data itself and use it to weight the joint fit (generalized least
squares / Bayesian inversion), rather than treating every point as an
independent, equally-weighted observation. This targets the same failure
mode documented in v3/v4: an unweighted (or iid-weighted) fit can still
show small residuals while its reported uncertainty is invalid, because
correlated noise carries less independent information than its raw point
count suggests.

- InSAR: each interferogram's baseline noise term is now drawn from a
  spatially correlated Gaussian random field (exponential covariance
  kernel, fixed correlation length, same marginal variance as the old
  i.i.d. term) over the shared 700-point cloud, instead of i.i.d. noise.
  The elevation-correlated (repairable) and large-scale (unrepairable)
  components are unchanged, so the existing atmospheric-noise QC
  classification and its calibration are unaffected -- only the baseline
  noise "shape" changed, not its per-point magnitude.
- GNSS: 3 continuous stations at different offsets, each carrying its own
  independent white noise (as in v4) PLUS a per-epoch, per-component
  common-mode error shared identically across all stations that epoch
  (a standard real GNSS network effect -- e.g. reference-frame or orbit
  error -- normally removed by common-mode stacking across the network).

20 interferograms: 10 ascending, 10 descending, each track formed from
11 independently-scheduled acquisition dates (~monthly spacing, offset
between tracks) as 10 sequential interferograms -- a realistic SBAS-style
stack, not a dense continuous time series.

Ascending geometry: incidence=39.1054 deg, heading=-11.876832 deg.
Descending geometry: incidence=39.1840 deg, heading=-168.27934 deg.
(Real geometry values from the author's own prior fieldwork in Iceland.)

Noise model, 6/8/6 split across the 20 interferograms:
- 6 "clean": spatially correlated baseline noise only.
- 8 "repairable": spatially correlated baseline noise PLUS an
  elevation-correlated linear term (simulating tropospheric delay
  correlated with topography), correctable via a phase-vs-elevation
  regression.
- 6 "unrepairable": spatially correlated baseline noise PLUS a
  large-amplitude, SHORT-correlation-length rough field NOT correlated
  with elevation (simulating turbulent atmospheric noise whose spatial
  frequency content resists any smooth/low-order correction) -- must be
  excluded.

Points are NOT a dense pixel grid but a quasi-random sample of ~700
points within the domain, matching the standard real-world practice of
downsampling (e.g. quadtree) an InSAR product before inversion.
"""

import numpy as np
import csv
import json
import os

GENERATOR_SEED = int(os.environ.get("GENERATOR_SEED", "7"))
RNG = np.random.default_rng(GENERATOR_SEED)

# ---------------------------------------------------------------------------
# TRUE (ground truth) source parameters
# ---------------------------------------------------------------------------
TRUE_X0_M = 0.0
TRUE_Y0_M = 0.0
TRUE_DEPTH_M = 4500.0
POISSON_RATIO = 0.25

ONSET_DAY = 60
N_DAYS = 365
DISPLACEMENT_RATE_M_PER_DAY_AT_SOURCE = 0.9 / (N_DAYS - 1 - ONSET_DAY)  # ~0.00296 m/day
TRUE_DV_RATE_M3_PER_DAY = (
    DISPLACEMENT_RATE_M_PER_DAY_AT_SOURCE * np.pi * TRUE_DEPTH_M**2 / (1 - POISSON_RATIO)
)

# ---------------------------------------------------------------------------
# Geometry (author's own values from Iceland fieldwork)
# ---------------------------------------------------------------------------
ASC_INCIDENCE_DEG = 39.1054
ASC_HEADING_DEG = -11.876832
DESC_INCIDENCE_DEG = 39.1840
DESC_HEADING_DEG = -168.27934

# ---------------------------------------------------------------------------
# Spatial sampling: ~700 quasi-random points (downsampled InSAR product),
# not a dense pixel grid
# ---------------------------------------------------------------------------
N_POINTS = 700
DOMAIN_HALF_WIDTH_M = 3500.0
x_pts = RNG.uniform(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, N_POINTS)
y_pts = RNG.uniform(-DOMAIN_HALF_WIDTH_M, DOMAIN_HALF_WIDTH_M, N_POINTS)

ELEV_PEAK_X_M = 300.0
ELEV_PEAK_Y_M = 200.0
PEAK_ELEV_M = 1500.0
DECAY_LENGTH_M = 1500.0
BASE_ELEV_M = 200.0
r_from_elev_peak = np.sqrt((x_pts - ELEV_PEAK_X_M)**2 + (y_pts - ELEV_PEAK_Y_M)**2)
elevation_pts = PEAK_ELEV_M * np.exp(-r_from_elev_peak / DECAY_LENGTH_M) + BASE_ELEV_M

# ---------------------------------------------------------------------------
# Spatially correlated InSAR baseline noise: exponential covariance kernel,
# same marginal sigma as the old i.i.d. term (0.002m) -- QC threshold
# calibration is unaffected since per-point magnitude is unchanged, only
# the correlation between points.
# ---------------------------------------------------------------------------
INSAR_CORR_SIGMA_M = 0.002
INSAR_CORR_LENGTH_M = 400.0


def _build_spatial_cholesky(x, y, sigma, length_scale):
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx**2 + dy**2)
    cov = sigma**2 * np.exp(-dist / length_scale)
    cov += 1e-9 * (sigma**2) * np.eye(len(x))  # jitter for numerical stability
    return np.linalg.cholesky(cov)


_SPATIAL_CHOL = _build_spatial_cholesky(x_pts, y_pts, INSAR_CORR_SIGMA_M, INSAR_CORR_LENGTH_M)


def draw_correlated_insar_field():
    z = RNG.standard_normal(N_POINTS)
    return _SPATIAL_CHOL @ z


# Unrepairable turbulent noise: genuinely rough (short correlation length,
# much shorter than the baseline field's), not a smooth deterministic trend.
# A smooth low-order trend (an orbital-ramp-like field) is exactly what a
# general trend-surface correction is designed to remove, so it does not
# represent noise that resists repair -- it is realistic turbulent
# atmospheric noise, which has real power at short spatial scales, that
# genuinely cannot be captured by any reasonable smooth correction without
# also removing real deformation signal.
UNREPAIRABLE_CORR_LENGTH_M = 120.0
_UNREPAIRABLE_CHOL = _build_spatial_cholesky(x_pts, y_pts, 1.0, UNREPAIRABLE_CORR_LENGTH_M)


def draw_rough_unrepairable_field():
    z = RNG.standard_normal(N_POINTS)
    field = _UNREPAIRABLE_CHOL @ z
    return field / np.std(field)  # unit marginal sigma, scaled by caller


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


def cumulative_dV_at_day(day, onset=ONSET_DAY, rate=TRUE_DV_RATE_M3_PER_DAY):
    return 0.0 if day < onset else rate * (day - onset)


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


# ---------------------------------------------------------------------------
# Build acquisition date lists (independent per track)
# ---------------------------------------------------------------------------
asc_dates = np.concatenate([
    np.linspace(0, 140, 6), np.linspace(210, N_DAYS - 1, 5)
]).astype(int)  # 70-day gap (140-210), 11 dates -> 10 interferograms
desc_dates = np.concatenate([
    np.linspace(6, 146, 6), np.linspace(216, N_DAYS - 7, 5)
]).astype(int)  # own 70-day gap (146-216), offset from ascending's

asc_pairs = [(asc_dates[i], asc_dates[i + 1]) for i in range(10)]
desc_pairs = [(desc_dates[i], desc_dates[i + 1]) for i in range(10)]

categories = ["clean"] * 5 + ["repairable"] * 8 + ["unrepairable"] * 6 + ["unwrap_jump"] * 1
RNG.shuffle(categories)
asc_categories = categories[:10]
desc_categories = categories[10:]

ELEV_CORR_COEFF_RANGE = (0.000006, 0.000012)
REPAIRABLE_TURBULENT_SIGMA_M = 0.004
UNREPAIRABLE_NOISE_SIGMA_M = 0.012

# Unwrapping-error defect: a discrete offset (a phase-ambiguity-sized jump,
# ~2.8cm -- realistic for a C-band radar) affecting points on one side of a
# random spatial split, riding on otherwise-clean correlated noise. Not
# correlated with elevation, so the existing phase-elevation correction
# cannot fix it -- and not smooth/large-scale like the unrepairable category,
# so a naive residual-magnitude read treats it like ordinary turbulent noise.
# Genuinely repairable (by detecting and correcting the discrete step), but
# only by a method that looks for a discontinuity, not the elevation
# regression already implemented for the "repairable" category.
UNWRAP_JUMP_SIZE_M = 0.028
_unwrap_jump_boundary_x = float(RNG.uniform(-1000.0, 1000.0))


def make_interferogram(day0, day1, incidence_deg, heading_deg, category):
    dV0 = cumulative_dV_at_day(day0)
    dV1 = cumulative_dV_at_day(day1)
    ux0, uy0, uz0 = mogi_displacement(x_pts, y_pts, TRUE_DEPTH_M, dV0)
    ux1, uy1, uz1 = mogi_displacement(x_pts, y_pts, TRUE_DEPTH_M, dV1)
    d_ux, d_uy, d_uz = ux1 - ux0, uy1 - uy0, uz1 - uz0

    los_true = los_projection_full(d_ux, d_uy, d_uz, incidence_deg, heading_deg)

    noise = draw_correlated_insar_field()
    if category == "repairable":
        coeff = RNG.uniform(*ELEV_CORR_COEFF_RANGE) * RNG.choice([-1, 1])
        elevation_term = coeff * (elevation_pts - BASE_ELEV_M)
        raw_field = x_pts + 0.7 * y_pts + 0.3 * (x_pts * y_pts / DOMAIN_HALF_WIDTH_M)
        raw_field = raw_field - np.mean(raw_field)
        raw_field = raw_field / np.std(raw_field) * REPAIRABLE_TURBULENT_SIGMA_M
        turbulent_overlay = raw_field * RNG.choice([-1, 1])
        noise = noise + elevation_term + turbulent_overlay
    elif category == "unrepairable":
        rough_field = draw_rough_unrepairable_field() * UNREPAIRABLE_NOISE_SIGMA_M
        noise = noise + rough_field
    elif category == "unwrap_jump":
        jump_sign = RNG.choice([-1, 1])
        jump = np.where(x_pts > _unwrap_jump_boundary_x, UNWRAP_JUMP_SIZE_M * jump_sign, 0.0)
        noise = noise + jump

    los_observed = los_true + noise
    return los_observed


records = []
answer_pairs = []
for i, (day0, day1) in enumerate(asc_pairs):
    ifg_id = f"ASC-{i+1:02d}"
    cat = asc_categories[i]
    los_observed = make_interferogram(day0, day1, ASC_INCIDENCE_DEG, ASC_HEADING_DEG, cat)
    for j in range(N_POINTS):
        records.append({
            "interferogram_id": ifg_id, "point_id": j,
            "x_m": round(x_pts[j], 2), "y_m": round(y_pts[j], 2),
            "elevation_m": round(elevation_pts[j], 2),
            "los_displacement_m": round(los_observed[j], 6),
        })
    answer_pairs.append({
        "interferogram_id": ifg_id, "orbit": "ascending",
        "day_start": int(day0), "day_end": int(day1),
        "incidence_deg": ASC_INCIDENCE_DEG, "heading_deg": ASC_HEADING_DEG,
        "true_category": cat,
    })

for i, (day0, day1) in enumerate(desc_pairs):
    ifg_id = f"DESC-{i+1:02d}"
    cat = desc_categories[i]
    los_observed = make_interferogram(day0, day1, DESC_INCIDENCE_DEG, DESC_HEADING_DEG, cat)
    for j in range(N_POINTS):
        records.append({
            "interferogram_id": ifg_id, "point_id": j,
            "x_m": round(x_pts[j], 2), "y_m": round(y_pts[j], 2),
            "elevation_m": round(elevation_pts[j], 2),
            "los_displacement_m": round(los_observed[j], 6),
        })
    answer_pairs.append({
        "interferogram_id": ifg_id, "orbit": "descending",
        "day_start": int(day0), "day_end": int(day1),
        "incidence_deg": DESC_INCIDENCE_DEG, "heading_deg": DESC_HEADING_DEG,
        "true_category": cat,
    })

# ---------------------------------------------------------------------------
# Held-out validation interferogram: same schema and noise character as a
# "clean" sequential interferogram, but explicitly NOT part of the
# sequential fitting stack (ID prefix "VAL-", not "ASC-"/"DESC-"). Exists
# to test predictive validation (was the fitted model derived from the
# sequential stack ever checked against a genuinely unseen observation?)
# rather than atmospheric-noise screening, which is unrelated.
# ---------------------------------------------------------------------------
VAL_DAY_START, VAL_DAY_END = 100, 180
VAL_ID = "VAL-01"
val_los_observed = make_interferogram(VAL_DAY_START, VAL_DAY_END, ASC_INCIDENCE_DEG, ASC_HEADING_DEG, "clean")
for j in range(N_POINTS):
    records.append({
        "interferogram_id": VAL_ID, "point_id": j,
        "x_m": round(x_pts[j], 2), "y_m": round(y_pts[j], 2),
        "elevation_m": round(elevation_pts[j], 2),
        "los_displacement_m": round(val_los_observed[j], 6),
    })
answer_pairs.append({
    "interferogram_id": VAL_ID, "orbit": "ascending",
    "day_start": VAL_DAY_START, "day_end": VAL_DAY_END,
    "incidence_deg": ASC_INCIDENCE_DEG, "heading_deg": ASC_HEADING_DEG,
    "true_category": "validation",
})

# ---------------------------------------------------------------------------
# GNSS network: 3 continuous stations, fixed locations, decimated to
# 10-day epochs. Each epoch/component gets ONE shared common-mode draw
# (identical across stations that epoch) plus independent per-station
# white noise.
# ---------------------------------------------------------------------------
GNSS_STATIONS = [
    {"id": "GNSS-01", "x_m": 900.0, "y_m": -700.0},
    {"id": "GNSS-02", "x_m": -1300.0, "y_m": 900.0},
    {"id": "GNSS-03", "x_m": 200.0, "y_m": 1600.0},
]
GNSS_WHITE_SIGMA_M = 0.003
GNSS_COMMON_MODE_SIGMA_M = 0.0025
GNSS_EPOCH_DAYS = np.arange(0, N_DAYS, 10)

gnss_records = []
for day in GNSS_EPOCH_DAYS:
    dV = cumulative_dV_at_day(int(day))
    common = GNSS_COMMON_MODE_SIGMA_M * RNG.standard_normal(3)  # shared per epoch: e, n, u
    for st in GNSS_STATIONS:
        ux, uy, uz = mogi_displacement(
            np.array([st["x_m"]]), np.array([st["y_m"]]), TRUE_DEPTH_M, dV, TRUE_X0_M, TRUE_Y0_M
        )
        wn = GNSS_WHITE_SIGMA_M * RNG.standard_normal(3)
        gnss_records.append({
            "station_id": st["id"], "day": int(day), "x_m": st["x_m"], "y_m": st["y_m"],
            "east_disp_m": round(float(ux[0]) + common[0] + wn[0], 6),
            "north_disp_m": round(float(uy[0]) + common[1] + wn[1], 6),
            "vertical_disp_m": round(float(uz[0]) + common[2] + wn[2], 6),
        })

PRIMARY_GNSS_ID = "GNSS-01"
primary = next(st for st in GNSS_STATIONS if st["id"] == PRIMARY_GNSS_ID)
true_final_ux, true_final_uy, true_final_uz = mogi_displacement(
    np.array([primary["x_m"]]), np.array([primary["y_m"]]), TRUE_DEPTH_M,
    cumulative_dV_at_day(N_DAYS - 1), TRUE_X0_M, TRUE_Y0_M,
)

# True (noise-free) LOS value at the validation interferogram's evaluation
# point (reuses the primary GNSS station's already-disclosed coordinates),
# for the verifier's accuracy backstop.
_val_dV0 = cumulative_dV_at_day(VAL_DAY_START)
_val_dV1 = cumulative_dV_at_day(VAL_DAY_END)
_val_ux0, _val_uy0, _val_uz0 = mogi_displacement(
    np.array([primary["x_m"]]), np.array([primary["y_m"]]), TRUE_DEPTH_M, _val_dV0, TRUE_X0_M, TRUE_Y0_M
)
_val_ux1, _val_uy1, _val_uz1 = mogi_displacement(
    np.array([primary["x_m"]]), np.array([primary["y_m"]]), TRUE_DEPTH_M, _val_dV1, TRUE_X0_M, TRUE_Y0_M
)
true_validation_los_m = float(los_projection_full(
    _val_ux1 - _val_ux0, _val_uy1 - _val_uy0, _val_uz1 - _val_uz0, ASC_INCIDENCE_DEG, ASC_HEADING_DEG
)[0])

# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("GENERATOR_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "interferograms.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
    w.writeheader()
    w.writerows(records)

with open(os.path.join(OUTPUT_DIR, "interferogram_metadata.csv"), "w", newline="") as f:
    fieldnames = ["interferogram_id", "orbit", "day_start", "day_end", "incidence_deg", "heading_deg"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for rec in answer_pairs:
        w.writerow({k: rec[k] for k in fieldnames})

with open(os.path.join(OUTPUT_DIR, "gps_stations.csv"), "w", newline="") as f:
    fieldnames = ["station_id", "day", "x_m", "y_m", "east_disp_m", "north_disp_m", "vertical_disp_m"]
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    w.writerows(gnss_records)

answer_key = {
    "true_x0_m": TRUE_X0_M,
    "true_y0_m": TRUE_Y0_M,
    "true_depth_m": TRUE_DEPTH_M,
    "true_dv_rate_m3_per_day": TRUE_DV_RATE_M3_PER_DAY,
    "true_cumulative_delta_v_m3": cumulative_dV_at_day(N_DAYS - 1),
    "onset_day": ONSET_DAY,
    "unrepairable_interferograms": [p["interferogram_id"] for p in answer_pairs if p["true_category"] == "unrepairable"],
    "repairable_interferograms": [p["interferogram_id"] for p in answer_pairs if p["true_category"] == "repairable"],
    "clean_interferograms": [p["interferogram_id"] for p in answer_pairs if p["true_category"] == "clean"],
    "gps_location_m": {"x_m": primary["x_m"], "y_m": primary["y_m"]},
    "true_gps_final_displacement_m": {
        "east_m": float(true_final_ux[0]),
        "north_m": float(true_final_uy[0]),
        "vertical_m": float(true_final_uz[0]),
    },
    "insar_true_correlation_length_m": INSAR_CORR_LENGTH_M,
    "insar_true_sill_m2": INSAR_CORR_SIGMA_M ** 2,
    "gnss_true_common_mode_sigma_m": GNSS_COMMON_MODE_SIGMA_M,
    "gnss_true_white_sigma_m": GNSS_WHITE_SIGMA_M,
    "gnss_stations_m": GNSS_STATIONS,
    "unwrap_jump_interferograms": [p["interferogram_id"] for p in answer_pairs if p["true_category"] == "unwrap_jump"],
    "unwrap_jump_size_m": UNWRAP_JUMP_SIZE_M,
    "unwrap_jump_boundary_x_m": _unwrap_jump_boundary_x,
    "validation_interferogram_id": VAL_ID,
    "validation_interferogram_evaluation_point_m": {"x_m": primary["x_m"], "y_m": primary["y_m"]},
    "validation_interferogram_window": {
        "day_start": VAL_DAY_START, "day_end": VAL_DAY_END,
        "incidence_deg": ASC_INCIDENCE_DEG, "heading_deg": ASC_HEADING_DEG,
    },
    "true_validation_interferogram_los_m": true_validation_los_m,
    "notes": (
        "Generated by generate_insar_data.py (v6). True Mogi source at "
        "x0=0, y0=0, depth=4500m. Onset day (60) known/disclosed; linear "
        "post-onset rate. InSAR baseline noise is spatially correlated "
        "(exponential kernel, correlation length 400m, same marginal "
        "sigma as the old i.i.d. term) -- QC classification and its "
        "calibration are unaffected. 3-station GNSS network with a "
        "shared per-epoch common-mode error plus independent per-station "
        "white noise. A correct inversion estimates both covariance "
        "structures from the data and weights the joint fit accordingly. "
        "One sequential interferogram carries a discrete unwrapping-style "
        "jump instead of the usual repairable/unrepairable noise. A "
        "held-out validation interferogram (VAL-01) is not part of the "
        "sequential fitting stack at all."
    ),
}
with open(os.path.join(OUTPUT_DIR, "answer_key.json"), "w") as f:
    json.dump(answer_key, f, indent=2)

print("Generated files in", OUTPUT_DIR)
print(json.dumps({k: v for k, v in answer_key.items() if k != "notes"}, indent=2))
