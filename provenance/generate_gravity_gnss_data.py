"""
generate_gravity_gnss_data.py -- ground-truth data generator for the
deformation-microgravity-volcano task.

AUTHOR'S PROVENANCE SCRIPT. Not executed by solve.py/test_outputs.py at
runtime; included for reviewer traceability.

True source: a Mogi (1958) point source at x0=200, y0=-150, depth=3000 m,
volume change dV=13e6 m3, density=2900 kg/m3 (magmatic recharge regime),
so mass change dM = density * dV.

Two epochs (pre-inflation, post-inflation) rather than a dense daily
series -- microgravity surveys are realistically episodic (campaign-style),
and this keeps the task's trap focused on spatial covariance weighting,
not temporal noise structure.

Confounds / traps:
- GNSS and gravity have very different measurement uncertainty SCALES
  (mm-level vs microGal-level).
- Noise is spatially correlated across nearby stations (exponential
  covariance in inter-station distance), for BOTH datasets independently.
- Gravity benchmarks are co-located with 8 of the 10 GNSS stations; the
  free-air correction uses each benchmark's co-located GNSS vertical
  displacement directly (no circular iteration needed), which is also
  the reason the two datasets end up statistically correlated: they
  share the same GNSS vertical-displacement noise realization.
"""

import numpy as np
import csv
import json
import os

RNG = np.random.default_rng(11)

# ---------------------------------------------------------------------------
# TRUE (ground truth) source parameters
# ---------------------------------------------------------------------------
TRUE_X0_M = 200.0
TRUE_Y0_M = -150.0
TRUE_DEPTH_M = 3000.0
TRUE_DELTA_V_M3 = 13e6
TRUE_DENSITY_KGM3 = 2900.0
TRUE_DELTA_M_KG = TRUE_DELTA_V_M3 * TRUE_DENSITY_KGM3

POISSON_RATIO = 0.25
G_GRAV = 6.674e-11
FREE_AIR_GRADIENT_MGAL_M = -0.3086  # standard sign: dg/dh (mGal per m of uplift)

N_DAYS = 490
ONSET_DAY = 90
RAMP_DAYS = 365

# ---------------------------------------------------------------------------
# Network: 10 GNSS stations; 8 gravity benchmarks co-located with 8 of them
# ---------------------------------------------------------------------------
station_ids = [f"GV-{i:02d}" for i in range(1, 11)]
station_xy = [
    (900, 300), (-700, 1100), (300, -1400), (2200, 900), (-1800, -1600),
    (1300, -1000), (-500, 400), (2500, -1300), (-2300, 700), (400, 2300),
]
GRAVITY_CO_LOCATED_STATIONS = station_ids[:8]  # GV-01 .. GV-08

GNSS_NOISE_SIGMA_M = 0.012        # ~12 mm formal uncertainty per station
GRAVITY_NOISE_SIGMA_MGAL = 0.003  # ~3 microGal formal uncertainty per benchmark
SPATIAL_CORR_LENGTH_M = 1200.0    # exponential covariance length scale


def mogi_displacement(x, y, depth, delta_v_m3, nu=POISSON_RATIO):
    r = np.sqrt(x**2 + y**2)
    R = np.sqrt(r**2 + depth**2)
    C = (1 - nu) * delta_v_m3 / np.pi
    u_z = C * depth / R**3
    u_r = C * r / R**3
    theta = np.arctan2(y, x)
    u_x = u_r * np.cos(theta)
    u_y = u_r * np.sin(theta)
    return u_x, u_y, u_z


def point_mass_gravity_effect(x, y, depth, mass_kg):
    r2 = x**2 + y**2
    denom = (r2 + depth**2) ** 1.5
    dg_si = G_GRAV * mass_kg * depth / denom
    return dg_si * 1e5  # mGal


def exponential_covariance(coords, sigma, length_scale):
    """Build a spatial covariance matrix: Cov[i,j] = sigma_i*sigma_j*exp(-d_ij/L)."""
    n = len(coords)
    cov = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = np.sqrt((coords[i][0] - coords[j][0]) ** 2 + (coords[i][1] - coords[j][1]) ** 2)
            cov[i, j] = sigma[i] * sigma[j] * np.exp(-d / length_scale)
    return cov


# ---------------------------------------------------------------------------
# Compute TRUE displacement/gravity at final epoch (day N_DAYS-1)
# ---------------------------------------------------------------------------
records_coords = []
true_disp = {}
for sid, (x, y) in zip(station_ids, station_xy):
    dx = x - TRUE_X0_M
    dy = y - TRUE_Y0_M
    ux, uy, uz = mogi_displacement(dx, dy, TRUE_DEPTH_M, TRUE_DELTA_V_M3)
    true_disp[sid] = (ux, uy, uz)
    records_coords.append({"station_id": sid, "x_m": x, "y_m": y})

# ---------------------------------------------------------------------------
# GNSS: two-epoch (pre/post) displacement, with spatially correlated noise
# ---------------------------------------------------------------------------
gnss_coords_list = [station_xy[station_ids.index(s)] for s in station_ids]
gnss_sigma = [GNSS_NOISE_SIGMA_M] * len(station_ids)
gnss_cov = exponential_covariance(gnss_coords_list, gnss_sigma, SPATIAL_CORR_LENGTH_M)
L_gnss = np.linalg.cholesky(gnss_cov + 1e-12 * np.eye(len(station_ids)))

noise_e = L_gnss @ RNG.standard_normal(len(station_ids))
noise_n = L_gnss @ RNG.standard_normal(len(station_ids))
noise_u = L_gnss @ RNG.standard_normal(len(station_ids))

gnss_records = []
for i, sid in enumerate(station_ids):
    ux, uy, uz = true_disp[sid]
    gnss_records.append({
        "station_id": sid,
        "disp_east_m": round(ux + noise_e[i], 6),
        "disp_north_m": round(uy + noise_n[i], 6),
        "disp_up_m": round(uz + noise_u[i], 6),
        "sigma_m": GNSS_NOISE_SIGMA_M,
    })

# ---------------------------------------------------------------------------
# Gravity: two-epoch change at co-located benchmarks, with spatially
# correlated noise (independent draw from the GNSS noise). NOTE: the
# observed (uncorrected) gravity noise is independent of GNSS noise, but
# the CORRECTED gravity change (after subtracting the free-air term) is
# NOT independent of the GNSS vertical displacement, because both use the
# same disp_up_m value/noise realization for that station.
# ---------------------------------------------------------------------------
gravity_coords_list = [station_xy[station_ids.index(s)] for s in GRAVITY_CO_LOCATED_STATIONS]
gravity_sigma = [GRAVITY_NOISE_SIGMA_MGAL] * len(GRAVITY_CO_LOCATED_STATIONS)
gravity_cov = exponential_covariance(gravity_coords_list, gravity_sigma, SPATIAL_CORR_LENGTH_M)
L_grav = np.linalg.cholesky(gravity_cov + 1e-12 * np.eye(len(GRAVITY_CO_LOCATED_STATIONS)))
noise_g = L_grav @ RNG.standard_normal(len(GRAVITY_CO_LOCATED_STATIONS))

# One gravity benchmark carries a plausible, undisclosed hydrological
# anomaly (e.g. a local water-table change) unrelated to the volcanic
# source -- a well-documented real confound in microgravity monitoring.
FLAGGED_BENCHMARK_INDEX = 4  # MG-05, unrecoverable (hydrological anomaly, must be excluded)
HYDRO_ANOMALY_MGAL = 0.022

gravity_records = []
for i, sid in enumerate(GRAVITY_CO_LOCATED_STATIONS):
    x, y = station_xy[station_ids.index(sid)]
    dx, dy = x - TRUE_X0_M, y - TRUE_Y0_M
    dg_mass = point_mass_gravity_effect(dx, dy, TRUE_DEPTH_M, TRUE_DELTA_M_KG)
    uz = true_disp[sid][2]
    dg_freeair = FREE_AIR_GRADIENT_MGAL_M * uz
    dg_observed = dg_mass + dg_freeair + noise_g[i]
    if i == FLAGGED_BENCHMARK_INDEX:
        dg_observed += HYDRO_ANOMALY_MGAL
    gravity_records.append({
        "benchmark_id": f"MG-{i+1:02d}",
        "co_located_gnss_station_id": sid,
        "gravity_change_mgal": round(dg_observed, 6),
        "sigma_mgal": GRAVITY_NOISE_SIGMA_MGAL,
    })

TRUE_FLAGGED_GRAVITY_BENCHMARK = f"MG-{FLAGGED_BENCHMARK_INDEX+1:02d}"

# ---------------------------------------------------------------------------
# Also generate a daily GNSS time series for onset-day detection (step 1)
# ---------------------------------------------------------------------------
days = np.arange(N_DAYS)
def opening_fraction(day):
    if day < ONSET_DAY:
        return 0.0
    elif day < ONSET_DAY + RAMP_DAYS:
        return (day - ONSET_DAY) / RAMP_DAYS
    else:
        return 1.0

timeseries_records = []
for day in days:
    frac = opening_fraction(day)
    ne = L_gnss @ RNG.standard_normal(len(station_ids))
    nn = L_gnss @ RNG.standard_normal(len(station_ids))
    nu = L_gnss @ RNG.standard_normal(len(station_ids))
    for i, sid in enumerate(station_ids):
        ux, uy, uz = true_disp[sid]
        timeseries_records.append({
            "station_id": sid, "day": int(day),
            "disp_east_m": round(ux * frac + ne[i], 6),
            "disp_north_m": round(uy * frac + nn[i], 6),
            "disp_up_m": round(uz * frac + nu[i], 6),
        })

# ---------------------------------------------------------------------------
# Write output files
# ---------------------------------------------------------------------------
OUTPUT_DIR = os.environ.get("GENERATOR_OUTPUT_DIR", ".")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(OUTPUT_DIR, "gnss_coords.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["station_id", "x_m", "y_m"])
    w.writeheader()
    w.writerows(records_coords)

with open(os.path.join(OUTPUT_DIR, "gnss_timeseries.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(timeseries_records[0].keys()))
    w.writeheader()
    w.writerows(timeseries_records)

# NOTE: gnss_displacement_total.csv is written for provenance/debugging only
# and must NOT be copied into task/environment/data/ (agent-visible public
# data). solve.py derives total displacement, and its per-station
# uncertainty, itself from gnss_timeseries.csv (the plateau-to-plateau
# window comparison in compute_total_displacement); it never reads this
# file. Shipping it to the agent would hand over a ready-made total
# displacement + formal sigma_m, short-circuiting exactly the derivation
# step (robust to end-of-record saturation, uncertainty from repeated
# noisy measurements rather than a given formal value) that is part of
# the task's intended difficulty.
with open(os.path.join(OUTPUT_DIR, "gnss_displacement_total.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gnss_records[0].keys()))
    w.writeheader()
    w.writerows(gnss_records)

with open(os.path.join(OUTPUT_DIR, "microgravity_stations.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(gravity_records[0].keys()))
    w.writeheader()
    w.writerows(gravity_records)

true_station_velocities = {}
for sid in station_ids:
    ux, uy, uz = true_disp[sid]
    true_station_velocities[sid] = {
        "velocity_up_m_per_day": uz / RAMP_DAYS,
        "velocity_east_m_per_day": ux / RAMP_DAYS,
        "velocity_north_m_per_day": uy / RAMP_DAYS,
    }

answer_key = {
    "true_x0_m": TRUE_X0_M,
    "true_y0_m": TRUE_Y0_M,
    "true_depth_m": TRUE_DEPTH_M,
    "true_delta_v_m3": TRUE_DELTA_V_M3,
    "true_delta_m_kg": TRUE_DELTA_M_KG,
    "true_density_kgm3": TRUE_DENSITY_KGM3,
    "true_process_classification": "magmatic_recharge",
    "onset_day": ONSET_DAY,
    "ramp_days": RAMP_DAYS,
    "spatial_corr_length_m": SPATIAL_CORR_LENGTH_M,
    "free_air_gradient_mgal_per_m": FREE_AIR_GRADIENT_MGAL_M,
    "true_station_velocities": true_station_velocities,
    "true_flagged_gravity_benchmark": TRUE_FLAGGED_GRAVITY_BENCHMARK,
    "hydro_anomaly_mgal": HYDRO_ANOMALY_MGAL,
    "notes": (
        "Generated by generate_gravity_gnss_data.py. True Mogi source at "
        "x0=200, y0=-150, depth=3000m, dV=13e6 m3, density=2900 kg/m3 "
        "(magmatic recharge regime). 90-day quiescent baseline, then a "
        "365-day (1-year) inflation ramp, matching a realistic annual "
        "microgravity survey campaign. Noise is spatially correlated "
        "(exponential covariance, length scale 1200 m) and independently "
        "drawn for GNSS and gravity datasets; the corrected gravity change "
        "is nonetheless statistically correlated with GNSS because both "
        "use the same vertical-displacement noise realization at "
        "co-located stations (free-air correction)."
    ),
}
with open(os.path.join(OUTPUT_DIR, "answer_key.json"), "w") as f:
    json.dump(answer_key, f, indent=2)

print("Generated files in", OUTPUT_DIR)
print(json.dumps(answer_key, indent=2))
