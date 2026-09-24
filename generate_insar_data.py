"""
generate_insar_data.py -- ground-truth data generator for the
insar-volcano-inversion task.

AUTHOR'S PROVENANCE SCRIPT. Not executed by solve.py/test_outputs.py at
runtime; included for reviewer traceability.

True source: a Mogi (1958) point source at x0=0, y0=0, depth=4500 m, with
a decaying-rate (exponential, saturating) inflation profile: onset at
day 60, initial rate 350 cm/yr, time constant tau=100 days, implying a
final cumulative displacement (at the point nearest the source) of ~96 cm.

20 interferograms: 10 ascending, 10 descending, each track formed from
11 independently-scheduled acquisition dates (~monthly spacing, offset
between tracks) as 10 sequential interferograms -- a realistic SBAS-style
stack, not a dense continuous time series.

Ascending geometry: incidence=39.1054 deg, heading=-11.876832 deg.
Descending geometry: incidence=39.1840 deg, heading=-168.27934 deg.
(Real geometry values from the author's own prior fieldwork in Iceland.)

Noise model, 6/8/6 split across the 20 interferograms:
- 6 "clean": small i.i.d. noise only.
- 8 "repairable": clean noise PLUS an elevation-correlated linear term
  (simulating tropospheric delay correlated with topography), correctable
  via a phase-vs-elevation regression.
- 6 "unrepairable": clean noise PLUS a large-amplitude smooth 2D random
  field NOT correlated with elevation (simulating turbulent atmospheric
  noise with no simple correctable structure) -- must be excluded.

Points are NOT a dense pixel grid but a quasi-random sample of ~700
points within the domain, matching the standard real-world practice of
downsampling (e.g. quadtree) an InSAR product before inversion.
"""

import numpy as np
import csv
import json
import os

RNG = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# TRUE (ground truth) source parameters
# ---------------------------------------------------------------------------
TRUE_X0_M = 0.0
TRUE_Y0_M = 0.0
TRUE_DEPTH_M = 4500.0
POISSON_RATIO = 0.25

# Onset day is KNOWN/disclosed (not estimated by the agent) -- avoids the
# severe joint spatial+temporal degeneracy found earlier when onset was
# an unknown fit alongside x0,y0,depth. Deformation grows LINEARLY
# (constant rate) after onset, no saturation.
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

# Synthetic volcanic-cone elevation model -- peak is OFFSET from the true
# Mogi source location (real volcanoes commonly have their topographic
# edifice displaced from the deep magmatic pressure source), so an
# elevation-correlated noise regression cannot be confused with the real
# deformation signal, which is centered at the source instead
ELEV_PEAK_X_M = 300.0
ELEV_PEAK_Y_M = 200.0
PEAK_ELEV_M = 1500.0
DECAY_LENGTH_M = 1500.0
BASE_ELEV_M = 200.0
r_from_elev_peak = np.sqrt((x_pts - ELEV_PEAK_X_M)**2 + (y_pts - ELEV_PEAK_Y_M)**2)
elevation_pts = PEAK_ELEV_M * np.exp(-r_from_elev_peak / DECAY_LENGTH_M) + BASE_ELEV_M


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
    """Linear (constant-rate) cumulative volume change since record start.
    Zero before onset, then grows linearly -- onset is KNOWN/disclosed,
    so this is not a shape parameter the agent must estimate."""
    return 0.0 if day < onset else rate * (day - onset)


def los_projection_full(u_e, u_n, u_u, incidence_deg, heading_deg):
    """Full LOS projection (Image 1) -- what a real satellite actually
    measures, including the true (if partial) sensitivity to north-south
    motion. Used for the true synthetic data and the primary direct-LOS
    inversion.
    """
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return (np.sin(theta) * np.cos(alpha) * u_e
            - np.sin(theta) * np.sin(alpha) * u_n
            - np.cos(theta) * u_u)


def los_projection_reduced(u_e, u_u, incidence_deg, heading_deg):
    """Reduced LOS projection (Image 2), assuming negligible north-south
    motion. Used ONLY for the secondary, descriptive vertical/east-west
    decomposition deliverable, where the 2-look-direction system must
    assume V_N=0 to be solvable for 2 unknowns.
    """
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return -np.cos(theta) * u_u + np.sin(theta) * np.cos(alpha) * u_e


# ---------------------------------------------------------------------------
# Build acquisition date lists (independent per track)
# ---------------------------------------------------------------------------
# Acquisition dates WITH a real temporal gap (no acquisitions at all
# during a stretch of the record) for each track -- the interferogram
# spanning that gap will have a much longer duration than its neighbors,
# and the true cumulative displacement during the un-observed stretch
# must be inferred, not directly measured.
asc_dates = np.concatenate([
    np.linspace(0, 140, 6), np.linspace(210, N_DAYS - 1, 5)
]).astype(int)  # 70-day gap (140-210), 11 dates -> 10 interferograms
desc_dates = np.concatenate([
    np.linspace(6, 146, 6), np.linspace(216, N_DAYS - 7, 5)
]).astype(int)  # own 70-day gap (146-216), offset from ascending's

asc_pairs = [(asc_dates[i], asc_dates[i + 1]) for i in range(10)]
desc_pairs = [(desc_dates[i], desc_dates[i + 1]) for i in range(10)]

# Two additional FULL-SPAN interferograms, both covering the SAME
# reference window (day 0 to day N_DAYS-1), used as the reliable channel
# for the spatial (x0,y0,depth,volume) inversion -- avoiding any
# dependence on the temporal shape parameters (onset/tau), which are only
# weakly identifiable from the sequential stack alone. These are always
# "clean" (small noise only).

# ---------------------------------------------------------------------------
# Assign noise category: 6 clean, 8 repairable, 6 unrepairable (mixed
# across ascending/descending)
# ---------------------------------------------------------------------------
categories = ["clean"] * 6 + ["repairable"] * 8 + ["unrepairable"] * 6
RNG.shuffle(categories)
asc_categories = categories[:10]
desc_categories = categories[10:]

CLEAN_NOISE_SIGMA_M = 0.002        # ~2mm i.i.d.
ELEV_CORR_COEFF_RANGE = (0.000006, 0.000012)  # m displacement per m elevation
                                                # (rescaled down ~7x from the
                                                # earlier full-span design,
                                                # since individual sequential
                                                # interferograms now carry a
                                                # much smaller true signal
                                                # (~2-8cm over ~28-90 days)
                                                # rather than ~0.9m over the
                                                # full record)
REPAIRABLE_TURBULENT_SIGMA_M = 0.004  # residual turbulent overlay,
                                        # riding on top of the elevation-linear
                                        # term -- correcting for elevation alone
                                        # does NOT perfectly recover the signal;
                                        # a real judgment call on the residual
                                        # noise level (still below threshold) is
                                        # required, not a blind mechanical fit
UNREPAIRABLE_NOISE_SIGMA_M = 0.012  # rescaled down to match smaller signal scale


def make_interferogram(day0, day1, incidence_deg, heading_deg, category):
    dV0 = cumulative_dV_at_day(day0)
    dV1 = cumulative_dV_at_day(day1)
    ux0, uy0, uz0 = mogi_displacement(x_pts, y_pts, TRUE_DEPTH_M, dV0)
    ux1, uy1, uz1 = mogi_displacement(x_pts, y_pts, TRUE_DEPTH_M, dV1)
    d_ux, d_uy, d_uz = ux1 - ux0, uy1 - uy0, uz1 - uz0

    los_true = los_projection_full(d_ux, d_uy, d_uz, incidence_deg, heading_deg)

    noise = CLEAN_NOISE_SIGMA_M * RNG.standard_normal(N_POINTS)
    if category == "repairable":
        coeff = RNG.uniform(*ELEV_CORR_COEFF_RANGE) * RNG.choice([-1, 1])
        elevation_term = coeff * (elevation_pts - BASE_ELEV_M)
        # smooth turbulent overlay riding on top of the elevation-driven
        # term -- normalized to a FIXED realized RMS (not left to random
        # coefficient-draw variance) so repairable interferograms have a
        # consistent, reliably-separable post-correction noise level
        raw_field = x_pts + 0.7 * y_pts + 0.3 * (x_pts * y_pts / DOMAIN_HALF_WIDTH_M)
        raw_field = raw_field - np.mean(raw_field)
        raw_field = raw_field / np.std(raw_field) * REPAIRABLE_TURBULENT_SIGMA_M
        turbulent_overlay = raw_field * RNG.choice([-1, 1])
        noise = noise + elevation_term + turbulent_overlay
    elif category == "unrepairable":
        # large-amplitude smooth field uncorrelated with elevation,
        # normalized to a FIXED realized RMS well above the correction
        # threshold regardless of any reasonable correction attempt
        raw_field = x_pts - 0.5 * y_pts + 0.4 * (x_pts * y_pts / DOMAIN_HALF_WIDTH_M)
        raw_field = raw_field - np.mean(raw_field)
        raw_field = raw_field / np.std(raw_field) * UNREPAIRABLE_NOISE_SIGMA_M
        noise = noise + raw_field * RNG.choice([-1, 1])

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
    "notes": (
        "Generated by generate_insar_data.py. True Mogi source at x0=0, "
        "y0=0, depth=4500m. Onset day (60) is KNOWN/disclosed to the "
        "agent -- deformation grows LINEARLY (constant rate) after onset, "
        "no saturation. Acquisition dates have a genuine ~70-day gap "
        "(different for each track) with no coverage at all, requiring "
        "interpolation across it. 700 quasi-random sample points "
        "(downsampled InSAR product). 6 clean / 8 repairable "
        "(elevation-correlated) / 6 unrepairable (turbulent) interferograms "
        "among the 20 sequential ones."
    ),
}
with open(os.path.join(OUTPUT_DIR, "answer_key.json"), "w") as f:
    json.dump(answer_key, f, indent=2)

print("Generated files in", OUTPUT_DIR)
print(json.dumps({k: v for k, v in answer_key.items() if k != "notes"}, indent=2))
