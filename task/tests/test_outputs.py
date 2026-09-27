"""
test_outputs.py -- verifier for the InSAR/GNSS volcano deformation
rate-change task (McTigue joint source inversion).

Every tolerance below was calibrated by running the reference solution
(task/solution/solve.py) against eight independent synthetic realizations
of the same scenario (different generator seeds) and taking roughly a
3-5x margin over the largest error/width actually observed. See
task/solution/process.md for the full calibration table, including the
finding that the source RADIUS is only weakly identified by this data
(the reference solution itself shows errors from ~5% to ~95% of the true
value across seeds, including one realization where the fit pins radius
at its search boundary) -- radius is therefore checked for format and
self-consistency only, not point-estimate accuracy, the same treatment
Task3-style tasks give other weakly-identified-by-design outputs.

Nothing here is a pasted expected value -- every check compares the
submission against task/tests/data/answer_key.json (the private ground
truth). task/tests/data/ also holds a private copy of the same public CSVs
shipped to the agent (interferograms.csv, interferogram_metadata.csv,
gnss_stations.csv, gnss_timeseries.csv) -- generate_data.py writes both
copies for the author's own reproducibility -- but this verifier does not
read them; every assertion here is against answer_key.json and the
submitted result.json alone.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RESULT_PATH = OUTPUT_DIR / "result.json"
ANSWER_KEY_PATH = TESTS_DIR / "data" / "answer_key.json"
PUBLIC_DATA_DIR = TESTS_DIR / "data"

REQUIRED_KEYS = {
    "x0_m", "y0_m", "depth_m", "radius_m",
    "x0_uncertainty_95", "y0_uncertainty_95", "depth_uncertainty_95", "radius_uncertainty_95",
    "vertical_rate_before_m_per_day",
    "vertical_rate_after_m_per_day",
    "rate_change_day",
    "cumulative_vertical_displacement_m",
    "volume_change_m3",
    "vertical_rate_before_uncertainty_95",
    "vertical_rate_after_uncertainty_95",
    "rate_change_day_uncertainty_95",
    "cumulative_vertical_displacement_uncertainty_95",
    "volume_change_uncertainty_95",
    "control_zone_cumulative_vertical_displacement_m",
    "insar_gnss_ramp_coefficients",
    "gnss_stations_used",
    "gnss_stations_excluded",
    "interferograms_with_unwrapping_correction",
    "interferograms_excluded_from_inversion",
    "insar_gnss_rmse_m",
    "poisson_ratio_assumed",
}

# Point-estimate tolerances (see process.md calibration table: max observed
# error across 8 seeds was 52.9 m / 52.1 m / 6.32% / 7.08% / 3.27% / 6.3
# days / 2.76% respectively for x0 / y0 / depth / rate-before / rate-after /
# rate-change-day / cumulative displacement).
LOCATION_ABS_TOL_M = 200.0
DEPTH_REL_TOL = 0.25
RATE_BEFORE_REL_TOL = 0.25
RATE_AFTER_REL_TOL = 0.15
RATE_CHANGE_DAY_ABS_TOL = 25.0
CUMULATIVE_REL_TOL = 0.12
VOLUME_CHANGE_REL_TOL = 0.40  # max observed across 8 seeds was 11.21%
CONTROL_ABS_TOL_M = 0.010

# CI sanity caps, re-measured across all 8 calibration seeds against the
# current parametric-bootstrap uncertainty method (see process.md for the
# full width table and the true-value coverage rate per quantity -- these
# caps only reject a degenerate/placeholder-wide interval, at roughly a
# 3-3.6x margin over the largest width the bootstrap actually produced;
# they are not, and are not meant to be, a coverage guarantee).
LOCATION_CI_WIDTH_CAP_M = 250.0           # max observed (8 seeds) ~70.2 m
DEPTH_CI_WIDTH_CAP_FRAC = 0.30            # max observed (8 seeds) ~8.8%
RATE_BEFORE_CI_WIDTH_CAP_FRAC = 0.55      # max observed (8 seeds) ~17.4%
RATE_AFTER_CI_WIDTH_CAP_FRAC = 0.30       # see note below -- NOT a ~3.5x margin
                                           # like its neighbors
RATE_CHANGE_DAY_CI_WIDTH_CAP = 50.0       # max observed (8 seeds) ~13.9 days
CUMULATIVE_CI_WIDTH_CAP_FRAC = 0.12       # max observed (8 seeds) ~3.4%
VOLUME_CHANGE_CI_WIDTH_CAP_FRAC = 0.55    # max observed (8 seeds) ~16.6%

RADIUS_CI_WIDTH_SANITY_CAP_M = 2.0e5  # radius is weakly identified (see above) and a
# genuinely honest CI can be tens of thousands of meters wide (observed up to ~50,000 m
# across 8 seeds); this only catches a degenerate placeholder (e.g. bounds of +/-1e9).

# rate_after gets the same weak-coverage treatment as radius, for a documented
# reason (process.md's coverage table): the reference bootstrap's own CI for
# rate_after covered the true value in only 2/8 calibration seeds, well below
# nominal -- unlike every other quantity above, which cleared 4/8 or better.
# That means the bootstrap itself is known to underestimate rate_after's true
# uncertainty, so using ITS width (even with the same ~3.5x margin used
# elsewhere) as the cap would risk rejecting a differently-implemented,
# honestly-wider CI for exactly the quantity where "honestly wider" is most
# likely to be correct. 30% keeps the cap meaningful (still rejects a
# degenerate/placeholder interval) without leaning on a margin computed from
# a method with demonstrated poor coverage for this specific quantity.
RMSE_SANITY_CAP_M = 0.03  # max observed across 8 seeds was ~0.015 m
MAX_GNSS_EXCLUDED = 4     # max observed across 8 seeds was 2
MAX_UNWRAP_FLAGGED = 4    # true count is 2; allow some slack for a
                          # differently-implemented method flagging one
                          # extra borderline interferogram
MAX_UNWRAP_EXCLUDED = 4   # same slack, for a solution that drops rather
                          # than repairs a defective interferogram


@pytest.fixture(scope="module")
def answer_key() -> dict:
    assert ANSWER_KEY_PATH.exists(), f"missing private answer key at {ANSWER_KEY_PATH}"
    return json.loads(ANSWER_KEY_PATH.read_text())


@pytest.fixture(scope="module")
def submitted() -> dict:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


def _load_csv(path: Path) -> list[dict]:
    with open(path) as fh:
        return list(csv.DictReader(fh))


@pytest.fixture(scope="module")
def public_data() -> dict:
    # The same public CSVs shipped to the agent, from this verifier's own
    # private copy -- used only by test_insar_gnss_rmse_is_consistent_with_ramp
    # below, to independently cross-check a self-reported number.
    return {
        "interferograms": _load_csv(PUBLIC_DATA_DIR / "interferograms.csv"),
        "interferogram_metadata": _load_csv(PUBLIC_DATA_DIR / "interferogram_metadata.csv"),
        "gnss_stations": _load_csv(PUBLIC_DATA_DIR / "gnss_stations.csv"),
        "gnss_timeseries": _load_csv(PUBLIC_DATA_DIR / "gnss_timeseries.csv"),
    }


def _ci(submitted, key):
    val = submitted[key]
    assert isinstance(val, list) and len(val) == 2, f"{key} must be a [lower, upper] pair"
    lo, hi = float(val[0]), float(val[1])
    assert lo < hi, f"{key} must have lower < upper, got {val}"
    return lo, hi


def _check_point_and_ci(submitted, point_key, ci_key, true_val, rel_tol=None, abs_tol=None,
                         ci_width_cap=None, ci_width_frac_cap=None):
    reported = float(submitted[point_key])
    if rel_tol is not None:
        rel_err = abs(reported - true_val) / abs(true_val)
        assert rel_err <= rel_tol, (
            f"{point_key}={reported} vs true={true_val} (rel err {rel_err:.3f} > {rel_tol})"
        )
    if abs_tol is not None:
        err = abs(reported - true_val)
        assert err <= abs_tol, f"{point_key}={reported} vs true={true_val} (abs err {err} > {abs_tol})"

    lo, hi = _ci(submitted, ci_key)
    assert lo <= reported <= hi, f"{point_key} point estimate outside its own reported CI"
    if ci_width_cap is not None:
        assert (hi - lo) <= ci_width_cap, f"{ci_key} implausibly wide: {hi - lo} > {ci_width_cap}"
    if ci_width_frac_cap is not None:
        cap = ci_width_frac_cap * abs(reported)
        assert (hi - lo) <= cap, f"{ci_key} implausibly wide: {hi - lo} > {cap}"
    return reported


def test_required_keys_present(submitted):
    missing = REQUIRED_KEYS - set(submitted)
    assert not missing, f"Missing required keys: {missing}"


def test_source_location_within_tolerance(submitted, answer_key):
    _check_point_and_ci(submitted, "x0_m", "x0_uncertainty_95", answer_key["true_x0_m"],
                         abs_tol=LOCATION_ABS_TOL_M, ci_width_cap=LOCATION_CI_WIDTH_CAP_M)
    _check_point_and_ci(submitted, "y0_m", "y0_uncertainty_95", answer_key["true_y0_m"],
                         abs_tol=LOCATION_ABS_TOL_M, ci_width_cap=LOCATION_CI_WIDTH_CAP_M)


def test_source_depth_within_tolerance(submitted, answer_key):
    _check_point_and_ci(submitted, "depth_m", "depth_uncertainty_95", answer_key["true_depth_m"],
                         rel_tol=DEPTH_REL_TOL, ci_width_frac_cap=DEPTH_CI_WIDTH_CAP_FRAC)


def test_source_radius_format_only(submitted, answer_key):
    """Radius is only weakly identified by this data (see process.md); it
    is checked for physical plausibility and self-consistency, not
    point-estimate accuracy -- the same treatment Task3-style tasks give
    other outputs that are 'checked only for format.'"""
    radius = float(submitted["radius_m"])
    depth = float(submitted["depth_m"])
    assert radius == radius and radius > 0, "radius_m must be a finite, positive number"
    assert radius < depth, "radius_m must be smaller than depth_m (a surfacing source is not physical here)"
    lo, hi = _ci(submitted, "radius_uncertainty_95")
    assert lo <= radius <= hi, "radius_m point estimate outside its own reported CI"
    assert (hi - lo) <= RADIUS_CI_WIDTH_SANITY_CAP_M, (
        f"radius_uncertainty_95 implausibly wide: {hi - lo} > {RADIUS_CI_WIDTH_SANITY_CAP_M}"
    )


def test_vertical_rate_before_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_vertical_rate_before_m_per_day_primary"]
    _check_point_and_ci(submitted, "vertical_rate_before_m_per_day", "vertical_rate_before_uncertainty_95",
                         true_val, rel_tol=RATE_BEFORE_REL_TOL, ci_width_frac_cap=RATE_BEFORE_CI_WIDTH_CAP_FRAC)


def test_vertical_rate_after_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_vertical_rate_after_m_per_day_primary"]
    _check_point_and_ci(submitted, "vertical_rate_after_m_per_day", "vertical_rate_after_uncertainty_95",
                         true_val, rel_tol=RATE_AFTER_REL_TOL, ci_width_frac_cap=RATE_AFTER_CI_WIDTH_CAP_FRAC)


def test_rate_accelerates(submitted):
    """The decision this task supports: has the inflation rate increased?
    (It has, by construction.) A submission that gets both rates close to
    their individual tolerances but in the wrong relative order has not
    actually answered the practitioner's question."""
    before = float(submitted["vertical_rate_before_m_per_day"])
    after = float(submitted["vertical_rate_after_m_per_day"])
    assert after > before, (
        f"vertical_rate_after ({after}) must exceed vertical_rate_before ({before}): "
        "the true scenario is an acceleration"
    )


def test_rate_change_day_within_tolerance(submitted, answer_key):
    _check_point_and_ci(submitted, "rate_change_day", "rate_change_day_uncertainty_95",
                         answer_key["true_rate_change_day"], abs_tol=RATE_CHANGE_DAY_ABS_TOL,
                         ci_width_cap=RATE_CHANGE_DAY_CI_WIDTH_CAP)


def test_cumulative_displacement_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_cumulative_vertical_displacement_m_primary"]
    _check_point_and_ci(submitted, "cumulative_vertical_displacement_m",
                         "cumulative_vertical_displacement_uncertainty_95", true_val,
                         rel_tol=CUMULATIVE_REL_TOL, ci_width_frac_cap=CUMULATIVE_CI_WIDTH_CAP_FRAC)


def test_volume_change_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_volume_change_m3"]
    _check_point_and_ci(submitted, "volume_change_m3", "volume_change_uncertainty_95", true_val,
                         rel_tol=VOLUME_CHANGE_REL_TOL, ci_width_frac_cap=VOLUME_CHANGE_CI_WIDTH_CAP_FRAC)


def test_control_zone_shows_no_meaningful_deformation(submitted, answer_key):
    """The control station sits far enough from the source that the true
    signal there is a couple of mm over the whole record -- a submission
    that (incorrectly) extrapolates the near-source source model to the
    far field, or otherwise hallucinates deformation from noise, will be
    off by much more than this tolerance."""
    true_val = answer_key["true_cumulative_vertical_displacement_m_control"]
    reported = float(submitted["control_zone_cumulative_vertical_displacement_m"])
    err = abs(reported - true_val)
    assert err <= CONTROL_ABS_TOL_M, (
        f"control_zone_cumulative_vertical_displacement_m={reported} vs true={true_val} "
        f"(abs err {err} > {CONTROL_ABS_TOL_M} m)"
    )


def test_gnss_bad_station_excluded(submitted, answer_key):
    excluded = set(submitted["gnss_stations_excluded"])
    used = set(submitted["gnss_stations_used"])
    all_ids = set(answer_key["all_gnss_station_ids"])

    assert excluded.issubset(all_ids), f"gnss_stations_excluded has unknown ids: {excluded - all_ids}"
    assert used.issubset(all_ids), f"gnss_stations_used has unknown ids: {used - all_ids}"
    assert not (excluded & used), "a station cannot be both used and excluded"

    bad_id = answer_key["bad_gnss_station_id"]
    assert bad_id in excluded, f"the known-defective station {bad_id} must be in gnss_stations_excluded"
    assert len(excluded) <= MAX_GNSS_EXCLUDED, (
        f"too many GNSS stations excluded ({len(excluded)}); "
        f"at most {MAX_GNSS_EXCLUDED} is defensible for this network"
    )


def test_unwrap_defects_detected(submitted, answer_key):
    # The instruction allows a genuinely defective interferogram to be
    # either repaired or excluded from the inversion -- both are valid
    # science. interferograms_with_unwrapping_correction covers the first;
    # interferograms_excluded_from_inversion covers the second. A true
    # defect just has to show up in one of the two, never both.
    flagged = submitted["interferograms_with_unwrapping_correction"]
    excluded = submitted["interferograms_excluded_from_inversion"]
    assert isinstance(flagged, list)
    assert isinstance(excluded, list)
    flagged_set = set(flagged)
    excluded_set = set(excluded)
    all_ids = set(answer_key["all_interferogram_ids"])
    assert flagged_set.issubset(all_ids), f"unknown interferogram ids: {flagged_set - all_ids}"
    assert excluded_set.issubset(all_ids), f"unknown interferogram ids: {excluded_set - all_ids}"
    assert not (flagged_set & excluded_set), (
        "an interferogram cannot be both corrected and excluded: "
        f"{flagged_set & excluded_set}"
    )

    true_defects = set(answer_key["unwrap_defect_interferograms"])
    handled_set = flagged_set | excluded_set
    assert true_defects.issubset(handled_set), (
        f"must flag or exclude the genuine unwrapping-error interferograms "
        f"{true_defects}, got flagged={flagged_set}, excluded={excluded_set}"
    )
    assert len(flagged_set) <= MAX_UNWRAP_FLAGGED, (
        f"too many interferograms flagged as unwrapping-corrected ({len(flagged_set)})"
    )
    assert len(excluded_set) <= MAX_UNWRAP_EXCLUDED, (
        f"too many interferograms excluded for unwrapping ({len(excluded_set)})"
    )


def test_poisson_ratio_matches_spec(submitted):
    assert abs(float(submitted["poisson_ratio_assumed"]) - 0.25) < 1e-6, (
        "poisson_ratio_assumed must equal 0.25, the disclosed modeling assumption"
    )


# insar_gnss_ramp_coefficients is checked for format/plausibility only, not
# against a ground-truth value, by design: generate_data.py never injects a
# deliberate systematic ramp. The ramp a solution fits is absorbing the
# realization of spatially-correlated atmospheric noise near the tie epochs
# (ATMOS_CORR_LENGTH_M in generate_data.py), which differs by seed and by
# which epochs/points a given implementation happens to use for the tie --
# there is no single "true" ramp coefficient to grade against the way there
# is for x0/depth/rate. Grading it against a fabricated answer-key value
# would not test anything real; format/sanity is the correct check here.
def test_ramp_coefficients_format(submitted):
    ramp = submitted["insar_gnss_ramp_coefficients"]
    assert isinstance(ramp, dict)
    for track in ("ascending", "descending"):
        assert track in ramp, f"insar_gnss_ramp_coefficients missing '{track}'"
        track_ramp = ramp[track]
        for k in ("constant_m", "gradient_x_m_per_m", "gradient_y_m_per_m"):
            assert k in track_ramp, f"insar_gnss_ramp_coefficients['{track}'] missing '{k}'"
            v = float(track_ramp[k])
            assert v == v and abs(v) < 1e6, f"insar_gnss_ramp_coefficients['{track}']['{k}'] is not sane"


# insar_gnss_rmse_m is checked via self-report against a sanity cap, not
# independently recomputed the "exact" way, by design: a full recomputation
# would require the verifier to re-run the InSAR-to-GNSS reconciliation
# itself (which epochs/points to compare, how to project GNSS into each
# track's LOS, which stations to trust) using ONE specific method -- the
# reference solution's. That would silently fail a differently-implemented
# but scientifically valid reconciliation, rather than testing the
# submission's science. The cap still does real work: it catches a
# submission that never applied a meaningful correction at all (e.g. a
# hardcoded/placeholder RMSE, or skipping the tie step), since the max
# observed across all 8 calibration seeds was ~0.015 m against a 0.03 m cap.
# See test_insar_gnss_rmse_is_consistent_with_ramp below for a separate,
# coarser but genuinely independent cross-check against the submission's
# own disclosed ramp and used stations.
def test_insar_gnss_reconciliation_rmse_is_sane(submitted):
    rmse = float(submitted["insar_gnss_rmse_m"])
    assert rmse == rmse and rmse >= 0.0, "insar_gnss_rmse_m must be a finite, non-negative number"
    assert rmse <= RMSE_SANITY_CAP_M, (
        f"insar_gnss_rmse_m={rmse} exceeds the sanity cap ({RMSE_SANITY_CAP_M} m); "
        "this indicates the InSAR-to-GNSS reference-frame correction was not "
        "meaningfully applied"
    )


RMSE_PROXY_K_NEAREST = 8
RMSE_PROXY_BAND_FACTOR = 5.0


def _los_projection_full(ue, un, uu, incidence_deg, heading_deg):
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return (np.sin(theta) * np.cos(alpha) * ue
            - np.sin(theta) * np.sin(alpha) * un
            - np.cos(theta) * uu)


def _ramp_value_from_submission(track_ramp: dict, x: float, y: float) -> float:
    return (float(track_ramp["constant_m"])
            + float(track_ramp["gradient_x_m_per_m"]) * x
            + float(track_ramp["gradient_y_m_per_m"]) * y)


def _earliest_clean_interferogram(meta_rows, orbit, defect_ids):
    rows = [m for m in meta_rows if m["orbit"] == orbit and m["interferogram_id"] not in defect_ids]
    rows.sort(key=lambda m: (int(m["day_start"]), int(m["day_end"])))
    return rows[0]


def _gnss_disp_at_day(rows, day):
    days = np.array([float(r["day"]) for r in rows])
    order = np.argsort(days)
    days = days[order]
    e = np.array([float(r["east_disp_m"]) for r in rows])[order]
    n = np.array([float(r["north_disp_m"]) for r in rows])[order]
    u = np.array([float(r["vertical_disp_m"]) for r in rows])[order]
    return np.interp(day, days, e), np.interp(day, days, n), np.interp(day, days, u)


# This is a SEPARATE, genuinely independent cross-check on insar_gnss_rmse_m
# (see the rationale above for why a full, exact recomputation isn't done).
# It sidesteps needing to reproduce the submission's own SBAS/reconciliation
# method entirely: for each track, it takes only that track's single
# EARLIEST interferogram known (from the private answer key) to be free of
# the injected unwrapping defect, so a "cumulative-since-epoch-0" InSAR value
# at any point is just that one interferogram's raw two-epoch measurement --
# no network inversion needed. At each station the submission itself
# reports using, it averages the K nearest raw InSAR points, applies the
# submission's OWN disclosed ramp at that averaged location, and compares
# against that same station's own two-epoch GNSS displacement (day_start to
# day_end) projected into the track's LOS. Pooled across all used stations
# and both tracks, this gives a coarse proxy RMSE.
#
# This proxy is deliberately noisier than a well-implemented submission's own
# reported RMSE -- it rests on a single interferogram pair per track instead
# of a full multi-epoch reconciliation, so individual-station noise (the
# genuinely noisy-but-honest near-field stations especially) isn't averaged
# down nearly as much. Calibration against the reference solution across 3
# seeds (13, 7, 42) showed this proxy landing anywhere from ~0.6x to ~1.4x
# the reference's own reported RMSE. RMSE_PROXY_BAND_FACTOR=5 leaves roughly
# 3.5x of headroom beyond that observed spread in both directions -- loose
# enough not to penalize a differently-implemented, honest reconciliation,
# while still catching a self-report that bears no relation to the
# submission's own disclosed ramp and stations (e.g. hardcoded or copied
# from an unrelated computation).
def test_insar_gnss_rmse_is_consistent_with_ramp(submitted, answer_key, public_data):
    ramp = submitted["insar_gnss_ramp_coefficients"]
    used = submitted["gnss_stations_used"]
    submitted_rmse = float(submitted["insar_gnss_rmse_m"])

    station_xy = {
        s["station_id"]: (float(s["x_m"]), float(s["y_m"]))
        for s in public_data["gnss_stations"]
    }
    ts_by_station: dict = {}
    for r in public_data["gnss_timeseries"]:
        ts_by_station.setdefault(r["station_id"], []).append(r)

    by_ifg: dict = {}
    for r in public_data["interferograms"]:
        by_ifg.setdefault(r["interferogram_id"], []).append(r)

    defect_ids = set(answer_key["unwrap_defect_interferograms"])

    residuals = []
    for orbit in ("ascending", "descending"):
        info = _earliest_clean_interferogram(public_data["interferogram_metadata"], orbit, defect_ids)
        ifg_id = info["interferogram_id"]
        day0, day1 = int(info["day_start"]), int(info["day_end"])
        incidence, heading = float(info["incidence_deg"]), float(info["heading_deg"])
        track_ramp = ramp[orbit]

        pts = by_ifg[ifg_id]
        xs = np.array([float(p["x_m"]) for p in pts])
        ys = np.array([float(p["y_m"]) for p in pts])
        los = np.array([float(p["los_displacement_m"]) for p in pts])

        for sid in used:
            if sid not in station_xy or sid not in ts_by_station:
                continue
            sx, sy = station_xy[sid]
            dist = np.sqrt((xs - sx) ** 2 + (ys - sy) ** 2)
            idx = np.argsort(dist)[:RMSE_PROXY_K_NEAREST]
            raw_insar = float(los[idx].mean())
            px, py = float(xs[idx].mean()), float(ys[idx].mean())
            corrected = raw_insar - _ramp_value_from_submission(track_ramp, px, py)

            e0, n0, u0 = _gnss_disp_at_day(ts_by_station[sid], day0)
            e1, n1, u1 = _gnss_disp_at_day(ts_by_station[sid], day1)
            predicted_gnss = float(_los_projection_full(e1 - e0, n1 - n0, u1 - u0, incidence, heading))

            residuals.append(corrected - predicted_gnss)

    assert residuals, "could not independently cross-check insar_gnss_rmse_m: no used station matched public data"
    proxy_rmse = float(np.sqrt(np.mean(np.square(residuals))))
    floor = 0.001  # avoid a degenerate band if the proxy itself lands near zero by chance
    lo = max(proxy_rmse, floor) / RMSE_PROXY_BAND_FACTOR
    hi = max(proxy_rmse, floor) * RMSE_PROXY_BAND_FACTOR
    assert lo <= submitted_rmse <= hi, (
        f"insar_gnss_rmse_m={submitted_rmse} is inconsistent with an independent proxy "
        f"({proxy_rmse:.5f} m) recomputed from the submission's own disclosed ramp "
        f"coefficients and used stations -- expected roughly {lo:.5f} to {hi:.5f} m"
    )
