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

Nothing here is a pasted expected value -- everything is recomputed from
task/tests/data/answer_key.json (the private ground truth) or from the
public data files also copied into task/tests/data/.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/workspace/output"))
TESTS_DIR = Path(os.environ.get("TESTS_DIR", "/tests"))
RESULT_PATH = OUTPUT_DIR / "result.json"
ANSWER_KEY_PATH = TESTS_DIR / "data" / "answer_key.json"

REQUIRED_KEYS = {
    "x0_m", "y0_m", "depth_m", "radius_m",
    "x0_uncertainty_95", "y0_uncertainty_95", "depth_uncertainty_95", "radius_uncertainty_95",
    "vertical_rate_before_m_per_day",
    "vertical_rate_after_m_per_day",
    "rate_change_day",
    "cumulative_vertical_displacement_m",
    "vertical_rate_before_uncertainty_95",
    "vertical_rate_after_uncertainty_95",
    "rate_change_day_uncertainty_95",
    "cumulative_vertical_displacement_uncertainty_95",
    "control_zone_cumulative_vertical_displacement_m",
    "insar_gnss_ramp_coefficients",
    "gnss_stations_used",
    "gnss_stations_excluded",
    "interferograms_with_unwrapping_correction",
    "insar_gnss_rmse_m",
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
CONTROL_ABS_TOL_M = 0.010

# CI sanity caps (max observed width/point-estimate ratio, or width in
# days, across 8 seeds).
LOCATION_CI_WIDTH_CAP_M = 800.0
DEPTH_CI_WIDTH_CAP_FRAC = 0.5
RATE_BEFORE_CI_WIDTH_CAP_FRAC = 0.5
RATE_AFTER_CI_WIDTH_CAP_FRAC = 0.3
RATE_CHANGE_DAY_CI_WIDTH_CAP = 80.0
CUMULATIVE_CI_WIDTH_CAP_FRAC = 0.15

RADIUS_CI_WIDTH_SANITY_CAP_M = 2.0e5  # radius is weakly identified (see above) and a
# genuinely honest CI can be tens of thousands of meters wide (observed up to ~50,000 m
# across 8 seeds); this only catches a degenerate placeholder (e.g. bounds of +/-1e9).
RMSE_SANITY_CAP_M = 0.03  # max observed across 8 seeds was ~0.015 m
MAX_GNSS_EXCLUDED = 4     # max observed across 8 seeds was 2
MAX_UNWRAP_FLAGGED = 4    # true count is 2; allow some slack for a
                          # differently-implemented method flagging one
                          # extra borderline interferogram


@pytest.fixture(scope="module")
def answer_key() -> dict:
    assert ANSWER_KEY_PATH.exists(), f"missing private answer key at {ANSWER_KEY_PATH}"
    return json.loads(ANSWER_KEY_PATH.read_text())


@pytest.fixture(scope="module")
def submitted() -> dict:
    assert RESULT_PATH.exists(), f"missing output: {RESULT_PATH}"
    return json.loads(RESULT_PATH.read_text())


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
    flagged = submitted["interferograms_with_unwrapping_correction"]
    assert isinstance(flagged, list)
    flagged_set = set(flagged)
    all_ids = set(answer_key["all_interferogram_ids"])
    assert flagged_set.issubset(all_ids), f"unknown interferogram ids: {flagged_set - all_ids}"

    true_defects = set(answer_key["unwrap_defect_interferograms"])
    assert true_defects.issubset(flagged_set), (
        f"must flag the genuine unwrapping-error interferograms {true_defects}, got {flagged_set}"
    )
    assert len(flagged_set) <= MAX_UNWRAP_FLAGGED, (
        f"too many interferograms flagged as unwrapping-corrected ({len(flagged_set)})"
    )


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


def test_insar_gnss_reconciliation_rmse_is_sane(submitted):
    rmse = float(submitted["insar_gnss_rmse_m"])
    assert rmse == rmse and rmse >= 0.0, "insar_gnss_rmse_m must be a finite, non-negative number"
    assert rmse <= RMSE_SANITY_CAP_M, (
        f"insar_gnss_rmse_m={rmse} exceeds the sanity cap ({RMSE_SANITY_CAP_M} m); "
        "this indicates the InSAR-to-GNSS reference-frame correction was not "
        "meaningfully applied"
    )
