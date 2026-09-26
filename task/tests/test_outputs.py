"""
test_outputs.py -- verifier for the InSAR/GNSS volcano deformation
rate-change task.

Every tolerance below was calibrated by running the reference solution
(task/solution/solve.py) against six independent synthetic realizations
of the same scenario (different generator seeds) and taking roughly a
3-5x margin over the largest error/width actually observed. See
task/solution/process.md for the calibration table. Nothing here is a
pasted expected value -- everything is recomputed from
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
# error across 6 seeds was 5.16% / 2.18% / 5 days / 1.53% respectively).
RATE_BEFORE_REL_TOL = 0.20
RATE_AFTER_REL_TOL = 0.10
RATE_CHANGE_DAY_ABS_TOL = 20.0
CUMULATIVE_REL_TOL = 0.10
CONTROL_ABS_TOL_M = 0.010

# CI sanity caps (max observed width/point-estimate ratio across 6 seeds:
# 0.124 / 0.050 / 15 days / 0.027 respectively).
RATE_BEFORE_CI_WIDTH_CAP_FRAC = 0.5
RATE_AFTER_CI_WIDTH_CAP_FRAC = 0.3
RATE_CHANGE_DAY_CI_WIDTH_CAP = 60.0
CUMULATIVE_CI_WIDTH_CAP_FRAC = 0.12

RMSE_SANITY_CAP_M = 0.03  # max observed across 6 seeds was ~0.012 m
MAX_GNSS_EXCLUDED = 4     # max observed across 6 seeds was 3
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


def test_required_keys_present(submitted):
    missing = REQUIRED_KEYS - set(submitted)
    assert not missing, f"Missing required keys: {missing}"


def test_vertical_rate_before_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_vertical_rate_before_m_per_day_primary"]
    reported = float(submitted["vertical_rate_before_m_per_day"])
    rel_err = abs(reported - true_val) / abs(true_val)
    assert rel_err <= RATE_BEFORE_REL_TOL, (
        f"vertical_rate_before_m_per_day={reported} vs true={true_val} "
        f"(rel err {rel_err:.3f} > {RATE_BEFORE_REL_TOL})"
    )
    lo, hi = _ci(submitted, "vertical_rate_before_uncertainty_95")
    assert lo <= reported <= hi, "vertical_rate_before point estimate outside its own CI"
    cap = RATE_BEFORE_CI_WIDTH_CAP_FRAC * abs(reported)
    assert (hi - lo) <= cap, f"vertical_rate_before CI implausibly wide: {hi - lo} > {cap}"


def test_vertical_rate_after_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_vertical_rate_after_m_per_day_primary"]
    reported = float(submitted["vertical_rate_after_m_per_day"])
    rel_err = abs(reported - true_val) / abs(true_val)
    assert rel_err <= RATE_AFTER_REL_TOL, (
        f"vertical_rate_after_m_per_day={reported} vs true={true_val} "
        f"(rel err {rel_err:.3f} > {RATE_AFTER_REL_TOL})"
    )
    lo, hi = _ci(submitted, "vertical_rate_after_uncertainty_95")
    assert lo <= reported <= hi, "vertical_rate_after point estimate outside its own CI"
    cap = RATE_AFTER_CI_WIDTH_CAP_FRAC * abs(reported)
    assert (hi - lo) <= cap, f"vertical_rate_after CI implausibly wide: {hi - lo} > {cap}"


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
    true_val = answer_key["true_rate_change_day"]
    reported = float(submitted["rate_change_day"])
    err = abs(reported - true_val)
    assert err <= RATE_CHANGE_DAY_ABS_TOL, (
        f"rate_change_day={reported} vs true={true_val} (err {err} > {RATE_CHANGE_DAY_ABS_TOL} days)"
    )
    lo, hi = _ci(submitted, "rate_change_day_uncertainty_95")
    assert lo <= reported <= hi, "rate_change_day point estimate outside its own CI"
    assert (hi - lo) <= RATE_CHANGE_DAY_CI_WIDTH_CAP, (
        f"rate_change_day CI implausibly wide: {hi - lo} > {RATE_CHANGE_DAY_CI_WIDTH_CAP}"
    )


def test_cumulative_displacement_within_tolerance(submitted, answer_key):
    true_val = answer_key["true_cumulative_vertical_displacement_m_primary"]
    reported = float(submitted["cumulative_vertical_displacement_m"])
    rel_err = abs(reported - true_val) / abs(true_val)
    assert rel_err <= CUMULATIVE_REL_TOL, (
        f"cumulative_vertical_displacement_m={reported} vs true={true_val} "
        f"(rel err {rel_err:.3f} > {CUMULATIVE_REL_TOL})"
    )
    lo, hi = _ci(submitted, "cumulative_vertical_displacement_uncertainty_95")
    assert lo <= reported <= hi, "cumulative_vertical_displacement point estimate outside its own CI"
    cap = CUMULATIVE_CI_WIDTH_CAP_FRAC * abs(reported)
    assert (hi - lo) <= cap, f"cumulative_vertical_displacement CI implausibly wide: {hi - lo} > {cap}"


def test_control_zone_shows_no_meaningful_deformation(submitted, answer_key):
    """The control station sits far enough from the source that the true
    signal there is ~1.6 mm over the whole record -- a submission that
    (incorrectly) extrapolates the near-source rate-change model to the
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
    for k in ("constant_m", "gradient_x_m_per_m", "gradient_y_m_per_m"):
        assert k in ramp, f"insar_gnss_ramp_coefficients missing '{k}'"
        v = float(ramp[k])
        assert v == v and abs(v) < 1e6, f"insar_gnss_ramp_coefficients['{k}'] is not a sane finite number"


def test_insar_gnss_reconciliation_rmse_is_sane(submitted):
    rmse = float(submitted["insar_gnss_rmse_m"])
    assert rmse == rmse and rmse >= 0.0, "insar_gnss_rmse_m must be a finite, non-negative number"
    assert rmse <= RMSE_SANITY_CAP_M, (
        f"insar_gnss_rmse_m={rmse} exceeds the sanity cap ({RMSE_SANITY_CAP_M} m); "
        "this indicates the InSAR-to-GNSS reference-frame correction was not "
        "meaningfully applied"
    )
