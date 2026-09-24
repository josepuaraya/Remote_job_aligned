"""
test_outputs.py -- verifier for the insar-volcano-inversion task (v2).

Recomputes the expected answer from the author's own private copy of the
true source parameters (tests/data/answer_key.json -- never shipped to the
agent), then checks the agent's submitted result.txt against it.

All tolerances/CI-width caps are anchored to the TRUE value from the
answer key, never to the agent's own submitted value.

Run with: pytest test_outputs.py
"""

import json
import os
import pytest

OUTPUT_PATH = os.environ.get("TASK_OUTPUT_PATH", "/workspace/output/result.txt")
ANSWER_KEY_PATH = os.environ.get(
    "TASK_ANSWER_KEY_PATH", os.path.join(os.path.dirname(__file__), "data", "answer_key.json")
)

REQUIRED_KEYS = {
    "interferograms_passed_filter", "interferograms_excluded", "interferograms_used_in_inversion",
    "x0_m", "x0_uncertainty_95", "y0_m", "y0_uncertainty_95",
    "depth_m", "depth_uncertainty_95",
    "volume_change_m3", "volume_change_uncertainty_95",
    "poisson_ratio_assumed",
    "vertical_east_west_decomposition_sample",
}

ALL_IDS = {f"ASC-{i:02d}" for i in range(1, 11)} | {f"DESC-{i:02d}" for i in range(1, 11)}

# Tolerances -- calibrated against 3 independent noise realizations of the
# author's own reference solution (see process.md for the calibration log).
# This task is genuinely harder than an earlier design iteration (see
# process.md): the joint fit combines many interferograms of varying
# duration (including one spanning a real ~70-day acquisition gap), and
# the formal (delta-method) uncertainty was found to underestimate the
# true error, requiring an empirically calibrated safety margin -- the
# caps below already include that margin.
LOCATION_ERROR_TOL_M = 200.0       # observed max ~64m point error, ~3x margin
DEPTH_REL_TOL = 0.10               # observed max 2.44% point error, ~4x margin
VOLUME_REL_TOL = 0.10              # observed max 2.29% point error, ~4x margin
MAX_EXCLUDED_COUNT = 10            # observed exactly 6 (perfect match) across 3 seeds, generous margin


@pytest.fixture(scope="module")
def answer_key():
    with open(ANSWER_KEY_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def submitted():
    assert os.path.exists(OUTPUT_PATH), f"Expected output file not found at {OUTPUT_PATH}"
    with open(OUTPUT_PATH) as f:
        content = f.read()
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        pytest.fail(f"result.txt is not valid JSON: {e}")


def test_required_keys_present(submitted):
    missing = REQUIRED_KEYS - submitted.keys()
    assert not missing, f"Missing required keys: {missing}"


def test_poisson_ratio_matches_spec(submitted):
    assert abs(submitted["poisson_ratio_assumed"] - 0.25) < 1e-6, (
        "instruction.md specifies Poisson's ratio 0.25; submitted value does not match"
    )


def test_unrepairable_interferograms_excluded(submitted, answer_key):
    """Every genuinely unrepairable interferogram must be excluded (blanket
    acceptance loses information); the total excluded count must stay
    reasonable (blanket rejection also loses information).
    """
    true_unrepairable = set(answer_key["unrepairable_interferograms"])
    submitted_excluded = set(submitted.get("interferograms_excluded", []))

    assert true_unrepairable.issubset(submitted_excluded), (
        f"Expected {true_unrepairable} to be excluded (genuinely unrepairable "
        f"atmospheric noise), but interferograms_excluded was {submitted_excluded}."
    )
    assert len(submitted_excluded) <= MAX_EXCLUDED_COUNT, (
        f"Excluded {len(submitted_excluded)} interferograms -- excluding "
        f"most/all of them is not a genuine QC finding."
    )
    assert submitted_excluded.issubset(ALL_IDS), (
        f"interferograms_excluded contains unexpected IDs: {submitted_excluded - ALL_IDS}"
    )


def test_used_in_inversion_is_consistent(submitted):
    """The interferograms actually used to derive the source parameters
    must be a non-empty subset of those that passed the QC filter.
    """
    passed = set(submitted.get("interferograms_passed_filter", []))
    used = set(submitted.get("interferograms_used_in_inversion", []))
    assert used, "interferograms_used_in_inversion must not be empty"
    assert used.issubset(passed), (
        f"interferograms_used_in_inversion ({used}) must be a subset of "
        f"interferograms_passed_filter ({passed})"
    )


def test_location_within_tolerance(submitted, answer_key):
    true_x0 = answer_key["true_x0_m"]
    true_y0 = answer_key["true_y0_m"]
    x0, y0 = submitted["x0_m"], submitted["y0_m"]
    error_m = ((x0 - true_x0) ** 2 + (y0 - true_y0) ** 2) ** 0.5
    assert error_m <= LOCATION_ERROR_TOL_M, (
        f"Horizontal location error {error_m:.1f} m exceeds tolerance {LOCATION_ERROR_TOL_M:.0f} m"
    )


def test_location_confidence_intervals_are_reasonable(submitted):
    """No tight width cap is enforced: a genuinely honest uncertainty
    method (e.g. one that properly accounts for spatially correlated
    atmospheric noise) could legitimately produce a wider interval than
    this task's own reference, and should not be penalized for being
    more conservative. This checks self-consistency (the point estimate
    falls within its own interval) plus a very generous sanity bound
    (width no more than 10x the point-estimate tolerance) that only
    catches a degenerate, uninformatively wide interval reported to
    trivially "pass" (e.g. [-1e9, 1e9]).
    """
    x0, y0 = submitted["x0_m"], submitted["y0_m"]
    x0_lo, x0_hi = submitted["x0_uncertainty_95"]
    y0_lo, y0_hi = submitted["y0_uncertainty_95"]
    assert x0_lo < x0_hi, "x0_uncertainty_95 must have lower < upper"
    assert y0_lo < y0_hi, "y0_uncertainty_95 must have lower < upper"
    assert x0_lo <= x0 <= x0_hi, f"x0 point estimate ({x0}) falls outside its own reported CI [{x0_lo}, {x0_hi}]"
    assert y0_lo <= y0 <= y0_hi, f"y0 point estimate ({y0}) falls outside its own reported CI [{y0_lo}, {y0_hi}]"
    sanity_cap = 10 * LOCATION_ERROR_TOL_M
    assert (x0_hi - x0_lo) <= sanity_cap, f"x0 CI implausibly wide: {x0_hi - x0_lo:.1f} m"
    assert (y0_hi - y0_lo) <= sanity_cap, f"y0 CI implausibly wide: {y0_hi - y0_lo:.1f} m"


def test_depth_within_tolerance(submitted, answer_key):
    true_depth = answer_key["true_depth_m"]
    reported_depth = submitted["depth_m"]

    rel_err = abs(reported_depth - true_depth) / true_depth
    assert rel_err <= DEPTH_REL_TOL, (
        f"Reported depth {reported_depth:.1f} m is {rel_err:.1%} off true depth "
        f"{true_depth:.1f} m (tolerance {DEPTH_REL_TOL:.0%})"
    )

    # No absolute width cap (see test_location_confidence_intervals_are_reasonable
    # for rationale) -- only self-consistency: the point estimate must fall
    # within its own reported interval.
    lo, hi = submitted["depth_uncertainty_95"]
    assert lo < hi, "depth_uncertainty_95 must have lower < upper"
    assert lo <= reported_depth <= hi, (
        f"depth_m ({reported_depth}) falls outside its own reported CI [{lo}, {hi}]"
    )
    sanity_cap = 10 * DEPTH_REL_TOL * true_depth
    assert (hi - lo) <= sanity_cap, f"Depth CI implausibly wide: {hi-lo:.2f} m (sanity cap {sanity_cap:.2f} m)"


def test_volume_within_tolerance(submitted, answer_key):
    """volume_change_m3 refers to the cumulative volume change from day 0
    to the last acquisition day, using the KNOWN/disclosed onset day and
    a linear (constant-rate) deformation model after onset.
    """
    true_vol = answer_key["true_cumulative_delta_v_m3"]
    reported_vol = submitted["volume_change_m3"]

    rel_err = abs(reported_vol - true_vol) / true_vol
    assert rel_err <= VOLUME_REL_TOL, (
        f"Reported volume {reported_vol:.0f} is {rel_err:.1%} off true cumulative volume "
        f"{true_vol:.0f} (tolerance {VOLUME_REL_TOL:.0%})"
    )

    lo, hi = submitted["volume_change_uncertainty_95"]
    assert lo < hi, "volume_change_uncertainty_95 must have lower < upper"
    assert lo <= reported_vol <= hi, (
        f"volume_change_m3 ({reported_vol}) falls outside its own reported CI [{lo}, {hi}]"
    )
    sanity_cap = 10 * VOLUME_REL_TOL * true_vol
    assert (hi - lo) <= sanity_cap, f"Volume CI implausibly wide: {hi-lo:.0f} (sanity cap {sanity_cap:.0f})"


def test_vertical_east_west_decomposition_format(submitted):
    """Descriptive deliverable -- checked for presence and sane
    structure/format only, not tightly toleranced.
    """
    decomp = submitted["vertical_east_west_decomposition_sample"]
    assert isinstance(decomp, list) and len(decomp) >= 5, (
        "vertical_east_west_decomposition_sample should be a list of several sample points"
    )
    for entry in decomp:
        assert "x_m" in entry and "vertical_m" in entry and "east_west_m" in entry
