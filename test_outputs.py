"""
test_outputs.py -- verifier for the insar-volcano-inversion task (v5).

Recomputes the expected answer from the author's own private copy of the
true source parameters (tests/data/answer_key.json -- never shipped to the
agent), then checks the agent's submitted result.txt against it.

All tolerances/CI-width caps are anchored to the TRUE value from the
answer key, never to the agent's own submitted value.

v4 change: adds an independent GPS-reconciliation check. The InSAR-only
tolerances below (location/depth/volume) are deliberately loose because
InSAR alone cannot tightly separate depth from cumulative volume change
in this geometry (domain half-width comparable to source depth). The GPS
station is unaffected by atmospheric noise and gives a genuinely
independent 3-component measurement; a source estimate must predict that
station's observed record to within a tight tolerance, not merely produce
small InSAR residuals. This is checked two ways: (1) self-consistency --
the agent's own reported gps_predicted_displacement_final_m must match
what its own submitted x0/y0/depth/volume_change_m3 actually predict
(catches a fabricated or un-updated number); (2) accuracy -- that
prediction must be close to the true (noise-free) GPS displacement.

v5 change: the InSAR atmospheric noise and the GNSS common-mode error are
now genuinely correlated (see generate_insar_data.py/solve.py), and the
agent must report its own estimate of both covariance structures
(insar_covariance_estimate, gnss_covariance_estimate). These are graded
loosely and deliberately: calibration against the reference solution
showed that a short-lag semivariogram fit to this
amount of data is only weakly identified (the fitted correlation length
tracks the optimizer's starting guess across a wide range, a known
geostatistics identifiability issue, not a defect in a particular
implementation) -- so the checks here only confirm the agent produced a
plausible, non-placeholder estimate, not a tight numeric match. The real
signal from this addition is (a) the required fields themselves, which a
naive per-point-independent fit is unlikely to produce sensible values
for at all, and (b) that the downstream point estimates/CIs (already
checked above) remain accurate under a joint fit that properly accounts
for the correlation, which was validated across 3 noise seeds.

Run with: pytest test_outputs.py
"""

import json
import math
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
    "gps_predicted_displacement_final_m",
    "insar_covariance_estimate",
    "gnss_covariance_estimate",
    "validation_interferogram_predicted_los_m",
    "vertical_east_west_decomposition_sample",
}

ALL_IDS = {f"ASC-{i:02d}" for i in range(1, 7)} | {f"DESC-{i:02d}" for i in range(1, 6)}

# Tolerances -- calibrated against 3 independent noise realizations of the
# author's own reference solution.
# This task is genuinely harder than an earlier design iteration: the
# joint fit combines many interferograms of varying
# duration (including one spanning a real ~70-day acquisition gap), and
# the formal (delta-method) uncertainty was found to underestimate the
# true error, requiring an empirically calibrated safety margin -- the
# caps below already include that margin.
LOCATION_ERROR_TOL_M = 200.0       # InSAR-only geometry under-resolves x0,y0 tightly
DEPTH_REL_TOL = 0.10               # InSAR-only geometry under-resolves depth tightly
VOLUME_REL_TOL = 0.10              # correlated with depth via the trade-off above
MAX_EXCLUDED_COUNT = 6              # observed 3-4 (true unrepairable=3, plus occasional conservative extra) across 5 seeds, with 11 total interferograms

# GPS reconciliation has two independent checks with different jobs:
# - Self-consistency is the primary, robust check: it is pure arithmetic
#   (does the submitted x0/y0/depth/volume_change_m3 actually predict the
#   submitted gps_predicted_displacement_final_m?), independent of noise
#   or methodology, so the tolerance is just numerical-precision-tight.
#   This is what catches an agent that reports a GPS-shaped number
#   without ever deriving it from its own fitted source (e.g. copying the
#   observed GPS record, or leaving a stale value from an InSAR-only fit).
# - Accuracy against the true record is a secondary backstop, deliberately
#   loose: calibrated runs show reasonable joint
#   InSAR+GPS weighting choices land in the ~0.2-1.3cm range on the
#   shipped dataset, while ignoring GPS entirely lands at ~1.5-4cm
#   depending on noise realization. A tight cm-level cutoff here would
#   also penalize legitimate differences in how much an agent chooses to
#   weight the GPS record, so this only catches a grossly wrong source.
GPS_SELF_CONSISTENCY_TOL_M = 0.001   # 1mm: submitted params must actually predict the submitted GPS number
GPS_ACCURACY_TOL_M = 0.05            # 5cm: generous backstop against a grossly wrong source

# Covariance-estimate plausibility bounds -- deliberately wide (see module
# docstring): calibration runs against the reference solution, sweeping the
# fit's own starting guess, landed anywhere in ~100-1500m (correlation
# length) and ~4e-6-1.1e-5 m^2 (sill) purely from the short-lag data's own
# identifiability limits -- these bounds are set with real margin around
# that spread, wide enough not to penalize a well-reasoned fit for landing
# on a different (but equally defensible) point of that ridge, while still
# catching a placeholder, zero, or wildly implausible value.
INSAR_CORR_LENGTH_BOUNDS_M = (30.0, 2500.0)
INSAR_SILL_BOUNDS_M2 = (5e-7, 5e-5)
GNSS_SIGMA_BOUNDS_M = (0.0003, 0.02)

# Held-out validation interferogram (VAL-01, not part of the sequential
# fitting stack) -- same two-check pattern as GPS reconciliation:
# self-consistency (pure arithmetic, tight) plus a generous accuracy
# backstop.
VAL_SELF_CONSISTENCY_TOL_M = 0.001   # 1mm: submitted params must actually predict the submitted value
VAL_ACCURACY_TOL_M = 0.02            # 2cm: generous backstop -- VAL-01 carries only small correlated noise


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


def mogi_displacement_enu(x, y, depth, delta_v_m3, x0, y0, nu=0.25):
    """Independent reimplementation (not imported from solve.py) used to
    recompute predictions from the agent's own submitted parameters."""
    dx, dy = x - x0, y - y0
    r = math.sqrt(dx**2 + dy**2)
    R = math.sqrt(r**2 + depth**2)
    C = (1 - nu) * delta_v_m3 / math.pi
    u_z = C * depth / R**3
    u_r = C * r / R**3
    theta = math.atan2(dy, dx)
    u_x = u_r * math.cos(theta)
    u_y = u_r * math.sin(theta)
    return u_x, u_y, u_z


def los_projection_full(u_e, u_n, u_u, incidence_deg, heading_deg):
    """Independent reimplementation (not imported from solve.py)."""
    theta = math.radians(incidence_deg)
    alpha = math.radians(heading_deg)
    return (math.sin(theta) * math.cos(alpha) * u_e
            - math.sin(theta) * math.sin(alpha) * u_n
            - math.cos(theta) * u_u)


def predicted_validation_los(x0, y0, depth, volume_change_m3, answer_key, onset_day=60.0, record_last_day=364.0):
    """Recompute what the submitted source parameters predict for the
    held-out validation interferogram's LOS displacement, deriving the
    implied constant post-onset rate from volume_change_m3 (disclosed to
    be cumulative from day 0 to record_last_day), exactly as solve.py does.
    """
    rate = volume_change_m3 / max(record_last_day - onset_day, 1e-9)
    win = answer_key["validation_interferogram_window"]
    point = answer_key["validation_interferogram_evaluation_point_m"]

    def eff_dur(day):
        return max(0.0, day - onset_day)

    dV0 = rate * eff_dur(win["day_start"])
    dV1 = rate * eff_dur(win["day_end"])
    ux0, uy0, uz0 = mogi_displacement_enu(point["x_m"], point["y_m"], depth, dV0, x0, y0)
    ux1, uy1, uz1 = mogi_displacement_enu(point["x_m"], point["y_m"], depth, dV1, x0, y0)
    return los_projection_full(ux1 - ux0, uy1 - uy0, uz1 - uz0, win["incidence_deg"], win["heading_deg"])


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


def test_gps_prediction_matches_submitted_source(submitted, answer_key):
    """The agent's own reported gps_predicted_displacement_final_m must
    actually be what its own submitted x0/y0/depth/volume_change_m3
    predict at the GPS station -- catches a number that was copied from
    the raw GPS record, left over from an earlier fit, or otherwise not
    genuinely derived from the final submitted source parameters.
    """
    gps_loc = answer_key["gps_location_m"]
    ux, uy, uz = mogi_displacement_enu(
        gps_loc["x_m"], gps_loc["y_m"],
        submitted["depth_m"], submitted["volume_change_m3"],
        submitted["x0_m"], submitted["y0_m"],
    )
    reported = submitted["gps_predicted_displacement_final_m"]
    for comp, predicted in (("east_m", ux), ("north_m", uy), ("vertical_m", uz)):
        assert comp in reported, f"gps_predicted_displacement_final_m missing '{comp}'"
        err = abs(reported[comp] - predicted)
        assert err <= GPS_SELF_CONSISTENCY_TOL_M, (
            f"gps_predicted_displacement_final_m.{comp} ({reported[comp]:.4f} m) does not match "
            f"what the submitted x0/y0/depth/volume_change_m3 actually predict ({predicted:.4f} m) "
            f"-- off by {err*1000:.1f} mm, tolerance {GPS_SELF_CONSISTENCY_TOL_M*1000:.0f} mm"
        )


def test_gps_prediction_matches_true_record(submitted, answer_key):
    """The submitted source must reconcile with the independent GPS
    station -- but this is a generous backstop (5cm), not a tight check,
    despite the GPS station's own measurement noise being small (mm-level)
    and uncorrelated with the InSAR atmospheric noise. Calibration found
    the depth/volume trade-off this targets is real but
    modest, and itself sensitive to legitimate differences in how an
    agent weights the GPS record against the InSAR data -- a tight
    cm-level cutoff would risk failing a genuinely correct joint
    inversion over an implementation choice. This only catches a grossly
    wrong source; the primary, tight check is self-consistency
    (test_gps_prediction_matches_submitted_source, 1mm), not this one.
    """
    true_final = answer_key["true_gps_final_displacement_m"]
    reported = submitted["gps_predicted_displacement_final_m"]
    for comp in ("east_m", "north_m", "vertical_m"):
        err = abs(reported[comp] - true_final[comp])
        assert err <= GPS_ACCURACY_TOL_M, (
            f"gps_predicted_displacement_final_m.{comp} ({reported[comp]:.4f} m) is {err*100:.1f} cm "
            f"off the true GPS-station displacement ({true_final[comp]:.4f} m), tolerance "
            f"{GPS_ACCURACY_TOL_M*100:.0f} cm -- the submitted source parameters do not reconcile "
            f"with the independent GPS record."
        )


def test_insar_covariance_estimate_is_plausible(submitted):
    """Loosely graded (see module docstring): the fitted correlation
    length/sill are only weakly identified from this amount of data, so
    this only confirms the agent produced a genuine, plausible estimate
    -- not that it tightly matches the true generator value.
    """
    est = submitted["insar_covariance_estimate"]
    assert "correlation_length_m" in est and "sill_m2" in est, (
        "insar_covariance_estimate must have 'correlation_length_m' and 'sill_m2'"
    )
    lo, hi = INSAR_CORR_LENGTH_BOUNDS_M
    assert lo <= est["correlation_length_m"] <= hi, (
        f"insar_covariance_estimate.correlation_length_m ({est['correlation_length_m']}) "
        f"is outside the plausible range [{lo}, {hi}] m"
    )
    lo, hi = INSAR_SILL_BOUNDS_M2
    assert lo <= est["sill_m2"] <= hi, (
        f"insar_covariance_estimate.sill_m2 ({est['sill_m2']}) is outside the plausible range [{lo}, {hi}] m^2"
    )


def test_gnss_covariance_estimate_is_plausible(submitted):
    """Same loose-grading rationale as test_insar_covariance_estimate_is_plausible."""
    est = submitted["gnss_covariance_estimate"]
    assert "common_mode_sigma_m" in est and "white_sigma_m" in est, (
        "gnss_covariance_estimate must have 'common_mode_sigma_m' and 'white_sigma_m'"
    )
    lo, hi = GNSS_SIGMA_BOUNDS_M
    assert lo <= est["common_mode_sigma_m"] <= hi, (
        f"gnss_covariance_estimate.common_mode_sigma_m ({est['common_mode_sigma_m']}) "
        f"is outside the plausible range [{lo}, {hi}] m"
    )
    assert lo <= est["white_sigma_m"] <= hi, (
        f"gnss_covariance_estimate.white_sigma_m ({est['white_sigma_m']}) "
        f"is outside the plausible range [{lo}, {hi}] m"
    )


def test_validation_interferogram_self_consistent(submitted, answer_key):
    """The agent's own reported validation_interferogram_predicted_los_m
    must actually be what its own submitted x0/y0/depth/volume_change_m3
    predict for the held-out validation interferogram (VAL-01, not part
    of the sequential fitting stack) -- pure arithmetic, so the tolerance
    is numerical-precision-tight. Catches a value that was not genuinely
    derived from the final submitted source parameters (e.g. copied from
    the observed VAL-01 record, or never actually computed).
    """
    predicted = predicted_validation_los(
        submitted["x0_m"], submitted["y0_m"], submitted["depth_m"], submitted["volume_change_m3"], answer_key,
    )
    reported = submitted["validation_interferogram_predicted_los_m"]
    err = abs(reported - predicted)
    assert err <= VAL_SELF_CONSISTENCY_TOL_M, (
        f"validation_interferogram_predicted_los_m ({reported:.4f} m) does not match what the "
        f"submitted x0/y0/depth/volume_change_m3 actually predict for VAL-01 ({predicted:.4f} m) "
        f"-- off by {err*1000:.1f} mm, tolerance {VAL_SELF_CONSISTENCY_TOL_M*1000:.0f} mm"
    )


def test_validation_interferogram_accuracy(submitted, answer_key):
    """The submitted source must predict the held-out validation
    interferogram reasonably well -- a generous backstop (VAL-01 carries
    only small correlated noise, no atmospheric contamination), not the
    primary discriminator. Primarily exercises predictive validation: was
    the fitted model ever checked against a genuinely unseen observation,
    not just the data it was fit to.
    """
    predicted = submitted["validation_interferogram_predicted_los_m"]
    true_val = answer_key["true_validation_interferogram_los_m"]
    err = abs(predicted - true_val)
    assert err <= VAL_ACCURACY_TOL_M, (
        f"validation_interferogram_predicted_los_m ({predicted:.4f} m) is {err*100:.1f} cm off "
        f"the true VAL-01 displacement ({true_val:.4f} m), tolerance {VAL_ACCURACY_TOL_M*100:.0f} cm"
    )


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
