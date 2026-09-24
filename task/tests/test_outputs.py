"""
test_outputs.py -- verifier for the deformation-microgravity-volcano task.

Recomputes the expected answer from the author's own private copy of the
true source parameters (tests/data/answer_key.json -- never shipped to the
agent), then checks the agent's submitted result.txt against it.

All tolerances/CI-width caps are anchored to the TRUE value from the
answer key, never to the agent's own submitted value -- this closes a
self-referential loophole found and fixed in an earlier task, where an
agent could inflate its own reported value to unlock a wider allowed
interval.

CI containment is graded with a small slack margin (see
`assert_ci_reasonable`) rather than a bare "true value in [lo, hi]" test:
a correctly calibrated 95% CI is *expected* to miss the true value on
roughly one submission in twenty even when the method and the
uncertainty propagation are entirely correct, since this task grades a
single fixed noise realization rather than averaging over repeated
draws. Treating every miss as a hard failure would occasionally fail a
genuinely correct submission for a reason that has nothing to do with
its science. The slack margin is deliberately much smaller than the CI
itself, so a systematically wrong or badly miscalibrated interval still
fails; only a near-miss from an honestly computed interval is tolerated.
The width cap is unaffected by the slack and still prevents an agent
from gaming containment by reporting an implausibly wide interval.

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
    "onset_day_estimate", "station_velocities",
    "x0_m", "x0_uncertainty_95", "y0_m", "y0_uncertainty_95",
    "depth_m", "depth_uncertainty_95",
    "volume_change_m3", "volume_change_uncertainty_95",
    "mass_change_kg", "mass_change_uncertainty_95",
    "density_kgm3", "density_uncertainty_95",
    "process_classification", "poisson_ratio_assumed",
    "flagged_gravity_benchmarks", "n_bootstrap_resamples",
}
VALID_CLASSIFICATIONS = {"magmatic_recharge", "non_magmatic_source", "magmatic_mass_loss", "unrealistic"}

DENSITY_RECHARGE_LOW = 1500.0
DENSITY_RECHARGE_HIGH = 3700.0

MIN_BOOTSTRAP_RESAMPLES = 300

# Tolerances -- calibrated against independent noise realizations of the
# author's own reference solution (see process.md for the full calibration
# log). Depth/volume are tightly constrained by the joint inversion;
# mass/density are noisier since they depend more on the lower-SNR gravity
# data, matching real joint GNSS-gravity inversion practice.
ONSET_DAY_TOL_DAYS = 15
LOCATION_ERROR_TOL_M = 90.0
LOCATION_CI_WIDTH_CAP_M = 90.0
DEPTH_REL_TOL = 0.12
DEPTH_CI_WIDTH_FRACTION = 0.08
VOLUME_REL_TOL = 0.15
VOLUME_CI_WIDTH_FRACTION = 0.12
DENSITY_CI_WIDTH_FRACTION = 0.6
MASS_CI_WIDTH_FRACTION = 0.6
MASS_REL_TOL = 0.35
VELOCITY_REL_TOL = 0.20

# Minimum CI width (as a fraction of the true value) for mass and density.
# Calibrated from an ablation of the reference solution with the spatial
# correlation length forced to ~0 (i.e. treating GNSS and gravity noise as
# independent, uncorrelated per-station values): across 3 independent noise
# seeds that naive analysis produces mass/density CIs of ~22-24% of the true
# value, versus ~37-39% for the correlation-aware analysis -- a large,
# consistent gap, because skipping the spatial + cross-dataset covariance
# makes the reported uncertainty overconfident (too narrow), not just
# differently centered. This floor sits between the two (comfortably above
# the naive ceiling, comfortably below the correlated floor observed across
# seeds), so it fails an analysis that skipped the covariance modeling on
# the actual output, rather than on a self-reported claim. x0/y0/depth/
# volume show the same direction of effect but a much smaller, noisier gap
# (~1.06-1.3x versus ~1.6-1.7x for mass/density) that isn't safely
# separable from seed-to-seed variation with only 3 calibration seeds, so
# no floor is imposed there -- only the existing maximum-width caps apply.
MASS_CI_MIN_WIDTH_FRACTION = 0.28
DENSITY_CI_MIN_WIDTH_FRACTION = 0.28

# Containment slack: fraction of the reported CI's own width that the true
# value is allowed to fall outside of and still count as "contained". This
# absorbs the ~5% chance that an honestly-calibrated 95% CI legitimately
# excludes the true value on this one fixed noise draw, without opening a
# loophole for a badly-centered or too-narrow interval (the width cap,
# applied to the RAW reported interval, is unaffected by this slack).
CI_CONTAINMENT_SLACK_FRACTION = 0.35


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


def assert_ci_reasonable(true_value, lo, hi, max_width, label, min_width=None):
    assert hi >= lo, f"{label}: reported CI is inverted [{lo}, {hi}]"
    width = hi - lo
    assert width <= max_width, (
        f"{label} CI too wide: {width:.4g} (cap {max_width:.4g}). A wide interval is not a "
        f"substitute for a real estimate of uncertainty."
    )
    if min_width is not None:
        assert width >= min_width, (
            f"{label} CI too narrow: {width:.4g} (floor {min_width:.4g}). An interval this "
            f"tight is only reachable by treating GNSS and gravity noise as independent and "
            f"ignoring their spatial and cross-dataset correlation, which understates the "
            f"true uncertainty rather than reflecting a more precise estimate."
        )
    slack = CI_CONTAINMENT_SLACK_FRACTION * width
    assert (lo - slack) <= true_value <= (hi + slack), (
        f"{label}: true value {true_value:.4g} falls outside the reported 95% CI "
        f"[{lo:.4g}, {hi:.4g}], even allowing a {CI_CONTAINMENT_SLACK_FRACTION:.0%} slack "
        f"margin of the CI's own width for legitimate sampling variability on a single "
        f"noise realization. This indicates the interval is centered or scaled wrong, not "
        f"just an unlucky draw."
    )


def test_required_keys_present(submitted):
    missing = REQUIRED_KEYS - submitted.keys()
    assert not missing, f"Missing required keys: {missing}"
    assert submitted["process_classification"] in VALID_CLASSIFICATIONS, (
        f"process_classification must be one of {VALID_CLASSIFICATIONS}, "
        f"got {submitted['process_classification']!r}"
    )


def test_poisson_ratio_matches_spec(submitted):
    assert abs(submitted["poisson_ratio_assumed"] - 0.25) < 1e-6, (
        "instruction.md specifies Poisson's ratio 0.25; submitted value does not match"
    )


def test_bootstrap_resample_count_meets_minimum(submitted):
    n = submitted["n_bootstrap_resamples"]
    assert isinstance(n, (int, float)) and not isinstance(n, bool), (
        f"n_bootstrap_resamples must be numeric, got {n!r}"
    )
    assert n >= MIN_BOOTSTRAP_RESAMPLES, (
        f"n_bootstrap_resamples={n} is below the {MIN_BOOTSTRAP_RESAMPLES} resamples "
        f"instruction.md requires for the confidence intervals"
    )


def test_onset_day_reasonable(submitted, answer_key):
    onset = submitted["onset_day_estimate"]
    true_onset = answer_key["onset_day"]
    assert abs(onset - true_onset) <= ONSET_DAY_TOL_DAYS, (
        f"onset_day_estimate={onset} is more than {ONSET_DAY_TOL_DAYS} days from "
        f"the true onset ({true_onset})"
    )


def test_process_classification_matches_truth(submitted, answer_key):
    assert submitted["process_classification"] == answer_key["true_process_classification"], (
        f"process_classification={submitted['process_classification']!r} does not match "
        f"the true classification ({answer_key['true_process_classification']!r}). This is "
        f"the central scientific judgment being tested."
    )


def test_density_equals_mass_over_volume(submitted):
    """The three reported quantities must be mutually consistent: density
    is not an independent measurement, it is mass change divided by
    volume change. A submission whose numbers don't multiply out this way
    could otherwise pass the individual mass/volume/density checks (each
    calibrated with its own tolerance) without the reported density
    actually having been derived from the reported mass and volume.
    """
    mass = submitted["mass_change_kg"]
    volume = submitted["volume_change_m3"]
    density = submitted["density_kgm3"]
    assert volume != 0, "volume_change_m3 must not be zero"
    expected_density = mass / volume
    rel_err = abs(density - expected_density) / abs(expected_density)
    assert rel_err <= 1e-3, (
        f"density_kgm3={density:.4g} is not consistent with "
        f"mass_change_kg/volume_change_m3={expected_density:.4g} "
        f"(relative difference {rel_err:.2%}); density must be derived from the "
        f"reported mass and volume, not reported independently"
    )


def test_classification_follows_from_density(submitted):
    """process_classification must be the bucket implied by the submitted
    density itself, using the thresholds given in instruction.md (0 and
    1500-3700 kg/m3 for magmatic recharge, 0-1500 for non-magmatic,
    negative for mass loss, >3700 unrealistic) -- not simply copied from
    prior knowledge of the correct answer while the reported density sits
    in a different bucket.
    """
    density = submitted["density_kgm3"]
    classification = submitted["process_classification"]
    if density < 0:
        expected = "magmatic_mass_loss"
    elif density <= DENSITY_RECHARGE_LOW:
        expected = "non_magmatic_source"
    elif density <= DENSITY_RECHARGE_HIGH:
        expected = "magmatic_recharge"
    else:
        expected = "unrealistic"
    assert classification == expected, (
        f"process_classification={classification!r} does not follow from the submitted "
        f"density_kgm3={density:.4g} (expected {expected!r} under the thresholds in "
        f"instruction.md)"
    )


def test_location_within_tolerance(submitted, answer_key):
    true_x0 = answer_key["true_x0_m"]
    true_y0 = answer_key["true_y0_m"]
    x0, y0 = submitted["x0_m"], submitted["y0_m"]
    error_m = ((x0 - true_x0) ** 2 + (y0 - true_y0) ** 2) ** 0.5
    assert error_m <= LOCATION_ERROR_TOL_M, (
        f"Horizontal location error {error_m:.1f} m exceeds tolerance {LOCATION_ERROR_TOL_M:.0f} m"
    )


def test_location_confidence_intervals_are_calibrated(submitted, answer_key):
    true_x0 = answer_key["true_x0_m"]
    true_y0 = answer_key["true_y0_m"]
    x0_lo, x0_hi = submitted["x0_uncertainty_95"]
    y0_lo, y0_hi = submitted["y0_uncertainty_95"]
    assert_ci_reasonable(true_x0, x0_lo, x0_hi, LOCATION_CI_WIDTH_CAP_M, "x0")
    assert_ci_reasonable(true_y0, y0_lo, y0_hi, LOCATION_CI_WIDTH_CAP_M, "y0")


def test_depth_within_tolerance(submitted, answer_key):
    true_depth = answer_key["true_depth_m"]
    reported_depth = submitted["depth_m"]

    rel_err = abs(reported_depth - true_depth) / true_depth
    assert rel_err <= DEPTH_REL_TOL, (
        f"Reported depth {reported_depth:.1f} m is {rel_err:.1%} off true depth "
        f"{true_depth:.1f} m (tolerance {DEPTH_REL_TOL:.0%})"
    )

    lo, hi = submitted["depth_uncertainty_95"]
    max_width = DEPTH_CI_WIDTH_FRACTION * true_depth
    assert_ci_reasonable(true_depth, lo, hi, max_width, "depth")


def test_volume_within_tolerance(submitted, answer_key):
    true_vol = answer_key["true_delta_v_m3"]
    reported_vol = submitted["volume_change_m3"]

    rel_err = abs(reported_vol - true_vol) / true_vol
    assert rel_err <= VOLUME_REL_TOL, (
        f"Reported volume {reported_vol:.0f} is {rel_err:.1%} off true volume "
        f"{true_vol:.0f} (tolerance {VOLUME_REL_TOL:.0%})"
    )

    lo, hi = submitted["volume_change_uncertainty_95"]
    max_width = VOLUME_CI_WIDTH_FRACTION * true_vol
    assert_ci_reasonable(true_vol, lo, hi, max_width, "volume")


def test_mass_within_tolerance_and_calibrated(submitted, answer_key):
    true_mass = answer_key["true_delta_m_kg"]
    reported_mass = submitted["mass_change_kg"]

    rel_err = abs(reported_mass - true_mass) / true_mass
    assert rel_err <= MASS_REL_TOL, (
        f"Reported mass {reported_mass:.0f} is {rel_err:.1%} off true mass "
        f"{true_mass:.0f} (tolerance {MASS_REL_TOL:.0%})"
    )

    lo, hi = submitted["mass_change_uncertainty_95"]
    max_width = MASS_CI_WIDTH_FRACTION * true_mass
    min_width = MASS_CI_MIN_WIDTH_FRACTION * true_mass
    assert_ci_reasonable(true_mass, lo, hi, max_width, "mass", min_width=min_width)


def test_density_confidence_interval_is_calibrated(submitted, answer_key):
    """Density itself is graded primarily via the classification check
    (test_process_classification_matches_truth) and the internal
    consistency check (test_density_equals_mass_over_volume); this check
    additionally verifies the reported uncertainty is honestly
    calibrated, not just the point estimate landing in the right bucket
    by luck.
    """
    true_density = answer_key["true_density_kgm3"]
    lo, hi = submitted["density_uncertainty_95"]
    max_width = DENSITY_CI_WIDTH_FRACTION * true_density
    min_width = DENSITY_CI_MIN_WIDTH_FRACTION * true_density
    assert_ci_reasonable(true_density, lo, hi, max_width, "density", min_width=min_width)


def test_station_velocities_present_for_all_stations(submitted):
    velocities = submitted["station_velocities"]
    assert isinstance(velocities, list) and len(velocities) == 10, (
        f"Expected 10 station velocity entries, got {len(velocities) if isinstance(velocities, list) else 'non-list'}"
    )
    for entry in velocities:
        assert "station_id" in entry and "velocity_m_per_day" in entry and "uncertainty_95" in entry


def test_station_velocities_match_truth(submitted, answer_key):
    """velocity_m_per_day is checked against the TRUE per-station rate
    (final displacement / ramp duration), not just for presence/format --
    calibrated with generous margin over the error observed from the
    reference solution's own velocity-fitting method.
    """
    true_velocities = answer_key["true_station_velocities"]
    for entry in submitted["station_velocities"]:
        sid = entry["station_id"]
        assert sid in true_velocities, f"Unknown station_id {sid!r} in station_velocities"
        true_v = true_velocities[sid]["velocity_up_m_per_day"]
        reported_v = entry["velocity_m_per_day"]
        rel_err = abs(reported_v - true_v) / abs(true_v)
        assert rel_err <= VELOCITY_REL_TOL, (
            f"Station {sid}: reported velocity {reported_v:.6f} m/day is "
            f"{rel_err:.1%} off the true rate {true_v:.6f} m/day "
            f"(tolerance {VELOCITY_REL_TOL:.0%})"
        )


def test_flagged_gravity_benchmark_identified(submitted, answer_key):
    """One gravity benchmark has a plausible hydrological anomaly unrelated
    to the volcanic source. A careful analysis should identify it (not
    blindly include every benchmark), while not flagging an excessive
    number (blanket suspicion loses valid information just as much as
    blanket acceptance).
    """
    true_flagged = answer_key.get("true_flagged_gravity_benchmark")
    submitted_flagged = set(submitted.get("flagged_gravity_benchmarks", []))

    assert true_flagged in submitted_flagged, (
        f"Expected {true_flagged!r} to be flagged as having unreliable data, "
        f"but flagged_gravity_benchmarks was {submitted_flagged}. This "
        f"benchmark has a real hydrological anomaly unrelated to the "
        f"volcanic source that biases mass/density estimates if included."
    )
    assert len(submitted_flagged) <= 2, (
        f"Flagged {len(submitted_flagged)} benchmarks ({submitted_flagged}) -- "
        f"flagging most/all benchmarks as unreliable is not a genuine QC "
        f"finding, it defeats the purpose of the check."
    )
