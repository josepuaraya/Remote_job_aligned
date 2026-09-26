# Reference workflow: InSAR/GNSS volcano deformation rate-change

## 1. Per-track SBAS network reconstruction

Ascending and descending interferograms are independent products: different
pixel clouds, different acquisition dates, different look geometry. The
network of epochs and interferogram edges for each track is reconstructed
directly from `interferogram_metadata.csv` (nothing about the network shape
is assumed or hardcoded). Both tracks carry redundant edges -- consecutive
pairs and skip-one pairs -- so every interior epoch sits inside at least one
closed triangle of interferograms.

## 2. Unwrapping-error screening via loop closure

For every closed triangle (i,j), (j,k), (i,k), the summed LOS change around
the loop should be ~0 up to ordinary noise (it is a telescoping sum of
cumulative displacement differences). Per point, per edge, closures are
pooled across every triangle that edge belongs to and split into two 1-D
clusters. A genuine discrete unwrapping defect shows a cluster pair
separated by close to one ambiguity cycle (the disclosed SAR system
constant), covering a spatially coherent subset of points (checked via a
k-NN neighborhood majority) -- ordinary atmospheric noise does not reproduce
both properties together.

A single triangle cannot say *which* of its three edges is defective (the
closure is a property of the whole loop): several edges bordering the true
defect typically pass the clustering screen too, since each shares exactly
one anomalous triangle with it. This is resolved by arbitration: every
candidate edge's proposed correction is applied as a trial, and the
network-wide sum of squared closures (across every triangle, not just the
ones used to find the candidate) is recomputed. Only the genuinely
defective edge fixes *every* triangle it participates in; a wrong candidate
only shares one triangle with the truth and leaves any other anomalous
triangle untouched. The candidate that minimizes the network-wide closure
sum-of-squares, and only if that minimum is a substantial (>50%) reduction
from the uncorrected baseline, is accepted and applied.

## 3. Per-point network inversion (SBAS) with estimated weights

Per point, per track, a weighted least-squares inversion recovers the
cumulative LOS displacement at each epoch relative to the track's first
epoch. Decorrelation drops a different, randomly-sized fraction of points
per interferogram, so each point's own subgraph of usable edges differs; an
epoch not reachable from epoch 0 through that point's own edges is left
`NaN` rather than silently reported as 0 (an unconstrained column's
minimum-norm least-squares "solution" looks like real data but is not --
this was caught during development by comparing reconstructed and true LOS
at a diagnostic point and finding some epochs pinned at exactly 0).

Interferogram weights are not assumed: an initial equal-weighted inversion's
residuals are pooled per interferogram to estimate its own noise variance
(variance-component estimation), and the inversion is re-run with those
weights. This recovers the atmospheric noise heterogeneity between
interferograms from the data's own redundancy, without needing to label any
interferogram "clean" or "noisy" in advance.

## 4. Spatial interpolation and vertical/east-west decomposition

Each track's per-epoch field is evaluated at the GNSS station coordinates
via a local first-order (planar) fit within each station's k nearest
points, not a plain inverse-distance average. This matters close to the
source: a smoothly curving field (as the true Mogi field is, especially
within a few hundred meters of the source) is systematically underestimated
by an inverse-distance average, which can never exceed the values it
averages, while a local plane fit is unbiased to first order as long as the
neighborhood is small relative to the source depth. This was caught the
same way as the inversion bug above: reconstructing the near-source
stations' time series and comparing directly against the known true model
showed the plain IDW estimate tracking only a fraction of the true rate.

The two tracks' interpolated series are then linearly aligned onto a common
day grid (restricted to each station's actual epoch overlap) and combined,
assuming negligible north-south sensitivity (a consequence of the
near-polar look geometry, matching the disclosed heading angles), into an
InSAR-derived vertical displacement time series at each station.

## 5. GNSS quality screening and the InSAR-to-GNSS reference correction

InSAR is only ever a *relative* measurement; comparing it to GNSS requires
knowing each station's InSAR-vs-GNSS offset first. Per station, the
residual (InSAR-derived vertical minus GNSS-observed vertical) is regressed
against time. A station whose residual trend is both statistically
significant (|z| > 3 against its own regression standard error) AND large
in absolute terms over its own record span (> 3.5 cm) is not explained by a
simple constant reference-frame offset -- it behaves like an independent,
unmodeled local process (e.g. monument instability) riding on top of the
volcanic signal, and is excluded before the next step. The magnitude
condition exists alongside the significance test because the multi-stage
reconstruction (network inversion + spatial interpolation + temporal
alignment) has its own real, non-adversarial precision floor of a few
centimeters at some stations; requiring both conditions avoids excluding
ordinary reconstruction imprecision while still catching a station with an
actual centimeter-per-month-scale unmodeled drift.

The remaining stations' residual *intercepts* (their reference-frame
offset, net of any drift) are fit with a planar ramp `a + b*x + c*y` via
ordinary least squares. This ramp is then subtracted from the InSAR field
everywhere, bringing it into the GNSS network's absolute reference frame --
this is the step that lets the final InSAR-vs-GNSS comparison be
meaningful at all, rather than dominated by an arbitrary per-track
reference offset.

## 6. Two-segment rate-change fit and the control-zone check

At the primary near-source station, the GNSS record and the corrected
InSAR-derived vertical series (pooled across both tracks' available
epochs) are combined into one weighted dataset and fit with a two-segment
linear model (unknown breakpoint) via grid search over the breakpoint day
(5-day steps) plus weighted linear least squares for the segment rates at
each candidate breakpoint. The same combined-series construction, but with
a single (not two-segment) linear fit, is applied at a designated far-field
station to produce the control-zone cumulative-displacement check --
if the rate-change model were (incorrectly) extrapolated to the far field,
or if noise were mistaken for signal there, this number would be far larger
than the true near-zero far-field signal.

95% confidence intervals for the rate-change parameters come from a
parametric bootstrap: synthetic residual noise is redrawn from each
observation's own estimated sigma (GNSS's disclosed formal uncertainty;
InSAR's pooled reconstruction-noise floor), and the full breakpoint-grid-
search-plus-refit is repeated per replicate.

## 7. Final reconciliation

The corrected InSAR-derived vertical series is compared against each
retained station's own GNSS record at the overlapping epochs, and the
pooled RMSE across all retained stations is reported as the final
InSAR/GNSS reconciliation check.

## Calibration (six independent noise realizations, same scenario)

The reference solution was re-run against six independent synthetic
realizations of the same scenario (generator seeds 1, 7, 13, 42, 99, 202 --
different atmospheric/GNSS noise draws and decorrelation masks, same true
source/rates/stations). Seed 13 is the seed shipped as the task's public
data and answer key.

| seed | rate-before err | rate-after err | rate-change-day err | cumulative err | control est. (true 1.57 mm) | GNSS excluded | unwrap flagged |
|---|---|---|---|---|---|---|---|
| 1   | 1.36% | 2.05% | 0 d | 0.98% | 2.98 mm | GNSS-03 | ASC-02, DESC-07 |
| 7   | 3.75% | 1.00% | 5 d | 0.71% | -0.65 mm | GNSS-03 | ASC-02, DESC-07 |
| 13  | 0.24% | 0.78% | 0 d | 0.61% | 1.62 mm | GNSS-03 | ASC-02, DESC-07 |
| 42  | 5.16% | 0.03% | 5 d | 0.55% | 0.46 mm | GNSS-03 | ASC-02, DESC-07 |
| 99  | 1.71% | 0.24% | 0 d | 0.70% | 0.74 mm | GNSS-03 | ASC-02, DESC-07 |
| 202 | 3.70% | 2.18% | 5 d | 1.53% | 0.23 mm | GNSS-03, GNSS-04, GNSS-05 | ASC-02, DESC-07 |

Point-estimate tolerances in `tests/test_outputs.py` are set at roughly a
3-5x margin over the largest error observed here (e.g. 20% vs. the observed
5.16% max for the pre-change rate; 20 days vs. the observed 5-day max for
the change day). The genuinely defective interferograms (ASC-02, DESC-07)
and GNSS station (GNSS-03) are correctly identified in every realization.
Seed 202 additionally excludes two legitimately good mid-field stations
(GNSS-04, GNSS-05) whose own reconstruction noise happened to cross the
drift-detection threshold in that particular noise draw; the verifier's
GNSS-exclusion check accordingly requires only that the known-bad station
is included and caps the total excluded count (4), rather than requiring an
exact match -- a stricter check would not survive the reference solution's
own honest seed-to-seed variability.

CI widths (as a fraction of the point estimate, or in days for the change
day) across the six seeds: rate-before 8.5-12.4%, rate-after 3.6-5.0%,
rate-change-day 10-15 days, cumulative displacement 2.1-2.7%. The verifier's
maximum-width sanity caps sit at roughly 3-4x these observed maxima, wide
enough to accept a more conservative but still genuine uncertainty method
without accepting a degenerate, uninformatively wide interval.

The verifier was also checked against seven deliberately bad submissions
(equal before/after rates with no acceleration, a maximally wide CI, all
stations excluded, no unwrapping correction reported, the known-bad station
left in, the control zone reported with the same magnitude as the volcano
signal, and an uncorrected InSAR-GNSS RMSE) -- every one is correctly
rejected.

## Two known implementation pitfalls found and fixed during development

- **Unconstrained epochs silently reported as zero.** The first version of
  the per-point network inversion solved for cumulative displacement at
  every epoch using a single design matrix sized to the whole track,
  regardless of whether a given point's own (decorrelation-thinned) edges
  actually reached that epoch. `numpy.lstsq`'s minimum-norm solution
  quietly assigns 0 to any entirely unconstrained column, which is
  indistinguishable from a real (near-zero) displacement unless checked.
  This was only caught by directly comparing a reconstructed point's time
  series against the true noise-free model and finding epochs pinned at
  exactly 0 with no gradual approach. Fixed by tracking, per point, which
  epochs are actually reachable from epoch 0 through that point's own edge
  set (a graph reachability check) and leaving the rest `NaN`.
- **Inverse-distance interpolation biased low near the source.** A plain
  IDW average from a sparse, uniformly-scattered point cloud systematically
  underestimated the near-source stations' true signal by 30-60%, which
  showed up indirectly as an apparently "drifting" GNSS station where none
  existed -- the interpolation error itself grows over time in proportion
  to the (also time-varying) true signal, mimicking exactly the kind of
  trend the GNSS quality screen in step 5 is designed to catch. Diagnosed
  by reconstructing the InSAR-derived series at a near-source station and
  comparing directly against the true model rather than only against GNSS
  (which has its own noise and could not by itself distinguish "GNSS is
  drifting" from "InSAR reconstruction is biased"). Fixed by switching to a
  local first-order (planar) interpolant, which is unbiased to first order
  for a smoothly curving field.
