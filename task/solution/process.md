# Reference workflow: InSAR/GNSS volcano deformation rate-change
# (McTigue joint source inversion)

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
the loop should be ~0 up to ordinary noise. Per point, per edge, closures
are pooled across every triangle that edge belongs to and split into two
1-D clusters. A genuine discrete unwrapping defect shows a cluster pair
separated by close to one ambiguity cycle (the disclosed SAR system
constant), covering a spatially coherent subset of points (a k-NN
neighborhood-majority check) -- ordinary atmospheric noise does not
reproduce both properties together.

A single triangle cannot say which of its three edges is defective (the
closure is a property of the whole loop): several edges bordering the true
defect typically pass the clustering screen too, since each shares exactly
one anomalous triangle with it. This is resolved by arbitration: every
candidate edge's proposed correction is applied as a trial, and the
network-wide sum of squared closures (across every triangle, not just the
ones used to find the candidate) is recomputed. Only the genuinely
defective edge fixes every triangle it participates in; a wrong candidate
only shares one triangle with the truth and leaves any other anomalous
triangle untouched.

## 3. Per-point network inversion (SBAS) with estimated weights

Per point, per track, a weighted least-squares inversion recovers the
cumulative LOS displacement at each epoch relative to the track's first
epoch. Decorrelation drops a different, randomly-sized fraction of points
per interferogram, so each point's own subgraph of usable edges differs; an
epoch not reachable from epoch 0 through that point's own edges is left
`NaN` rather than silently reported as 0 (an unconstrained column's
minimum-norm least-squares "solution" looks like real data but is not).
Interferogram weights are estimated from an initial equal-weighted
inversion's own residuals (variance-component estimation), not assumed.

## 4. GNSS-to-InSAR reconciliation, done in InSAR's own native LOS domain

InSAR is only ever a relative measurement, and ascending/descending are two
independent products with two independent reference-frame errors. Both
facts argue against decomposing to vertical before correcting: each GNSS
station's own (E, N, U) record is instead projected into EACH track's LOS
geometry separately (los_projection_full), compared directly against that
track's own raw interpolated LOS at the station (a local first-order
planar fit to the k nearest pixels, not a plain inverse-distance average --
see the pitfalls section below for why that distinction matters). This
gives one independent residual time series per station PER TRACK.

A station is excluded only when BOTH tracks' independent residual trends
agree it is significant (|z| > 3) AND large in absolute terms over its own
record span (> 3.5 cm) -- a real local process (e.g. monument instability)
projects into both LOS geometries since the two tracks' incidence angles
are nearly identical, whereas one track's own reconstruction noise
generally does not coincidentally reproduce the same trend in the other,
independently-processed track. The magnitude condition exists because the
multi-stage reconstruction (network inversion + spatial interpolation) has
its own real, non-adversarial precision floor of a few centimeters at some
stations; requiring both significance AND magnitude, in both tracks
independently, avoids excluding ordinary reconstruction imprecision.

The remaining stations' residual intercepts are fit with a SEPARATE planar
ramp `a + b*x + c*y` per track (each track has its own independent
reference-frame/orbital error), via *weighted* least squares -- each
station's contribution is weighted by the inverse of its own intercept's
standard error, so a station that reports itself as imprecise (large
sigma) is down-weighted in this small (~9-point, 3-parameter) regression
rather than treated as equally informative as a precise one. This ramp is
then subtracted from that track's LOS field everywhere.

## 5. Joint McTigue source + two-segment rate-history inversion

A single McTigue (1987) finite spherical source -- location (x0, y0),
depth, radius, and a two-segment (unknown breakpoint) volume-rate history
-- is fit by nonlinear least squares (`scipy.optimize.least_squares`, trust
-region-reflective, box-bounded) against a stacked, weighted residual
vector: every retained GNSS station's full (E, N, U) record (weighted by
its own disclosed per-epoch sigma) plus a near-source sample of the
now-corrected InSAR LOS pixels from both tracks (weighted by each track's
own pooled reconstruction-noise estimate from step 3).

The breakpoint day is included as a continuous 7th parameter (not a
discrete grid search) so the final Jacobian captures its sensitivity too.
Fitting proceeds in two passes: a GNSS-only pass (multi-started over
several breakpoint-day initial guesses) gives a first location estimate;
the near-field InSAR sample is then drawn from around that estimate and
folded in for a second, refined pass. The initial (x0, y0) guess itself is
a displacement-magnitude-weighted centroid of the retained stations --
stations with negligible signal contribute negligible weight, which lets a
generic starting point find the source without hardcoding which stations
are "near-field."

95% confidence intervals use the delta method: the covariance of the
7 fitted parameters comes from `(J^T J)^-1` at the final solution's
Jacobian (residuals are already whitened by construction), and any derived
quantity (the vertical rate at a named station, cumulative displacement)
gets its own CI via a numerical gradient of that quantity with respect to
all 7 parameters, propagated through the same covariance matrix.

## 6. Final reconciliation

The corrected per-track LOS field is compared against each retained
station's own GNSS record (projected into that track's geometry) at the
overlapping epochs, and the pooled RMSE across both tracks and all
retained stations is reported as the final InSAR/GNSS reconciliation
check.

## Calibration (eight independent noise realizations, same scenario)

The reference solution was re-run against eight independent synthetic
realizations of the same scenario (generator seeds 1, 7, 13, 42, 99, 202,
555, 777 -- different noise draws and decorrelation masks, same true
source/rates/stations). Seed 13 is the seed shipped as the task's public
data and answer key.

| seed | x0 err (m) | y0 err (m) | depth err | radius err | rate-before err | rate-after err | t_break err | cumulative err | GNSS excluded | unwrap flagged |
|---|---|---|---|---|---|---|---|---|---|---|
| 1   | 4.0  | 10.1 | 5.23% | 8.85%  | 5.27% | 0.07% | 2.0 d | 1.16% | GNSS-03 | ASC-02, DESC-07 |
| 7   | 14.2 | 12.6 | 2.90% | 10.17% | 3.60% | 1.29% | 0.3 d | 0.17% | GNSS-03 | ASC-02, DESC-07 |
| 13  | 18.4 | 2.5  | 1.41% | 4.93%  | 2.96% | 2.03% | 4.1 d | 1.34% | GNSS-03 | ASC-02, DESC-07 |
| 42  | 2.0  | 1.1  | 0.54% | 26.25% | 0.03% | 0.91% | 2.9 d | 0.01% | GNSS-03 | ASC-02, DESC-07 |
| 99  | 2.1  | 24.5 | 3.64% | 21.13% | 7.08% | 1.11% | 6.3 d | 0.14% | GNSS-03 | ASC-02, DESC-07 |
| 202 | 52.9 | 0.2  | 0.11% | 95.00% | 0.56% | 3.27% | 3.2 d | 2.76% | GNSS-03, GNSS-05 | ASC-02, DESC-07 |
| 555 | 1.8  | 52.1 | 1.07% | 9.93%  | 3.21% | 0.11% | 2.5 d | 0.52% | GNSS-03 | ASC-02, DESC-07 |
| 777 | 13.5 | 25.0 | 6.32% | 35.11% | 7.04% | 1.48% | 3.6 d | 0.43% | GNSS-03 | ASC-02, DESC-07 |

Point-estimate tolerances in `tests/test_outputs.py` are set at roughly a
3-5x margin over the largest error observed here for every quantity except
radius (see below). The genuinely defective interferograms (ASC-02,
DESC-07) and GNSS station (GNSS-03) are correctly identified in every
realization; seed 202 additionally excludes one legitimately good
mid-field station (GNSS-05) whose own reconstruction noise happened to
cross the drift-detection threshold in that particular noise draw -- the
verifier's GNSS-exclusion check accordingly requires only that the
known-bad station is included and caps the total excluded count, rather
than requiring an exact match.

**Radius is only weakly identified by this data.** Seed 202 pins the
fitted radius at its search-space lower bound (50 m vs. a true 1000 m, a
95% error) while every other fitted quantity in that same seed lands
well within tolerance -- a textbook sign of a flat/degenerate direction in
the parameter space, not a broken fit. The delta-method CI correctly
reflects this: in that seed it reports a width of about 49,000 m, two
orders of magnitude wider than the ~50-130 m widths seen in well-behaved
seeds. Because the correction radius mostly enters as a small (order
5-25%) perturbation on top of an otherwise Mogi-like field, and the other
six free parameters (especially depth and both rates) can substantially
compensate for a misjudged radius without a large cost in fit quality, no
point-estimate tolerance on radius would both (a) accept this reference
solution's own honest behavior across all eight seeds and (b) be tight
enough to mean anything. `radius_m` is therefore checked only for format,
physical plausibility (0 < radius < depth), and CI self-consistency --
the same treatment this task gives other outputs where only the shape of
the answer, not its precise value, is verified.

## Ablations run to validate (not just assert) where the difficulty lives

Three specific "an agent might take a shortcut here" hypotheses were each
tested directly, by deliberately breaking one mechanism and comparing the
result against the properly-implemented pipeline on the same seed-13
data. Two did not hold up under measurement; one did, and the pipeline
described above reflects that finding.

- **McTigue vs. a pure-Mogi fit (radius pinned near zero).** Refitting the
  remaining 6 parameters with radius fixed at 55 m produced x0/y0/depth
  errors of the same order as the full McTigue fit (in some cases
  slightly better on individual quantities), because the other free
  parameters compensate for the missing near-field correction. This
  hypothesis is not supported by evidence and is not relied upon as a
  difficulty mechanism here.
- **Equal-weighting vs. per-station-sigma-weighting the GNSS data in the
  joint inversion.** With ~9 stations and ~1,400 near-field InSAR
  observations feeding one nonlinear fit, two stations reporting
  4-5x-elevated (but unbiased) noise did not have enough leverage to
  meaningfully change the result either way. Also not relied upon.
- **Equal-weighting vs. per-station-sigma-weighting the planar RAMP fit**
  (step 4). This one held up: on seed 13, x0 error dropped from 48.0 m
  (equal-weighted ramp) to 18.4 m (properly weighted), and the primary
  station's pre-breakpoint rate error dropped from 6.77% to 2.96% -- both
  roughly a 2-3x improvement. Unlike the joint inversion, the ramp fit has
  very little redundancy (about 9 points, 3 parameters), so a noisy
  station given equal weight there has real leverage. This is the
  validated reason `fit_ramp` in `solve.py` is a weighted, not ordinary,
  least-squares fit, and it is the mechanism the noisy near-field stations
  (GNSS-09, GNSS-10) are actually included to exercise.

The honest upshot: the task's difficulty does not rest on a single
dramatic "gotcha." It rests on getting several smaller, individually
modest-but-real judgment calls right at once (loop-closure arbitration,
per-track LOS-domain correction, weighted ramp fitting, honest treatment
of a weakly-identified parameter) well enough that the cumulative result
lands inside the calibrated tolerances above.

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
  underestimated the near-source stations' true signal, which showed up
  indirectly as an apparently "drifting" GNSS station where none existed --
  the interpolation error itself grows over time in proportion to the
  (also time-varying) true signal, mimicking exactly the kind of trend the
  GNSS quality screen in step 4 is designed to catch. Diagnosed by
  reconstructing the InSAR-derived series at a near-source station and
  comparing directly against the true model rather than only against GNSS
  (which has its own noise and could not by itself distinguish "GNSS is
  drifting" from "InSAR reconstruction is biased"). Fixed by switching to a
  local first-order (planar) interpolant, which is unbiased to first order
  for a smoothly curving field, and later compounded by discovering (via
  the same true-vs-reconstructed comparison) that combining ascending and
  descending into vertical *before* correcting each track's own reference
  offset was a second, related error -- fixed by moving the GNSS
  reconciliation into each track's native LOS domain (step 4 above).
