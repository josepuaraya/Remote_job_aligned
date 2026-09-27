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
record span (> 5 cm, `DRIFT_MAGNITUDE_FLOOR_M` in solve.py) -- a real local
process (e.g. monument instability)
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

95% confidence intervals come from a parametric bootstrap (300 replicates),
not a delta method -- see the calibration section below for why an earlier
delta-method version (covariance from `(J^T J)^-1` at the final fit's
Jacobian) was replaced. Each replicate redraws noise at the raw-data level
-- each interferogram's own estimated variance (from step 3's
variance-component estimation), each GNSS observation's own disclosed
sigma -- and reruns the entire pipeline above (SBAS reconstruction,
unwrapping screen, GNSS screen, ramp fit, joint inversion) on the
perturbed data. The reported CI for every quantity is the 2.5th/97.5th
percentile of that replicate distribution. `run_full_pipeline()` is the
single function both the point estimate and every bootstrap replicate
call, so the two can never silently diverge.

## 6. Final reconciliation

The corrected per-track LOS field is compared against each retained
station's own GNSS record (projected into that track's geometry) at the
overlapping epochs, and the pooled RMSE across both tracks and all
retained stations is reported as the final InSAR/GNSS reconciliation
check.

**Why `insar_gnss_ramp_coefficients` and `insar_gnss_rmse_m` are graded as
format/sanity checks, not against a ground-truth value.** `generate_data.py`
never injects a deliberate systematic ramp -- the ramp above is fitted to
absorb the realization of spatially-correlated atmospheric noise near the
tie epochs (`ATMOS_CORR_LENGTH_M`), which differs by seed and by which
epochs/points a given implementation happens to use for the tie. There is
no single "true" ramp coefficient the way there is a true `x0_m` or
`depth_m`, so the verifier checks format and plausibility only
(`test_ramp_coefficients_format`). Likewise, independently recomputing
`insar_gnss_rmse_m` in the verifier would require re-running the
InSAR-to-GNSS reconciliation using one specific method -- the reference
solution's choice of epochs, station screening, and LOS projection -- which
would fail a differently-implemented but scientifically valid
reconciliation rather than testing the submission's science. The verifier
instead checks the self-reported RMSE against a sanity cap
(`test_insar_gnss_reconciliation_rmse_is_sane`, `RMSE_SANITY_CAP_M = 0.03`
m vs. ~0.015 m observed at worst across the 8 calibration seeds), which
still catches a submission that skipped the tie step or hardcoded a
placeholder value.

## Calibration (eight independent noise realizations, same scenario)

The reference solution was re-run against eight independent synthetic
realizations of the same scenario (generator seeds 1, 7, 13, 42, 99, 202,
555, 777 -- different noise draws and decorrelation masks, same true
source/rates/stations). Seed 13 is the seed shipped as the task's public
data and answer key.

| seed | x0 err (m) | y0 err (m) | depth err | radius err | rate-before err | rate-after err | t_break err | cumulative err | volume-change err | GNSS excluded | unwrap flagged |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1   | 4.0  | 10.1 | 5.23% | 8.85%  | 5.27% | 0.07% | 2.0 d | 1.16% | 11.21% | GNSS-03 | ASC-02, DESC-07 |
| 7   | 14.2 | 12.6 | 2.90% | 10.17% | 3.60% | 1.29% | 0.3 d | 0.17% | 4.72%  | GNSS-03 | ASC-02, DESC-07 |
| 13  | 18.4 | 2.5  | 1.41% | 4.93%  | 2.96% | 2.03% | 4.1 d | 1.34% | 3.61%  | GNSS-03 | ASC-02, DESC-07 |
| 42  | 2.0  | 1.1  | 0.54% | 26.25% | 0.03% | 0.91% | 2.9 d | 0.01% | 4.69%  | GNSS-03 | ASC-02, DESC-07 |
| 99  | 2.1  | 24.5 | 3.64% | 21.13% | 7.08% | 1.11% | 6.3 d | 0.14% | 3.64%  | GNSS-03 | ASC-02, DESC-07 |
| 202 | 52.9 | 0.2  | 0.11% | 95.00% | 0.56% | 3.27% | 3.2 d | 2.76% | 3.52%  | GNSS-03, GNSS-05 | ASC-02, DESC-07 |
| 555 | 1.8  | 52.1 | 1.07% | 9.93%  | 3.21% | 0.11% | 2.5 d | 0.52% | 0.18%  | GNSS-03 | ASC-02, DESC-07 |
| 777 | 13.5 | 25.0 | 6.32% | 35.11% | 7.04% | 1.48% | 3.6 d | 0.43% | 6.74%  | GNSS-03 | ASC-02, DESC-07 |

`volume_change_m3` is the McTigue source's total cumulative volume change
(day 0 to the last day of record) -- the same `cumulative_dv_at(...)`
quantity already computed internally to get the primary station's
cumulative vertical displacement, simply reported directly rather than
projected through a station's geometry. Its tolerance (40%, ~4x the
observed 11.21% max) is looser than the station-level cumulative
displacement's, because it inherits the same rate/breakpoint uncertainty
without benefiting from cancellation against a specific station's own
sensitivity factor.

Point-estimate tolerances in `tests/test_outputs.py` are set at roughly a
3-5x margin over the largest error observed here for every quantity except
radius (see below).

**The uncertainty method itself went through two revisions, both caught by
questions asked of this document rather than by calibration alone.**

*Revision 1 (stale caps).* The joint McTigue nonlinear fit initially used
delta-method CIs (covariance from `(J^T J)^-1` at the final fit's
Jacobian). These are much tighter than an earlier per-station bootstrap
design's widths, and the CI sanity caps -- written against that earlier
design and never re-measured after the rewrite -- ended up 30-140x looser
than the delta-method pipeline actually produced (e.g. an 800 m
location-CI cap against an actual observed maximum of 6.7 m). Caught by
review, fixed by re-measuring every cap directly against the delta-method
output.

*Revision 2 (the delta method itself was wrong).* Re-measuring the caps
made the next question obvious: does the true value actually fall inside
these "95%" intervals close to 95% of the time? Checked directly across
the 8 calibration seeds, the delta-method CIs' true coverage was far
below nominal -- depth 1/8, rate-before 1/8, rate-change-day 1/8, volume
change 1/8, x0 3/8, y0 3/8, radius 2/8, rate-after 2/8, cumulative
displacement 3/8. The delta method only reflects uncertainty conditional
on the SBAS reconstruction and ramp fit being exact, known inputs --
structurally, it cannot see the uncertainty those upstream steps
themselves carry, which is why it was consistently too narrow. Replaced
with the full-pipeline parametric bootstrap described above (resample
raw-data noise, rerun everything, take percentiles), which does propagate
that upstream uncertainty. Re-measured coverage after the fix, same 8
seeds:

| quantity | coverage (of 8) | max CI width observed | verifier cap | margin |
|---|---|---|---|---|
| x0 | 7/8 | 70.2 m | 250 m | ~3.6x |
| y0 | 5/8 | 25.4 m | 250 m | ~9.8x |
| depth | 4/8 | 8.8% of point | 30% | ~3.4x |
| radius | 5/8 | up to ~1770% of point (seed 202) | 200,000 m | n/a |
| rate-before | 6/8 | 17.4% of point | 55% | ~3.2x |
| rate-after | 2/8 | 2.3% of point | 30% | n/a (see below) |
| rate-change day | 6/8 | 13.9 d | 50 d | ~3.6x |
| cumulative displacement | 7/8 | 3.4% of point | 12% | ~3.5x |
| volume change | 5/8 | 16.6% of point | 55% | ~3.3x |

This is a large, real improvement over the delta method (which achieved
1-3/8 coverage for nearly every quantity) but it is not a claim of
achieving genuine 95% coverage across the board -- `rate-after`'s 2/8 is
still poor, most likely because the record's post-breakpoint segment is
shorter and the bootstrap's per-observation noise resampling may not
fully capture whatever additional structure the two-segment fit is
sensitive to there. This is disclosed rather than hidden: the width caps
above are set from the ACTUAL widths the bootstrap produces (with a
~3-3.6x margin against a degenerate/placeholder interval), not tuned to
manufacture a coverage number, and a submission is graded on whether its
CI is self-consistent and sanely sized, not on a coverage property that
cannot be checked from a single realization.

These caps were checked against four deliberately too-wide (but
individually plausible-looking, e.g. depth +/-10%) submissions to confirm
each is correctly rejected, and against the genuine reference output on
all 8 seeds to confirm none of them are tight enough to reject an honest
result. Radius keeps its own, much looser cap (200,000 m, unrelated to the
~3x margin pattern above) for the reason given below -- it is not an
oversight, it reflects genuine weak identifiability, and its bootstrap
width in seed 202 alone (about 49,000 m, two-to-three orders of magnitude
wider than well-behaved seeds) would blow through a normally-scaled cap.

`rate-after` gets the same "not a ~3x margin" treatment as radius, for a
distinct but related reason: its own 2/8 coverage means the bootstrap width
itself is known to be too narrow for this quantity, so a ~3.5x margin over
a too-narrow number is not a safe bound the way it is for the other
quantities (whose bootstraps cleared 4/8+ coverage). Rather than encode a
cap derived from a demonstrably under-covering distribution, `rate-after`'s
cap (30% of the point estimate) is set loose enough to not reject a
genuinely honest, appropriately-wider CI, while still catching a degenerate
placeholder -- the same posture as radius, applied for the analogous
reason.

The genuinely defective interferograms (ASC-02,
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
the parameter space, not a broken fit. The bootstrap CI correctly
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

**`insar_gnss_rmse_m` is self-reported against a sanity cap, not
independently recomputed -- tried the opposite, reverted after trajectory
evidence.** An independent cross-check was added at one point: for each
track, take the single earliest interferogram known (from the private
answer key) to be free of the injected unwrapping defect -- a raw two-epoch
measurement needing no SBAS reconstruction -- average the 8 nearest points
at each station the submission reports using, apply the submission's own
disclosed ramp, and compare against that station's own two-epoch GNSS
displacement projected into the track's LOS. Calibrated against the
reference solution (seeds 13, 7, 42) this proxy landed at 0.6-1.4x the
reference's own reported RMSE, so the self-report was required to fall
within 5x of the proxy in either direction.

Live trajectory review across 12 agent trials found this rejected 7 of 8
claude-opus/codex trials, all of which had every other check correct
(source location, depth, rates, volume, GNSS QC, unwrapping-defect
handling) and used a MORE rigorous method than this reference solution: a
GNSS-regularized joint inversion that ties InSAR to GNSS during time-series
reconstruction rather than via a separate post-hoc planar ramp. Their
genuinely tighter tie residuals (self-reported RMSE 5-9x below the proxy)
were flagged as "inconsistent" purely because the proxy's own noise floor
-- inherent to using one unaveraged interferogram pair -- sits around
15-20 mm, well above what a proper multi-epoch reconciliation can
legitimately achieve (GNSS sigmas here go down to 2.5 mm; averaging over
the full record pushes a good submission's real RMSE toward that floor,
not the proxy's noisier one). No band width fixes this: it isn't a
calibration error, it's that the proxy's architecture (post-hoc ramp on raw
single-pair pixel data) only matches one specific reconciliation style and
can't validate a method that ties GNSS a different way. Reverted to
self-report + sanity cap. This is the same category of risk flagged
(theoretically, before this) every time a self-reported field was proposed
for independent recomputation -- confirmed here with real trajectory data
rather than argued from first principles.

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
