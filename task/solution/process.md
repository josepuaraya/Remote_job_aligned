# Reference workflow: deformation-microgravity-volcano

## 1. Onset detection and station velocities

For each candidate onset day (grid, 5-day step), fit a two-segment model
(flat baseline, then linear ramp starting at that day) to every station's
vertical time series and sum the SSE across all 10 stations. The onset
day that minimizes the joint SSE is taken as the shared inflation onset.
Re-fitting the same segmented model at the selected onset day gives each
station's vertical velocity and its standard error from the regression.

The total (pre -> post) displacement used later in the inversion is
computed independently of this ramp fit, as a plateau-to-plateau
difference (mean of the last 40 days minus mean of the first 40 days),
because the true signal in this network saturates before the record
ends; extrapolating the ramp model would overshoot. Its uncertainty is
propagated from the standard error of each window mean.

## 2. Free-air correction, and why it couples the two datasets

Gravity benchmarks are co-located with 8 of the 10 GNSS stations. The
free-air correction removes the elevation-change contribution to the
observed gravity change using each benchmark's own co-located GNSS
vertical displacement:

    dg_mass = dg_observed - (free_air_gradient) * disp_up

Because `disp_up` is itself a noisy GNSS measurement (not a known
constant), the corrected gravity value inherits GNSS uncertainty on top
of the gravimeter's own instrument noise. Worse, it is *the same*
`disp_up` value used directly as a GNSS observation in the inversion, so
the corrected-gravity residual and the GNSS-up residual at that station
are not independent -- they share a noise source. Treating the two
datasets as independent when building the weighting understates the
joint uncertainty and biases how much weight gravity gets relative to
GNSS.

## 3. Variance-covariance matrices

Both datasets show spatially correlated noise (an exponential-decay
covariance in inter-station distance is a standard model for both GNSS
common-mode and gravity survey error). The correlation length for each
dataset is estimated empirically from an initial (diagonal-weighted)
fit's residuals via a variogram-style regression of normalized residual
cross-products against inter-station distance, then used to rebuild a
full exponential covariance matrix for that dataset.

On top of that spatial structure, the free-air correction step above
adds:
- extra variance to each corrected gravity observation,
  `(free_air_gradient)^2 * Var(disp_up)`;
- an explicit cross-covariance block between corrected gravity and GNSS
  vertical displacement, `-free_air_gradient * Cov(disp_up_i, disp_up_j)`,
  nonzero for every station pair where the gravity benchmark is
  co-located with one of them.

These three pieces (GNSS spatial block, gravity spatial block, and the
free-air cross-covariance) are assembled into one joint covariance
matrix over the full stacked observation vector
`[gnss_east, gnss_north, gnss_up, corrected_gravity]`, rather than two
independent block-diagonal matrices. This is refit via generalized least
squares (GLS) using the Cholesky factor of that joint matrix to whiten
residuals.

## 4. Joint inversion

A Mogi (1958) point source (Poisson's ratio 0.25) predicts GNSS
displacement; a point-mass gravity term (`dg = G*dM*d/(r^2+d^2)^1.5`)
predicts corrected gravity change. Both share the same source location
and depth. The five parameters (x0, y0, depth, dV, dM) are fit jointly
by nonlinear least squares against the whitened joint residual vector,
with multi-start restarts to reduce sensitivity to the initial guess.

## 5. Excluding an unreliable gravity benchmark before the final fit

After the GLS fit, each gravity benchmark's residual is standardized by
its own marginal standard deviation from the fitted joint covariance
(which already accounts for the benchmark's instrument noise, its
spatial correlation with its neighbors, and the propagated GNSS
uncertainty at its location). A benchmark whose standardized residual
still exceeds |z| = 2 is not explained by a model that already accounts
for all of that -- it behaves like an independent local anomaly (e.g. a
hydrological signal) riding on top of the volcanic source signal, not
like volcano-driven signal with ordinary noise. One benchmark in this
network is flagged this way. Excluding it and refitting is necessary
because leaving it in biases the mass (and therefore the density)
estimate directly: it is a live number added to that benchmark's
gravity change, not just extra scatter that averages out.

## 6. Density and classification

`density = dM / dV` from the final (post-exclusion) fit. The result is
classified against the density thresholds given in `instruction.md`
(negative: mass loss; 0-1500 kg/m3: non-magmatic process, e.g. gas
exsolution or hydrothermal pressurization; 1500-3700 kg/m3: magmatic
recharge; above 3700 kg/m3: not a physically defensible density for
this kind of source, flagged as unrealistic).

## 7. Uncertainty via parametric bootstrap

95% confidence intervals are produced by resampling from the SAME joint
covariance matrix used in the fit: each replicate draws a single
correlated residual vector from the full joint multivariate normal
(preserving both the within-dataset spatial correlation and the
GNSS-gravity cross-correlation from step 3), refits the correlation
length from that replicate's own residuals rather than reusing the
point estimate, rebuilds the joint covariance, and refits. This
propagates not knowing the true correlation length exactly, on top of
ordinary parameter uncertainty. At least 300 successful replicates are
required; the reference solution runs more to allow for occasional
non-convergent replicates and still clear that floor.

## 8. Validation

The reference solution was run against three independent noise
realizations of the same synthetic network (different generator seeds)
to check that tolerances accept a correctly-implemented but differently
seeded run, and that the flagged-benchmark and classification logic are
not artifacts of one particular draw. All three passed the verifier
end-to-end.
