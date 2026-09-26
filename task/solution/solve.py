"""
solve.py -- reference solution for the InSAR/GNSS volcano deformation
rate-change task.

Pipeline
--------
1. Reconstruct each track's own SBAS acquisition network (epochs + edges)
   directly from interferogram_metadata.csv -- nothing about the network
   shape is hardcoded.
2. Screen each track for a phase-unwrapping error using triplet (loop)
   closure: for every closed triangle of interferograms (i,j),(j,k),(i,k)
   the summed LOS change around the loop should be zero up to ordinary
   noise. A defect shows up as a loop misclosure clustered near a nonzero
   integer multiple of the disclosed ambiguity quantum, on a spatially
   coherent subset of points -- both properties are checked before a
   correction is accepted, since ordinary atmospheric noise can produce
   an occasional large misclosure at an isolated point but not a
   spatially coherent one at exactly a cycle multiple.
3. Per-point, per-track SBAS network inversion (weighted least squares)
   recovers a cumulative-LOS time series at each pixel, referenced to
   that track's own first epoch. Interferogram weights are estimated from
   the data itself (variance-component estimation from an initial
   equal-weighted inversion's residuals), not assumed.
4. GNSS is the only absolute-reference dataset here, so InSAR is tied to
   it in InSAR's OWN native domain, per track, not after decomposing to
   vertical: each GNSS station's own (E, N, U) record is projected into
   EACH track's LOS geometry separately (a station has one predicted LOS
   value per track), and compared against that track's raw interpolated
   LOS at the station. A station is excluded only when BOTH tracks'
   independent residual trends agree it is significant and large (a real
   local process shows up in both geometries; track-specific
   reconstruction noise generally does not). The remaining stations'
   residual offsets are fit with a separate planar ramp a + b*x + c*y per
   track (each track has its own independent reference-frame/orbital
   error) and removed from that track's LOS field everywhere -- only then
   are the two tracks decomposed into vertical/east-west.
5. A joint nonlinear inversion (GNSS, weighted by each station's own
   reported uncertainty, plus a near-source sample of the now-corrected
   InSAR LOS pixels from both tracks) fits a single McTigue (1987) finite
   spherical source -- location, depth, radius, and a two-segment
   (unknown breakpoint) volume-rate history -- rather than treating any
   one station as "the" answer. A noisy-but-unbiased station is
   down-weighted by its own disclosed sigma, not excluded; only a station
   whose error is NOT explained by its own reported uncertainty (step 4)
   is dropped. 95% CIs use the delta method from the fit's Jacobian.
6. The fitted source's predicted vertical rate/displacement at the
   primary and control stations, and a final InSAR-vs-GNSS LOS RMSE at
   the retained stations, are reported as the reconciliation checks.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(WORKSPACE_DIR, "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(WORKSPACE_DIR, "output"))

AMBIGUITY_QUANTUM_M = 0.028  # disclosed SAR system constant (half-wavelength)
PRIMARY_STATION_ID = "GNSS-01"
CONTROL_STATION_ID = "GNSS-07"
POISSON_RATIO = 0.25
INSAR_NEAR_FIELD_RADIUS_M = 4000.0
INSAR_MAX_POINTS_PER_TRACK = 90


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_csv(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_data():
    ifg = load_csv(os.path.join(DATA_DIR, "interferograms.csv"))
    meta = load_csv(os.path.join(DATA_DIR, "interferogram_metadata.csv"))
    stations = load_csv(os.path.join(DATA_DIR, "gnss_stations.csv"))
    gnss = load_csv(os.path.join(DATA_DIR, "gnss_timeseries.csv"))
    return ifg, meta, stations, gnss


def los_projection_full(u_e, u_n, u_u, incidence_deg, heading_deg):
    theta = np.radians(incidence_deg)
    alpha = np.radians(heading_deg)
    return (
        np.sin(theta) * np.cos(alpha) * u_e
        - np.sin(theta) * np.sin(alpha) * u_n
        - np.cos(theta) * u_u
    )


def mctigue_displacement(x, y, depth, delta_v_m3, radius, x0, y0, nu=POISSON_RATIO):
    """McTigue (1987) finite spherical source. Reduces exactly to the Mogi
    point-source result as radius -> 0 (dV = pi*radius**3*(dP/mu), the
    standard pressure/volume relation for a pressurized sphere), which is
    both a physical sanity check and why a Mogi-only fit systematically
    misses near-field displacement for a source whose radius is not
    negligible relative to its depth."""
    dx, dy = x - x0, y - y0
    rho = np.sqrt(dx ** 2 + dy ** 2)
    R = np.sqrt(rho ** 2 + depth ** 2)
    dP_mu = delta_v_m3 / (np.pi * radius ** 3)
    a_d = radius / depth
    f1 = (depth ** 3) / (R ** 3)
    c1 = a_d ** 3 / (7.0 - 5.0 * nu)
    uzbar = a_d ** 3 * (1 - nu) * f1 * (1 - c1 * (0.5 * (1 + nu) - 3.75 * (2 - nu) * f1))
    u_z = uzbar * dP_mu * depth
    u_r = u_z * (rho / depth)
    theta_ang = np.arctan2(dy, dx)
    u_x = u_r * np.cos(theta_ang)
    u_y = u_r * np.sin(theta_ang)
    return u_x, u_y, u_z


# ---------------------------------------------------------------------------
# Per-track network reconstruction
# ---------------------------------------------------------------------------
class Track:
    def __init__(self, name, meta_rows, ifg_rows):
        self.name = name
        meta_rows = [m for m in meta_rows if m["orbit"] == name]
        self.incidence_deg = float(meta_rows[0]["incidence_deg"])
        self.heading_deg = float(meta_rows[0]["heading_deg"])

        days = sorted({int(m["day_start"]) for m in meta_rows} | {int(m["day_end"]) for m in meta_rows})
        self.epoch_days = days
        self.day_to_epoch = {d: i for i, d in enumerate(days)}
        self.n_epochs = len(days)

        self.edges = {}  # ifg_id -> (i, j)
        for m in meta_rows:
            i = self.day_to_epoch[int(m["day_start"])]
            j = self.day_to_epoch[int(m["day_end"])]
            self.edges[m["interferogram_id"]] = (i, j)

        rows = [r for r in ifg_rows if r["interferogram_id"] in self.edges]
        by_ifg = defaultdict(list)
        for r in rows:
            by_ifg[r["interferogram_id"]].append(r)

        all_point_ids = sorted({int(r["point_id"]) for r in rows})
        self.point_id_index = {pid: k for k, pid in enumerate(all_point_ids)}
        self.n_points = len(all_point_ids)
        self.x = np.zeros(self.n_points)
        self.y = np.zeros(self.n_points)
        seen = np.zeros(self.n_points, dtype=bool)

        # los[ifg_id] -> dict(point_index -> value); also point coordinates
        self.los = {}
        for ifg_id, recs in by_ifg.items():
            d = {}
            for r in recs:
                pidx = self.point_id_index[int(r["point_id"])]
                d[pidx] = float(r["los_displacement_m"])
                if not seen[pidx]:
                    self.x[pidx] = float(r["x_m"])
                    self.y[pidx] = float(r["y_m"])
                    seen[pidx] = True
            self.los[ifg_id] = d

        self.ifg_ids = sorted(self.edges.keys(), key=lambda k: self.edges[k])


def find_triangles(track):
    """All (edge_ij, edge_jk, edge_ik) triples closing a loop i->j->k->i."""
    edges_by_pair = {v: k for k, v in track.edges.items()}
    triangles = []
    edge_list = list(track.edges.items())
    for id_ij, (i, j) in edge_list:
        for id_jk, (j2, k) in edge_list:
            if j2 != j or k <= j:
                continue
            id_ik = edges_by_pair.get((i, k))
            if id_ik is not None:
                triangles.append((id_ij, id_jk, id_ik, i, j, k))
    return triangles


def spatial_coherence_fraction(x, y, mask, k=6):
    if mask.sum() < k + 1 or (~mask).sum() < 1:
        return 0.0
    idx = np.where(mask)[0]
    xs, ys = x[idx], y[idx]
    frac_hits = []
    all_x, all_y = x, y
    for m, (px, py) in zip(idx, zip(xs, ys)):
        d2 = (all_x - px) ** 2 + (all_y - py) ** 2
        nn = np.argsort(d2)[1:k + 1]
        frac_hits.append(mask[nn].mean())
    return float(np.mean(frac_hits))


def _two_cluster_split(values, quantum):
    """1-D two-means clustering, initialized at 0 and +/- one ambiguity
    cycle. Returns (centers, assignment, inertia) for the better of the
    two sign hypotheses."""
    best = None
    for init2 in (quantum, -quantum):
        c = np.array([0.0, init2])
        assign = np.zeros(len(values), dtype=int)
        for _ in range(25):
            d0 = np.abs(values - c[0])
            d1 = np.abs(values - c[1])
            new_assign = (d1 < d0).astype(int)
            if (new_assign == assign).all() and _ > 0:
                assign = new_assign
                break
            assign = new_assign
            for g in (0, 1):
                if (assign == g).any():
                    c[g] = values[assign == g].mean()
        inertia = float(np.sum((values - c[assign]) ** 2))
        if best is None or inertia < best[0]:
            best = (inertia, c.copy(), assign.copy())
    return best[1], best[2], best[0]


def _all_triangle_closures(track, los, triangles):
    total = 0.0
    for id_ij, id_jk, id_ik, i, j, k in triangles:
        common = set(los[id_ij]) & set(los[id_jk]) & set(los[id_ik])
        for p in common:
            c = los[id_ij][p] + los[id_jk][p] - los[id_ik][p]
            total += c * c
    return total


def detect_and_correct_unwrap_defect(track):
    """Loop-closure screen. Returns (corrected_los_dict, defect_edge_id_or_None).

    Stage 1 (screen): for each edge, per-point loop closures (pooled over
    every closed triangle containing that edge) are split into two 1-D
    clusters. A genuine unwrapping defect shows up as a cluster pair
    separated by close to one ambiguity cycle, with the displaced cluster
    occupying a SPATIALLY COHERENT subset of points (a k-NN neighborhood
    majority) -- ordinary atmospheric noise does not reproduce both
    properties at once.

    Stage 1 alone is not sufficient: an edge that merely shares ONE
    triangle with the true defective edge inherits the same apparent
    anomaly in that single triangle (the closure is a property of the
    whole loop, not separable edge-by-edge from one triangle alone), so
    several edges around the true defect typically all pass stage 1.
    Stage 2 (arbitrate) resolves this the way it would be resolved by
    hand: the genuinely defective edge is the one that, once corrected,
    brings EVERY triangle in the network back near zero closure -- a
    wrong candidate only shared by one triangle fixes that triangle but
    leaves any other anomalous triangle untouched. The candidate with the
    lowest total network-wide closure sum-of-squares after its proposed
    correction is accepted.
    """
    triangles = find_triangles(track)
    edge_point_contribs = defaultdict(lambda: defaultdict(list))  # ifg_id -> point -> [contribs]
    for id_ij, id_jk, id_ik, i, j, k in triangles:
        common = set(track.los[id_ij]) & set(track.los[id_jk]) & set(track.los[id_ik])
        for p in common:
            closure = track.los[id_ij][p] + track.los[id_jk][p] - track.los[id_ik][p]
            edge_point_contribs[id_ij][p].append(closure)
            edge_point_contribs[id_jk][p].append(closure)
            edge_point_contribs[id_ik][p].append(-closure)

    candidates = []  # (ifg_id, mask_points, n_cycles, score)
    for ifg_id, point_map in edge_point_contribs.items():
        if len(point_map) < 20:
            continue
        pts = np.array(sorted(point_map.keys()))
        vals = np.array([np.mean(point_map[p]) for p in pts])

        centers, assign, _ = _two_cluster_split(vals, AMBIGUITY_QUANTUM_M)
        sep = abs(centers[1] - centers[0])
        if not (0.5 * AMBIGUITY_QUANTUM_M < sep < 1.5 * AMBIGUITY_QUANTUM_M):
            continue
        far_cluster = int(np.argmax(np.abs(centers)))
        local_mask = assign == far_cluster
        frac = np.mean(local_mask)
        if not (0.1 < frac < 0.9):
            continue

        # Coherence is checked only among the points this edge actually has
        # data for (decorrelation already thins the cloud unevenly; testing
        # against the full point cloud would falsely penalize coherence
        # wherever a neighbor simply has no data for this edge/triangle).
        coherence = spatial_coherence_fraction(track.x[pts], track.y[pts], local_mask)
        if coherence <= 0.6:
            continue

        n_cycles = float(np.round(centers[far_cluster] / AMBIGUITY_QUANTUM_M))
        candidates.append((ifg_id, pts[local_mask], n_cycles, coherence * sep))

    corrected_los = {ifg_id: dict(d) for ifg_id, d in track.los.items()}
    if not candidates:
        return corrected_los, None

    baseline_sse = _all_triangle_closures(track, track.los, triangles)
    best_edge, best_mask_points, best_n, best_sse = None, None, 0, baseline_sse
    for ifg_id, mask_points, n_cycles, _score in candidates:
        trial = {k: dict(v) for k, v in track.los.items()}
        for p in mask_points:
            if p in trial[ifg_id]:
                trial[ifg_id][p] -= n_cycles * AMBIGUITY_QUANTUM_M
        sse = _all_triangle_closures(track, trial, triangles)
        if sse < best_sse:
            best_edge, best_mask_points, best_n, best_sse = ifg_id, mask_points, n_cycles, sse

    # Only accept if the winning correction actually reduces network-wide
    # closure substantially (guards against accepting a spurious candidate
    # when nothing beats leaving the data uncorrected).
    if best_edge is not None and best_sse < 0.5 * baseline_sse:
        for p in best_mask_points:
            if p in corrected_los[best_edge]:
                corrected_los[best_edge][p] -= best_n * AMBIGUITY_QUANTUM_M
        return corrected_los, best_edge
    return corrected_los, None


# ---------------------------------------------------------------------------
# SBAS network inversion (weighted least squares, per point)
# ---------------------------------------------------------------------------
def _reachable_from_zero(n_epochs, edges):
    """BFS over the point's own edge set. An epoch NOT reachable from
    epoch 0 through edges this point actually has data for is not
    determined by this point's data at all -- a naive lstsq over ALL
    epochs would silently return 0 for it (an unconstrained column's
    minimum-norm solution), which looks like real data but isn't."""
    adj = defaultdict(list)
    for _ifg_id, i, j in edges:
        adj[i].append(j)
        adj[j].append(i)
    seen = {0}
    frontier = [0]
    while frontier:
        nxt = []
        for node in frontier:
            for nb in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    nxt.append(nb)
        frontier = nxt
    return seen


def invert_time_series(track, los, weights=None):
    """Returns cum[point_idx, epoch_idx] cumulative LOS relative to epoch 0,
    NaN for any epoch not actually connected to epoch 0 through this
    point's own (decorrelation-thinned) edge set."""
    cum = np.full((track.n_points, track.n_epochs), np.nan)
    cum[:, 0] = 0.0
    edge_list = list(track.edges.items())
    point_to_edges = defaultdict(list)
    for ifg_id, (i, j) in edge_list:
        for p in los[ifg_id]:
            point_to_edges[p].append((ifg_id, i, j))

    w = weights or {ifg_id: 1.0 for ifg_id, _ in edge_list}
    for p, edges in point_to_edges.items():
        if len(edges) == 0:
            continue
        reachable = sorted(_reachable_from_zero(track.n_epochs, edges) - {0})
        if not reachable:
            continue
        col_index = {epoch: c for c, epoch in enumerate(reachable)}
        usable_edges = [(ifg_id, i, j) for ifg_id, i, j in edges if i in col_index or j in col_index]
        G = np.zeros((len(usable_edges), len(reachable)))
        d = np.zeros(len(usable_edges))
        sw = np.zeros(len(usable_edges))
        for r, (ifg_id, i, j) in enumerate(usable_edges):
            if j in col_index:
                G[r, col_index[j]] += 1.0
            if i in col_index:
                G[r, col_index[i]] -= 1.0
            d[r] = los[ifg_id][p]
            sw[r] = np.sqrt(w[ifg_id])
        Gw = G * sw[:, None]
        dw = d * sw
        sol, *_ = np.linalg.lstsq(Gw, dw, rcond=None)
        for epoch, c in col_index.items():
            cum[p, epoch] = sol[c]
    return cum


def estimate_edge_variances(track, los, cum):
    resid_sq = defaultdict(list)
    for ifg_id, (i, j) in track.edges.items():
        for p, val in los[ifg_id].items():
            ci, cj = cum[p, i], cum[p, j]
            if np.isnan(ci) or np.isnan(cj):
                continue
            resid_sq[ifg_id].append((val - (cj - ci)) ** 2)
    variances = {}
    floor = 1e-7
    for ifg_id, vals in resid_sq.items():
        variances[ifg_id] = max(float(np.mean(vals)), floor) if vals else floor
    return variances


def run_track_pipeline(track):
    corrected_los, defect_edge = detect_and_correct_unwrap_defect(track)
    cum0 = invert_time_series(track, corrected_los)
    variances = estimate_edge_variances(track, corrected_los, cum0)
    weights = {k: 1.0 / v for k, v in variances.items()}
    cum1 = invert_time_series(track, corrected_los, weights=weights)
    return cum1, defect_edge, variances


# ---------------------------------------------------------------------------
# Interpolation of a track's per-epoch field onto arbitrary (x, y) points
# ---------------------------------------------------------------------------
def local_linear_interpolate(track, cum, query_x, query_y, k=18):
    """Local first-order (planar) fit within each query point's k nearest
    neighbors, evaluated at the query location. A plain inverse-distance
    average is biased low near a strongly curved source (it can never
    exceed the values it averages), whereas a local plane fit is unbiased
    to first order for a smoothly curving field as long as the
    neighborhood is small enough to be locally near-linear -- true here
    since the neighborhoods are a small fraction of the source depth."""
    out = np.full((len(query_x), track.n_epochs), np.nan)
    for qi, (qx, qy) in enumerate(zip(query_x, query_y)):
        d2 = (track.x - qx) ** 2 + (track.y - qy) ** 2
        order = np.argsort(d2)[:k]
        dx = track.x[order] - qx
        dy = track.y[order] - qy
        dist = np.sqrt(d2[order]) + 1e-6
        w = 1.0 / dist
        A = np.vstack([np.ones_like(dx), dx, dy]).T
        for e in range(track.n_epochs):
            vals = cum[order, e]
            valid = ~np.isnan(vals)
            if valid.sum() < 4:
                continue
            Aw = A[valid] * w[valid, None]
            bw = vals[valid] * w[valid]
            coef, *_ = np.linalg.lstsq(Aw, bw, rcond=None)
            out[qi, e] = coef[0]
    return out



# ---------------------------------------------------------------------------
# GNSS loading
# ---------------------------------------------------------------------------
def load_gnss(stations_rows, gnss_rows):
    stations = {r["station_id"]: (float(r["x_m"]), float(r["y_m"])) for r in stations_rows}
    series = defaultdict(list)  # station_id -> list of (day, e, n, u, sig_e, sig_n, sig_u)
    for r in gnss_rows:
        series[r["station_id"]].append((
            int(r["day"]), float(r["east_disp_m"]), float(r["north_disp_m"]), float(r["vertical_disp_m"]),
            float(r["east_sigma_m"]), float(r["north_sigma_m"]), float(r["vertical_sigma_m"]),
        ))
    for sid in series:
        series[sid].sort(key=lambda t: t[0])
    return stations, series


# ---------------------------------------------------------------------------
# Per-track ramp estimation and correction (native LOS domain)
# ---------------------------------------------------------------------------
def fit_line(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    sigma2 = np.sum(resid ** 2) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return coef[0], coef[1], np.sqrt(max(cov[0, 0], 0.0)), np.sqrt(max(cov[1, 1], 0.0))


def gnss_los_at_track_epochs(track, gdays, ge, gn, gu):
    """Project a station's own (E, N, U) record into this track's LOS
    geometry, then sample it at the track's own epoch days."""
    pred_los_daily = los_projection_full(ge, gn, gu, track.incidence_deg, track.heading_deg)
    epoch_days = np.array(track.epoch_days, dtype=float)
    return np.interp(epoch_days, gdays, pred_los_daily)


DRIFT_Z_THRESHOLD = 3.0
DRIFT_MAGNITUDE_FLOOR_M = 0.05  # below this, a significant-but-small slope is
# attributed to ordinary SBAS/interpolation reconstruction imprecision, not a
# genuine unmodeled local process; only a trend that is BOTH statistically
# significant AND large enough in absolute terms, IN BOTH TRACKS
# INDEPENDENTLY, is treated as a bad station. Requiring agreement between two
# independently-processed tracks is what separates a real local process
# (which projects into both LOS geometries, since the incidence angles are
# similar) from one track's own reconstruction noise.


def per_track_station_screen(track, raw_at_stations, station_ids, stations, gnss_series):
    """Returns dict sid -> (intercept, intercept_se, exceeds_threshold: bool)."""
    out = {}
    for k, sid in enumerate(station_ids):
        raw = raw_at_stations[k]
        valid = ~np.isnan(raw)
        if valid.sum() < 5:
            continue
        epoch_days = np.array(track.epoch_days, dtype=float)[valid]
        gdays = np.array([r[0] for r in gnss_series[sid]], dtype=float)
        ge = np.array([r[1] for r in gnss_series[sid]])
        gn = np.array([r[2] for r in gnss_series[sid]])
        gu = np.array([r[3] for r in gnss_series[sid]])
        pred_los = gnss_los_at_track_epochs(track, gdays, ge, gn, gu)[valid]
        resid = raw[valid] - pred_los
        c0, c1, se0, se1 = fit_line(epoch_days, resid)
        z = c1 / se1 if se1 > 0 else 0.0
        total_drift = c1 * (epoch_days.max() - epoch_days.min())
        exceeds = abs(z) > DRIFT_Z_THRESHOLD and abs(total_drift) > DRIFT_MAGNITUDE_FLOOR_M
        out[sid] = (c0, se0, exceeds)
    return out


def fit_ramp(intercepts, intercept_ses, used_ids, stations):
    """Weighted least squares: a station whose own residual-intercept is
    poorly determined (large se0, e.g. a genuinely noisy but unbiased
    station) is down-weighted rather than treated as equally informative
    as a precise one. With only ~n_stations points and 3 free parameters,
    this small regression has far less redundancy than the joint source
    inversion, so an unweighted noisy outlier has real leverage here."""
    xs = np.array([stations[sid][0] for sid in used_ids])
    ys = np.array([stations[sid][1] for sid in used_ids])
    offs = np.array([intercepts[sid] for sid in used_ids])
    ses = np.array([max(intercept_ses[sid], 1e-6) for sid in used_ids])
    w = 1.0 / ses
    A = np.vstack([np.ones_like(xs), xs, ys]).T * w[:, None]
    b = offs * w
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef


def ramp_value(coef, x, y):
    return coef[0] + coef[1] * x + coef[2] * y


# ---------------------------------------------------------------------------
# Joint McTigue (1987) source + two-segment rate-history inversion
# ---------------------------------------------------------------------------
PARAM_NAMES = ["x0", "y0", "depth", "radius", "rate1", "rate2", "t_break"]


def cumulative_dv_at(day, rate1, rate2, t_break):
    day = np.asarray(day, dtype=float)
    before = rate1 * day
    after = rate1 * t_break + rate2 * (day - t_break)
    return np.where(day <= t_break, before, after)


def model_enu(theta, x, y, day):
    x0, y0, depth, radius, rate1, rate2, t_break = theta
    dv = cumulative_dv_at(day, rate1, rate2, t_break)
    return mctigue_displacement(x, y, depth, dv, radius, x0, y0)


def build_residual_fn(gnss_data, insar_data):
    def resid_fn(theta):
        parts = []
        for g in gnss_data:
            ux, uy, uz = model_enu(theta, g["x"], g["y"], g["day"])
            parts.append((ux - g["e"]) / g["sige"])
            parts.append((uy - g["n"]) / g["sign"])
            parts.append((uz - g["u"]) / g["sigu"])
        for ins in insar_data:
            ux, uy, uz = model_enu(theta, ins["x"], ins["y"], ins["day"])
            pred_los = los_projection_full(ux, uy, uz, ins["incidence"], ins["heading"])
            parts.append((pred_los - ins["los"]) / ins["sigma"])
        return np.concatenate(parts) if parts else np.zeros(0)
    return resid_fn


def bounds_for(record_last_day):
    lower = [-6000.0, -6000.0, 300.0, 50.0, 0.0, 0.0, 15.0]
    upper = [6000.0, 6000.0, 6000.0, 1800.0, 5.0e5, 5.0e5, record_last_day - 15.0]
    return np.array(lower), np.array(upper)


def initial_guess(used_gnss_stations, stations, gnss_series, record_last_day):
    """Displacement-magnitude-weighted centroid for (x0, y0); generic
    starting values for depth/radius; rate scale from a rough sensitivity
    back-out at the highest-signal station -- no true parameter values are
    used, only what is observable from the visible data."""
    weights, xs, ys, best_sid, best_w = [], [], [], None, -1.0
    for sid in used_gnss_stations:
        rows = gnss_series[sid]
        u = np.array([r[3] for r in rows])
        w = abs(float(u[-1] - u[0]))
        weights.append(w)
        xs.append(stations[sid][0])
        ys.append(stations[sid][1])
        if w > best_w:
            best_w, best_sid = w, sid
    weights = np.array(weights)
    if weights.sum() <= 0:
        x0 = float(np.mean(xs))
        y0 = float(np.mean(ys))
    else:
        x0 = float(np.sum(weights * np.array(xs)) / weights.sum())
        y0 = float(np.sum(weights * np.array(ys)) / weights.sum())
    depth0, radius0 = 2000.0, 500.0

    rows = gnss_series[best_sid]
    days = np.array([r[0] for r in rows], dtype=float)
    u = np.array([r[3] for r in rows])
    rough_slope, _intercept = np.polyfit(days, u, 1)
    sx, sy = stations[best_sid]
    sensitivity = mctigue_displacement(
        np.array([sx]), np.array([sy]), depth0, np.array([1.0]), radius0, x0, y0
    )[2][0]
    sensitivity = sensitivity if abs(sensitivity) > 1e-12 else 1e-12
    rate_scale = max(abs(float(rough_slope) / sensitivity), 100.0)
    return np.array([x0, y0, depth0, radius0, rate_scale, 2.0 * rate_scale, record_last_day / 2.0])


def run_joint_inversion(gnss_data, insar_data, used_gnss_stations, stations, gnss_series, record_last_day):
    lower, upper = bounds_for(record_last_day)
    theta0 = np.clip(
        initial_guess(used_gnss_stations, stations, gnss_series, record_last_day), lower, upper
    )

    tb_candidates = np.linspace(record_last_day * 0.15, record_last_day * 0.85, 5)
    gnss_only_resid = build_residual_fn(gnss_data, [])

    best = None
    for tb in tb_candidates:
        start = theta0.copy()
        start[6] = tb
        start = np.clip(start, lower, upper)
        try:
            res = least_squares(gnss_only_resid, start, bounds=(lower, upper), method="trf")
        except Exception:
            continue
        cost = float(np.sum(res.fun ** 2))
        if best is None or cost < best[0]:
            best = (cost, res.x)
    pass1_theta = best[1] if best is not None else theta0

    full_resid = build_residual_fn(gnss_data, insar_data)
    best2 = None
    for perturb in (0.0, 0.1, -0.1):
        start = np.clip(pass1_theta * (1.0 + perturb * np.array([0, 0, 1, 1, 1, 1, 0])), lower, upper)
        try:
            res = least_squares(full_resid, start, bounds=(lower, upper), method="trf")
        except Exception:
            continue
        cost = float(np.sum(res.fun ** 2))
        if best2 is None or cost < best2[0]:
            best2 = (cost, res)
    final = best2[1]

    jac = final.jac
    try:
        cov = np.linalg.pinv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = np.eye(len(final.x)) * 1e6
    return final.x, cov


Z95 = 1.959963984540054


def delta_ci(f, theta_hat, cov_theta):
    theta_hat = np.asarray(theta_hat, dtype=float)
    n = len(theta_hat)
    grad = np.zeros(n)
    for i in range(n):
        step = 1e-5 * max(abs(theta_hat[i]), 1.0)
        tp, tm = theta_hat.copy(), theta_hat.copy()
        tp[i] += step
        tm[i] -= step
        grad[i] = (f(tp) - f(tm)) / (2 * step)
    var = float(grad @ cov_theta @ grad)
    se = float(np.sqrt(max(var, 0.0)))
    val = float(f(theta_hat))
    return val, [val - Z95 * se, val + Z95 * se]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ifg_rows, meta_rows, station_rows, gnss_rows = load_data()

    asc = Track("ascending", meta_rows, ifg_rows)
    desc = Track("descending", meta_rows, ifg_rows)

    cum_asc, defect_asc, var_asc = run_track_pipeline(asc)
    cum_desc, defect_desc, var_desc = run_track_pipeline(desc)
    defect_ids = [d for d in (defect_asc, defect_desc) if d is not None]

    stations, gnss_series = load_gnss(station_rows, gnss_rows)
    station_ids = sorted(stations.keys())
    sx = np.array([stations[s][0] for s in station_ids])
    sy = np.array([stations[s][1] for s in station_ids])
    record_last_day = max(int(r["day"]) for r in gnss_rows)

    raw_asc_at_stations = local_linear_interpolate(asc, cum_asc, sx, sy)
    raw_desc_at_stations = local_linear_interpolate(desc, cum_desc, sx, sy)

    # ---------------- Per-track GNSS screening (native LOS domain) ---------
    asc_screen = per_track_station_screen(asc, raw_asc_at_stations, station_ids, stations, gnss_series)
    desc_screen = per_track_station_screen(desc, raw_desc_at_stations, station_ids, stations, gnss_series)

    excluded_stations = sorted(
        sid for sid in station_ids
        if asc_screen.get(sid, (0.0, 0.0, False))[2] and desc_screen.get(sid, (0.0, 0.0, False))[2]
    )
    used_stations = [sid for sid in station_ids if sid not in excluded_stations]

    asc_intercepts = {sid: asc_screen[sid][0] for sid in used_stations if sid in asc_screen}
    asc_intercept_ses = {sid: asc_screen[sid][1] for sid in used_stations if sid in asc_screen}
    desc_intercepts = {sid: desc_screen[sid][0] for sid in used_stations if sid in desc_screen}
    desc_intercept_ses = {sid: desc_screen[sid][1] for sid in used_stations if sid in desc_screen}
    ramp_asc = fit_ramp(asc_intercepts, asc_intercept_ses, list(asc_intercepts), stations)
    ramp_desc = fit_ramp(desc_intercepts, desc_intercept_ses, list(desc_intercepts), stations)

    # ---------------- Apply per-track ramp correction everywhere -----------
    corrected_cum_asc = cum_asc - ramp_value(ramp_asc, asc.x, asc.y)[:, None]
    corrected_cum_desc = cum_desc - ramp_value(ramp_desc, desc.x, desc.y)[:, None]
    corrected_asc_at_stations = raw_asc_at_stations - ramp_value(ramp_asc, sx, sy)[:, None]
    corrected_desc_at_stations = raw_desc_at_stations - ramp_value(ramp_desc, sx, sy)[:, None]

    # ---------------- Assemble GNSS data for the joint inversion -----------
    gnss_data = []
    for sid in used_stations:
        rows = gnss_series[sid]
        gnss_data.append({
            "x": stations[sid][0], "y": stations[sid][1],
            "day": np.array([r[0] for r in rows], dtype=float),
            "e": np.array([r[1] for r in rows]), "n": np.array([r[2] for r in rows]),
            "u": np.array([r[3] for r in rows]),
            "sige": np.array([r[4] for r in rows]), "sign": np.array([r[5] for r in rows]),
            "sigu": np.array([r[6] for r in rows]),
        })

    # ---------------- Assemble a near-field InSAR sample --------------------
    x0_guess, y0_guess = initial_guess(used_stations, stations, gnss_series, record_last_day)[:2]
    insar_pooled_sigma_asc = float(np.sqrt(np.mean(list(var_asc.values())))) if var_asc else 0.005
    insar_pooled_sigma_desc = float(np.sqrt(np.mean(list(var_desc.values())))) if var_desc else 0.005

    def sample_insar(track, corrected_cum, pooled_sigma):
        """One batched entry per track (arrays, not one dict per point-epoch
        pair) so the model is evaluated vectorized, not looped in Python."""
        d2 = (track.x - x0_guess) ** 2 + (track.y - y0_guess) ** 2
        near = np.where(d2 < INSAR_NEAR_FIELD_RADIUS_M ** 2)[0]
        if len(near) > INSAR_MAX_POINTS_PER_TRACK:
            near = np.random.default_rng(0).choice(near, size=INSAR_MAX_POINTS_PER_TRACK, replace=False)
        xs, ys, days, los_vals = [], [], [], []
        for p in near:
            for e, day in enumerate(track.epoch_days):
                val = corrected_cum[p, e]
                if np.isnan(val):
                    continue
                xs.append(track.x[p])
                ys.append(track.y[p])
                days.append(float(day))
                los_vals.append(val)
        if not xs:
            return None
        return {
            "x": np.array(xs), "y": np.array(ys), "day": np.array(days),
            "los": np.array(los_vals), "sigma": pooled_sigma,
            "incidence": track.incidence_deg, "heading": track.heading_deg,
        }

    insar_data = [
        d for d in (
            sample_insar(asc, corrected_cum_asc, insar_pooled_sigma_asc),
            sample_insar(desc, corrected_cum_desc, insar_pooled_sigma_desc),
        ) if d is not None
    ]

    # ---------------- Joint McTigue source + rate-history inversion --------
    theta_hat, cov_theta = run_joint_inversion(
        gnss_data, insar_data, used_stations, stations, gnss_series, record_last_day
    )

    px, py = stations[PRIMARY_STATION_ID]
    cx, cy = stations[CONTROL_STATION_ID]

    def f_param(i):
        return lambda th: th[i]

    def f_rate_before(th):
        return mctigue_displacement(np.array([px]), np.array([py]), th[2], np.array([th[4]]), th[3], th[0], th[1])[2][0]

    def f_rate_after(th):
        return mctigue_displacement(np.array([px]), np.array([py]), th[2], np.array([th[5]]), th[3], th[0], th[1])[2][0]

    def f_cumulative(x, y):
        def f(th):
            dv = cumulative_dv_at(record_last_day, th[4], th[5], th[6])
            return mctigue_displacement(np.array([x]), np.array([y]), th[2], np.array([dv]), th[3], th[0], th[1])[2][0]
        return f

    x0_hat, x0_ci = delta_ci(f_param(0), theta_hat, cov_theta)
    y0_hat, y0_ci = delta_ci(f_param(1), theta_hat, cov_theta)
    depth_hat, depth_ci = delta_ci(f_param(2), theta_hat, cov_theta)
    radius_hat, radius_ci = delta_ci(f_param(3), theta_hat, cov_theta)
    tb_hat, tb_ci = delta_ci(f_param(6), theta_hat, cov_theta)
    rate_before_hat, rate_before_ci = delta_ci(f_rate_before, theta_hat, cov_theta)
    rate_after_hat, rate_after_ci = delta_ci(f_rate_after, theta_hat, cov_theta)
    cum_primary_hat, cum_primary_ci = delta_ci(f_cumulative(px, py), theta_hat, cov_theta)
    cum_control_hat, _ = delta_ci(f_cumulative(cx, cy), theta_hat, cov_theta)

    # ---------------- Final InSAR-vs-GNSS reconciliation (native LOS) ------
    sq_errors = []
    for track, corrected_at_stations, screen in (
        (asc, corrected_asc_at_stations, asc_screen), (desc, corrected_desc_at_stations, desc_screen)
    ):
        epoch_days = np.array(track.epoch_days, dtype=float)
        for k, sid in enumerate(station_ids):
            if sid not in used_stations:
                continue
            raw = corrected_at_stations[k]
            valid = ~np.isnan(raw)
            if valid.sum() == 0:
                continue
            rows = gnss_series[sid]
            gdays = np.array([r[0] for r in rows], dtype=float)
            ge = np.array([r[1] for r in rows])
            gn = np.array([r[2] for r in rows])
            gu = np.array([r[3] for r in rows])
            pred_los = gnss_los_at_track_epochs(track, gdays, ge, gn, gu)
            sq_errors.extend(((raw[valid] - pred_los[valid]) ** 2).tolist())
    rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")

    result = {
        "x0_m": x0_hat, "y0_m": y0_hat, "depth_m": depth_hat, "radius_m": radius_hat,
        "x0_uncertainty_95": x0_ci, "y0_uncertainty_95": y0_ci,
        "depth_uncertainty_95": depth_ci, "radius_uncertainty_95": radius_ci,
        "vertical_rate_before_m_per_day": rate_before_hat,
        "vertical_rate_after_m_per_day": rate_after_hat,
        "rate_change_day": tb_hat,
        "cumulative_vertical_displacement_m": cum_primary_hat,
        "vertical_rate_before_uncertainty_95": rate_before_ci,
        "vertical_rate_after_uncertainty_95": rate_after_ci,
        "rate_change_day_uncertainty_95": tb_ci,
        "cumulative_vertical_displacement_uncertainty_95": cum_primary_ci,
        "control_zone_cumulative_vertical_displacement_m": cum_control_hat,
        "insar_gnss_ramp_coefficients": {
            "ascending": {"constant_m": float(ramp_asc[0]), "gradient_x_m_per_m": float(ramp_asc[1]),
                          "gradient_y_m_per_m": float(ramp_asc[2])},
            "descending": {"constant_m": float(ramp_desc[0]), "gradient_x_m_per_m": float(ramp_desc[1]),
                           "gradient_y_m_per_m": float(ramp_desc[2])},
        },
        "gnss_stations_used": used_stations,
        "gnss_stations_excluded": excluded_stations,
        "interferograms_with_unwrapping_correction": defect_ids,
        "insar_gnss_rmse_m": rmse,
    }

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, "result.json"), "w") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)

    return result


if __name__ == "__main__":
    main()
