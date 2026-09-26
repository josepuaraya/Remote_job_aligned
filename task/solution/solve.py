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
4. Each track's per-epoch field is interpolated (inverse-distance,
   nearest points) onto the GNSS station coordinates, then the two
   tracks are combined (assuming negligible north-south sensitivity,
   consistent with the near-polar look geometry) into an InSAR-derived
   vertical displacement time series at each station.
5. Because InSAR is only ever a relative measurement, each station's own
   InSAR-vs-GNSS residual is regressed against time; a station whose
   residual carries a statistically significant trend is not explained
   by a simple reference-frame offset (it is behaving like an
   independent, non-volcanic local process) and is excluded before the
   next step. The remaining stations' residual offsets are fit with a
   planar ramp a + b*x + c*y (the InSAR reference-frame/orbital term),
   which is then removed from the InSAR field everywhere.
6. At the primary near-source station, the GNSS record and the corrected
   InSAR-derived vertical series are combined and fit with a two-segment
   (unknown breakpoint) linear model via grid search + weighted least
   squares, giving the rate before/after the change and the change day.
   The same combined-series approach at a designated far-field station
   gives the control-zone check. 95% CIs come from a parametric
   bootstrap using each source's own estimated noise.
7. A final InSAR-vs-GNSS RMSE (post-correction, at the retained stations)
   is reported as the reconciliation check.
"""
from __future__ import annotations

import csv
import json
import os
from collections import defaultdict

import numpy as np

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "/workspace")
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(WORKSPACE_DIR, "data"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(WORKSPACE_DIR, "output"))

RNG = np.random.default_rng(4242)

AMBIGUITY_QUANTUM_M = 0.028  # disclosed SAR system constant (half-wavelength)
PRIMARY_STATION_ID = "GNSS-01"
CONTROL_STATION_ID = "GNSS-07"
N_BOOTSTRAP = 300


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


def decompose_vertical_eastwest(asc_track, asc_at_station, desc_track, desc_at_station, common_days):
    """Linear-interpolate each track's per-epoch series onto common_days, then
    solve the 2x2 LOS system (ignoring north) for (east_west, vertical)."""
    theta_a, alpha_a = np.radians(asc_track.incidence_deg), np.radians(asc_track.heading_deg)
    theta_d, alpha_d = np.radians(desc_track.incidence_deg), np.radians(desc_track.heading_deg)
    A1, C1 = np.sin(theta_a) * np.cos(alpha_a), np.cos(theta_a)
    A2, C2 = np.sin(theta_d) * np.cos(alpha_d), np.cos(theta_d)
    det = A1 * (-C2) - A2 * (-C1)

    asc_days = np.array(asc_track.epoch_days, dtype=float)
    desc_days = np.array(desc_track.epoch_days, dtype=float)
    valid_asc = ~np.isnan(asc_at_station)
    valid_desc = ~np.isnan(desc_at_station)

    lo = max(asc_days[valid_asc].min() if valid_asc.any() else np.inf,
             desc_days[valid_desc].min() if valid_desc.any() else np.inf)
    hi = min(asc_days[valid_asc].max() if valid_asc.any() else -np.inf,
             desc_days[valid_desc].max() if valid_desc.any() else -np.inf)
    days = np.array([d for d in common_days if lo <= d <= hi], dtype=float)
    if len(days) == 0:
        return np.array([]), np.array([]), np.array([])

    los_a = np.interp(days, asc_days[valid_asc], asc_at_station[valid_asc])
    los_d = np.interp(days, desc_days[valid_desc], desc_at_station[valid_desc])

    u_e = (los_a * (-C2) - los_d * (-C1)) / det
    u_u = (A1 * los_d - A2 * los_a) / det
    return days, u_e, u_u


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
# Ramp / offset estimation and correction
# ---------------------------------------------------------------------------
def station_insar_gnss_residuals(days, insar_u, gnss_days, gnss_u):
    gnss_at = np.interp(days, gnss_days, gnss_u)
    return days, insar_u - gnss_at


def fit_line(x, y):
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    dof = max(len(x) - 2, 1)
    sigma2 = np.sum(resid ** 2) / dof
    cov = sigma2 * np.linalg.inv(A.T @ A)
    return coef[0], coef[1], np.sqrt(max(cov[1, 1], 0.0))


DRIFT_Z_THRESHOLD = 3.0
DRIFT_MAGNITUDE_FLOOR_M = 0.05  # below this, a significant-but-small slope is
# attributed to ordinary SBAS/interpolation reconstruction imprecision, not a
# genuine unmodeled local process; only a trend that is BOTH statistically
# significant AND large enough in absolute terms is treated as a bad station.


def screen_and_fit_ramp(station_offsets, station_xy):
    """station_offsets: dict sid -> (days, residuals). Returns (ramp_coefs,
    used_ids, excluded_ids, intercepts)."""
    intercepts, slopes, slope_se, total_drift, ids = {}, {}, {}, {}, []
    for sid, (days, resid) in station_offsets.items():
        if len(days) < 5:
            continue
        c0, c1, se1 = fit_line(days.astype(float), resid)
        intercepts[sid], slopes[sid], slope_se[sid] = c0, c1, se1
        total_drift[sid] = c1 * (days.max() - days.min())
        ids.append(sid)

    z = {sid: slopes[sid] / slope_se[sid] if slope_se[sid] > 0 else 0.0 for sid in ids}
    excluded = [
        sid for sid in ids
        if abs(z[sid]) > DRIFT_Z_THRESHOLD and abs(total_drift[sid]) > DRIFT_MAGNITUDE_FLOOR_M
    ]
    used = [sid for sid in ids if sid not in excluded]

    xs = np.array([station_xy[sid][0] for sid in used])
    ys = np.array([station_xy[sid][1] for sid in used])
    offs = np.array([intercepts[sid] for sid in used])
    A = np.vstack([np.ones_like(xs), xs, ys]).T
    coef, *_ = np.linalg.lstsq(A, offs, rcond=None)
    return coef, used, excluded, intercepts


def ramp_value(coef, x, y):
    return coef[0] + coef[1] * x + coef[2] * y


# ---------------------------------------------------------------------------
# Two-segment (unknown breakpoint) weighted regression
# ---------------------------------------------------------------------------
def segmented_design(t, tb):
    before = np.minimum(t, tb)
    after = np.maximum(t - tb, 0.0)
    return np.vstack([np.ones_like(t), before, after]).T


def fit_segmented(t, y, sigma, tb_grid):
    best = None
    w = 1.0 / np.maximum(sigma, 1e-6)
    for tb in tb_grid:
        A = segmented_design(t, tb) * w[:, None]
        b = y * w
        coef, *_ = np.linalg.lstsq(A, b, rcond=None)
        resid = b - A @ coef
        sse = float(np.sum(resid ** 2))
        if best is None or sse < best[0]:
            best = (sse, tb, coef)
    return best  # (sse, tb, [c0, rate1, rate2])


def fit_single_rate(t, y, sigma):
    w = 1.0 / np.maximum(sigma, 1e-6)
    A = np.vstack([np.ones_like(t), t]).T * w[:, None]
    b = y * w
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef  # [c0, rate]


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

    asc_at_stations = local_linear_interpolate(asc, cum_asc, sx, sy)
    desc_at_stations = local_linear_interpolate(desc, cum_desc, sx, sy)

    all_days = sorted(set(asc.epoch_days) | set(desc.epoch_days))

    insar_vertical = {}  # sid -> (days, u_u)
    for k, sid in enumerate(station_ids):
        days, u_e, u_u = decompose_vertical_eastwest(
            asc, asc_at_stations[k], desc, desc_at_stations[k], all_days
        )
        insar_vertical[sid] = (days, u_u)

    station_offsets = {}
    for sid in station_ids:
        days_i, u_u = insar_vertical[sid]
        if len(days_i) < 5:
            continue
        gdays = np.array([r[0] for r in gnss_series[sid]])
        gu = np.array([r[3] for r in gnss_series[sid]])
        d, resid = station_insar_gnss_residuals(days_i, u_u, gdays, gu)
        station_offsets[sid] = (d, resid)

    ramp_coef, used_stations, excluded_stations, intercepts = screen_and_fit_ramp(
        station_offsets, stations
    )

    def corrected_insar_series(sid):
        days_i, u_u = insar_vertical[sid]
        if len(days_i) == 0:
            return days_i, u_u
        x, y = stations[sid]
        return days_i, u_u - ramp_value(ramp_coef, x, y)

    def combined_series(sid):
        days_i, u_u_corr = corrected_insar_series(sid)
        gdays = np.array([r[0] for r in gnss_series[sid]])
        gu = np.array([r[3] for r in gnss_series[sid]])
        gsig = np.array([r[6] for r in gnss_series[sid]])
        insar_sigma = 0.004  # conservative pooled InSAR-derived-vertical noise floor
        t = np.concatenate([gdays.astype(float), days_i.astype(float)])
        y = np.concatenate([gu, u_u_corr])
        sig = np.concatenate([gsig, np.full(len(days_i), insar_sigma)])
        order = np.argsort(t)
        return t[order], y[order], sig[order]

    record_last_day = max(int(r["day"]) for r in gnss_rows)
    tb_grid = np.arange(30, record_last_day - 30, 5.0)

    t_p, y_p, sig_p = combined_series(PRIMARY_STATION_ID)
    sse, tb_hat, coef_p = fit_segmented(t_p, y_p, sig_p, tb_grid)
    c0_hat, rate1_hat, rate2_hat = coef_p
    cumulative_primary = rate1_hat * tb_hat + rate2_hat * (record_last_day - tb_hat)

    t_c, y_c, sig_c = combined_series(CONTROL_STATION_ID)
    coef_c = fit_single_rate(t_c, y_c, sig_c)
    cumulative_control = coef_c[1] * record_last_day

    # ---------------- Bootstrap CIs (parametric, using estimated sigmas) ----
    boot_rate1, boot_rate2, boot_tb, boot_cum = [], [], [], []
    for _ in range(N_BOOTSTRAP):
        y_star = y_p + sig_p * RNG.standard_normal(len(y_p))
        _, tb_s, coef_s = fit_segmented(t_p, y_star, sig_p, tb_grid)
        boot_rate1.append(coef_s[1])
        boot_rate2.append(coef_s[2])
        boot_tb.append(tb_s)
        boot_cum.append(coef_s[1] * tb_s + coef_s[2] * (record_last_day - tb_s))

    def ci95(vals):
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return [float(lo), float(hi)]

    # ---------------- Final InSAR-vs-GNSS reconciliation --------------------
    sq_errors = []
    for sid in used_stations:
        days_i, u_u_corr = corrected_insar_series(sid)
        gdays = np.array([r[0] for r in gnss_series[sid]])
        gu = np.array([r[3] for r in gnss_series[sid]])
        gu_at = np.interp(days_i, gdays, gu)
        sq_errors.extend(((u_u_corr - gu_at) ** 2).tolist())
    rmse = float(np.sqrt(np.mean(sq_errors))) if sq_errors else float("nan")

    result = {
        "vertical_rate_before_m_per_day": float(rate1_hat),
        "vertical_rate_after_m_per_day": float(rate2_hat),
        "rate_change_day": float(tb_hat),
        "cumulative_vertical_displacement_m": float(cumulative_primary),
        "vertical_rate_before_uncertainty_95": ci95(boot_rate1),
        "vertical_rate_after_uncertainty_95": ci95(boot_rate2),
        "rate_change_day_uncertainty_95": ci95(boot_tb),
        "cumulative_vertical_displacement_uncertainty_95": ci95(boot_cum),
        "control_zone_cumulative_vertical_displacement_m": float(cumulative_control),
        "insar_gnss_ramp_coefficients": {
            "constant_m": float(ramp_coef[0]),
            "gradient_x_m_per_m": float(ramp_coef[1]),
            "gradient_y_m_per_m": float(ramp_coef[2]),
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
