"""
H(dT1,dT2) = (U*LMTD)^(-beta) Tangent-Plane (convex, underestimating) Envelope
================================================================================
H is the "area-cost kernel" you multiply by G=Q^beta (via McCormick) to get
A^beta = G*H, replacing the old chain
    dT1,dT2 -> LMTD (envelope) -> A=Q/(U*LMTD) (McCormick) -> Cost=f(A) (SOS2)
with
    dT1,dT2 -> H=(U*LMTD)^-beta (envelope, THIS FILE)   \\
    Q       -> G=Q^beta (1D SOS2)                        }-> P=G*H (McCormick) -> Cost = a + b*P
i.e. two tightly-bound one-directional relaxations feeding a single, ordinary
bilinear McCormick term, instead of three independently-loose relaxations
chained through two free intermediate variables (LMTD, A).

H is jointly CONVEX in (dT1,dT2) for all beta>0 (verified numerically, not just
assumed) -> tangent planes lie BELOW H everywhere -> the valid envelope is
max-over-planes, and it UNDERESTIMATES H (and hence A^beta, hence Area, hence
Cost) everywhere. That direction matters physically: an underestimated Area is
an undersized/infeasible exchanger once you plug real numbers back in. Since
you're doing fixed-topology NLP with the true equations afterward, the MILP's
job here is just to pick a good topology -- so the target is "small enough
error that topology selection doesn't flip", not "provably safe in one
direction". Keep that in mind when you set error_threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ──────────────────────────────────────────────────────────────────────
# Core LMTD value + gradient (reused from the original tester)
# ──────────────────────────────────────────────────────────────────────
def lmtd_and_grad(dT1, dT2, tol=0.1):
    dT1 = np.atleast_1d(np.asarray(dT1, dtype=float))
    dT2 = np.atleast_1d(np.asarray(dT2, dtype=float))
    scalar_input = dT1.size == 1 and dT2.size == 1

    dT1b, dT2b = np.broadcast_arrays(dT1, dT2)
    L = np.empty_like(dT1b, dtype=float)
    dLd1 = np.empty_like(dT1b, dtype=float)
    dLd2 = np.empty_like(dT1b, dtype=float)

    S = dT1b + dT2b
    D = dT1b - dT2b
    x = D / S

    close = np.abs(dT1b - dT2b) < tol
    far = ~close

    if np.any(close):
        xc = x[close]; Sc = S[close]; xc2 = xc * xc
        L[close] = 0.5 * Sc * (1.0 - xc2 / 3.0)
        dLd1[close] = 0.5 - xc / 3.0 + xc2 / 6.0
        dLd2[close] = 0.5 + xc / 3.0 + xc2 / 6.0

    if np.any(far):
        d1, d2 = dT1b[far], dT2b[far]
        ln_ratio = np.log(d1) - np.log(d2)
        L[far] = (d1 - d2) / ln_ratio
        dLd1[far] = (ln_ratio - (d1 - d2) / d1) / ln_ratio**2
        dLd2[far] = (-ln_ratio + (d1 - d2) / d2) / ln_ratio**2

    if scalar_input:
        return float(L.flat[0]), float(dLd1.flat[0]), float(dLd2.flat[0])
    return L, dLd1, dLd2


# ──────────────────────────────────────────────────────────────────────
# H = (U*LMTD)^(-beta) value + gradient (chain rule off LMTD)
# ──────────────────────────────────────────────────────────────────────
def H_and_grad(dT1, dT2, beta, U=1.0, tol=0.1):
    """
    H(dT1,dT2) = (U*LMTD(dT1,dT2))^(-beta)
    dH/ddTi    = -beta * U^-beta * LMTD^(-beta-1) * dLMTD/ddTi
    """
    L, dLd1, dLd2 = lmtd_and_grad(dT1, dT2, tol)
    Uinv_beta = U ** (-beta)
    Hval = Uinv_beta * L ** (-beta)
    coef = -beta * Uinv_beta * L ** (-beta - 1.0)
    dHd1 = coef * dLd1
    dHd2 = coef * dLd2
    if np.isscalar(dT1) and np.isscalar(dT2):
        return float(Hval), float(dHd1), float(dHd2)
    return Hval, dHd1, dHd2


def H_true(dT1, dT2, beta, U=1.0, tol=0.1):
    H, _, _ = H_and_grad(dT1, dT2, beta, U, tol)
    return H


def tangent_plane_coeffs_H(dT1_0, dT2_0, beta, U=1.0, tol=1e-6):
    """
    Returns (a0,a1,a2) s.t. H(dT1,dT2) >= a0 + a1*dT1 + a2*dT2  (H is convex,
    so every tangent plane is a valid GLOBAL underestimator, not just local).
    """
    H0, g1, g2 = H_and_grad(dT1_0, dT2_0, beta, U, tol)
    a0 = H0 - g1 * dT1_0 - g2 * dT2_0
    return a0, g1, g2


# ──────────────────────────────────────────────────────────────────────
# Envelope = MAX over tangent planes (valid because H is convex)
# ──────────────────────────────────────────────────────────────────────
def envelope_value_max(dT1, dT2, planes):
    dT1 = np.asarray(dT1, dtype=float)
    dT2 = np.asarray(dT2, dtype=float)
    a0 = np.array([p[0] for p in planes])
    a1 = np.array([p[1] for p in planes])
    a2 = np.array([p[2] for p in planes])
    vals = a0 + a1 * dT1[..., None] + a2 * dT2[..., None]
    return vals.max(axis=-1)


def build_initial_planes_H(dT_lo, dT_hi, beta, U=1.0, N_init=3, use_symmetry=True):
    """Small geomspace seed grid, same symmetry trick as the LMTD tester."""
    grid = np.geomspace(dT_lo, dT_hi, N_init)
    planes = []
    if use_symmetry:
        for i in range(N_init):
            for j in range(i, N_init):
                x, y = grid[i], grid[j]
                a0, a1, a2 = tangent_plane_coeffs_H(x, y, beta, U)
                planes.append((a0, a1, a2))
                if i != j:
                    planes.append((a0, a2, a1))
    else:
        for x in grid:
            for y in grid:
                planes.append(tangent_plane_coeffs_H(x, y, beta, U))
    return planes, grid


# ──────────────────────────────────────────────────────────────────────
# Validation: dense uniform grid + random log-uniform points combined
# (per your request: "1000x1000 uniform+random")
# ──────────────────────────────────────────────────────────────────────
def make_test_points(dT_lo, dT_hi, uniform_N=1000, n_random=1_000_000, seed=0,
                      include_boundary=True, n_per_edge=300):
    """
    Builds a combined test-point set:
      - uniform_N x uniform_N geomspace grid (structured coverage)
      - n_random points drawn log-uniformly over the square (unstructured
        coverage, catches facet-switching curves a grid can straddle)
      - optional boundary + dT1==dT2 diagonal points (worst-case-prone
        locations: within a single active facet, error = H - plane is convex,
        so its max over that facet sits on the facet's boundary)
    Returns (D1, D2) flat arrays.
    """
    g = np.geomspace(dT_lo, dT_hi, uniform_N)
    D1u, D2u = np.meshgrid(g, g, indexing="ij")
    D1u, D2u = D1u.ravel(), D2u.ravel()

    rng = np.random.default_rng(seed)
    logs = rng.uniform(np.log(dT_lo), np.log(dT_hi), size=(n_random, 2))
    D1r, D2r = np.exp(logs[:, 0]), np.exp(logs[:, 1])

    parts1 = [D1u, D1r]
    parts2 = [D2u, D2r]

    if include_boundary:
        line = np.geomspace(dT_lo, dT_hi, n_per_edge)
        lo = np.full(n_per_edge, dT_lo)
        hi = np.full(n_per_edge, dT_hi)
        parts1 += [lo, hi, line, line, line]
        parts2 += [line, line, lo, hi, line]

    return np.concatenate(parts1), np.concatenate(parts2)


def evaluate_envelope(planes, dT1, dT2, beta, U=1.0):
    """Returns dict of error stats for a given point set."""
    Ht = H_true(dT1, dT2, beta, U)
    Ha = envelope_value_max(dT1, dT2, planes)
    err = Ht - Ha                       # should be >= ~0 (underestimate)
    rel_err = err / np.maximum(Ht, 1e-12)
    idx = int(np.argmax(rel_err))
    n_violations = int(np.sum(err < -1e-9))   # sign-consistency check
    return {
        "dT1": dT1, "dT2": dT2, "H_true": Ht, "H_approx": Ha,
        "err": err, "rel_err": rel_err,
        "max_err": float(np.max(err)), "max_rel_err": float(rel_err[idx]),
        "avg_err": float(np.mean(err)), "avg_rel_err": float(np.mean(rel_err)),
        "worst_point": (float(dT1[idx]), float(dT2[idx])),
        "n_sign_violations": n_violations,
    }


# ──────────────────────────────────────────────────────────────────────
# Greedy adaptive refinement: add a plane at the current worst point,
# repeat until max relative error <= threshold (minimizes plane count
# for the target accuracy, unlike growing a uniform N_G x N_G grid).
# ──────────────────────────────────────────────────────────────────────
def greedy_refine_H(dT_lo, dT_hi, beta, U=1.0,
                     error_threshold=0.01, max_planes=500,
                     N_init=3, use_symmetry=True,
                     oracle_uniform_N=150, oracle_random_pts=20_000,
                     oracle_seed=0, resample_every=25, verbose=True):
    """
    Phase 1: greedily add tangent planes at whatever point in a (fixed-ish,
    periodically-resampled) oracle test set has the largest relative error,
    until the oracle-measured max relative error <= error_threshold.

    resample_every>0 redraws the random half of the oracle set every that
    many iterations, so the greedy search doesn't overfit to one random draw.
    """
    planes, _ = build_initial_planes_H(dT_lo, dT_hi, beta, U, N_init, use_symmetry)
    history = []
    D1o, D2o = make_test_points(dT_lo, dT_hi, oracle_uniform_N, oracle_random_pts,
                                 seed=oracle_seed, include_boundary=True)

    for it in range(max_planes):
        if resample_every and it > 0 and it % resample_every == 0:
            D1o, D2o = make_test_points(dT_lo, dT_hi, oracle_uniform_N, oracle_random_pts,
                                         seed=oracle_seed + it, include_boundary=True)

        res = evaluate_envelope(planes, D1o, D2o, beta, U)
        history.append({"n_planes": len(planes), "max_rel_err": res["max_rel_err"]})

        if verbose and (it % 10 == 0 or res["max_rel_err"] <= error_threshold):
            print(f"  iter={it:4d}  planes={len(planes):4d}  "
                  f"max_rel_err={res['max_rel_err']*100:8.4f}%  "
                  f"at (dT1={res['worst_point'][0]:.3f}, dT2={res['worst_point'][1]:.3f})")

        if res["max_rel_err"] <= error_threshold:
            return planes, history

        x0, y0 = res["worst_point"]
        a0, a1, a2 = tangent_plane_coeffs_H(x0, y0, beta, U)
        planes.append((a0, a1, a2))
        if use_symmetry and abs(x0 - y0) > 1e-6:
            planes.append((a0, a2, a1))

    print(f"  WARNING: max_planes={max_planes} reached without hitting "
          f"error_threshold={error_threshold*100:.2f}% (oracle set).")
    return planes, history


def select_H_envelope(dT_lo, dT_hi, beta, U=1.0, error_threshold=0.01,
                       max_planes=500, N_init=3, use_symmetry=True,
                       oracle_uniform_N=150, oracle_random_pts=20_000,
                       final_uniform_N=1000, final_random_pts=1_000_000,
                       max_outer_rounds=4, verbose=True):
    """
    Two-phase selection:
      1. greedy_refine_H against a moderate oracle set (fast, drives the
         search) until the oracle says max_rel_err <= error_threshold.
      2. Independent, much larger final validation (default 1000x1000
         uniform + 1e6 random, per your spec). If the oracle under-sampled
         and the true max error is still above threshold, feed the fresh
         worst point back into another greedy round and repeat (bounded by
         max_outer_rounds) instead of silently trusting a too-small oracle.
    Returns (planes, final_validation_result, history).
    """
    planes = None
    full_history = []
    for round_i in range(max_outer_rounds):
        if verbose:
            print(f"\n=== Outer round {round_i+1} ===")
        if planes is None:
            planes, hist = greedy_refine_H(
                dT_lo, dT_hi, beta, U, error_threshold, max_planes,
                N_init, use_symmetry, oracle_uniform_N, oracle_random_pts,
                oracle_seed=round_i * 1000, verbose=verbose)
        else:
            # resume greedy refinement seeded with the existing planes
            hist = []
            D1o, D2o = make_test_points(dT_lo, dT_hi, oracle_uniform_N, oracle_random_pts,
                                         seed=round_i * 1000, include_boundary=True)
            for it in range(max_planes):
                res = evaluate_envelope(planes, D1o, D2o, beta, U)
                hist.append({"n_planes": len(planes), "max_rel_err": res["max_rel_err"]})
                if res["max_rel_err"] <= error_threshold:
                    break
                x0, y0 = res["worst_point"]
                a0, a1, a2 = tangent_plane_coeffs_H(x0, y0, beta, U)
                planes.append((a0, a1, a2))
                if use_symmetry and abs(x0 - y0) > 1e-6:
                    planes.append((a0, a2, a1))
        full_history += hist

        if verbose:
            print(f"Validating against final grid "
                  f"({final_uniform_N}x{final_uniform_N} uniform + {final_random_pts} random)...")
        D1f, D2f = make_test_points(dT_lo, dT_hi, final_uniform_N, final_random_pts,
                                     seed=999 + round_i)
        final_result = evaluate_envelope(planes, D1f, D2f, beta, U)
        if verbose:
            print(f"  -> {len(planes)} planes, final max_rel_err="
                  f"{final_result['max_rel_err']*100:.4f}% "
                  f"at (dT1={final_result['worst_point'][0]:.3f}, "
                  f"dT2={final_result['worst_point'][1]:.3f})")

        if final_result["max_rel_err"] <= error_threshold:
            return planes, final_result, full_history

        if verbose:
            print("  Final grid found a worse point than the oracle did -> "
                  "adding it and doing another refinement round.")

    print(f"  WARNING: target {error_threshold*100:.2f}% not confirmed after "
          f"{max_outer_rounds} outer rounds. Returning best available "
          f"({len(planes)} planes, {final_result['max_rel_err']*100:.3f}%).")
    return planes, final_result, full_history


# ──────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────
def plot_3d_comparison(dT_lo, dT_hi, planes, beta, U=1.0, plot_N=100,
                        title=None, angles=((25, -60), (25, 30), (60, -45), (10, -110))):
    g = np.geomspace(dT_lo, dT_hi, plot_N)
    D1, D2 = np.meshgrid(g, g, indexing="ij")
    Ht = H_true(D1, D2, beta, U)
    Ha = envelope_value_max(D1, D2, planes)

    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(D1, D2, Ht, alpha=0.55, cmap="viridis", edgecolor="none")
        ax.plot_wireframe(D1, D2, Ha, color="red", linewidth=0.4, rstride=5, cstride=5)
        ax.set_xlabel("dT1"); ax.set_ylabel("dT2"); ax.set_zlabel("H")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle((title or f"H=(U*LMTD)^-{beta}: true vs. envelope (max-of-planes)")
                 + "  (viridis=true, red wireframe=envelope, envelope <= true)")
    fig.tight_layout()
    return fig


def plot_error_surface(dT_lo, dT_hi, planes, beta, U=1.0, plot_N=100,
                        angles=((25, -60), (60, -45))):
    g = np.geomspace(dT_lo, dT_hi, plot_N)
    D1, D2 = np.meshgrid(g, g, indexing="ij")
    Ht = H_true(D1, D2, beta, U)
    Ha = envelope_value_max(D1, D2, planes)
    rel_err = (Ht - Ha) / np.maximum(Ht, 1e-12)

    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(D1, D2, rel_err * 100, cmap="inferno")
        ax.set_xlabel("dT1"); ax.set_ylabel("dT2"); ax.set_zlabel("Rel. error (%)")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle("Underestimation error (%) of H envelope over (dT1, dT2)")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────
def analyze_H(dT_hi, delta_tmin, beta, U=1.0, error_threshold=0.01,
              max_planes=500, N_init=3, use_symmetry=True,
              oracle_uniform_N=150, oracle_random_pts=20_000,
              final_uniform_N=1000, final_random_pts=1_000_000,
              make_plots=True):
    dT_lo = delta_tmin
    if dT_hi <= dT_lo:
        raise ValueError(f"Infeasible: dT_hi ({dT_hi}) <= dT_lo ({dT_lo}).")

    print(f"dT range: [{dT_lo:.4f}, {dT_hi:.4f}]   beta={beta}   U={U}\n")
    planes, result, history = select_H_envelope(
        dT_lo, dT_hi, beta, U, error_threshold, max_planes, N_init, use_symmetry,
        oracle_uniform_N, oracle_random_pts, final_uniform_N, final_random_pts)

    print(f"\n--- Result ---")
    print(f"Planes              : {len(planes)}")
    print(f"Max relative error  : {result['max_rel_err']*100:.4f}%  at "
          f"(dT1={result['worst_point'][0]:.3f}, dT2={result['worst_point'][1]:.3f})")
    print(f"Avg relative error  : {result['avg_rel_err']*100:.4f}%")
    print(f"Sign violations     : {result['n_sign_violations']} "
          f"({'OK - strictly underestimating' if result['n_sign_violations']==0 else 'WARNING'})")

    if make_plots:
        plot_3d_comparison(dT_lo, dT_hi, planes, beta, U)
        plot_error_surface(dT_lo, dT_hi, planes, beta, U)
        plt.show()

    return {"planes": planes, "result": result, "history": history,
            "dT_lo": dT_lo, "dT_hi": dT_hi, "beta": beta, "U": U}


if __name__ == "__main__":
    Tin_H = 220.0
    Tin_C = 140.0
    delta_tmin = 10.0
    beta = 0.6

    out = analyze_H(
        Tin_H - Tin_C, delta_tmin, beta, U=1.0,
        error_threshold=0.01,
        max_planes=500,
        N_init=3,
        oracle_uniform_N=150, oracle_random_pts=20_000,
        final_uniform_N=1000, final_random_pts=1_000_000,
        make_plots=False,   # flip to True if you want the 3D plots
    )