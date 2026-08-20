"""
-beta*ln(LMTD) Tangent-Plane (1st-order Taylor) Hyperplane Tester
====================================================================
Builds supporting-hyperplane (tangent-plane) outer approximations for

    f(dT1, dT2) = -cost_beta * ln(LMTD(dT1, dT2))
    LMTD(dT1, dT2) = (dT1 - dT2) / ln(dT1/dT2)

for use in a Pyomo/SCIP HENS MILP, instead of approximating LMTD
itself.

Why the envelope direction flips vs. the old LMTD-only script
---------------------------------------------------------------
LMTD is concave and positive on dT1,dT2 > 0, so tangent planes to LMTD
sit ABOVE it -> min-of-planes gives a valid overestimator.

ln(.) is concave and increasing, so ln(LMTD) is also concave (concave
composed with concave-increasing is concave). Negating flips
concavity, so f = -cost_beta*ln(LMTD) is CONVEX (for cost_beta > 0).
Tangent planes to a convex function sit BELOW it, so the correct
outer approximation is now the MAX over planes (the tightest valid
lower bound / supporting-hyperplane relaxation), not the min. This
is the standard convex-underestimator ("f_hat >= a0 + a1*dT1 +
a2*dT2" per plane) formulation for a MILP epigraph reformulation.

All error/validation bookkeeping below has been flipped to match:
err = f_true - f_approx, and a valid envelope has err >= 0 everywhere
(the piecewise-linear max-of-planes never exceeds the true convex
function).
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)


# ──────────────────────────────────────────────────────────────────────
# Core LMTD value + analytic gradient (dT1 == dT2 singularity handled)
# ──────────────────────────────────────────────────────────────────────
def lmtd_and_grad(dT1, dT2, tol=0.1):
    """
    Vectorized LMTD value + partial derivatives w.r.t. dT1, dT2.
    Near dT1 == dT2 (removable 0/0 singularity), uses the analytic
    Taylor-series limit: LMTD -> dT, gradient -> (0.5, 0.5).

    This is an internal building block: f = -cost_beta*ln(LMTD) and
    its gradient are obtained from this via the chain rule in
    neg_beta_ln_lmtd_and_grad() below, so the same near-diagonal
    handling automatically carries over to f.
    """
    dT1 = np.atleast_1d(np.asarray(dT1, dtype=float))
    dT2 = np.atleast_1d(np.asarray(dT2, dtype=float))
    scalar_input = dT1.size == 1 and dT2.size == 1 and np.isscalar(dT1[0])

    dT1b, dT2b = np.broadcast_arrays(dT1, dT2)
    L = np.empty_like(dT1b, dtype=float)
    dLd1 = np.empty_like(dT1b, dtype=float)
    dLd2 = np.empty_like(dT1b, dtype=float)

    close = np.abs(dT1b - dT2b) < tol
    far = ~close

    if np.any(close):
        avg = 0.5 * (dT1b[close] + dT2b[close])
        L[close] = avg
        dLd1[close] = 0.5
        dLd2[close] = 0.5

    if np.any(far):
        d1, d2 = dT1b[far], dT2b[far]
        ln_ratio = np.log(d1) - np.log(d2)
        L[far] = (d1 - d2) / ln_ratio
        dLd1[far] = (ln_ratio - (d1 - d2) / d1) / ln_ratio**2
        dLd2[far] = (-ln_ratio + (d1 - d2) / d2) / ln_ratio**2

    if scalar_input:
        return float(L.flat[0]), float(dLd1.flat[0]), float(dLd2.flat[0])
    return L, dLd1, dLd2


def lmtd_value(dT1, dT2, tol=0.1):
    L, _, _ = lmtd_and_grad(dT1, dT2, tol)
    return L


# ──────────────────────────────────────────────────────────────────────
# f = -cost_beta * ln(LMTD) value + gradient, via chain rule on LMTD
# ──────────────────────────────────────────────────────────────────────
def neg_beta_ln_lmtd_and_grad(dT1, dT2, cost_beta, tol=0.1):
    """
    Vectorized f = -cost_beta*ln(LMTD(dT1,dT2)) value + partial
    derivatives w.r.t. dT1, dT2.

    Obtained via chain rule from lmtd_and_grad:
        f      = -cost_beta * ln(L)
        df/dx  = -cost_beta * (dL/dx) / L
        df/dy  = -cost_beta * (dL/dy) / L

    This is algebraically identical to the closed forms
        df/dx = cost_beta*((x - x*ln(x/y) - y) / (x*(x-y)*ln(x/y)))
        df/dy = cost_beta*((y + y*ln(x/y) - x) / (y*(x-y)*ln(x/y)))
    (verified numerically), and automatically inherits the
    near-diagonal (dT1 ~ dT2) handling from lmtd_and_grad: at dT1 ==
    dT2 == T exactly, the limit is f -> -cost_beta*ln(T) and
    df/dx == df/dy -> -cost_beta/(2T).
    """
    L, dLd1, dLd2 = lmtd_and_grad(dT1, dT2, tol)
    f = -cost_beta * np.log(L)
    dfd1 = -cost_beta * dLd1 / L
    dfd2 = -cost_beta * dLd2 / L
    return f, dfd1, dfd2


def f_true(dT1, dT2, cost_beta, tol=0.1):
    f, _, _ = neg_beta_ln_lmtd_and_grad(dT1, dT2, cost_beta, tol)
    return f


def tangent_plane_coeffs(dT1_0, dT2_0, cost_beta, tol=1e-6):
    """Returns (a0,a1,a2) s.t. f(dT1,dT2) >= a0 + a1*dT1 + a2*dT2 (f convex)."""
    f0, g1, g2 = neg_beta_ln_lmtd_and_grad(dT1_0, dT2_0, cost_beta, tol)
    a0 = f0 - g1 * dT1_0 - g2 * dT2_0
    return a0, g1, g2


# ──────────────────────────────────────────────────────────────────────
# Hyperplane construction (with symmetry-exploiting evaluation count)
# ──────────────────────────────────────────────────────────────────────
def build_tangent_hyperplanes(dT_lo, dT_hi, N_G, cost_beta, use_symmetry=True):
    """
    Builds a family of tangent-plane coefficients over a geomspace
    N_G x N_G grid spanning [dT_lo, dT_hi] for BOTH dT1 and dT2 (same
    range for both, matching the MILP preprocessing convention).

    use_symmetry=True: exploits f(dT1,dT2) == f(dT2,dT1) (LMTD is
    symmetric, and so is -cost_beta*ln(LMTD)). Only the
    upper-triangular N_G*(N_G+1)/2 points need an actual gradient
    evaluation; each off-diagonal point's mirror plane is obtained for
    free by swapping (a1,a2). Produces the same N_G^2-plane envelope
    with fewer evaluations (e.g. N_G=7 -> 28 evaluations -> 49 planes).

    Returns (planes, grid) where planes is a list of (a0,a1,a2) tuples.
    """
    grid = np.geomspace(dT_lo, dT_hi, N_G)
    planes = []

    if use_symmetry:
        for i in range(N_G):
            for j in range(i, N_G):
                x, y = grid[i], grid[j]
                a0, a1, a2 = tangent_plane_coeffs(x, y, cost_beta)
                planes.append((a0, a1, a2))
                if i != j:
                    planes.append((a0, a2, a1))  # mirror via symmetry
    else:
        for x in grid:
            for y in grid:
                planes.append(tangent_plane_coeffs(x, y, cost_beta))

    return planes, grid


def envelope_value(dT1, dT2, planes):
    """
    Piecewise-linear outer approximation = MAX over all tangent planes.
    (f is convex, so the tightest valid lower-bound envelope is the
    max of its supporting hyperplanes -- this is the flip vs. the
    old LMTD-only script, which used min because LMTD is concave.)
    """
    dT1 = np.asarray(dT1, dtype=float)
    dT2 = np.asarray(dT2, dtype=float)
    a0 = np.array([p[0] for p in planes])
    a1 = np.array([p[1] for p in planes])
    a2 = np.array([p[2] for p in planes])
    vals = a0 + a1 * dT1[..., None] + a2 * dT2[..., None]
    return vals.max(axis=-1)


# ──────────────────────────────────────────────────────────────────────
# Dense-grid validation
# ──────────────────────────────────────────────────────────────────────
def test_against_dense_grid(planes, dT_lo, dT_hi, cost_beta, dense_N=100, use_geomspace=True):
    """
    Evaluates true f = -cost_beta*ln(LMTD) vs. the hyperplane envelope
    over a dense dense_N x dense_N test grid and reports error
    statistics.
    """
    g = np.geomspace(dT_lo, dT_hi, dense_N) if use_geomspace else np.linspace(dT_lo, dT_hi, dense_N)
    D1, D2 = np.meshgrid(g, g, indexing="ij")

    F_true = f_true(D1, D2, cost_beta)
    F_approx = envelope_value(D1, D2, planes)

    err = F_true - F_approx                        # should be >= ~0 (envelope underestimates)
    rel_err = err / np.maximum(np.abs(F_true), 1e-9)

    idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)
    n_violations = int(np.sum(err < -1e-6))         # sign-consistency check

    return {
        "grid": g, "D1": D1, "D2": D2, "L_true": F_true, "L_approx": F_approx,
        "err": err, "rel_err": rel_err,
        "max_err": float(np.max(err)),
        "max_rel_err": float(np.max(rel_err)),
        "avg_err": float(np.mean(err)),
        "avg_rel_err": float(np.mean(rel_err)),
        "worst_point": (float(D1[idx]), float(D2[idx])),
        "n_sign_violations": n_violations,
    }


# ──────────────────────────────────────────────────────────────────────
# Structured (boundary + diagonal) validation points
#
# Within any region where a single tangent plane is the active
# maximum, error(dT1,dT2) = f(dT1,dT2) - plane(dT1,dT2) is a CONVEX
# function (f itself is convex) MINUS an affine one -> still CONVEX
# on that region. A convex function's max over a region is always on
# the region's boundary, not the interior. So the true worst-case
# error lives on the domain edges, the dT1==dT2 diagonal, or a
# plane-switching curve -- a uniform/dense grid can straddle these
# without ever sampling exactly on them.
# ──────────────────────────────────────────────────────────────────────
def build_boundary_and_diagonal_points(dT_lo, dT_hi, n_per_edge=200, use_geomspace=True):
    """
    Targeted test points along the four domain edges + the dT1==dT2
    diagonal, to catch the structurally likely worst-case error
    locations that a uniform interior grid can miss.
    """
    line = (np.geomspace(dT_lo, dT_hi, n_per_edge) if use_geomspace
            else np.linspace(dT_lo, dT_hi, n_per_edge))
    lo = np.full(n_per_edge, dT_lo)
    hi = np.full(n_per_edge, dT_hi)

    # edges: dT1=lo, dT1=hi, dT2=lo, dT2=hi  +  diagonal dT1==dT2
    dT1_pts = np.concatenate([lo, hi, line, line, line])
    dT2_pts = np.concatenate([line, line, lo, hi, line])
    return dT1_pts, dT2_pts


def test_against_structured_points(planes, dT1_pts, dT2_pts, cost_beta):
    """Same error stats as test_against_dense_grid, but for a flat point set."""
    F_true = f_true(dT1_pts, dT2_pts, cost_beta)
    F_approx = envelope_value(dT1_pts, dT2_pts, planes)
    err = F_true - F_approx
    rel_err = err / np.maximum(np.abs(F_true), 1e-9)
    idx = int(np.argmax(rel_err))
    n_violations = int(np.sum(err < -1e-6))
    return {
        "dT1": dT1_pts, "dT2": dT2_pts, "L_true": F_true, "L_approx": F_approx,
        "err": err, "rel_err": rel_err,
        "max_err": float(np.max(err)), "max_rel_err": float(np.max(rel_err)),
        "avg_err": float(np.mean(err)), "avg_rel_err": float(np.mean(rel_err)),
        "worst_point": (float(dT1_pts[idx]), float(dT2_pts[idx])),
        "n_sign_violations": n_violations,
    }


def full_validation(planes, dT_lo, dT_hi, cost_beta, dense_N=100, n_per_edge=200, use_geomspace=True):
    """
    Combines the dense interior grid with targeted boundary/diagonal
    sampling. max_err / max_rel_err reflect the WORST of both sources
    (this is the number you should trust as "worst case"); avg_err /
    avg_rel_err stay sourced from the dense grid only, since boundary
    points would otherwise bias the "typical case" statistic.
    """
    dense_result = test_against_dense_grid(planes, dT_lo, dT_hi, cost_beta, dense_N, use_geomspace)
    dT1_b, dT2_b = build_boundary_and_diagonal_points(dT_lo, dT_hi, n_per_edge, use_geomspace)
    boundary_result = test_against_structured_points(planes, dT1_b, dT2_b, cost_beta)

    if dense_result["max_rel_err"] >= boundary_result["max_rel_err"]:
        worst_source, worst_point = "dense_grid", dense_result["worst_point"]
    else:
        worst_source, worst_point = "boundary/diagonal", boundary_result["worst_point"]

    return {
        "dense": dense_result,
        "boundary": boundary_result,
        "max_err": max(dense_result["max_err"], boundary_result["max_err"]),
        "max_rel_err": max(dense_result["max_rel_err"], boundary_result["max_rel_err"]),
        "avg_err": dense_result["avg_err"],
        "avg_rel_err": dense_result["avg_rel_err"],
        "worst_point": worst_point,
        "worst_source": worst_source,
        "n_sign_violations": dense_result["n_sign_violations"] + boundary_result["n_sign_violations"],
    }


def dense_grid_convergence_check(planes, dT_lo, dT_hi, cost_beta,
                                  dense_N_list=(50, 100, 200, 400, 800),
                                  n_per_edge=200, verbose=True):
    """
    Sweeps the TEST grid resolution (dense_N) itself and reports how
    max_rel_err changes -- validates that dense_N is fine enough,
    the same way select_grid_size_dynamic validates N_G. If the
    reported max keeps climbing as dense_N grows, your current
    dense_N is understating the true worst-case error.
    """
    history = []
    for dense_N in dense_N_list:
        result = full_validation(planes, dT_lo, dT_hi, cost_beta, dense_N=dense_N, n_per_edge=n_per_edge)
        history.append({"dense_N": dense_N, "max_rel_err": result["max_rel_err"],
                         "worst_source": result["worst_source"], "worst_point": result["worst_point"]})
        if verbose:
            print(f"  dense_N={dense_N:4d}  max_rel_err={result['max_rel_err']*100:.4f}%  "
                  f"(worst on {result['worst_source']}, at {result['worst_point']})")

    if len(history) >= 2:
        drift = abs(history[-1]["max_rel_err"] - history[-2]["max_rel_err"])
        if verbose:
            if drift < 1e-4:
                print(f"  -> stabilized (change over last step: {drift*100:.5f} pts). "
                      f"dense_N={history[-2]['dense_N']} was likely already sufficient.")
            else:
                print(f"  -> STILL DRIFTING (change over last step: {drift*100:.5f} pts). "
                      f"Consider testing beyond dense_N={dense_N_list[-1]}.")
    return history


# ──────────────────────────────────────────────────────────────────────
# Dynamic grid-size selection
# ──────────────────────────────────────────────────────────────────────
def select_grid_size_dynamic(dT_lo, dT_hi, cost_beta, error_threshold=0.02,
                              convergence_tol=1e-3, N_start=3, N_max=20,
                              dense_N=100, n_per_edge=200, use_symmetry=True, verbose=True):
    """
    Grows N_G until (a) max relative error -- across the dense interior
    grid AND the boundary/diagonal points (see full_validation) -- is
    <= error_threshold, AND (b) the change in that max relative error
    from the previous N_G is <= convergence_tol (diminishing returns).
    Falls back to the best N_G found (with a warning) if N_max is hit
    without satisfying both conditions.

    Returns (N_G, planes, validation_result, history).
    """
    history = []
    prev_max_rel = None
    best = None

    for N_G in range(N_start, N_max + 1):
        planes, _grid = build_tangent_hyperplanes(dT_lo, dT_hi, N_G, cost_beta, use_symmetry)
        result = full_validation(planes, dT_lo, dT_hi, cost_beta, dense_N=dense_N, n_per_edge=n_per_edge)
        history.append({"N_G": N_G, "n_planes": len(planes),
                         **{k: result[k] for k in
                            ("max_err", "max_rel_err", "avg_err", "avg_rel_err",
                             "worst_point", "worst_source", "n_sign_violations")}})

        if verbose:
            print(f"  N_G={N_G:2d} ({len(planes):3d} planes)  "
                  f"max_rel_err={result['max_rel_err']*100:7.4f}%  "
                  f"avg_rel_err={result['avg_rel_err']*100:7.4f}%  "
                  f"sign_violations={result['n_sign_violations']}")

        if best is None or result["max_rel_err"] < best["max_rel_err"]:
            best = {"N_G": N_G, "planes": planes, **result}

        converged = (prev_max_rel is not None and
                     abs(prev_max_rel - result["max_rel_err"]) <= convergence_tol)
        acceptable = result["max_rel_err"] <= error_threshold

        if converged and acceptable:
            return N_G, planes, result, history

        prev_max_rel = result["max_rel_err"]

    print(f"  WARNING: target error {error_threshold*100:.2f}% not reached within "
          f"N_max={N_max}. Returning best found: N_G={best['N_G']} "
          f"(max_rel_err={best['max_rel_err']*100:.3f}%).")
    return best["N_G"], best["planes"], best, history


# ──────────────────────────────────────────────────────────────────────
# 3D plotting
# ──────────────────────────────────────────────────────────────────────
def plot_3d_comparison(result, title="-beta*ln(LMTD): true surface vs. tangent-plane envelope",
                        angles=((25, -60), (25, 30), (60, -45), (10, -110))):
    D1, D2, L_true, L_approx = result["D1"], result["D2"], result["L_true"], result["L_approx"]
    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(D1, D2, L_true, alpha=0.55, cmap="viridis", edgecolor="none")
        ax.plot_wireframe(D1, D2, L_approx, color="red", linewidth=0.4, rstride=5, cstride=5)
        ax.set_xlabel("dT1"); ax.set_ylabel("dT2"); ax.set_zlabel("-beta*ln(LMTD)")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle(title + "  (green/viridis=true, red wireframe=envelope)")
    fig.tight_layout()
    return fig


def plot_error_surface(result, angles=((25, -60), (60, -45))):
    D1, D2, rel_err = result["D1"], result["D2"], result["rel_err"]
    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))
    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        surf = ax.plot_surface(D1, D2, rel_err * 100, cmap="inferno")
        ax.set_xlabel("dT1"); ax.set_ylabel("dT2"); ax.set_zlabel("Rel. error (%)")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)
    fig.suptitle("Relative approximation error (%) over (dT1, dT2), err = f_true - f_envelope")
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────────────────────────────
# Top-level driver
# ──────────────────────────────────────────────────────────────────────
def analyze_match(dT_hi, delta_tmin, cost_beta,
                   mode="dynamic", N_G_fixed=8,
                   error_threshold=0.02, convergence_tol=1e-3,
                   N_start=3, N_max=20, dense_N=100, n_per_edge=200,
                   use_symmetry=True, make_plots=True,
                   check_dense_N_convergence=False,
                   dense_N_sweep=(50, 100, 200, 400, 800)):
    """
    mode='dynamic' -> grow N_G until converged & below error_threshold
    mode='fixed'   -> use N_G_fixed directly, still fully validated

    Builds a supporting-hyperplane (max-of-planes) outer approximation
    of f(dT1,dT2) = -cost_beta*ln(LMTD(dT1,dT2)), which is CONVEX, so
    the envelope is a valid lower bound (f_true >= envelope
    everywhere) -- the natural form for a MILP epigraph reformulation
    (f_hat >= a0 + a1*dT1 + a2*dT2 per plane, f_hat minimized).

    Validation combines the dense interior grid with targeted
    boundary/diagonal sampling (see full_validation) -- the max
    relative error reported is the worst of both sources, since the
    true worst case for a tangent-plane envelope structurally sits on
    a boundary or plane-switching curve, not necessarily an interior
    grid point.

    Set check_dense_N_convergence=True to first sweep dense_N itself
    (dense_N_sweep) and confirm the reported error has stabilized
    before trusting the chosen dense_N.
    """
    dT_lo = delta_tmin
    if dT_hi <= dT_lo:
        raise ValueError(f"Infeasible/degenerate match: dT_hi ({dT_hi}) <= dT_lo ({dT_lo}). "
                          f"Check Tin_H > Tin_C + delta_tmin.")

    print(f"dT range: [{dT_lo:.4f}, {dT_hi:.4f}]   cost_beta={cost_beta}\n")

    if mode == "dynamic":
        print("Running dynamic grid-size sweep...")
        N_G, planes, result, history = select_grid_size_dynamic(
            dT_lo, dT_hi, cost_beta, error_threshold, convergence_tol,
            N_start, N_max, dense_N, n_per_edge, use_symmetry)
    elif mode == "fixed":
        N_G = N_G_fixed
        planes, _grid = build_tangent_hyperplanes(dT_lo, dT_hi, N_G, cost_beta, use_symmetry)
        result = full_validation(planes, dT_lo, dT_hi, cost_beta, dense_N=dense_N, n_per_edge=n_per_edge)
        history = None
    else:
        raise ValueError("mode must be 'dynamic' or 'fixed'")

    if check_dense_N_convergence:
        print("\nChecking whether dense_N is fine enough (test-grid resolution sweep)...")
        dense_grid_convergence_check(planes, dT_lo, dT_hi, cost_beta, dense_N_sweep, n_per_edge)

    print(f"\n--- Result (N_G={N_G}, {len(planes)} tangent planes, "
          f"{dense_N}x{dense_N} interior grid + {n_per_edge}/edge boundary+diagonal) ---")
    print(f"Max abs error       : {result['max_err']:.6f}")
    print(f"Max relative error  : {result['max_rel_err']*100:.4f}%  "
          f"at (dT1={result['worst_point'][0]:.3f}, dT2={result['worst_point'][1]:.3f})  "
          f"[worst on: {result['worst_source']}]")
    print(f"  (interior grid max : {result['dense']['max_rel_err']*100:.4f}%)")
    print(f"  (boundary/diag max : {result['boundary']['max_rel_err']*100:.4f}%)")
    print(f"Avg abs error       : {result['avg_err']:.6f}")
    print(f"Avg relative error  : {result['avg_rel_err']*100:.4f}%  (interior grid only)")
    print(f"Sign violations     : {result['n_sign_violations']} "
          f"({'OK - envelope <= true everywhere' if result['n_sign_violations'] == 0 else 'WARNING: envelope exceeded true value somewhere!'})")

    if result["max_rel_err"] > error_threshold:
        print(f"WARNING: max relative error exceeds threshold ({error_threshold*100:.2f}%).")

    if make_plots:
        plot_3d_comparison(result["dense"])
        plot_error_surface(result["dense"])
        plt.show()

    return {"N_G": N_G, "planes": planes, "result": result, "history": history,
            "dT_lo": dT_lo, "dT_hi": dT_hi, "cost_beta": cost_beta}


if __name__ == "__main__":
    # Edit these for the match you want to validate:
    Tin_H = 220.0
    Tin_C = 140.0
    delta_tmin = 10.0
    cost_beta = 1.0

    out = analyze_match(
        Tin_H - Tin_C, delta_tmin, cost_beta,
        mode="dynamic",           # or "fixed"
        N_G_fixed=10,
        error_threshold=0.01,     # 1% max relative error target
        convergence_tol=0.001,    # stop growing grid once error stalls
        N_start=3, N_max=20,
        dense_N=100,
        use_symmetry=True,
        make_plots=True,
    )