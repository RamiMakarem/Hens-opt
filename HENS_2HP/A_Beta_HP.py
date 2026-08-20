"""
A_Beta Tangent-Plane (1st-order Taylor) Hyperplane Generator & Tester
===================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


# ──────────────────────────────────────────────────────────────────────
# Core Mathematical & Hyperplane Generators
# ──────────────────────────────────────────────────────────────────────
def lmtd_val(d1, d2):
    """Computes exact LMTD for given temperature differences (d1, d2)."""
    if abs(d1 - d2) < 1e-4:
        return 0.5 * (d1 + d2)
    return (d1 - d2) / np.log(d1 / d2)


def abeta_and_grad(Q, LMTD, beta=0.6, U=1.0):
    """Computes f = (Q / (U * LMTD))^beta and its gradients w.r.t Q and LMTD."""
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    L = np.atleast_1d(np.asarray(LMTD, dtype=float))
    scalar_input = Q.size == 1 and L.size == 1 and np.isscalar(Q[0])

    Qb, Lb = np.broadcast_arrays(Q, L)
    
    f = (Qb / (U * Lb)) ** beta
    df_dQ = beta * f / Qb
    df_dL = -beta * f / Lb

    if scalar_input:
        return float(f.flat[0]), float(df_dQ.flat[0]), float(df_dL.flat[0])
    return f, df_dQ, df_dL


def abeta_tangent_plane_coeffs(Q0, L0, beta=0.6, U=1.0):
    """Returns (a0, a1, a2) s.t. A^beta(Q, LMTD) ≈ a0 + a1*Q + a2*LMTD."""
    f0, g1, g2 = abeta_and_grad(Q0, L0, beta=beta, U=U)
    a0 = f0  # f0 - g1*Q0 - g2*L0 simplifies analytically to f0
    return a0, g1, g2


def build_abeta_hyperplanes(Q_lo, Q_hi, L_lo, L_hi, N_Q, N_L, beta=0.6, U=1.0):
    """Builds tangent-plane coefficients over a grid spanning [Q_lo, Q_hi] x [L_lo, L_hi]."""
    grid_Q = np.geomspace(Q_lo, Q_hi, N_Q)
    grid_L = np.geomspace(L_lo, L_hi, N_L)
    planes = []

    for q in grid_Q:
        for l in grid_L:
            planes.append(abeta_tangent_plane_coeffs(q, l, beta=beta, U=U))

    return planes, grid_Q, grid_L


def abeta_true(Q, LMTD, beta=0.6, U=1.0):
    """Computes exact A^beta = (Q / (U * LMTD))^beta."""
    return (np.asarray(Q, dtype=float) / (U * np.asarray(LMTD, dtype=float))) ** beta


def local_plane_value_abeta(Q, LMTD, grid_Q, grid_L, planes):
    """Evaluates query points using tangent plane centered at nearest grid point."""
    Q = np.asarray(Q, dtype=float)
    L = np.asarray(LMTD, dtype=float)

    idx_Q = np.abs(np.log(Q[..., None]) - np.log(grid_Q)).argmin(axis=-1)
    idx_L = np.abs(np.log(L[..., None]) - np.log(grid_L)).argmin(axis=-1)

    N_L = len(grid_L)
    plane_idx = idx_Q * N_L + idx_L

    a0 = np.array([planes[k][0] for k in plane_idx.flat]).reshape(Q.shape)
    a1 = np.array([planes[k][1] for k in plane_idx.flat]).reshape(Q.shape)
    a2 = np.array([planes[k][2] for k in plane_idx.flat]).reshape(Q.shape)

    return a0 + a1 * Q + a2 * L


# ──────────────────────────────────────────────────────────────────────
# Validation & Error Analysis Functions
# ──────────────────────────────────────────────────────────────────────
def test_against_dense_grid_abeta(grid_Q, grid_L, planes, Q_lo, Q_hi, L_lo, L_hi,
                                  beta=0.6, U=1.0, dense_N=100):
    """Evaluates true A^beta vs. piecewise local tangent planes over dense interior grid."""
    g_Q = np.geomspace(Q_lo, Q_hi, dense_N)
    g_L = np.geomspace(L_lo, L_hi, dense_N)
    QQ, LL = np.meshgrid(g_Q, g_L, indexing="ij")

    A_true = abeta_true(QQ, LL, beta=beta, U=U)
    A_approx = local_plane_value_abeta(QQ, LL, grid_Q, grid_L, planes)

    abs_err = np.abs(A_approx - A_true)
    rel_err = abs_err / np.maximum(A_true, 1e-9)
    idx = np.unravel_index(np.argmax(rel_err), rel_err.shape)

    return {
        "D_Q": QQ, "D_L": LL, "A_true": A_true, "A_approx": A_approx,
        "err": A_approx - A_true, "rel_err": rel_err,
        "max_err": float(np.max(abs_err)),
        "max_rel_err": float(np.max(rel_err)),
        "avg_err": float(np.mean(abs_err)),
        "avg_rel_err": float(np.mean(rel_err)),
        "worst_point": (float(QQ[idx]), float(LL[idx])),
    }


def build_boundary_points_abeta(Q_lo, Q_hi, L_lo, L_hi, n_per_edge=200):
    """Targeted sampling along the four domain edges."""
    line_Q = np.geomspace(Q_lo, Q_hi, n_per_edge)
    line_L = np.geomspace(L_lo, L_hi, n_per_edge)

    Q_pts = np.concatenate([np.full(n_per_edge, Q_lo), np.full(n_per_edge, Q_hi), line_Q, line_Q])
    L_pts = np.concatenate([line_L, line_L, np.full(n_per_edge, L_lo), np.full(n_per_edge, L_hi)])

    return Q_pts, L_pts


def test_against_structured_points_abeta(grid_Q, grid_L, planes, Q_pts, L_pts, beta=0.6, U=1.0):
    """Evaluates error stats over boundary point arrays."""
    A_true = abeta_true(Q_pts, L_pts, beta=beta, U=U)
    A_approx = local_plane_value_abeta(Q_pts, L_pts, grid_Q, grid_L, planes)

    abs_err = np.abs(A_approx - A_true)
    rel_err = abs_err / np.maximum(A_true, 1e-9)
    idx = int(np.argmax(rel_err))

    return {
        "Q": Q_pts, "L": L_pts, "A_true": A_true, "A_approx": A_approx,
        "err": A_approx - A_true, "rel_err": rel_err,
        "max_err": float(np.max(abs_err)),
        "max_rel_err": float(np.max(rel_err)),
        "avg_err": float(np.mean(abs_err)),
        "avg_rel_err": float(np.mean(rel_err)),
        "worst_point": (float(Q_pts[idx]), float(L_pts[idx])),
    }


def full_validation_abeta(grid_Q, grid_L, planes, Q_lo, Q_hi, L_lo, L_hi,
                          beta=0.6, U=1.0, dense_N=100, n_per_edge=200):
    """Combines interior grid and domain boundary evaluations."""
    dense_res = test_against_dense_grid_abeta(
        grid_Q, grid_L, planes, Q_lo, Q_hi, L_lo, L_hi, beta, U, dense_N
    )
    Q_b, L_b = build_boundary_points_abeta(Q_lo, Q_hi, L_lo, L_hi, n_per_edge)
    boundary_res = test_against_structured_points_abeta(
        grid_Q, grid_L, planes, Q_b, L_b, beta, U
    )

    if dense_res["max_rel_err"] >= boundary_res["max_rel_err"]:
        worst_source, worst_point = "dense_grid", dense_res["worst_point"]
    else:
        worst_source, worst_point = "boundary", boundary_res["worst_point"]

    return {
        "dense": dense_res,
        "boundary": boundary_res,
        "max_err": max(dense_res["max_err"], boundary_res["max_err"]),
        "max_rel_err": max(dense_res["max_rel_err"], boundary_res["max_rel_err"]),
        "avg_err": dense_res["avg_err"],
        "avg_rel_err": dense_res["avg_rel_err"],
        "worst_point": worst_point,
        "worst_source": worst_source,
    }


def select_grid_size_dynamic_abeta(Q_lo, Q_hi, L_lo, L_hi, beta=0.6, U=1.0,
                                   error_threshold=0.01, convergence_tol=1e-4,
                                   N_start=3, N_max=20, dense_N=100, n_per_edge=200, verbose=False):
    """Grows grid resolution N_G dynamically until max relative error <= error_threshold."""
    history = []
    prev_max_rel = None
    best = None

    for N_G in range(N_start, N_max + 1):
        planes, grid_Q, grid_L = build_abeta_hyperplanes(
            Q_lo, Q_hi, L_lo, L_hi, N_Q=N_G, N_L=N_G, beta=beta, U=U
        )
        result = full_validation_abeta(
            grid_Q, grid_L, planes, Q_lo, Q_hi, L_lo, L_hi,
            beta=beta, U=U, dense_N=dense_N, n_per_edge=n_per_edge
        )

        history.append({
            "N_G": N_G, "n_planes": len(planes),
            **{k: result[k] for k in ("max_err", "max_rel_err", "avg_err", "avg_rel_err",
                                     "worst_point", "worst_source")}
        })

        if verbose:
            print(f"  N_G={N_G:2d} ({len(planes):3d} planes)  "
                  f"max_rel_err={result['max_rel_err']*100:7.4f}%  "
                  f"avg_rel_err={result['avg_rel_err']*100:7.4f}%  "
                  f"(worst on {result['worst_source']} at Q={result['worst_point'][0]:.1f}, L={result['worst_point'][1]:.1f})")

        if best is None or result["max_rel_err"] < best["max_rel_err"]:
            best = {"N_G": N_G, "planes": planes, "grid_Q": grid_Q, "grid_L": grid_L, **result}

        converged = (prev_max_rel is not None and
                     abs(prev_max_rel - result["max_rel_err"]) <= convergence_tol)
        acceptable = result["max_rel_err"] <= error_threshold

        if converged and acceptable:
            return N_G, planes, grid_Q, grid_L, result, history

        prev_max_rel = result["max_rel_err"]

    return best["N_G"], best["planes"], best["grid_Q"], best["grid_L"], best, history


# ──────────────────────────────────────────────────────────────────────
# Preprocessing Entry Point (Used by Pyomo Script)
# ──────────────────────────────────────────────────────────────────────
def generate_abeta_hyperplanes_for_match(Q_lo, Q_hi, L_lo, L_hi, U, beta=0.6,
                                          N_G=5, mode="fixed",
                                          error_threshold=0.01, N_max=15):
    """Preprocessing entry point: generates planes for a single match (i, j)."""
    if mode == "dynamic":
        N_G, planes, grid_Q, grid_L, _, _ = select_grid_size_dynamic_abeta(
            Q_lo, Q_hi, L_lo, L_hi, beta=beta, U=U,
            error_threshold=error_threshold, N_start=3, N_max=N_max, verbose=False
        )
    else:
        planes, grid_Q, grid_L = build_abeta_hyperplanes(
            Q_lo, Q_hi, L_lo, L_hi, N_Q=N_G, N_L=N_G, beta=beta, U=U
        )
    return planes, grid_Q, grid_L


# ──────────────────────────────────────────────────────────────────────
# 3D Visual Testing & Analysis (Run Standalone)
# ──────────────────────────────────────────────────────────────────────
def plot_3d_comparison_abeta(dense_result, title="A^beta: True surface vs. Local Hyperplane Envelope",
                             angles=((25, -60), (25, 30), (60, -45), (10, -110))):
    """Plots true A^beta surface vs. piecewise local tangent plane envelope."""
    D_Q, D_L = dense_result["D_Q"], dense_result["D_L"]
    A_true, A_approx = dense_result["A_true"], dense_result["A_approx"]

    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))

    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(D_Q, D_L, A_true, alpha=0.55, cmap="viridis", edgecolor="none")
        ax.plot_wireframe(D_Q, D_L, A_approx, color="red", linewidth=0.4, rstride=5, cstride=5)
        ax.set_xlabel("Q (kW)")
        ax.set_ylabel("LMTD (K)")
        ax.set_zlabel("A^beta")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle(title + "  (viridis=true surface, red wireframe=tangent planes)")
    fig.tight_layout()
    return fig


def plot_error_surface_abeta(dense_result, angles=((25, -60), (60, -45))):
    """Plots relative approximation error (%) surface over (Q, LMTD)."""
    D_Q, D_L = dense_result["D_Q"], dense_result["D_L"]
    rel_err = dense_result["rel_err"]

    n = len(angles)
    fig = plt.figure(figsize=(6 * n, 6))

    for k, (elev, azim) in enumerate(angles):
        ax = fig.add_subplot(1, n, k + 1, projection="3d")
        ax.plot_surface(D_Q, D_L, rel_err * 100, cmap="inferno")
        ax.set_xlabel("Q (kW)")
        ax.set_ylabel("LMTD (K)")
        ax.set_zlabel("Rel. error (%)")
        ax.set_title(f"elev={elev}, azim={azim}")
        ax.view_init(elev=elev, azim=azim)

    fig.suptitle("Relative approximation error (%) over (Q, LMTD)")
    fig.tight_layout()
    return fig


def analyze_abeta_match(Q_lo, Q_hi, L_lo, L_hi, beta=0.6, U=1.0,
                        mode="dynamic", N_G_fixed=8,
                        error_threshold=0.01, convergence_tol=1e-4,
                        N_start=3, N_max=15, dense_N=100, n_per_edge=200,
                        make_plots=True):
    """Top-level visual and analytical testing driver for A^beta(Q, LMTD)."""
    if Q_hi <= Q_lo or L_hi <= L_lo:
        raise ValueError("Invalid domain bounds: upper limits must exceed lower limits.")

    print(f"Domain: Q in [{Q_lo:.1f}, {Q_hi:.1f}], LMTD in [{L_lo:.1f}, {L_hi:.1f}] (beta={beta}, U={U})\n")

    if mode == "dynamic":
        print("Running dynamic grid-size sweep...")
        N_G, planes, grid_Q, grid_L, result, history = select_grid_size_dynamic_abeta(
            Q_lo, Q_hi, L_lo, L_hi, beta=beta, U=U,
            error_threshold=error_threshold, convergence_tol=convergence_tol,
            N_start=N_start, N_max=N_max, dense_N=dense_N, n_per_edge=n_per_edge,
            verbose=True
        )
    elif mode == "fixed":
        N_G = N_G_fixed
        planes, grid_Q, grid_L = build_abeta_hyperplanes(
            Q_lo, Q_hi, L_lo, L_hi, N_Q=N_G, N_L=N_G, beta=beta, U=U
        )
        result = full_validation_abeta(
            grid_Q, grid_L, planes, Q_lo, Q_hi, L_lo, L_hi,
            beta=beta, U=U, dense_N=dense_N, n_per_edge=n_per_edge
        )
        history = None
    else:
        raise ValueError("mode must be 'dynamic' or 'fixed'")

    print(f"\n--- Result (N_G={N_G}, {len(planes)} tangent planes) ---")
    print(f"Max abs error       : {result['max_err']:.6f}")
    print(f"Max relative error  : {result['max_rel_err']*100:.4f}%  "
          f"at (Q={result['worst_point'][0]:.2f}, LMTD={result['worst_point'][1]:.2f})")

    if make_plots:
        plot_3d_comparison_abeta(result["dense"])
        plot_error_surface_abeta(result["dense"])
        plt.show()

    return {
        "N_G": N_G, "planes": planes, "grid_Q": grid_Q, "grid_L": grid_L,
        "result": result, "history": history
    }


if __name__ == "__main__":
    # Test run when executing `python A_Beta_HP.py`
    Q_lo, Q_hi = 100.0, 5000.0   # kW
    L_lo, L_hi = 10.0, 100.0     # K
    beta = 0.6
    U = 0.5                      # kW / (m^2 K)

    analyze_abeta_match(
        Q_lo, Q_hi, L_lo, L_hi, beta, U,
        mode="dynamic",
        error_threshold=0.01,    # Target 1% max relative error
        convergence_tol=0.001,
        N_start=3, N_max=15,
        make_plots=True
    )