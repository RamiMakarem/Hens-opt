import pandas as pd
import warnings
from types import SimpleNamespace
import numpy as np
import pyomo.environ as pyo
import math

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: LMTD and its partial derivatives
# ═══════════════════════════════════════════════════════════════════════════════

def _lmtd(dT1: float, dT2: float) -> float:
    """True log-mean temperature difference. Both inputs clamped to ≥1e-3 for B&B stability."""
    dT1 = dT1
    dT2 = dT2
    if abs(dT1 - dT2) < 1e-2:
        return (dT1+dT2)/2
    return (dT1 - dT2) / np.log(dT1 / dT2)


def _dlmtd(dT1: float, dT2: float):
    """
    Partial derivatives of LMTD. Both inputs clamped to ≥1e-3 for B&B stability.
    Returns (∂LMTD/∂dT1, ∂LMTD/∂dT2).
    """
    dT1 = max(dT1, 1e-3)
    dT2 = max(dT2, 1e-3)
    if abs(dT1 - dT2) < 1e-6:
        return 0.5, 0.5
    lnr = np.log(dT1 / dT2)
    d1 = (lnr * dT1 - (dT1 - dT2)) / (dT1 * lnr ** 2)
    d2 = (-lnr * dT2 + (dT1 - dT2)) / (dT2 * lnr ** 2)
    return d1, d2


def _area_cut(Q0: float, dT10: float, dT20: float,
              Q_coef: float, dT1_coef: float, dT2_coef: float):
    """
    Returns (rhs_constant, c_Q, c_dT1, c_dT2) for the linearisation cut:

        AR  ≥  c_Q·Q  +  c_dT1·dT1  +  c_dT2·dT2  +  rhs_constant

    where AR = A·U (reduced area).
    """
    L0        = _lmtd(dT10, dT20)
    dL1, dL2  = _dlmtd(dT10, dT20)
    AR0       = Q0 / L0

    cQ   = 1.0 / L0
    cdT1 = -Q0 * dL1 / L0 ** 2
    cdT2 = -Q0 * dL2 / L0 ** 2
    rhs  = AR0 - cQ * Q0 - cdT1 * dT10 - cdT2 * dT20

    return rhs, cQ, cdT1, cdT2


def _area_cut_2d(Q0: float, LMTD0: float):
    """
    2D linearisation cut for AR = Q / LMTD, treating LMTD as an already-known
    variable (piecewise-exact via the 2D SOS2 grid below) instead of
    re-deriving it from (dT1, dT2) inside the cut itself.

        AR  ≥  c_Q · Q  +  c_LMTD · LMTD  +  rhs_constant

    Q/LMTD is jointly convex in (Q, LMTD) for LMTD > 0, so this tangent
    plane -- evaluated at any point (Q0, LMTD0) -- is a valid global
    under-estimator everywhere in the feasible region, exactly like the
    old 3D cut was for (Q, dT1, dT2). This is the "2D OA" replacement for
    the old "3D OA": the non-linear (dT1, dT2) -> LMTD mapping is resolved
    once via the SOS2 grid, and the OA cuts only have to handle the
    remaining (and genuinely convex) Q/LMTD relationship.
    """
    LMTD0 = max(LMTD0, 1e-6)
    Q0    = max(Q0, 0.0)

    AR0    = Q0 / LMTD0
    c_Q    = 1.0 / LMTD0
    c_LMTD = -Q0 / LMTD0 ** 2
    rhs_constant = AR0 - c_Q * Q0 - c_LMTD * LMTD0   # simplifies exactly to AR0

    return rhs_constant, c_Q, c_LMTD


def _lmtd_2d_grid(dT1_lo, dT1_hi, dT2_lo, dT2_hi, n_pts=5):
    """
    Build a geometric-spacing (n_pts × n_pts) grid over (dT1, dT2) and the
    exact LMTD evaluated at every grid node. Used to encode LMTD(dT1, dT2)
    piecewise-exactly (2D SOS2 interpolation) instead of only supplying
    linear tangent cuts to the true non-linear surface.

    Returns (T1_grid, T2_grid, LMTD_grid) where LMTD_grid[n, m] =
    _lmtd(T1_grid[n], T2_grid[m]).
    """
    lo1 = max(dT1_lo, 1e-1)
    hi1 = max(dT1_hi, lo1 + 1.0)
    lo2 = max(dT2_lo, 1e-1)
    hi2 = max(dT2_hi, lo2 + 1.0)

    T1_grid = np.geomspace(lo1, hi1, n_pts)
    T2_grid = np.geomspace(lo2, hi2, n_pts)

    LMTD_grid = np.empty((n_pts, n_pts))
    for n in range(n_pts):
        for m in range(n_pts):
            LMTD_grid[n, m] = _lmtd(T1_grid[n], T2_grid[m])

    return T1_grid, T2_grid, LMTD_grid

def _lmtd_1d(dT1_lo, dT1_hi, dT2, n_pts=5):
    """
    Build a geometric-spacing (n_pts × n_pts) grid over (dT1, dT2) and the
    exact LMTD evaluated at every grid node. Used to encode LMTD(dT1, dT2)
    piecewise-exactly (2D SOS2 interpolation) instead of only supplying
    linear tangent cuts to the true non-linear surface.

    Returns (T1_grid, T2_grid, LMTD_grid) where LMTD_grid[n, m] =
    _lmtd(T1_grid[n], T2_grid[m]).
    """
    lo1 = max(dT1_lo, 1e-1)
    hi1 = max(dT1_hi, lo1 + 1.0)

    T1_grid = np.geomspace(lo1, hi1, n_pts)

    LMTD_grid = np.empty((n_pts))
    for n in range(n_pts):
        LMTD_grid[n] = _lmtd(T1_grid[n], dT2)

    return T1_grid, dT2, LMTD_grid

def _cost_tangent(A_bp: float, beta: float, cost_b: float, payback: float):
    """
    Tangent line of  f(A) = (cost_b/payback) · A^beta  at breakpoint A_bp.
    Returns (slope, intercept):  y_ijk ≥ slope·A_ijk + intercept
    """
    cb = cost_b / payback
    sl = cb * beta * A_bp ** (beta - 1)
    ic = cb * A_bp ** beta - sl * A_bp
    return sl, ic

def generate_hens_sos2_grid(dT1_lo, dT1_hi, dT2_lo, dT2_hi, Q_max, U, a, b, beta, Q_min=1e-3, n_pts=5):
    """
    Builds 1D coordinate vectors for LMTD and Q, and 2D matrices for Area and Cost.
    
    Indices:
      l -> LMTD index (0 to n_pts-1)
      p -> Q index    (0 to n_pts-1)
    
    Returns:
      lmtd_grid : 1D array of shape (n_pts,)
      Q_grid    : 1D array of shape (n_pts,)
      A_grid    : 2D array of shape (n_pts, n_pts) -> [l, p]
      Cost_grid : 2D array of shape (n_pts, n_pts) -> [l, p]
    """
    # 1. Direct LMTD Bounds via Monotonicity
    lmtd_min = _lmtd(dT1_lo, dT2_lo)
    lmtd_max = _lmtd(dT1_hi, dT2_hi)
    
    lmtd_min = max(lmtd_min, 1e-1)
    lmtd_max = max(lmtd_max, lmtd_min + 1.0)
    
    # 2. 1D Grid Sampling
    # - np.geomspace for LMTD gives denser points near lmtd_min (high 1/LMTD curvature)
    # - np.linspace for Q because Area is linear w.r.t Q
    lmtd_grid = np.geomspace(lmtd_min, lmtd_max, n_pts)
    Q_grid = np.linspace(max(Q_min, 1e-3), Q_max, n_pts)
    
    # 3. 2D Area Matrix via Broadcasting [Shape: (n_pts, n_pts)]
    # Rows (l): LMTD, Columns (p): Q
    A_grid = Q_grid[None, :] / (U * lmtd_grid[:, None])
    
    # 4. 2D Capital Cost Matrix
    Cost_grid = b * np.power(A_grid, beta)
    
    return lmtd_grid, Q_grid, A_grid, Cost_grid

def _normalise_utility_list(specs_raw, side):
    if specs_raw is None:
        return []
    if isinstance(specs_raw, dict):
        specs_raw = [specs_raw]

    out = []
    for idx, u in enumerate(specs_raw):
        u = dict(u)   # shallow copy so we don't mutate caller's data
        u["role"] = side
        u.setdefault("uid", u.get("id", f"{side}_util_{idx}"))
        u["max_flowrate"] = float(u.get("max_flowrate") or np.inf)

        utype = u.get("type", "sensible")

        if utype == "steam":
            # Pure isothermal condensation / evaporation
            T_s = float(u["T_steam"])
            lam = float(u.get("lambda_vap", 2000.0))   # kJ/kg default
            u["T_supply"]  = T_s
            u["T_return"]  = T_s   # isothermal
            u["Q_per_kg"]  = lam   # Q = m * λ

        elif utype in ("sensible", "stream"):
            Tin  = float(u["Tin"])
            Tout = float(u["Tout"])
            cp   = float(u.get("cp", 4.18))
            u["T_supply"] = Tin
            u["T_return"] = Tout
            u["Q_per_kg"] = cp * abs(Tin - Tout)

        elif utype == "combined":
            # Mixed latent + sensible (e.g. superheated steam that condenses then subcools)
            # Heating utility example:  Tin_vapor → T_phase (desuper, Cp_vap)
            #                           T_phase (condensation, λ)
            #                           T_phase → Tout_liquid (subcool, Cp_liq)
            Tin      = float(u["Tin"])
            T_phase  = float(u["T_phase"])
            Tout     = float(u["Tout"])
            lam      = float(u.get("lambda_vap", 2000.0))
            cp_vap   = float(u.get("cp_vap", 0.0))
            cp_liq   = float(u.get("cp_liq", 0.0))

            # Q_per_kg = total heat released / absorbed per kg
            Q = 0.0
            if side == "hot":
                # cooling path: Tin > T_phase > Tout
                if Tin > T_phase:
                    Q += cp_vap * (Tin - T_phase)
                Q += lam
                if T_phase > Tout:
                    Q += cp_liq * (T_phase - Tout)
            else:
                # heating path for cold utility: Tin < T_phase < Tout
                if T_phase > Tin:
                    Q += cp_liq * (T_phase - Tin)
                Q += lam
                if Tout > T_phase:
                    Q += cp_vap * (Tout - T_phase)

            u["T_supply"]  = Tin
            u["T_return"]  = Tout
            u["Q_per_kg"]  = max(Q, 1e-6)
            u["T_phase"]   = T_phase   # kept for elbow constraint

        else:
            raise ValueError(f"Unknown utility type {utype!r} in {side} utility spec.")

        # Infer role from temperatures if not already set by caller
        # Steam (isothermal) is always valid regardless of T_supply vs T_return
        if utype != "steam":
            if side == "hot" and u["T_supply"] <= u["T_return"]:
                raise ValueError(
                    f"Hot utility '{u['uid']}' has T_supply ({u['T_supply']}) ≤ T_return ({u['T_return']}). "
                    f"Hot sensible utilities must cool down (supply temp > return temp)."
                )
            if side == "cold" and u["T_supply"] >= u["T_return"]:
                raise ValueError(
                    f"Cold utility '{u['uid']}' has T_supply ({u['T_supply']}) ≥ T_return ({u['T_return']}). "
                    f"Cold utilities must heat up (supply temp < return temp)."
                )
        out.append(u)
    return out

from H_HP_Tester import select_H_envelope

# ==============================================================================
# EXACT O(1) SOS2 GRID SIZE CALCULATORS
# ==============================================================================
def calculate_exact_N_A(
    A_min: float,
    A_max: float,
    target_rel_error: float = 0.01,
    beta: float = 0.5,
) -> int:
    """Calculate the minimum number of SOS2 intervals (N) needed to guarantee

    a target maximum relative error for y = x^beta over [x_min, x_max].

    Parameters
    ----------
    x_min : float
        Lower bound of x (must be > 0).
    x_max : float
        Upper bound of x (must be > x_min).
    target_rel_error : float, optional
        Maximum allowable relative error (e.g., 0.01 for 1%). Default is 0.01.
    beta : float, optional
        Power exponent in y = x^beta (0 < beta < 1). Default is 0.5 (worst case).

    Returns
    -------
    int
        Number of piecewise linear intervals N. (Grid points P = N + 1).
    """
    if math.isclose(beta, 1.0, abs_tol=BETA_ONE_TOL):
        # y = x^1 is already exactly linear -- there is no fitting error to
        # bound, and beta*(1-beta) == 0 would divide by zero below. Callers
        # should route beta == 1 matches to the linear Cost = cost_b*A path
        # instead of calling this function at all (see pre_process_milp).
        raise ValueError(
            "calculate_exact_N_A called with beta == 1: y = x^beta is exactly "
            "linear here, so no SOS2 grid is needed -- use the linear "
            "Cost = cost_b * A path instead of this fitter."
        )

    # Geometric growth ratio step: (r - 1) ≈ sqrt(8 * epsilon / (beta * (1 - beta)))
    r_minus_1 = math.sqrt((8.0 * target_rel_error) / (beta * (1 - beta)))

    # Exact log step: ln(r) = ln(1 + (r - 1))
    ln_r = math.log1p(r_minus_1)

    # N = ceil( ln(x_max / x_min) / ln_r )
    N = math.ceil(math.log(A_max / A_min) / ln_r)

    return N

# ════════════════════════════════════════════════════════════════════
# H = (U*LMTD)^(-beta) hyperplane configuration
#
# Replaces the old LMTD tangent-plane config. H is jointly CONVEX in
# (dT1,dT2) (verified numerically for beta in (0, 1.5]), so its tangent
# planes are valid GLOBAL UNDERESTIMATORS everywhere -- the envelope is
# max-over-planes, not min-over-planes like the old concave-LMTD
# envelope was. This replaces the old dT1,dT2 -> LMTD -> (McCormick) ->
# A=Q/(U*LMTD) chain, whose free, loosely-bounded LMTD/A variables were
# the root cause of the 90-2000% area/cost errors (see chat notes).
#
# NOTE: select_H_envelope's final validation (final_uniform_N x
# final_uniform_N uniform grid + final_random_pts random points) runs
# ONCE PER FEASIBLE (i,j) MATCH below. The standalone-tester defaults
# (1000x1000 + 1e6 random) are far too expensive to repeat I*J times
# during preprocessing, so H_FINAL_UNIFORM_N/H_FINAL_RANDOM_PTS are
# turned down here. Spot-check a few matches at 1000/1e6 directly via
# H_HP_Tester.analyze_H if you want the original tester's confidence
# level rather than this cheaper per-match default.
# ════════════════════════════════════════════════════════════════════
H_ERROR_THRESHOLD    = 0.01        # target max relative error (1%)
H_MAX_PLANES         = 300
H_N_INIT             = 3
H_USE_SYMMETRY       = True
H_ORACLE_UNIFORM_N   = 80          # oracle grid used *during* greedy refinement
H_ORACLE_RANDOM_PTS  = 4_000
H_FINAL_UNIFORM_N    = 150         # cheaper per-match stand-in for the tester's 1000
H_FINAL_RANDOM_PTS   = 20_000      # cheaper per-match stand-in for the tester's 1e6
H_MAX_OUTER_ROUNDS   = 3
target_rel_err        = 0.01

# Tolerance for treating cost_beta as exactly 1.0 (Cost = cost_b * A is then
# already linear in A, so the whole A -> Cost SOS2 machinery is skipped for
# that match -- see the beta_linear split in pre_process_milp/build_model).
BETA_ONE_TOL          = 1e-9


def pre_process_milp(Hsap, Csap, delta_tmin, qh, qc,
                       cu_hot=80.0,
                       cu_cold=15.0,
                       U_overall=0.5,
                       U_matrix=None,
                       cost_a=5500.0,
                       cost_b=150.0,
                       cost_beta=1.0,
                       payback=1,
                       hours_per_year=8600,
                       utility_specs=None,
                       N_G_process=8,
                       N_G_util=8,
                       Q_floor_frac=0.02):
    EPSILON = 1e-4

    I = len(Hsap)
    J = len(Csap)
    K = I + J - 1
    S = K - 1

    Hi = range(I)
    Hj = range(J)
    Hs = range(S)

    HID = [r[0] for r in Hsap]; CID = [r[0] for r in Csap]
    CP_H = [r[3] for r in Hsap]; CP_C = [r[3] for r in Csap]
    Tin_H = [r[1] for r in Hsap]; Tin_C = [r[1] for r in Csap]
    Tout_H = [r[2] for r in Hsap]; Tout_C = [r[2] for r in Csap]

    dTH = [abs(Tout_H[i] - Tin_H[i]) for i in Hi]
    dTC = [abs(Tout_C[j] - Tin_C[j]) for j in Hj]
    Q_H_total = [CP_H[i] * dTH[i] for i in Hi]
    Q_C_total = [CP_C[j] * dTC[j] for j in Hj]

    if utility_specs is None:
        utility_specs = {}

    hot_utils = _normalise_utility_list(utility_specs.get("hot", None), "hot")
    n_HU = len(hot_utils)
    hot_Q_per_kg = np.array([u["Q_per_kg"] for u in hot_utils])
    hot_max_flow = np.array([u["max_flowrate"] for u in hot_utils])
    T_HU_supply = [u["T_supply"] for u in hot_utils]
    T_HU_return = [u["T_return"] for u in hot_utils]
    hot_is_combined = np.array([u["type"] == "combined" for u in hot_utils])
    hot_T_phase = np.array([u.get("T_phase", np.nan) for u in hot_utils])
    hot_cp_vap = np.array([u.get("cp_vap", 0.0) for u in hot_utils])

    cold_utils = _normalise_utility_list(utility_specs.get("cold", None), "cold")
    n_CU = len(cold_utils)
    cold_Q_per_kg = np.array([u["Q_per_kg"] for u in cold_utils])
    cold_max_flow = np.array([u["max_flowrate"] for u in cold_utils])
    T_CU_supply = [v["T_supply"] for v in cold_utils]
    T_CU_return = [u["T_return"] for u in cold_utils]
    cold_is_combined = np.array([u["type"] == "combined" for u in cold_utils])
    cold_T_phase = np.array([u.get("T_phase", np.nan) for u in cold_utils])
    cold_cp_liq = np.array([u.get("cp_liq", 0.0) for u in cold_utils])

    def _util_opex(u, default_cost):
        return float(u.get("cost_per_kw", default_cost))
    hu_opex = [_util_opex(u, cu_hot) for u in hot_utils]
    cu_opex = [_util_opex(v, cu_cold) for v in cold_utils]

    total_rows = I + n_CU
    total_cols = J + n_HU
    if U_matrix is not None:
        U_tmp = np.array(U_matrix, dtype=float)
        if U_tmp.shape == (total_rows, total_cols):
            U_mat = U_tmp
        elif U_tmp.shape == (I + 1, J + 1) and n_HU == 1 and n_CU == 1:
            U_mat = np.full((total_rows, total_cols), U_overall, dtype=float)
            U_mat[:I, :J] = U_tmp[:I, :J]
            U_mat[I, :J] = U_tmp[I, :J]
            U_mat[:I, J] = U_tmp[:I, J]
        elif U_tmp.shape == (I, J):
            U_mat = np.full((total_rows, total_cols), U_overall, dtype=float)
            U_mat[:I, :J] = U_tmp
        else:
            U_mat = np.full((total_rows, total_cols), U_overall, dtype=float)
            r = min(U_tmp.shape[0], total_rows)
            c = min(U_tmp.shape[1], total_cols)
            U_mat[:r, :c] = U_tmp[:r, :c]
    else:
        U_mat = np.full((total_rows, total_cols), U_overall, dtype=float)

    for i in Hi:
        if Tin_H[i] <= Tout_H[i]:
            raise ValueError(f"Hot stream {HID[i]}: inlet temperature ({Tin_H[i]}) must be greater than outlet ({Tout_H[i]}).")
    for j in Hj:
        if Tout_C[j] <= Tin_C[j]:
            raise ValueError(f"Cold stream {CID[j]}: outlet temperature ({Tout_C[j]}) must be greater than inlet ({Tin_C[j]}).")

    Q_match_max = {(i, j): min(Q_H_total[i], Q_C_total[j]) for i in Hi for j in Hj}

    feasible_match = {(i, j): (Tin_H[i] > Tin_C[j] + delta_tmin) for i in Hi for j in Hj}
    n_infeasible = sum(1 for v in feasible_match.values() if not v)
    if n_infeasible:
        infeas_names = [f"({HID[i]},{CID[j]})" for i in Hi for j in Hj if not feasible_match[i, j]]
        warnings.warn(f"The following {n_infeasible} match(es) are thermodynamically infeasible "
                      f"and will be excluded: {', '.join(infeas_names)}")

    dT1_lo = {(i, j): delta_tmin for i in Hi for j in Hj}
    dT2_lo = {(i, j): delta_tmin for i in Hi for j in Hj}

    def _dT_hi_safe(diff, delta_tmin, factor=2.0):
        return diff if diff >= delta_tmin else factor * delta_tmin

    dT1_hi = {(i, j): _dT_hi_safe(Tin_H[i] - Tin_C[j], delta_tmin) for i in Hi for j in Hj}
    dT2_hi = {(i, j): _dT_hi_safe(Tin_H[i] - Tin_C[j], delta_tmin) for i in Hi for j in Hj}

    # ════════════════════════════════════════════════════════════════
    # Per-match bounds: Q, H=(U*LMTD)^-beta, G=Q^beta, and their
    # McCormick product P = G*H = A^beta.
    #
    # This replaces A_min/A_max/A_bounds, W_bounds (the free A*LMTD
    # bilinear this whole redesign exists to eliminate -- see chat
    # notes), and the A -> Cost SOS2 grid. There is no separate A->Cost
    # SOS2 grid anymore: Cost = cost_b * P is already exactly LINEAR in
    # P once G and H each individually approximate Q^beta and
    # (U*LMTD)^-beta -- the nonlinearity was fully absorbed into G, H,
    # and the single G*H McCormick term, so no third piecewise stage is
    # needed the way A -> Cost used to be.
    # ════════════════════════════════════════════════════════════════

    H_min, H_max = {}, {}
    G_min, G_max = {}, {}
    lmtd_bounds, H_bounds, G_bounds, P_bounds = {}, {}, {}, {}
    Q_grid_G, G_grid = {}, {}
    Q_bounds, VarCost_bounds = {}, {}
    N_G_diag = {}
    M_dt = {}
    # (i,j) -> True iff cost_beta rounds to exactly 1.0 for this match, i.e.
    # G = Q^1 = Q is already exact and linear -- no SOS2 grid needed, G is
    # just the Q variable itself. Those matches skip the G SOS2 grid and
    # Constraints.py substitutes Q directly for G in the G*H McCormick
    # product. See m.LinearIJK / m.NonlinearIJK in build_model().
    beta_linear = {}

    for i in Hi:
        for j in Hj:
            if not feasible_match[i, j]:
                Q_bounds[i,j]= (0.0,0.0)
                continue

            beta = cost_beta[i, j] if isinstance(cost_beta, dict) else cost_beta
            is_linear = math.isclose(beta, 1.0, abs_tol=BETA_ONE_TOL)
            beta_linear[i, j] = is_linear

            M_dt[i, j] = max(delta_tmin, Tin_H[i] - Tin_C[j]) + max(delta_tmin, Tout_H[i] - Tout_C[j]) + 50

            # --- 1. Heat duty Q bounds (unchanged) ---
            q_max = Q_match_max[i, j]
            q_min = max(Q_floor_frac * q_max, EPSILON)
            Q_bounds[i, j] = (q_min, q_max)

            # --- 2. LMTD bounds (unchanged; still needed to bound H) ---
            U_ij = U_mat[i, j]
            lmtd_min = max(delta_tmin, 1e-3)
            lmtd_max = max(dT1_hi[i, j], delta_tmin)
            lmtd_bounds[i, j] = (lmtd_min, lmtd_max)

            # --- 3. H = (U*LMTD)^-beta bounds. H is DECREASING in LMTD
            # (unlike A, which was increasing in Q and decreasing in
            # LMTD the same way) -- so H's max sits at lmtd_min and its
            # min at lmtd_max, same pairing direction as the old A_max/
            # A_min, just on H instead of A. ---
            H_max[i, j] = (U_ij * lmtd_min) ** (-beta)
            H_min[i, j] = (U_ij * lmtd_max) ** (-beta)
            H_bounds[i, j] = (H_min[i, j], H_max[i, j])

            # --- 4. G = Q^beta bounds/grid ---
            if is_linear:
                # beta == 1: G = Q^1 = Q, exactly linear -- no SOS2 grid,
                # no breakpoints, no lam_G entries for this match. G_Pt
                # in build_model() below simply contains no (i,j,*,*)
                # tuples for it.
                N_G_diag[i, j] = 0
                Q_grid_G[i, j] = np.array([])
                G_grid[i, j] = np.array([])
                G_min[i, j], G_max[i, j] = q_min, q_max
            else:
                N_G = calculate_exact_N_A(q_min, q_max, target_rel_err, beta)
                q_g = np.geomspace(q_min, q_max, num=N_G)
                g_g = q_g ** beta
                N_G_diag[i, j] = N_G

                Q_grid_G[i, j] = q_g
                G_grid[i, j] = g_g
                G_min[i, j], G_max[i, j] = float(g_g[0]), float(g_g[-1])
            G_bounds[i, j] = (G_min[i, j], G_max[i, j])

            # --- 5. P = G*H = A^beta bounds. Both G and H are >= 0 and
            # each individually increasing/decreasing monotonically in
            # their own single variable, so the achievable P range is
            # just the corner product [Gmin*Hmin, Gmax*Hmax]. The actual
            # 4-inequality McCormick relaxation of P=G*H gets built from
            # (G_min,G_max,H_min,H_max) in Constraints.py -- this is the
            # ONE remaining relaxation gap in the whole A^beta pipeline,
            # replacing the three compounding gaps (LMTD envelope +
            # A=Q/LMTD McCormick + A->Cost SOS2) that used to exist. ---
            P_lo = G_min[i, j] * H_min[i, j]
            P_hi = G_max[i, j] * H_max[i, j]
            P_bounds[i, j] = (P_lo, P_hi)
            VarCost_bounds[i, j] = (cost_b * P_lo, cost_b * P_hi)


    # ════════════════════════════════════════════════════════════════
    # H(dT1,dT2) = (U*LMTD)^-beta tangent-plane hyperplanes, built and
    # validated per feasible match, in preprocess (before the MILP).
    # These are true supporting hyperplanes of the CONVEX H surface
    # (H >= a0 + a1*dT1 + a2*dT2 everywhere) -- direction is FLIPPED
    # from the old LMTD planes (LMTD <= ... everywhere, concave). Unlike
    # the old LMTD-only planes, H depends on beta AND U, so it is
    # rebuilt per match using that match's own beta/U rather than being
    # pure (dT1,dT2) geometry shared across matches with the same range.
    # ════════════════════════════════════════════════════════════════
    h_planes = {}     # (i,j) -> list of (a0,a1,a2) for f = H(dT1,dT2)
    h_N_planes = {}   # (i,j) -> number of planes actually used for that match
    h_diag = {}       # (i,j) -> dict of error/QA stats for logging

    for i in Hi:
        for j in Hj:
            if not feasible_match[i, j]:
                continue

            beta = cost_beta[i, j] if isinstance(cost_beta, dict) else cost_beta
            U_ij = U_mat[i, j]
            dT_lo_ij = dT1_lo[i, j]   # == dT2_lo[i, j] by construction
            dT_hi_ij = dT1_hi[i, j]   # == dT2_hi[i, j] by construction

            planes_ij, val_result, _hist = select_H_envelope(
                    dT_lo_ij, dT_hi_ij, beta, U_ij,
                    error_threshold=H_ERROR_THRESHOLD,
                    max_planes=H_MAX_PLANES,
                    N_init=H_N_INIT,
                    use_symmetry=H_USE_SYMMETRY,
                    oracle_uniform_N=H_ORACLE_UNIFORM_N,
                    oracle_random_pts=H_ORACLE_RANDOM_PTS,
                    final_uniform_N=H_FINAL_UNIFORM_N,
                    final_random_pts=H_FINAL_RANDOM_PTS,
                    max_outer_rounds=H_MAX_OUTER_ROUNDS,
                    verbose=False,
                )

            if val_result['n_sign_violations'] > 0:
                warnings.warn(
                    f"H hyperplanes for match ({HID[i]},{CID[j]}) are NOT "
                    f"strictly underestimating (envelope rose above the true convex "
                    f"function) at {val_result['n_sign_violations']} test point(s) "
                    f"-- check numerical tolerance near dT1==dT2.")

            if val_result["max_rel_err"] > H_ERROR_THRESHOLD:
                warnings.warn(
                    f"H hyperplane max relative error for match "
                    f"({HID[i]},{CID[j]}) = {val_result['max_rel_err']*100:.3f}% "
                    f"exceeds threshold {H_ERROR_THRESHOLD*100:.2f}% "
                    f"({len(planes_ij)} planes). Consider raising H_MAX_PLANES, "
                    f"H_ORACLE_*, or H_FINAL_UNIFORM_N/H_FINAL_RANDOM_PTS.")

            h_planes[i, j] = planes_ij
            h_N_planes[i, j] = len(planes_ij)
            h_diag[i, j] = {
                "max_rel_err": val_result["max_rel_err"],
                "avg_rel_err": val_result["avg_rel_err"],
                "worst_point": val_result["worst_point"],
                "n_planes": len(planes_ij),
                "n_sign_violations": val_result["n_sign_violations"],
                # NOTE: unlike the old lmtd_diag, this does not carry a
                # ready-made 2D "dense" meshgrid for plotting (select_H_
                # envelope's validation set mixes uniform+random+boundary
                # points as flat arrays, not a clean grid). Report.py's
                # 3D true-vs-envelope plotting will need a small update
                # (e.g. call H_HP_Tester.plot_3d_comparison directly with
                # this match's dT range/planes/beta/U) -- flagged for the
                # Report.py pass, not fixed here.
            }

    # ---- 1D Q-grid -> A^beta (single SOS2 link), HU heating cold streams ----
    N_GU = N_G_util
    Q_g_hu, Abeta_g_hu, Vcost_g_hu = {}, {}, {}
    A_max_hu, feasible_hu, Q_max_hu, dT1_HU = {}, {}, {}, {}
    dTmax_HU={}
    T_hu_in_eff  = np.zeros(n_HU)
    T_hu_out_eff = np.zeros(n_HU)

    def _hu_area_beta(Q, Tuin, Tuo, Tco, CPc, U, beta):
        Q = np.asarray(Q, dtype=float)
        dT1 = Tuin - Tco
        dT2 = Tuo - Tco + Q / CPc

        assert np.all(dT1 > 0) and np.all(dT2 > 0), \
            "Non-positive approach temperature reached _hu_area_beta — check upstream feasibility filtering"

        denom = dT1 - dT2
        near_pinch = np.abs(denom) < 0.1

        # away from the pinch, ordinary LMTD; at the pinch, the exact limit
        denom_calc = np.where(near_pinch, 1.0, denom)          # placeholder, discarded by np.where below
        ratio_calc = np.where(near_pinch, 1.0, dT1 / dT2)       # placeholder, discarded by np.where below
        lmtd_general = denom_calc / np.log(ratio_calc)
        lmtd_limit = (dT1 + dT2) / 2.0
        lmtd = np.where(near_pinch, lmtd_limit, lmtd_general)

        A = Q / (U * lmtd)
        return A ** beta

    for hu in range(n_HU):
        T_hu_in = hot_T_phase[hu] if hot_is_combined[hu] and not np.isnan(hot_T_phase[hu]) else T_HU_supply[hu]
        T_hu_out = hot_T_phase[hu] if hot_is_combined[hu] and not np.isnan(hot_T_phase[hu]) else T_HU_return[hu]
        max_q_hu = hot_max_flow[hu] * hot_Q_per_kg[hu] if hot_max_flow[hu] > 0 else np.inf
        T_hu_in_eff[hu]  = T_hu_in
        T_hu_out_eff[hu] = T_hu_out
        for j in Hj:
            dT1_hu = T_hu_in - Tout_C[j]
            dT1_HU[hu, j] = dT1_hu

            U_hu_j = U_mat[I + hu, j] if ((I + hu) < U_mat.shape[0] and j < U_mat.shape[1]) else U_overall
            Qmax = min(Q_C_total[j], max_q_hu)
            Q_max_hu[hu, j] = Qmax

            dTmax_HU[hu,j]=max(dT1_hu,T_hu_out-Tout_C[j]+Qmax/CP_C[j])

            # Q needed so that dT2 = Tuo - Tco + Q/CPc >= delta_tmin
            Q_min_feas = max(0.0, CP_C[j] * (delta_tmin - (T_hu_out - Tout_C[j])))

            is_feasible = (dT1_hu >= delta_tmin) and (Qmax > Q_min_feas) and np.isfinite(Qmax) and Qmax > 0
            feasible_hu[hu, j] = is_feasible

            if not is_feasible:
                Q_g_hu[hu, j]     = np.zeros(N_GU + 1)
                Abeta_g_hu[hu, j] = np.zeros(N_GU + 1)
                Vcost_g_hu[hu, j] = np.zeros(N_GU + 1)
                A_max_hu[hu, j] = 0.0
                continue

            Q_floor = max(Q_min_feas, Q_floor_frac * Qmax, EPSILON)
            Q_grid_hu = np.geomspace(Q_floor, Qmax, N_GU)
            Abeta_grid_hu = _hu_area_beta(Q_grid_hu, T_hu_in, T_hu_out, Tout_C[j], CP_C[j], U_hu_j, cost_beta)
            Abeta_grid_hu[0] = 0.0

            Q_g_hu[hu, j] = Q_grid_hu
            Abeta_g_hu[hu, j] = Abeta_grid_hu
            Vcost_g_hu[hu, j] = cost_b * Abeta_grid_hu
            A_max_hu[hu, j] = Abeta_grid_hu[-1] ** (1.0 / cost_beta) if Abeta_grid_hu[-1] > 0 else 0.0

    # ---- 1D Q-grid -> A^beta (single SOS2 link), CU cooling hot streams ----
    Q_g_cu, Abeta_g_cu, Vcost_g_cu = {}, {}, {}
    A_max_cu, feasible_cu, Q_max_cu, dT2_CU = {}, {}, {}, {}
    dTmax_CU={}
    T_cu_in_eff  = np.zeros(n_CU)
    T_cu_out_eff = np.zeros(n_CU)

    def _cu_area_beta(Q, Tuin, Tuo, Tho, CPh, U, beta):
        Q = np.asarray(Q, dtype=float)
        dT1 = Tho + Q / CPh - Tuo
        dT2 = Tho - Tuin

        assert np.all(dT1 > 0) and np.all(dT2 > 0), \
            "Non-positive approach temperature reached _cu_area_beta — check upstream feasibility filtering"

        denom = dT1 - dT2
        near_pinch = np.abs(denom) < 0.1

        denom_calc = np.where(near_pinch, 1.0, denom)        # placeholder, discarded by np.where below
        ratio_calc = np.where(near_pinch, 1.0, dT1 / dT2)     # placeholder, discarded by np.where below
        lmtd_general = denom_calc / np.log(ratio_calc)
        lmtd_limit = (dT1 + dT2) / 2.0
        lmtd = np.where(near_pinch, lmtd_limit, lmtd_general)

        A = Q / (U * lmtd)
        return A ** beta
    
    for cu in range(n_CU):
        T_cu_in = cold_T_phase[cu] if cold_is_combined[cu] and not np.isnan(cold_T_phase[cu]) else T_CU_supply[cu]
        T_cu_out = cold_T_phase[cu] if cold_is_combined[cu] and not np.isnan(cold_T_phase[cu]) else T_CU_return[cu]
        max_q_cu = cold_max_flow[cu] * cold_Q_per_kg[cu] if cold_max_flow[cu] > 0 else np.inf
        T_cu_in_eff[cu]  = T_cu_in
        T_cu_out_eff[cu] = T_cu_out

        for i in Hi:
            dT2_cu = Tout_H[i] - T_cu_in
            dT2_CU[cu, i] = dT2_cu

            U_cu_i = U_mat[i, J + cu] if (i < U_mat.shape[0] and (J + cu) < U_mat.shape[1]) else U_overall
            Qmax = min(Q_H_total[i], max_q_cu)
            Q_max_cu[cu, i] = Qmax

            dTmax_CU[cu,i]=max(dT2_CU[cu, i],Tout_H[i]-T_cu_out+Qmax/CP_H[i])

            # Q needed so that dT1 = Tho + Q/CPh - Tuo >= delta_tmin
            Q_min_feas = max(0.0, CP_H[i] * (delta_tmin - (Tout_H[i] - T_cu_out)))

            is_feasible = (dT2_cu >= delta_tmin) and (Qmax > Q_min_feas) and np.isfinite(Qmax) and Qmax > 0
            feasible_cu[cu, i] = is_feasible

            if not is_feasible:
                Q_g_cu[cu, i] = np.zeros(N_GU + 1)
                Abeta_g_cu[cu, i] = np.zeros(N_GU + 1)
                Vcost_g_cu[cu, i] = np.zeros(N_GU + 1)
                A_max_cu[cu, i] = 0.0
                continue

            Q_floor = max(Q_min_feas, Q_floor_frac * Qmax, EPSILON)
            Q_grid_cu = np.geomspace(Q_floor, Qmax, N_GU)
            Abeta_grid_cu = _cu_area_beta(Q_grid_cu, T_cu_in, T_cu_out, Tout_H[i], CP_H[i], U_cu_i, cost_beta)
            Abeta_grid_cu[0] = 0.0

            Q_g_cu[cu, i] = Q_grid_cu
            Abeta_g_cu[cu, i] = Abeta_grid_cu
            Vcost_g_cu[cu, i] = cost_b * Abeta_grid_cu
            A_max_cu[cu, i] = Abeta_grid_cu[-1] ** (1.0 / cost_beta) if Abeta_grid_cu[-1] > 0 else 0.0
    data = SimpleNamespace(**locals())
    return data

from Variables import build_variables
from Constraints import build_constraints
from Objective import build_objective

def build_model(data):
    m = pyo.ConcreteModel()

    # ---- Core index sets----
    m.Hi = pyo.Set(initialize=data.Hi)            # hot process streams
    m.Hj = pyo.Set(initialize=data.Hj)            # cold process streams
    m.Knodes = pyo.RangeSet(0, data.K - 1)        # temperature-node index, 0..K-1 (0-based, matches pre_process_milp)
    m.Hs = pyo.Set(initialize=data.Hs)                 # stage index, 0..S-1 (matches pre_process_milp)

    # ---- Utility & piecewise-linear helper sets ----
    m.HU = pyo.RangeSet(0, data.n_HU-1)
    m.CU = pyo.RangeSet(0, data.n_CU-1)
    m.GU0 = pyo.RangeSet(0, data.N_GU-1)

    # ---- Feasible (i, j) match sets, reused by constraints.py & objective.py ----
    feasible_pairs = [(i, j) for i in m.Hi for j in m.Hj if data.feasible_match[i, j]]
    m.FeasiblePairs = pyo.Set(initialize=feasible_pairs, dimen=2)
    m.FeasibleIJK = pyo.Set(
        initialize=[(i, j, k) for (i, j) in feasible_pairs for k in m.Hs],
        dimen=3,)
    
    # ---- H=(U*LMTD)^-beta Hyperplane Cuts ----
    # Renamed from the old m.NegBetaLnLMTD_cut_index (data.lmtd_planes) to
    # m.H_cut_index (data.h_planes). DIRECTION FLIP for Step 3: these must
    # be built in Constraints.py as H_ijk >= a0 + a1*dT1_ijk + a2*dT2_ijk
    # (underestimate), not the old LMTD_ijk <= a0 + ... (overestimate).
    m.H_cut_index = pyo.Set(dimen=4, initialize=[
            (i, j, k, p)
            for (i, j, k) in m.FeasibleIJK
            for p in range(len(data.h_planes[i, j]))
        ])

    feasible_hu = [(u, j) for u in m.HU for j in m.Hj if data.feasible_hu[u, j]]
    m.FeasibleHU = pyo.Set(initialize=feasible_hu, dimen=2)
    feasible_cu = [(v, i) for v in m.CU for i in m.Hi if data.feasible_cu[v, i]]
    m.FeasibleCU = pyo.Set(initialize=feasible_cu, dimen=2)

    # ---- breakpoint params, restricted to FeasibleHU / FeasibleCU ----
    m.Qbp_HU = pyo.Param(
        m.FeasibleHU, m.GU0, mutable=False,
        initialize=lambda m, u, j, k: float(data.Q_g_hu[u, j][k]))
    m.Cbp_HU = pyo.Param(
        m.FeasibleHU, m.GU0, mutable=False,
        initialize=lambda m, u, j, k: float(data.Vcost_g_hu[u, j][k]))
    m.Qbp_CU = pyo.Param(
        m.FeasibleCU, m.GU0, mutable=False,
        initialize=lambda m, v, i, k: float(data.Q_g_cu[v, i][k]))
    m.Cbp_CU = pyo.Param(
        m.FeasibleCU, m.GU0, mutable=False,
        initialize=lambda m, v, i, k: float(data.Vcost_g_cu[v, i][k]))

    # first nonzero breakpoint = semicontinuous "must be at least this if on" threshold
    m.Qthresh_HU = pyo.Param(m.FeasibleHU, initialize=lambda m, u, j: float(data.Q_g_hu[u, j][1]))
    m.Qthresh_CU = pyo.Param(m.FeasibleCU, initialize=lambda m, v, i: float(data.Q_g_cu[v, i][1]))

    # --- Index sets for G=Q^beta SOS2 weights ---
    # Renamed from the old m.A_Pt (data.A_grid, the A -> Cost SOS2) to
    # m.G_Pt (data.G_grid). There is no separate A->Cost SOS2 anymore --
    # Cost = cost_b*P = cost_b*G*H is already linear in P once G and H
    # exist (see the Per-match bounds block above) -- so this index now
    # only serves the G=Q^beta interpolation feeding the G*H McCormick
    # term, not a second downstream piecewise stage.
    g_pt_init = [
        (i, j, k, p)
        for (i, j, k) in m.FeasibleIJK
        for p in range(len(data.G_grid[i, j]))
        ]
    m.G_Pt = pyo.Set(dimen=4, initialize=g_pt_init)

    # --- Split FeasibleIJK by whether this match's cost_beta == 1 ---
    # LinearIJK matches get G == Q directly (Constraints.py substitutes Q
    # for G in the G*H McCormick product); NonlinearIJK matches keep the
    # G=Q^beta SOS2. This mirrors G_Pt already being empty for LinearIJK
    # matches (data.G_grid[i,j] == [] there), so no lam_G variables get
    # created for them either.
    m.LinearIJK = pyo.Set(
        dimen=3,
        initialize=[(i, j, k) for (i, j, k) in m.FeasibleIJK if data.beta_linear[i, j]],
    )
    m.NonlinearIJK = pyo.Set(
        dimen=3,
        initialize=[(i, j, k) for (i, j, k) in m.FeasibleIJK if not data.beta_linear[i, j]],
    )

    build_variables(m, data)
    build_constraints(m, data)
    build_objective(m, data)

    return m