import pandas as pd
import warnings
from types import SimpleNamespace
import numpy as np
import pyomo.environ as pyo
import math

# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Gray-code segment sets for the 2D DLOG piecewise-linear formulation
# (copied from DLOG_Reducedton-1Bin.py -- that file's name contains a hyphen,
# so it cannot be imported as a normal Python module; kept byte-for-byte
# identical to the reference implementation there).
# ═══════════════════════════════════════════════════════════════════════════════

def gray_code_sets(n):
    """
    For n segments indexed 0..n-1, return (B, N1, N0) where:
      B      = number of bits = ceil(log2(n))  (0 if n <= 1)
      N1[b]  = set of segment indices whose Gray code has bit b = 1
      N0[b]  = complement of N1[b] within {0,...,n-1}
    Uses the standard reflected binary Gray code: code(t) = t ^ (t >> 1).
    """
    B = 0 if n <= 1 else math.ceil(math.log2(n))
    codes = [t ^ (t >> 1) for t in range(n)]
    N1, N0 = [], []
    for b in range(B):
        s1 = {t for t in range(n) if (codes[t] >> b) & 1}
        N1.append(s1)
        N0.append(set(range(n)) - s1)
    return B, N1, N0


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

from Neg_ln_LMTD_HP_Tester import analyze_match

# ════════════════════════════════════════════════════════════════════
# LMTD hyperplane configuration
# ════════════════════════════════════════════════════════════════════
LMTD_GRID_MODE       = "dynamic"  # "dynamic" or "fixed"
LMTD_N_G_FIXED       = 10          # used only if LMTD_GRID_MODE == "fixed"
LMTD_ERROR_THRESHOLD = 0.01        # target max relative error (1%)
LMTD_CONVERGENCE_TOL = 1e-3        # stop growing grid once error stalls
LMTD_N_START          = 3
LMTD_N_MAX            = 20
LMTD_DENSE_N          = 100        # interior test-grid resolution (validated
                                    # sufficient via dense_grid_convergence_check
                                    # in the tester -- see chat notes)
LMTD_N_PER_EDGE       = 200        # boundary/diagonal test points per edge
LMTD_USE_SYMMETRY     = True

# ════════════════════════════════════════════════════════════════════
# Config: replaces N_G_process-driven DLOG sizing
# ════════════════════════════════════════════════════════════════════
LNQ_ERROR_THRESHOLD   = 0.01
LNQ_CONVERGENCE_TOL   = 1e-3
LNQ_N_START           = 2
LNQ_N_MAX             = 20

ABETA_ERROR_THRESHOLD = 0.01
ABETA_CONVERGENCE_TOL = 1e-3
ABETA_N_START         = 2
ABETA_N_MAX           = 30


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

    N_G = N_G_process
    dT1g, dT2g = {}, {}
    Q_g = {}
    A_max = {}

    for i in Hi:
        for j in Hj:
            if not feasible_match[i, j]:
                dT1g[i, j] = dT2g[i, j] = np.linspace(EPSILON, 1.0, N_G)
                Q_g[i, j] = np.zeros(N_G)
                A_max[i, j] = EPSILON
                continue

            dT1g[i, j] = np.geomspace(dT1_lo[i, j], dT1_hi[i, j], N_G)
            dT2g[i, j] = np.geomspace(dT2_lo[i, j], dT2_hi[i, j], N_G)

            Q_floor = max(Q_floor_frac * Q_match_max[i, j], EPSILON)
            Q_g[i, j] = np.linspace(Q_floor, Q_match_max[i, j], N_G)

            U_ij = U_mat[i, j]
            lmtd_min = max(delta_tmin, 1e-3)
            A_max[i, j] = Q_match_max[i, j] / (U_ij * lmtd_min)

    # ════════════════════════════════════════════════════════════════
    # NEW: -cost_beta*ln(LMTD) tangent-plane (1st-order Taylor)
    # hyperplanes, built and validated per feasible match, in
    # preprocess (before the MILP).
    #
    # f(dT1,dT2) = -cost_beta*ln(LMTD(dT1,dT2)) is CONVEX (LMTD is
    # concave, ln is concave-increasing so ln(LMTD) is concave,
    # negating flips it to convex). Tangent planes to a convex
    # function sit BELOW it, so the valid outer approximation is the
    # MAX over planes, giving a supporting-hyperplane LOWER bound:
    #     f(dT1,dT2) >= a0 + a1*dT1 + a2*dT2   for every plane
    #     f_hat = max over planes   (epigraph var, minimized in the MILP)
    # Since this f term enters the log-linearized area-cost equation as
    # +(-cost_beta*ln(LMTD)) alongside +cost_beta*ln(Q) etc., f_hat <=
    # f_true means the linearized log-cost is UNDERESTIMATED -- the
    # same "optimistic" direction as before, just reached via the
    # opposite (max, not min) plane-combination rule. Pair with the
    # solution-pool / true-equation re-evaluation step downstream
    # before trusting the chosen topology.
    #
    # Validation combines a dense interior grid with targeted
    # boundary/diagonal sampling (full_validation): within a single
    # tangent plane's active region, error = f - plane is convex minus
    # affine = still convex, so its true max sits on a boundary or
    # plane-switching curve, not necessarily an interior grid point.
    # max_rel_err below reflects the worst of both sources.
    # ════════════════════════════════════════════════════════════════
    lmtd_planes = {}   # (i,j) -> list of (a0,a1,a2) for f = -cost_beta*ln(LMTD)
    lmtd_N_G = {}       # (i,j) -> N_G actually used for that match
    lmtd_diag = {}       # (i,j) -> dict of error/QA stats for logging

    for i in Hi:
        for j in Hj:
            if not feasible_match[i, j]:
                continue

            dT_lo_ij = dT1_lo[i, j]   # == dT2_lo[i, j] by construction
            dT_hi_ij = dT1_hi[i, j]   # == dT2_hi[i, j] by construction

            cost_beta_ij = cost_beta

            match_result = analyze_match(
                    dT_hi_ij, delta_tmin, cost_beta_ij,
                    mode=LMTD_GRID_MODE,
                    N_G_fixed=LMTD_N_G_FIXED,
                    error_threshold=LMTD_ERROR_THRESHOLD,
                    convergence_tol=LMTD_CONVERGENCE_TOL,
                    N_start=LMTD_N_START, N_max=LMTD_N_MAX,
                    dense_N=LMTD_DENSE_N, n_per_edge=LMTD_N_PER_EDGE,
                    use_symmetry=LMTD_USE_SYMMETRY,
                    make_plots=False,
                )
            N_G_ij = match_result["N_G"]
            planes_ij = match_result["planes"]
            val_result = match_result["result"]

            if val_result['n_sign_violations'] > 0:
                warnings.warn(
                    f"-beta*ln(LMTD) hyperplanes for match ({HID[i]},{CID[j]}) are NOT "
                    f"strictly underestimating (envelope exceeded the true convex "
                    f"function) at {val_result['n_sign_violations']} test point(s) "
                    f"-- check numerical tolerance near dT1==dT2.")

            if val_result["max_rel_err"] > LMTD_ERROR_THRESHOLD:
                warnings.warn(
                    f"-beta*ln(LMTD) hyperplane max relative error for match "
                    f"({HID[i]},{CID[j]}) = {val_result['max_rel_err']*100:.3f}% "
                    f"(worst on {val_result['worst_source']}) exceeds threshold "
                    f"{LMTD_ERROR_THRESHOLD*100:.2f}% (N_G={N_G_ij}, "
                    f"{len(planes_ij)} planes). Consider raising LMTD_N_MAX.")

            lmtd_planes[i, j] = planes_ij
            lmtd_N_G[i, j] = N_G_ij
            lmtd_diag[i, j] = {
                "max_err": val_result["max_err"],
                "max_rel_err": val_result["max_rel_err"],
                "avg_err": val_result["avg_err"],
                "avg_rel_err": val_result["avg_rel_err"],
                "worst_point": val_result["worst_point"],
                "worst_source": val_result["worst_source"],
                "interior_max_rel_err": val_result["dense"]["max_rel_err"],
                "boundary_max_rel_err": val_result["boundary"]["max_rel_err"],
                "n_planes": len(planes_ij),
                "n_sign_violations": val_result["n_sign_violations"],
            }

    # ════════════════════════════════════════════════════════════════════
    #   (Q, LMTD) -> Cost grid with:
    #   (2) Q -> LnQ            : 1D tangent lines (concave overestimate)
    #   (4) A_beta -> Cost      : 1D SOS2 fit of cost_a + cost_b*exp(x)
    #   A_beta itself (step 3) is just a linear equality in the model:
    #   A_beta[i,j,k] == cost_beta*LnQ[i,j,k] - cost_beta*ln(U_ij)
    #                    + NegBetaLnLMTD[i,j,k]
    # ════════════════════════════════════════════════════════════════════

    # ---- (2) Q -> ln(Q), 1D tangent-line (concave overestimator) envelope ----
    def ln_tangent_line_coeffs(x0):
        """Tangent line to ln(x) at x0: ln(x) <= b0 + b1*x, tight at x0."""
        b1 = 1.0 / x0
        b0 = np.log(x0) - b1 * x0
        return b0, b1

    def build_lnQ_tangent_lines(Q_lo, Q_hi, N_G):
        """
        ln is concave -> tangent lines sit at/above it everywhere ->
        valid outer approx is MIN over lines:
            ln(Q) <= b0 + b1*Q   for every line
            LnQ_hat = min over lines
        """
        grid = np.geomspace(Q_lo, Q_hi, N_G)
        return [ln_tangent_line_coeffs(Q0) for Q0 in grid]

    def lnQ_envelope_value(Q, lines):
        Q = np.asarray(Q, dtype=float)
        b0 = np.array([l[0] for l in lines])
        b1 = np.array([l[1] for l in lines])
        return (b0 + b1 * Q[..., None]).min(axis=-1)

    def validate_lnQ_lines(lines, Q_lo, Q_hi, dense_N=300):
        Qg = np.geomspace(Q_lo, Q_hi, dense_N)
        true_val = np.log(Qg)
        approx = lnQ_envelope_value(Qg, lines)
        err = approx - true_val                       # should be >= 0 (overestimate)
        rel_err = err / np.maximum(np.abs(true_val), 1e-9)
        idx = int(np.argmax(rel_err))
        return {
            "max_err": float(np.max(err)),
            "max_rel_err": float(np.max(rel_err)),
            "n_sign_violations": int(np.sum(err < -1e-6)),
            "worst_Q": float(Qg[idx]),
        }

    def select_lnQ_grid_size_dynamic(Q_lo, Q_hi, error_threshold, convergence_tol,
                                      N_start, N_max):
        prev = None
        best = None
        for N_G in range(N_start, N_max + 1):
            lines = build_lnQ_tangent_lines(Q_lo, Q_hi, N_G)
            val = validate_lnQ_lines(lines, Q_lo, Q_hi)
            if best is None or val["max_rel_err"] < best["max_rel_err"]:
                best = {"N_G": N_G, "lines": lines, **val}
            converged = prev is not None and abs(prev - val["max_rel_err"]) <= convergence_tol
            if converged and val["max_rel_err"] <= error_threshold:
                return N_G, lines, val
            prev = val["max_rel_err"]
        warnings.warn(f"ln(Q) tangent-line target error {error_threshold*100:.2f}% not "
                       f"reached within N_max={N_max}; using best found N_G={best['N_G']} "
                       f"(max_rel_err={best['max_rel_err']*100:.3f}%).")
        return best["N_G"], best["lines"], best

    # ---- (4) A_beta -> Cost, 1D SOS2 fit of cost_a + cost_b*exp(x) ----
    def cost_of_Abeta(x):
        return cost_a + cost_b * np.exp(x)

    def build_Abeta_cost_sos2(Abeta_lo, Abeta_hi, n_pts):
        xs = np.linspace(Abeta_lo, Abeta_hi, n_pts)
        return xs, cost_of_Abeta(xs)

    def validate_Abeta_cost_sos2(xs, dense_N=300):
        cs = cost_of_Abeta(xs)
        xg = np.linspace(xs[0], xs[-1], dense_N)
        true_val = cost_of_Abeta(xg)
        approx = np.interp(xg, xs, cs)
        err = approx - true_val                        # chord of convex fn >= true
        rel_err = err / np.maximum(np.abs(true_val), 1e-9)
        idx = int(np.argmax(rel_err))
        return {
            "max_err": float(np.max(err)),
            "max_rel_err": float(np.max(rel_err)),
            "n_sign_violations": int(np.sum(err < -1e-6)),
            "worst_x": float(xg[idx]),
        }

    def select_Abeta_grid_size_dynamic(Abeta_lo, Abeta_hi, error_threshold, convergence_tol,
                                        N_start, N_max):
        prev = None
        best = None
        for n_pts in range(N_start, N_max + 1):
            xs, _cs = build_Abeta_cost_sos2(Abeta_lo, Abeta_hi, n_pts)
            val = validate_Abeta_cost_sos2(xs)
            if best is None or val["max_rel_err"] < best["max_rel_err"]:
                best = {"n_pts": n_pts, "xs": xs, **val}
            converged = prev is not None and abs(prev - val["max_rel_err"]) <= convergence_tol
            if converged and val["max_rel_err"] <= error_threshold:
                return n_pts, xs, val
            prev = val["max_rel_err"]
        warnings.warn(f"A_beta->Cost SOS2 target error {error_threshold*100:.2f}% not "
                       f"reached within N_max={N_max}; using best found n_pts={best['n_pts']} "
                       f"(max_rel_err={best['max_rel_err']*100:.3f}%).")
        return best["n_pts"], best["xs"], best

    # ---- per-match build ----
    lnQ_lines, lnQ_N_G, lnQ_diag = {}, {}, {}
    Abeta_grid, Abeta_cost_grid, Abeta_n_pts, Abeta_diag = {}, {}, {}, {}
    Q_lo_lnq,Q_hi_lnq={},{}
    for i in Hi:
        for j in Hj:
            if not feasible_match[i, j]:
                continue

            U_ij = U_mat[i, j]
            Q_floor_ij = max(Q_floor_frac * Q_match_max[i, j], EPSILON)
            Q_hi_ij = Q_match_max[i, j]

            # --- (2) ln(Q) tangent lines ---
            N_G_lnQ, lines_ij, val_lnQ = select_lnQ_grid_size_dynamic(
                Q_floor_ij, Q_hi_ij,
                LNQ_ERROR_THRESHOLD, LNQ_CONVERGENCE_TOL,
                LNQ_N_START, LNQ_N_MAX)

            if val_lnQ["n_sign_violations"] > 0:
                warnings.warn(
                    f"ln(Q) tangent lines for match ({HID[i]},{CID[j]}) are NOT strictly "
                    f"overestimating at {val_lnQ['n_sign_violations']} test point(s).")
            if val_lnQ["max_rel_err"] > LNQ_ERROR_THRESHOLD:
                warnings.warn(
                    f"ln(Q) tangent-line max relative error for match ({HID[i]},{CID[j]}) "
                    f"= {val_lnQ['max_rel_err']*100:.3f}% exceeds threshold "
                    f"{LNQ_ERROR_THRESHOLD*100:.2f}% (N_G={N_G_lnQ}).")

            lnQ_lines[i, j] = lines_ij
            lnQ_N_G[i, j] = N_G_lnQ
            lnQ_diag[i, j] = val_lnQ
            Q_lo_lnq[i,j]=Q_floor_ij
            Q_hi_lnq[i,j]=Q_hi_ij

            # --- implied range of A_beta = beta*ln(Q) - beta*ln(U) + NegBetaLnLMTD ---
            dT_lo_ij = dT1_lo[i, j]
            dT_hi_ij = dT1_hi[i, j]
            negbetalnlmtd_lo = -cost_beta * np.log(dT_hi_ij)   # most negative, at dT_hi
            negbetalnlmtd_hi = -cost_beta * np.log(dT_lo_ij)   # most positive, at dT_lo

            Abeta_lo_ij = cost_beta * np.log(Q_floor_ij) - cost_beta * np.log(U_ij) + negbetalnlmtd_lo
            Abeta_hi_ij = cost_beta * np.log(Q_hi_ij)    - cost_beta * np.log(U_ij) + negbetalnlmtd_hi
            if Abeta_lo_ij > 0 or Abeta_hi_ij < 0:
                print(f"Match ({HID[i]},{CID[j]}): Abeta range [{Abeta_lo_ij:.3f}, {Abeta_hi_ij:.3f}] "
                    f"does not straddle 0 -> old big-M is undersized for this match")
            if Abeta_hi_ij <= Abeta_lo_ij:
                raise ValueError(f"Degenerate A_beta range for match ({HID[i]},{CID[j]}): "
                                  f"[{Abeta_lo_ij}, {Abeta_hi_ij}]")

            # --- (4) A_beta -> Cost SOS2 breakpoints ---
            n_pts_ij, xs_ij, val_Abeta = select_Abeta_grid_size_dynamic(
                Abeta_lo_ij, Abeta_hi_ij,
                ABETA_ERROR_THRESHOLD, ABETA_CONVERGENCE_TOL,
                ABETA_N_START, ABETA_N_MAX)

            if val_Abeta["max_rel_err"] > ABETA_ERROR_THRESHOLD:
                warnings.warn(
                    f"A_beta->Cost SOS2 max relative error for match ({HID[i]},{CID[j]}) "
                    f"= {val_Abeta['max_rel_err']*100:.3f}% exceeds threshold "
                    f"{ABETA_ERROR_THRESHOLD*100:.2f}% (n_pts={n_pts_ij}).")

            Abeta_grid[i, j] = xs_ij
            Abeta_cost_grid[i, j] = cost_of_Abeta(xs_ij)
            Abeta_n_pts[i, j] = n_pts_ij
            Abeta_diag[i, j] = val_Abeta

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
            Q_grid = np.linspace(Q_floor, Qmax, N_GU)
            Abeta_grid_hu = _hu_area_beta(Q_grid, T_hu_in, T_hu_out, Tout_C[j], CP_C[j], U_hu_j, cost_beta)
            Abeta_grid_hu[0] = 0.0

            Q_g_hu[hu, j] = Q_grid
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
            Q_grid = np.linspace(Q_floor, Qmax, N_GU)
            Abeta_grid_cu = _cu_area_beta(Q_grid, T_cu_in, T_cu_out, Tout_H[i], CP_H[i], U_cu_i, cost_beta)
            Abeta_grid_cu[0] = 0.0

            Q_g_cu[cu, i] = Q_grid
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
    m.G0 = pyo.RangeSet(0, data.N_G)             # 0 = dummy slack, 1..N_G = real breakpoints
    m.GU0 = pyo.RangeSet(0, data.N_GU-1)

    # ---- Feasible (i, j) match sets, reused by constraints.py & objective.py ----
    feasible_pairs = [(i, j) for i in m.Hi for j in m.Hj if data.feasible_match[i, j]]
    m.FeasiblePairs = pyo.Set(initialize=feasible_pairs, dimen=2)
    m.FeasibleIJK = pyo.Set(
        initialize=[(i, j, k) for (i, j) in feasible_pairs for k in m.Hs],
        dimen=3,)
    
    # ---- LMTD Hyperplane Cuts ----
    m.NegBetaLnLMTD_cut_index = pyo.Set(dimen=4, initialize=[
            (i, j, k, p)
            for (i, j, k) in m.FeasibleIJK
            for p in range(len(data.lmtd_planes[i, j]))
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

    # ---- (2) Q -> ln(Q): 1D tangent-line index set & breakpoint... wait, these are
    # lines not breakpoints, so store line coefficients directly as params ----
    m.LnQ_cut_index = pyo.Set(dimen=4, initialize=[
        (i, j, k, p)
        for (i, j, k) in m.FeasibleIJK
        for p in range(len(data.lnQ_lines[i, j]))
    ])
    m.LnQ_b0 = pyo.Param(
        m.FeasiblePairs, pyo.RangeSet(0, max(len(v) for v in data.lnQ_lines.values()) - 1),
        mutable=False, default=0.0,
        initialize=lambda m, i, j, p: float(data.lnQ_lines[i, j][p][0]) if p < len(data.lnQ_lines[i, j]) else 0.0)
    m.LnQ_b1 = pyo.Param(
        m.FeasiblePairs, pyo.RangeSet(0, max(len(v) for v in data.lnQ_lines.values()) - 1),
        mutable=False, default=0.0,
        initialize=lambda m, i, j, p: float(data.lnQ_lines[i, j][p][1]) if p < len(data.lnQ_lines[i, j]) else 0.0)

    # ---- (4) A_beta -> Cost: 1D SOS2 index set & breakpoint params ----
    m.Abeta_bp_index = pyo.Set(dimen=3, initialize=[
        (i, j, p)
        for (i, j) in m.FeasiblePairs
        for p in range(len(data.Abeta_grid[i, j]))
    ])
    m.Abeta_Pt = pyo.Set(dimen=4, initialize=[
        (i, j, k, p)
        for (i, j, k) in m.FeasibleIJK
        for p in range(len(data.Abeta_grid[i, j]))
    ])

    m.Abeta_bp = pyo.Param(
        m.Abeta_bp_index, mutable=False,
        initialize=lambda m, i, j, p: float(data.Abeta_grid[i, j][p]))
    m.Cost_bp = pyo.Param(
        m.Abeta_bp_index, mutable=False,
        initialize=lambda m, i, j, p: float(data.Abeta_cost_grid[i, j][p]))

    build_variables(m, data)
    build_constraints(m, data)
    build_objective(m, data)

    return m