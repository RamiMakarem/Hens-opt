import glob
import math
import os
import tempfile
import warnings

import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from types import SimpleNamespace

from Pre_process import _lmtd, pre_process_milp, build_model
from H_HP_Tester import H_true as _H_true_exact


# ═══════════════════════════════════════════════════════════════════════════
# SOLVE
# ═══════════════════════════════════════════════════════════════════════════

class _SimpleSolverStatus:
    """Minimal duck-typed stand-in for a Pyomo SolverResults.solver block --
    just enough (.termination_condition, .primal_bound) for
    extract_hens_results()'s status handling and solve_hen_milp_pool()'s own
    `results.solver.termination_condition` check to work identically to the
    real thing, for the code path that solves via PySCIPOpt directly
    instead of through Pyomo's own solver interface."""
    def __init__(self, termination_condition, primal_bound=None):
        self.termination_condition = termination_condition
        self.primal_bound = primal_bound


class _SimplePyomoResults:
    def __init__(self, termination_condition, primal_bound=None):
        self.solver = _SimpleSolverStatus(termination_condition, primal_bound)


# Maps PySCIPOpt's Model.getStatus() strings onto the pyomo.opt
# TerminationCondition values the rest of this codebase already branches on.
_SCIP_STATUS_MAP = {
    "optimal":       TerminationCondition.optimal,
    "gaplimit":      TerminationCondition.optimal,      # solved to the requested gap
    "timelimit":     TerminationCondition.maxTimeLimit,
    "nodelimit":     TerminationCondition.maxTimeLimit,
    "totalnodelimit": TerminationCondition.maxTimeLimit,
    "stallnodelimit": TerminationCondition.maxTimeLimit,
    "memlimit":      TerminationCondition.maxTimeLimit,
    "timelimit ":    TerminationCondition.maxTimeLimit,
    "userinterrupt": TerminationCondition.maxTimeLimit,
    "infeasible":    TerminationCondition.infeasible,
    "unbounded":     TerminationCondition.unbounded,
    "inforunbd":     TerminationCondition.infeasibleOrUnbounded,
}


def _solve_scip_pool_pyscipopt(m, time_limit, gap, tee, pool_size):
    """
    Solve `m` with SCIP's native Python bindings (PySCIPOpt) instead of the
    ASL/NL interface Pyomo normally drives SCIP through, specifically so the
    solver's actual in-memory solution pool can be read back afterward.

    Why this exists: SCIP's solution-pool export is an interactive-shell
    dialog command ("write allsolutions <file>"), not a settable .set-file
    parameter -- so a previous version of this code that set
    solver.options['write/allsolutions'] on the ASL-based 'scip' Pyomo
    plugin was silently doing nothing (SCIP just logs an "unknown
    parameter" warning and keeps solving). That's exactly why a run that
    logged "Primal Bound ... (20 solutions)" still only ever produced one
    NLP refinement downstream -- there was no working export mechanism,
    so extract_all_scip_solutions() correctly found zero pool files and
    fell back to the incumbent alone.

    PySCIPOpt's Model.getSols() reads SCIP's solution storage directly, so
    it doesn't depend on any file-export mechanism at all. This function
    writes `m` to a temporary .lp file (symbolic_solver_labels=True, so
    variable names round-trip), solves that file's problem in a fresh
    PySCIPOpt Model, and returns every solution found.

    Returns
    -------
    best_var_dict : {var_name: value} for the best solution, or None if
        SCIP found no feasible solution at all.
    pool : list[{var_name: value}], one entry per solution in SCIP's
        solution storage (including the best one).
    status : pyomo.opt.TerminationCondition
    obj_val : the best solution's objective value, or None.

    Raises ImportError if PySCIPOpt isn't installed, or any other
    Exception PySCIPOpt itself raises (e.g. an invalid parameter name on
    an older SCIP build) -- callers are expected to catch both and fall
    back to the regular ASL-based single-incumbent solve.
    """
    import pyscipopt  # raises ImportError here if not installed

    fd, lp_path = tempfile.mkstemp(suffix=".lp")
    os.close(fd)
    try:
        m.write(lp_path, io_options={"symbolic_solver_labels": True})

        scip_m = pyscipopt.Model()
        if not tee:
            scip_m.hideOutput()
        scip_m.readProblem(lp_path)

        # Real, settable SCIP parameters only -- unlike the ASL .set-file
        # route, PySCIPOpt raises immediately on an unrecognized parameter
        # name instead of silently ignoring it, so only params known to be
        # genuine SCIP options are set here.
        scip_m.setParam("limits/time", float(time_limit))
        scip_m.setParam("limits/gap", float(gap))
        scip_m.setParam("limits/maxsol", int(pool_size))
        scip_m.setParam("limits/maxorigsol", int(pool_size))
        scip_m.setParam("presolving/maxrounds", -1)

        # Re-apply the same high branching priority solve_model() gives the
        # active-match binaries on the normal ASL path, so pool search
        # behaves consistently either way. Matched by name since the LP
        # file was written with symbolic_solver_labels=True.
        if hasattr(m, "FeasibleIJK") and hasattr(m, "z"):
            priority_names = {m.z[i, j, k].name for (i, j, k) in m.FeasibleIJK}
            if priority_names:
                for v in scip_m.getVars():
                    if v.name in priority_names:
                        scip_m.setVarBranchPriority(v, 1000)

        scip_m.optimize()

        if scip_m.getNSols() == 0:
            raw_status = scip_m.getStatus()
            status = _SCIP_STATUS_MAP.get(raw_status, TerminationCondition.other)
            return None, [], status, None

        scip_vars = scip_m.getVars()
        best_sol = scip_m.getBestSol()

        def _sol_to_dict(sol):
            return {v.name: scip_m.getSolVal(sol, v) for v in scip_vars}

        pool = [_sol_to_dict(s) for s in scip_m.getSols()]
        best_dict = _sol_to_dict(best_sol)
        obj_val = scip_m.getSolObjVal(best_sol)

        raw_status = scip_m.getStatus()
        status = _SCIP_STATUS_MAP.get(raw_status, TerminationCondition.optimal)

        return best_dict, pool, status, obj_val
    finally:
        try:
            os.remove(lp_path)
        except OSError:
            pass


def solve_model(m, solver_name='scip', time_limit=500, gap=0.001, tee=True,
                 collect_solution_pool=False, pool_dir=None, pool_size=50):
    """
    Solve the HENS MILP using Pyomo's 'scip' solver interface.

    `solver_name` is kept as a parameter (defaulting to 'scip') in case you
    want to point it at a different SCIP install/interface later, but no
    other solver is tried as a fallback.

    collect_solution_pool : bool, default False
        Opt-in flag. When True, solves via PySCIPOpt (SCIP's native Python
        bindings) instead of the usual ASL interface, so every solution in
        SCIP's own solution pool can be read back in-memory afterward (via
        `m._scip_solution_pool`, consumed by `extract_all_scip_solutions`)
        -- not just the final incumbent. Requires `pip install pyscipopt`
        (and a matching SCIP install/SCIPOPTDIR if not using a
        self-contained wheel); if it's unavailable or fails for any
        reason, this transparently falls back to the normal single-
        incumbent ASL solve and warns rather than raising, so pool
        exploration degrades gracefully instead of breaking the run.
    pool_dir : str, optional
        Kept for backward compatibility with a now-unused file-export path
        (SCIP's actual pool dump is an interactive-shell-only command, not
        a settable parameter, so nothing is ever written here anymore --
        see `_solve_scip_pool_pyscipopt`'s docstring). Unused otherwise.
    pool_size : int, default 50
        Upper bound on how many solutions SCIP is asked to retain
        internally when `collect_solution_pool=True` -- maps to SCIP's own
        'limits/maxsol' / 'limits/maxorigsol' parameters.
    """
    if collect_solution_pool:
        try:
            best_dict, pool, status, obj_val = _solve_scip_pool_pyscipopt(
                m, time_limit, gap, tee, pool_size)
            m._scip_solution_pool = pool
            if best_dict is None:
                return _SimplePyomoResults(status, primal_bound=None)
            _load_solution_into_model(m, best_dict)
            if tee:
                print(f"PySCIPOpt solution pool: {len(pool)} solution(s) retained "
                      f"(best objective = {obj_val:,.2f}).")
            return _SimplePyomoResults(status, primal_bound=obj_val)
        except ImportError:
            warnings.warn(
                "solve_model: collect_solution_pool=True requires PySCIPOpt "
                "('pip install pyscipopt', plus a matching SCIP install/"
                "SCIPOPTDIR if not using a self-contained wheel), which "
                "isn't available here -- falling back to a normal single-"
                "incumbent SCIP solve via the ASL interface (solution-pool "
                "exploration disabled for this run).")
            m._scip_solution_pool = []
        except Exception as exc:
            warnings.warn(
                f"solve_model: PySCIPOpt solution-pool solve failed ({exc}); "
                f"falling back to a normal single-incumbent SCIP solve via "
                f"the ASL interface.")
            m._scip_solution_pool = []

    m.priority = pyo.Suffix(direction=pyo.Suffix.EXPORT)

    # 2. Assign high branching priority to all active binary match variables
    for (i, j, k) in m.FeasibleIJK:
        m.priority[m.z[i, j, k]] = 1000

    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' is not available to Pyomo. "
            f"Make sure the `scip` executable is on PATH.")

    # Enable aggressive primal heuristics to find good upper bounds quickly
    solver.options['heuristics/emphasis'] = 'A'  
    # Focus presolving on bound tightening
    solver.options['presolving/maxrounds'] = -1
    solver.options['limits/time'] = time_limit
    solver.options['limits/gap'] = gap

    results = solver.solve(m, tee=tee)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# EXTRACT
# ═══════════════════════════════════════════════════════════════════════════

def extract_hens_results(m, data=None, solver_results=None, verbose=True):
    """
    Extracts Pyomo HENS results, calculates thermodynamic physical errors (LMTD, Area, Cost)
    and DLOG/SOS2 linearization errors directly from model `m` and optional `data` namespace.

    Returns a comprehensive dictionary containing all edge maps, temperature profiles,
    utility duties, cost breakdowns, and error metrics.
    """
    Q_THRESH = 1e-4

    # ── 1. Safe Variable / Parameter Getter Helpers ───────────────────────────
    def _get_attr(names, default=None):
        for name in names:
            if hasattr(m, name):
                return getattr(m, name)
            if data is not None and hasattr(data, name):
                return getattr(data, name)
        return default

    def _get_val(names, idx_tuple, default=0.0):
        attr = _get_attr(names)
        if attr is not None:
            try:
                val = pyo.value(attr[idx_tuple])
                return float(val) if val is not None else default
            except (KeyError, ValueError, TypeError):
                pass
        return default

    # ── 2. Extract Sets & Core Parameters ─────────────────────────────────────
    Hi = list(_get_attr(["Hi", "I"], []))
    Hj = list(_get_attr(["Hj", "J"], []))
    Hs = list(_get_attr(["Hs", "S"], []))
    Knodes = list(_get_attr(["Knodes", "K"], range(len(Hs) + 1)))

    I, J, S = len(Hi), len(Hj), len(Hs)
    K = len(Knodes) - 1

    # Stream IDs -- m.HID/m.CID are never defined as Pyomo components, so
    # this always falls through to the plain Python lists in `data`.
    HID = [str(pyo.value(m.HID[i])) if hasattr(m, "HID") else (data.HID[i] if data and hasattr(data, "HID") else f"H{i+1}") for i in range(I)]
    CID = [str(pyo.value(m.CID[j])) if hasattr(m, "CID") else (data.CID[j] if data and hasattr(data, "CID") else f"C{j+1}") for j in range(J)]

    # Model Cost & Process Parameters
    delta_tmin = float(pyo.value(m.delta_tmin) if hasattr(m, "delta_tmin") else getattr(data, "delta_tmin", 10.0))
    payback = float(pyo.value(m.payback) if hasattr(m, "payback") else getattr(data, "payback", 1.0))
    cost_a = float(pyo.value(m.cost_a) if hasattr(m, "cost_a") else getattr(data, "cost_a", 5500.0))
    cost_b = float(pyo.value(m.cost_b) if hasattr(m, "cost_b") else getattr(data, "cost_b", 150.0))
    cost_beta = float(pyo.value(m.cost_beta) if hasattr(m, "cost_beta") else getattr(data, "cost_beta", 1.0))
    U_overall = float(pyo.value(m.U_overall) if hasattr(m, "U_overall") else getattr(data, "U_overall", 0.5))

    def get_U(i_idx, j_idx):
        if hasattr(m, "U") and (Hi[i_idx], Hj[j_idx]) in m.U:
            return float(pyo.value(m.U[Hi[i_idx], Hj[j_idx]]))
        elif hasattr(m, "U_mat"):
            return float(pyo.value(m.U_mat[i_idx, j_idx]))
        elif data is not None and hasattr(data, "U_mat"):
            return float(data.U_mat[i_idx, j_idx])
        return U_overall

    # ── 3. Extract Temperatures ───────────────────────────────────────────────
    T_hot = np.zeros((I, K + 1))
    T_cold = np.zeros((J, K + 1))

    for i_idx, i in enumerate(Hi):
        for k_idx, k in enumerate(Knodes):
            T_hot[i_idx, k_idx] = _get_val(["TH", "th"], (i, k))

    for j_idx, j in enumerate(Hj):
        for k_idx, k in enumerate(Knodes):
            T_cold[j_idx, k_idx] = _get_val(["TC", "tc"], (j, k))

    Tout_H = [float(T_hot[i, -1]) for i in range(I)]
    Tout_C = [float(T_cold[j, 0]) for j in range(J)]

    # ── 4. Process HEX Sizing & Error Analysis ────────────────────────────────
    edges = []
    Q_arr = np.zeros((I, J, S))

    for i_idx, i in enumerate(Hi):
        for j_idx, j in enumerate(Hj):
            for s_idx, s in enumerate(Hs):
                Q = _get_val(["Q", "q"], (i, j, s))
                Q_arr[i_idx, j_idx, s_idx] = Q

                if Q > Q_THRESH:
                    U_ij = get_U(i_idx, j_idx)
                    z_val = _get_val(["z", "Z"], (i, j, s), default=1.0)

                    # Per-match beta: pre_process_milp accepts either a
                    # single scalar cost_beta or a {(i,j): beta} dict, and
                    # Pre_process.py itself resolves the match-specific
                    # value with this exact isinstance check. Solve_extract
                    # used to be able to ignore this entirely (m.A was read
                    # straight off the model, beta-independent); it can't
                    # anymore, since Area is now RECOVERED from A^beta
                    # (=m.P) via A = P^(1/beta) below, so the wrong beta
                    # here would silently corrupt Area/Cost for any run
                    # using a per-match beta dict.
                    beta_ij = (data.cost_beta[i, j]
                               if isinstance(data.cost_beta, dict) else cost_beta)

                    # A. Physical True Values (Thermodynamic Rigorous) --
                    # unchanged in method: dT1_true/dT2_true still come
                    # straight off the model's own dT1/dT2 (Block 8), which
                    # were never approximated (only LMTD/H were).
                    dT1_calc =T_hot[i_idx, s_idx] - T_cold[j_idx, s_idx]
                    dT2_calc =T_hot[i_idx, s_idx + 1] - T_cold[j_idx, s_idx + 1]
                    dT1_true = _get_val(["dT1"], (i, j, s), default=dT1_calc)
                    dT2_true = _get_val(["dT2"], (i, j, s), default=dT2_calc)
                    dT1_model=dT1_true
                    dT2_model=dT2_true
                    lmtd_true = _lmtd(dT1_true, dT2_true)
                    area_true = Q / (U_ij * max(lmtd_true, 1e-4))
                    Abeta_true = area_true ** beta_ij
                    cost_true = cost_a + cost_b * Abeta_true

                    # B. Decision Variables from Pyomo Model
                    #
                    # m.lmtd / m.A / m.W no longer exist (see Variables.py /
                    # Constraints.py's H/G/P redesign). LMTD is no longer a
                    # decision variable at all -- it is now a purely derived
                    # physical quantity, computed the same way "true" LMTD
                    # is (from dT1_model/dT2_model), so LMTD_model and
                    # LMTD_true below are now IDENTICAL by construction
                    # (err_lmtd_* is kept as a trivial ~0 sanity check, not
                    # a real linearisation gap anymore -- that gap moved to
                    # H, see Block D). Area is recovered from A^beta (m.P)
                    # since there is no longer a separate Area variable.
                    cap_fixed_per_hex = cost_a
                    cost_model_var = _get_val(["Cost", "cost"], (i, j, s), default=None)

                    lmtd_model = _lmtd(dT1_model, dT2_model)
                    H_model = _get_val(["H"], (i, j, s), default=None)
                    Abeta_model = _get_val(["P"], (i, j, s), default=None)   # P = A^beta
                    # G = Q^beta: LinearIJK matches (beta==1) have no G
                    # variable at all -- Constraints.py substitutes Q
                    # directly for G there, so do the same here.
                    if data.beta_linear[i, j]:
                        G_model = Q
                    else:
                        G_model = _get_val(["G"], (i, j, s), default=Q ** beta_ij)

                    area_model = (Abeta_model ** (1.0 / beta_ij)
                                  if Abeta_model is not None and Abeta_model > 0 else 0.0)
                    Q_model = Q if Q is not None else None

                    cost_model = cap_fixed_per_hex + cost_model_var

                    # McCormick relaxation gap: how far P (=m.P, the
                    # McCormick-relaxed A^beta) is from the EXACT bilinear
                    # product G_model*H_model at the model's own (G, H)
                    # values. This replaces the old W-vs-A*LMTD diagnostic
                    # one level down the new chain (P=G*H instead of
                    # W=A*lmtd) -- m.W no longer exists, so this metric is
                    # renamed rather than aliased.
                    if Abeta_model is not None and H_model is not None and G_model is not None:
                        P_true_GxH = G_model * H_model
                        err_P_abs = Abeta_model - P_true_GxH
                        err_P_pct = (abs(err_P_abs) / max(abs(P_true_GxH), 1e-6)) * 100.0
                    else:
                        P_true_GxH = None
                        err_P_abs = 0.0
                        err_P_pct = 0.0

                    # C. Physical Errors (model decision variables vs. rigorous thermodynamics)
                    err_lmtd_abs = lmtd_model - lmtd_true
                    err_lmtd_pct = (abs(err_lmtd_abs) / max(lmtd_true, 1e-4)) * 100.0

                    err_area_abs = area_model - area_true
                    err_area_pct = (abs(err_area_abs) / max(area_true, 1e-4)) * 100.0

                    err_cost_abs = cost_model - cost_true
                    err_cost_pct = (abs(err_cost_abs) / max(cost_true, 1e-4)) * 100.0

                    # D. Linearisation / Envelope Fitting Errors
                    #    (i)  H tangent-plane OA gap: BLOCK 9's cuts
                    #         UNDER-estimate the true, convex
                    #         H=(U*LMTD)^-beta surface; this is the gap
                    #         between the model's own H[i,j,k] and the
                    #         exact H evaluated at the model's own dT1/dT2.
                    #         Direct replacement for the old
                    #         exact_lmtd_at_model_dT / sos2_lmtd_fit_err
                    #         diagnostic (LMTD's grid is gone -- H's
                    #         tangent-plane envelope is the new OA gap).
                    H_true_at_model_dT = _H_true_exact(dT1_model, dT2_model, beta_ij, U_ij)
                    err_H_fit_abs = H_model - H_true_at_model_dT if H_model is not None else 0.0
                    err_H_fit_pct = (abs(err_H_fit_abs) / max(H_true_at_model_dT, 1e-9)) * 100.0

                    #    (ii) G=Q^beta SOS2 fit error: gap between the
                    #         SOS2-interpolated G[i,j,k] and the exact
                    #         Q^beta evaluated at the model's own Q --
                    #         i.e. the residual PWL interpolation error of
                    #         the new 1D G grid at the chosen operating
                    #         point. Direct replacement for the old DLOG
                    #         2D Cost-grid fit error (that 2D grid is gone;
                    #         the only piecewise-nonlinear fit left on the
                    #         Q-side is this 1D one). Always ~0 for
                    #         LinearIJK matches, since G==Q there exactly.
                    G_true_at_model_Q = Q ** beta_ij
                    err_G_fit_abs = G_model - G_true_at_model_Q
                    err_G_fit_pct = (abs(err_G_fit_abs) / max(G_true_at_model_Q, 1e-9)) * 100.0

                    edges.append({
                        "hot": HID[i_idx],
                        "cold": CID[j_idx],
                        "stage": s_idx + 1 if isinstance(s, int) else s,
                        "hot_id": HID[i_idx],
                        "cold_id": CID[j_idx],
                        "Q": round(Q, 4),
                        "A": round(area_model, 2),
                        "active": bool(z_val > 0.5),
                        # New: H=(U*LMTD)^-beta, G=Q^beta, and P=A^beta --
                        # the variables that replaced lmtd/A/W. Kept both
                        # as raw values (for warm-starting/inspection) and
                        # as the McCormick-gap error pair below.
                        "H": round(H_model, 6) if H_model is not None else None,
                        "H_true": round(H_true_at_model_dT, 6),
                        "G": round(G_model, 4) if G_model is not None else None,
                        "G_true": round(G_true_at_model_Q, 4),
                        "Abeta": round(Abeta_model, 6) if Abeta_model is not None else None,
                        "Abeta_true": round(Abeta_true, 6),
                        "P_true_GxH": round(P_true_GxH, 6) if P_true_GxH is not None else None,
                        "err_P_abs": round(err_P_abs, 6),
                        "err_P_pct": round(err_P_pct, 2),
                        "err_H_fit_abs": round(err_H_fit_abs, 6),
                        "err_H_fit_pct": round(err_H_fit_pct, 2),
                        "err_G_fit_abs": round(err_G_fit_abs, 4),
                        "err_G_fit_pct": round(err_G_fit_pct, 2),
                        # Plain names -- as-designed (model) values, used by
                        # the UI and as NLP warm-start values. Same field
                        # names as before the redesign.
                        "LMTD": round(lmtd_model, 2),
                        "Area_m2": round(area_model, 2),
                        "CapCost_$": round(cost_model, 0),
                        # True Thermodynamics
                        "LMTD_true": round(lmtd_true, 2),
                        "Area_m2_true": round(area_true, 2),
                        "Cost_true_$": round(cost_true, 0),
                        # Model Variables
                        "LMTD_model": round(lmtd_model, 2),
                        "Area_m2_model": round(area_model, 2),
                        "Cost_model_$": round(cost_model, 0),
                        # Errors
                        "err_lmtd_abs": round(err_lmtd_abs, 3),
                        "err_lmtd_pct": round(err_lmtd_pct, 2),
                        "err_area_abs": round(err_area_abs, 3),
                        "err_area_pct": round(err_area_pct, 2),
                        "err_cost_abs": round(err_cost_abs, 2),
                        "err_cost_pct": round(err_cost_pct, 2),})

    # Split fractions
    split_hot  = [[[0.0] * J for _ in range(S)] for _ in range(I)]
    split_cold = [[[0.0] * I for _ in range(S)] for _ in range(J)]

    for i in Hi:
        for k in Hs:
            total = sum(Q_arr[i, j, k] for j in Hj)
            for j in Hj:
                if total > Q_THRESH:
                    split_hot[i][k][j] = Q_arr[i, j, k] / total

    for j in Hj:
        for k in Hs:
            total = sum(Q_arr[i, j, k] for i in Hi)
            for i in Hi:
                if total > Q_THRESH:
                    split_cold[j][k][i] = Q_arr[i, j, k] / total

    # ── 5. Utility Exchangers & OPEX ─────────────────────────────────────────
    HU_set = list(_get_attr(["HU"], []))
    CU_set = list(_get_attr(["CU"], []))

    QH_agg = [0.0] * J
    QC_agg = [0.0] * I
    util_hex_edges = []
    ann_util = 0.0

    # Multi-utility mapping if HU / CU sets are defined
    if HU_set or CU_set:
        for u_idx, u in enumerate(HU_set):
            hu_cost = _get_val(["hu_opex"], u, default=0.0)
            # Effective supply/return temps (phase-adjusted for combined
            # utilities); precomputed once per utility in Pre_Process.py.
            T_hu_in = float(data.T_hu_in_eff[u]) if data is not None and hasattr(data, "T_hu_in_eff") else 250.0
            T_hu_out = float(data.T_hu_out_eff[u]) if data is not None and hasattr(data, "T_hu_out_eff") else 250.0

            for j_idx, j in enumerate(Hj):
                Q_uj = _get_val(["QHU"], (u, j))
                if Q_uj > Q_THRESH:
                    QH_agg[j_idx] += Q_uj
                    ann_util += Q_uj * hu_cost

                    # dT1 is Q-independent (precomputed in data.dT1_HU);
                    # dT2 depends on Q_uj -- matches _hu_area_beta() exactly.
                    CPc = data.CP_C[j_idx] if data is not None and hasattr(data, "CP_C") else 1.0
                    dT1_u = data.dT1_HU[u, j] if data is not None and hasattr(data, "dT1_HU") else (T_hu_in - Tout_C[j_idx])
                    dT2_u = T_hu_out - Tout_C[j_idx] + Q_uj / max(CPc, 1e-9)
                    dT1_u = max(dT1_u, delta_tmin)
                    dT2_u = max(dT2_u, delta_tmin)
                    lmtd_u = max(_lmtd(dT1_u, dT2_u), 1e-4)

                    U_hu_j = U_overall
                    if data is not None and hasattr(data, "U_mat") and hasattr(data, "I"):
                        row, col = data.I + u, j
                        if row < data.U_mat.shape[0] and col < data.U_mat.shape[1]:
                            U_hu_j = float(data.U_mat[row, col])

                    area_u = Q_uj / (U_hu_j * lmtd_u)
                    # Pull the DLOG/SOS2-linked variable cost straight from
                    # the model where available (now that BLOCK 11's link
                    # constraints are active); fall back to the formula only
                    # if Cost_HU isn't present at all.
                    cost_var_model = _get_val(["Cost_HU"], (u, j), default=(cost_b * (area_u ** cost_beta)))
                    cap_ann = cost_a + cost_var_model
                    util_hex_edges.append({
                        "type": "hot",
                        "utility": str(u),
                        "hot": str(u),
                        "cold": CID[j_idx],
                        "stream": j_idx,
                        "stream_id": CID[j_idx],
                        "Q": round(Q_uj, 4),
                        "A": round(area_u, 2),
                        "LMTD": round(lmtd_u, 2),
                        "Area_m2": round(area_u, 2),
                        "CapCost_$": round(cap_ann, 0),})

        for v_idx, v in enumerate(CU_set):
            cu_cost = _get_val(["cu_opex"], v, default=0.0)
            T_cu_in = float(data.T_cu_in_eff[v]) if data is not None and hasattr(data, "T_cu_in_eff") else 20.0
            T_cu_out = float(data.T_cu_out_eff[v]) if data is not None and hasattr(data, "T_cu_out_eff") else 30.0

            for i_idx, i in enumerate(Hi):
                Q_vi = _get_val(["QCU"], (v, i))
                if Q_vi > Q_THRESH:
                    QC_agg[i_idx] += Q_vi
                    ann_util += Q_vi * cu_cost

                    # dT2 is Q-independent (precomputed in data.dT2_CU);
                    # dT1 depends on Q_vi -- matches _cu_area_beta() exactly.
                    CPh = data.CP_H[i_idx] if data is not None and hasattr(data, "CP_H") else 1.0
                    dT2_u = data.dT2_CU[v, i] if data is not None and hasattr(data, "dT2_CU") else (Tout_H[i_idx] - T_cu_in)
                    dT1_u = Tout_H[i_idx] + Q_vi / max(CPh, 1e-9) - T_cu_out
                    dT1_u = max(dT1_u, delta_tmin)
                    dT2_u = max(dT2_u, delta_tmin)
                    lmtd_u = max(_lmtd(dT1_u, dT2_u), 1e-4)

                    U_cu_i = U_overall
                    if data is not None and hasattr(data, "U_mat") and hasattr(data, "J"):
                        row, col = i, data.J + v
                        if row < data.U_mat.shape[0] and col < data.U_mat.shape[1]:
                            U_cu_i = float(data.U_mat[row, col])

                    area_u = Q_vi / (U_cu_i * lmtd_u)
                    cost_var_model = _get_val(["Cost_CU"], (v, i), default=(cost_b * (area_u ** cost_beta)))
                    cap_ann = cost_a + cost_var_model
                    util_hex_edges.append({
                        "type": "cold",
                        "utility": str(v),
                        "hot": HID[i_idx],
                        "cold": str(v),
                        "stream": i_idx,
                        "stream_id": HID[i_idx],
                        "Q": round(Q_vi, 4),
                        "A": round(area_u, 2),
                        "LMTD": round(lmtd_u, 2),
                        "Area_m2": round(area_u, 2),
                        "CapCost_$": round(cap_ann, 0),})
    else:
        # Direct stream utility mapping (q_hu / q_cu) -- kept for
        # compatibility with older/simplified model variants that don't
        # use the HU/CU utility-option sets at all.
        hu_opex_unit = getattr(data, "hu_opex", [80.0])[0] if data else 80.0
        cu_opex_unit = getattr(data, "cu_opex", [20.0])[0] if data else 20.0

        for j_idx, j in enumerate(Hj):
            q_hu_val = _get_val(["q_hu", "QHU"], j)
            QH_agg[j_idx] = round(q_hu_val, 2)
            if q_hu_val > Q_THRESH:
                a_hu = _get_val(["area_hu", "A_HU"], j)
                util_hex_edges.append({
                    "type": "hot",
                    "stream": j_idx,
                    "stream_id": HID[j_idx],
                    "Q": round(q_hu_val, 2),
                    "A": round(a_hu, 2),
                    "Area_m2": round(a_hu, 2),
                    "CapCost_$": round(cost_a + cost_b * (a_hu ** cost_beta), 0) if a_hu > 0 else 0.0,})
                ann_util += q_hu_val * hu_opex_unit

        for i_idx, i in enumerate(Hi):
            q_cu_val = _get_val(["q_cu", "QCU"], i)
            QC_agg[i_idx] = round(q_cu_val, 2)
            if q_cu_val > Q_THRESH:
                a_cu = _get_val(["area_cu", "A_CU"], i)
                util_hex_edges.append({
                    "type": "cold",
                    "stream": i_idx,
                    "stream_id": CID[i_idx],
                    "Q": round(q_cu_val, 2),
                    "A": round(a_cu, 2),
                    "Area_m2": round(a_cu, 2),
                    "CapCost_$": round(cost_a + cost_b * (a_cu ** cost_beta), 0) if a_cu > 0 else 0.0,})
                ann_util += q_cu_val * cu_opex_unit

    # ── 6. Cost Aggregation & Objective Values ────────────────────────────────
    ann_cap_process = sum(e["Cost_true_$"] for e in edges)
    ann_cap_util = sum(e.get("CapCost_$", 0.0) for e in util_hex_edges)

    ann_cap = ann_cap_process + ann_cap_util
    tac_true = ann_util + ann_cap

    obj_val = None
    for obj_name in ("obj", "OBJ", "objective"):
        if hasattr(m, obj_name):
            try:
                obj_val = float(pyo.value(getattr(m, obj_name)))
                break
            except (ValueError, TypeError):
                pass
    if obj_val is None:
        # Fall back to the recomputed true TAC so downstream error-report
        # math never crashes on a None value.
        obj_val = tac_true
    # ── 7. Solver Status ──────────────────────────────────────────────────────
    if isinstance(solver_results, dict):
        status_str = str(solver_results.get("status", "UNKNOWN")).upper()
    elif solver_results is not None and hasattr(solver_results, "solver"):
        status_str = str(solver_results.solver.termination_condition).upper()
    else:
        status_str = "UNKNOWN"
    no_incumbent = status_str in ("MAXTIMELIMIT", "TIMELIMIT") and (
    solver_results is None
    or not hasattr(solver_results, "solver")
    or getattr(solver_results.solver, "primal_bound", None) in (None, float("inf"), 1e20))

    if no_incumbent:
        if verbose:
            print("⚠️  Solver hit the time limit with NO feasible solution found.")
            print("    Nothing to extract — try a longer time limit, better heuristics,")
            print("    or check whether a feasible network exists for this problem size.")
        return {
            "edges": [], "util_hex_edges": [], "TAC": None, "TAC_true": None,
            "milp_obj": None, "converged": False, "solver_status": status_str,
            # Renamed from "lmtd_diag"/data.lmtd_diag: Pre_process.py's own
            # per-match diagnostic dict is now h_diag (H tangent-plane
            # validation stats), since lmtd_diag no longer exists there.
            "no_solution_found": True, "h_diag": getattr(data, "h_diag", {}),}
    converged = status_str in ["OPTIMAL", "LOCALLY_SOLVED", "LOCALLYSOLVED"]

    # ── 8. Error Summary Calculation ─────────────────────────────────────────
    if edges:
        max_lmtd_err_pct = max(e["err_lmtd_pct"] for e in edges)
        mean_lmtd_err_pct = float(np.mean([e["err_lmtd_pct"] for e in edges]))

        max_area_err_pct = max(e["err_area_pct"] for e in edges)
        mean_area_err_pct = float(np.mean([e["err_area_pct"] for e in edges]))

        max_cost_err_pct = max(e["err_cost_pct"] for e in edges)
        mean_cost_err_pct = float(np.mean([e["err_cost_pct"] for e in edges]))

        # H tangent-plane OA fit gap (replaces max_sos2_lmtd_err)
        max_H_fit_err_abs = max(abs(e["err_H_fit_abs"]) for e in edges)
        max_H_fit_err_pct = max(e["err_H_fit_pct"] for e in edges)
        mean_H_fit_err_pct = float(np.mean([e["err_H_fit_pct"] for e in edges]))

        # G=Q^beta SOS2 fit gap (replaces max_sos2_cost_err)
        max_G_fit_err_abs = max(abs(e["err_G_fit_abs"]) for e in edges)
        max_G_fit_err_pct = max(e["err_G_fit_pct"] for e in edges)
        mean_G_fit_err_pct = float(np.mean([e["err_G_fit_pct"] for e in edges]))

        # McCormick P vs G*H relaxation gap (replaces max/mean_W_err_*)
        max_P_err_pct = max(e["err_P_pct"] for e in edges)
        mean_P_err_pct = float(np.mean([e["err_P_pct"] for e in edges]))
        max_P_err_abs = max(abs(e["err_P_abs"]) for e in edges)
    else:
        max_lmtd_err_pct = mean_lmtd_err_pct = max_area_err_pct = mean_area_err_pct = 0.0
        max_cost_err_pct = mean_cost_err_pct = 0.0
        max_H_fit_err_abs = max_H_fit_err_pct = mean_H_fit_err_pct = 0.0
        max_G_fit_err_abs = max_G_fit_err_pct = mean_G_fit_err_pct = 0.0
        max_P_err_pct = mean_P_err_pct = max_P_err_abs = 0.0

    # ── 9. Optional Terminal Report Printout ──────────────────────────────────
    if verbose:
        print("\n ── Optimization Solution & Error Report ─────────────────────────────")
        print(f"  Solver Status            : {status_str}")
        print(f"  Linearised MILP Objective: ${obj_val:,.0f}/yr")
        print(f"  True Exact TAC           : ${tac_true:,.0f}/yr")
        print(f"  Total Objective Error    : {abs(obj_val - tac_true) / max(tac_true, 1.0) * 100:.2f}%")
        print("  --------------------------------------------------------------------")
        print(f"  LMTD Error  (Model vs True): Max = {max_lmtd_err_pct:.2f}% | Mean = {mean_lmtd_err_pct:.2f}%")
        print(f"  Area Error  (Model vs True): Max = {max_area_err_pct:.2f}% | Mean = {mean_area_err_pct:.2f}%")
        print(f"  Cost Error  (Model vs True): Max = {max_cost_err_pct:.2f}% | Mean = {mean_cost_err_pct:.2f}%")
        print(f"  H Envelope Fit (OA gap)    : Max = {max_H_fit_err_pct:.2f}% ({max_H_fit_err_abs:.4f}) | Mean = {mean_H_fit_err_pct:.2f}%")
        print(f"  G=Q^beta SOS2 Fit Gap      : Max = {max_G_fit_err_pct:.2f}% ({max_G_fit_err_abs:.4f}) | Mean = {mean_G_fit_err_pct:.2f}%")
        print(f"  P vs G*H (McCormick Gap)   : Max = {max_P_err_pct:.2f}% ({max_P_err_abs:.6f}) | Mean = {mean_P_err_pct:.2f}%")
        print(f"  Active Process Exchangers  : {len(edges)} | Utility Exchangers: {len(util_hex_edges)}")

        if edges:
            print("\n ── Individual Process HEX Error Breakdown ─────────────────────────")
            for e in edges:
                print(f"  HEX ({e['hot_id']} -> {e['cold_id']}, Stage {e['stage']}): Q = {e['Q']} kW")
                print(f"    ├─ LMTD (Model/True): {e['LMTD_model']} / {e['LMTD_true']} K  (Err: {e['err_lmtd_pct']}%)")
                print(f"    ├─ Area (Model/True): {e['Area_m2_model']} / {e['Area_m2_true']} m² (Err: {e['err_area_pct']}%)")
                print(f"    ├─ Cost (Model/True): ${e['Cost_model_$']} / ${e['Cost_true_$']} (Err: {e['err_cost_pct']}%)")
                print(f"    ├─ H (Model/True at model dT): {e['H']} / {e['H_true']} (Err: {e['err_H_fit_pct']}%)")
                print(f"    ├─ G (Model/True at model Q) : {e['G']} / {e['G_true']} (Err: {e['err_G_fit_pct']}%)")
                print(f"    └─ P=A^beta (Model/G×H)      : {e['Abeta']} / {e['P_true_GxH']} (Err: {e['err_P_pct']}%)")

    # ── 10. Complete Combined Dictionary Return ──────────────────────────────
    return {
        # Edge and Utility Lists
        "edges":            edges,
        "util_hex_edges":   util_hex_edges,
        "hex_map":          {(e["hot_id"], e["cold_id"], e["stage"]): e["Q"] for e in edges},

        # Heat Loads & Temperatures
        "QH":               QH_agg,
        "QC":               QC_agg,
        "T_hot":            T_hot.tolist(),
        "T_cold":           T_cold.tolist(),
        "Tout_H":           Tout_H,
        "Tout_C":           Tout_C,
        "split_hot":        split_hot,
        "split_cold":       split_cold,
        "HIDs":             HID,
        "CIDs":             CID,

        # Cost Breakdowns
        "TAC":              round(tac_true, 0),
        "TAC_true":         round(tac_true, 0),
        "ann_util_cost":    round(ann_util, 0),
        "ann_cap_cost":     round(ann_cap, 0),
        "ann_cap_process":  round(ann_cap_process, 0),
        "ann_cap_util_hex": round(ann_cap_util, 0),
        "milp_obj":         round(obj_val, 0),

        # PWL/OA fit diagnostics per (i,j) process match, straight from
        # Pre_process.py's own pre-solve validation (max/avg error, worst
        # point, sign-violation count, etc.) -- kept here so the report can
        # show exactly how good/bad the H tangent-plane envelope was for
        # the matches that ended up active in this network, with zero
        # extra computation (it's already been computed once). Renamed
        # from "lmtd_diag"/data.lmtd_diag -- Pre_process.py's per-match
        # diagnostic dict is now h_diag (see select_H_envelope).
        "h_diag":           getattr(data, "h_diag", {}),

        # Model Parameters
        "cu_hot":           getattr(data, "cu_hot"),
        "cu_cold":          getattr(data, "cu_cold"),
        "U_overall":        U_overall,
        "U_matrix":         data.U_mat.tolist() if data and hasattr(data, "U_mat") and isinstance(data.U_mat, np.ndarray) else getattr(data, "U_mat", None),
        "cost_a":           cost_a,
        "cost_b":           cost_b,
        "cost_beta":        cost_beta,
        "utility_specs":    getattr(data, "utility_specs", None),
        "hot_utils":        getattr(data, "hot_utils", None),
        "cold_utils":       getattr(data, "cold_utils", None),
        "dlog_N_G_process": getattr(data, "N_G_process", 8),

        # Status
        "solver_status":    status_str,
        "converged":        converged,

        # Detailed Error Analysis Summary
        "error_summary": {
            "max_lmtd_err_pct":  round(max_lmtd_err_pct, 2),
            "mean_lmtd_err_pct": round(mean_lmtd_err_pct, 2),
            "max_area_err_pct":  round(max_area_err_pct, 2),
            "mean_area_err_pct": round(mean_area_err_pct, 2),
            "max_cost_err_pct":  round(max_cost_err_pct, 2),
            "mean_cost_err_pct": round(mean_cost_err_pct, 2),
            # H tangent-plane OA fit gap (replaces max_sos2_lmtd_err)
            "max_H_fit_err_abs":  round(max_H_fit_err_abs, 6),
            "max_H_fit_err_pct":  round(max_H_fit_err_pct, 2),
            "mean_H_fit_err_pct": round(mean_H_fit_err_pct, 2),
            # G=Q^beta SOS2 fit gap (replaces max_sos2_cost_err)
            "max_G_fit_err_abs":  round(max_G_fit_err_abs, 4),
            "max_G_fit_err_pct":  round(max_G_fit_err_pct, 2),
            "mean_G_fit_err_pct": round(mean_G_fit_err_pct, 2),
            # McCormick P vs G*H relaxation gap (replaces max/mean_W_err_*)
            # -- how far the model's P=A^beta is from the exact bilinear
            # product at its own G/H values.
            "max_P_err_pct":     round(max_P_err_pct, 2),
            "mean_P_err_pct":    round(mean_P_err_pct, 2),
            "max_P_err_abs":     round(max_P_err_abs, 6),},}


# ═══════════════════════════════════════════════════════════════════════════
# SOLUTION POOL EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════
#
# solve_model(..., collect_solution_pool=True) asks SCIP to export every
# feasible primal solution it finds (not just the final incumbent) as .sol
# files on disk. The functions below scan for those files, replay each
# solution's full variable assignment onto the *same* already-built Pyomo
# model `m`, and re-run extract_hens_results() against it -- so every
# distinct topology SCIP ever visited gets a proper results dict, ready to
# be handed to Core.refine_all_topologies_nlp() for exact-NLP ranking.
#
# Nothing here touches the model's formulation/constraints; it only reads
# and temporarily overwrites Var *values* on the model that was already
# built and solved, then restores the solver's own incumbent afterward.

def _parse_scip_sol_blocks(text):
    """
    Parse the contents of one SCIP .sol (or concatenated multi-solution
    pool) file into a list of {var_name: value} dicts, one per solution.

    Handles both:
      - a single-solution file (one "solution status:"/"objective value:"
        header followed by "name  value" lines), and
      - a concatenated pool file where multiple such blocks appear one
        after another (as written by SCIP's "write allsolutions").

    Lines that aren't a recognizable "<name> <value> [...]" pair (headers,
    blank lines, comments) are skipped rather than raising -- a single
    malformed line shouldn't take down parsing of an otherwise-good file.
    """
    blocks = []
    current = {}
    seen_header = False

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        low = line.lower()
        if low.startswith("solution status"):
            # Start of a new solution block -- flush whatever we had.
            if seen_header and current:
                blocks.append(current)
            current = {}
            seen_header = True
            continue
        if low.startswith("objective value") or low.startswith("no solution") \
                or low.startswith("primal solution") or low.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            val = float(parts[1])
        except ValueError:
            continue
        current[name] = val

    if current:
        blocks.append(current)
    return blocks


def _lookup_sol_value(var_name, var_dict):
    """Look up a Pyomo Var's name in a parsed .sol/pool var_dict, tolerating
    the character-mangling Pyomo's LP writer applies to symbolic labels.

    FIX: this previously only swapped brackets for parentheses (or vice
    versa), which is what an AMPL .sol file needs -- but the in-memory
    solution-pool path (_solve_scip_pool_pyscipopt) writes `m` out to a
    .lp file first, and Pyomo's LP writer mangles indexed names much more
    aggressively than that: "[" -> "(", "]" -> ")", AND every internal
    comma between multi-dimensional index components -> "_". E.g. the
    Pyomo name "Q[0,1,2]" is written to the .lp file -- and therefore
    shows up as a PySCIPOpt variable name, and as a var_dict key here --
    as "Q(0_1_2)", not "Q(0,1,2)". Every multi-index Var in this model
    (TH, TC, Q, QHU, QCU, ...) has 2+ index components, so the old
    bracket/paren-only swap never matched anything on that path: every
    lookup silently returned None, every var loaded on `m` stayed at its
    pre-solve (uninitialized) value, and every downstream `pyo.value(...)`
    call on it raised "No value for uninitialized VarData object ..." --
    exactly the flood of errors and the $0/yr TAC report seen upstream of
    this crash. The LP-mangled form is now tried explicitly, in addition
    to the exact name and the old paren/bracket-swap variants (kept for
    the legacy AMPL .sol-file path, which never mangles commas).
    """
    if var_name in var_dict:
        return var_dict[var_name]

    if "[" in var_name and var_name.endswith("]"):
        head, idx = var_name.split("[", 1)
        idx = idx[:-1]
        mangled = f"{head}({idx.replace(',', '_')})"
        if mangled in var_dict:
            return var_dict[mangled]

    alt = var_name.replace("(", "[").replace(")", "]")
    if alt in var_dict:
        return var_dict[alt]
    alt2 = var_name.replace("[", "(").replace("]", ")")
    return var_dict.get(alt2)


def _snapshot_var_values(m):
    """[(VarData, current value), ...] for every active Var on `m`, so the
    solver's own incumbent can be restored after scratch-loading pool
    members onto the same model object.

    FIX: this previously built a {VarData: value} dict, using each
    VarData object itself as the dict key. That raised
    "TypeError: unhashable type: 'VarData'" the moment this function ran
    (see the traceback: _extract_solutions_from_var_dicts ->
    _snapshot_var_values), aborting the whole solution-pool extraction --
    and, since it's called unconditionally before any pool member is even
    loaded, it broke every run that reached this point regardless of the
    _lookup_sol_value fix above. A dict was never actually needed here:
    the only consumer (the restore loop a few dozen lines below) just
    iterates (vardata, value) pairs, which a plain list supports exactly
    as well while never hashing a VarData at all.
    """
    return [
        (vardata, vardata.value)
        for varobj in m.component_objects(pyo.Var, active=True)
        for vardata in varobj.values()]


def _load_solution_into_model(m, var_dict):
    """
    Overwrite every (non-fixed) Var on `m` whose name is found in
    `var_dict` with that value. Binary/integer vars are rounded to the
    nearest int (SCIP sometimes prints e.g. 0.9999997 / 1.0000002).

    Returns (n_loaded, n_missing).
    """
    n_loaded, n_missing = 0, 0
    for varobj in m.component_objects(pyo.Var, active=True):
        for idx in varobj:
            vardata = varobj[idx]
            if vardata.is_fixed():
                continue
            val = _lookup_sol_value(vardata.name, var_dict)
            if val is None:
                n_missing += 1
                continue
            if vardata.is_binary() or vardata.is_integer():
                val = round(val)
            vardata.set_value(val)
            n_loaded += 1
    return n_loaded, n_missing


def _topology_fingerprint(m, Q_thresh=1.0):
    """
    A hashable fingerprint of which process matches / utility assignments
    are "active" (binary ~1, or -- if no binaries exist for that index --
    duty above Q_thresh) in the model's *current* variable values. Used to
    deduplicate solution-pool members that differ only in continuous
    values (Q, T, A, ...) but describe the exact same discrete topology,
    so Core.refine_all_topologies_nlp never wastes an IPOPT solve on two
    entries that would fix the identical structure.
    """
    active = []

    if hasattr(m, "z"):
        for idx in m.z:
            try:
                if pyo.value(m.z[idx]) > 0.5:
                    key_idx = idx if isinstance(idx, tuple) else (idx,)
                    active.append(("z",) + key_idx)
            except (ValueError, TypeError):
                continue

    for name in ("yHU", "y_HU", "YHU"):
        if hasattr(m, name):
            comp = getattr(m, name)
            for idx in comp:
                try:
                    if pyo.value(comp[idx]) > 0.5:
                        key_idx = idx if isinstance(idx, tuple) else (idx,)
                        active.append(("yHU",) + key_idx)
                except (ValueError, TypeError):
                    continue
            break

    for name in ("yCU", "y_CU", "YCU"):
        if hasattr(m, name):
            comp = getattr(m, name)
            for idx in comp:
                try:
                    if pyo.value(comp[idx]) > 0.5:
                        key_idx = idx if isinstance(idx, tuple) else (idx,)
                        active.append(("yCU",) + key_idx)
                except (ValueError, TypeError):
                    continue
            break

    return frozenset(active)


def _extract_solutions_from_var_dicts(m, data, var_dicts, labels, Q_thresh, dedupe, verbose):
    """
    Shared core of extract_all_scip_solutions: given a list of
    {var_name: value} dicts (one per pool solution) and a same-length list
    of human-readable labels for each, load each onto `m` in turn,
    fingerprint/dedupe, and re-run extract_hens_results(). Restores `m`'s
    original variable values once done, regardless of the source of
    `var_dicts` (in-memory PySCIPOpt pool or parsed .sol files).
    """
    snapshot = _snapshot_var_values(m)
    all_results = []
    seen_topologies = set()

    for var_dict, label in zip(var_dicts, labels):
        if not var_dict:
            continue
        try:
            n_loaded, n_missing = _load_solution_into_model(m, var_dict)
            if n_loaded == 0:
                continue

            topo_key = _topology_fingerprint(m, Q_thresh=Q_thresh)
            if dedupe and topo_key in seen_topologies:
                continue
            seen_topologies.add(topo_key)

            res = extract_hens_results(m, data=data, solver_results=None, verbose=False)
            if res.get("no_solution_found"):
                continue
            res["pool_file"] = label
            res["topology_key"] = topo_key
            all_results.append(res)
        except Exception as exc:
            warnings.warn(f"extract_all_scip_solutions: failed to extract solution "
                           f"'{label}' ({exc}); skipping that one solution.")
            continue

    # Put the model back exactly how solve_model() left it.
    # FIX: `snapshot` is now a list of (vardata, value) pairs, not a dict
    # (see _snapshot_var_values), so this just iterates it directly
    # instead of calling the now-nonexistent `.items()`.
    for vardata, val in snapshot:
        if val is not None:
            vardata.set_value(val)

    return all_results


def extract_all_scip_solutions(m, data=None, pool_dir=None, pattern="*.sol",
                                Q_thresh=1.0, dedupe=True, verbose=True):
    """
    Reconstruct every distinct feasible topology SCIP found while solving
    `m`, and extract a results dict (same shape as extract_hens_results()'s
    return value) for each one.

    Two sources are tried, in order:

      1. `m._scip_solution_pool` -- an in-memory list of {var_name: value}
         dicts set by solve_model(..., collect_solution_pool=True), which
         reads SCIP's own solution storage directly via PySCIPOpt's
         Model.getSols(). This is the real, working mechanism and is used
         whenever it's present (even if empty, meaning PySCIPOpt solved
         but found only the incumbent, or pool collection wasn't
         requested at all).
      2. A legacy fallback: `.sol` files matching `pattern` in `pool_dir`.
         SCIP's actual multi-solution export is an interactive-shell-only
         dialog command ("write allsolutions <file>"), not something the
         ASL-based Pyomo 'scip' plugin can trigger via solver.options, so
         in practice this path will almost always find nothing -- it's
         kept only in case some other SCIP interface/version in your
         environment does drop files there.

    Each pool member is loaded straight onto `m` -- no re-solve, no
    rebuild -- so this is cheap; extract_hens_results() is simply re-run
    against each pool member's variable values in turn. `m`'s own
    incumbent (whatever solve_model() last loaded onto it) is restored
    once every pool member has been processed, so calling this right
    after solve_model() is safe and leaves `m` exactly as it was.

    Parameters
    ----------
    m, data : the same Pyomo model / preprocessed-data namespace already
        passed to extract_hens_results() for the incumbent solution.
    pool_dir, pattern : only used by the legacy file-scan fallback (2).
    Q_thresh : float. Threshold used both to build each topology's
        fingerprint (for dedup) and passed through however
        extract_hens_results() itself defines "active".
    dedupe : bool. If True (default), solution-pool members that resolve
        to the exact same discrete topology (see _topology_fingerprint)
        are only extracted once.
    verbose : bool. Print a one-line summary at the end.

    Returns
    -------
    List[dict] -- one entry per distinct feasible topology found, each
    shaped exactly like extract_hens_results()'s return value, plus:
        "pool_file"    -- which pool solution (index, or .sol filename)
                           this topology came from.
        "topology_key" -- the frozenset fingerprint from
                           _topology_fingerprint(), handy for Core.py to
                           dedupe again downstream if needed.

    Returns an empty list (never raises) if no pool solutions are
    available from either source -- so callers can safely fall back to
    just using the single incumbent result in that case.
    """
    in_memory_pool = getattr(m, "_scip_solution_pool", None)

    if in_memory_pool:
        labels = [f"pool_solution_{idx}" for idx in range(len(in_memory_pool))]
        all_results = _extract_solutions_from_var_dicts(
            m, data, in_memory_pool, labels, Q_thresh, dedupe, verbose)

        if verbose:
            print(f"extract_all_scip_solutions: extracted {len(all_results)} distinct "
                  f"topolog{'y' if len(all_results) == 1 else 'ies'} from "
                  f"{len(in_memory_pool)} in-memory PySCIPOpt pool solution(s).")
        return all_results

    if in_memory_pool is not None:
        # collect_solution_pool=True was used and PySCIPOpt solved
        # successfully, but its pool only ever contained the incumbent
        # (n_sols == 1) -- nothing extra to extract, and there's no point
        # falling through to the file-scan below (there's nothing there
        # either in that code path).
        if verbose:
            print("extract_all_scip_solutions: PySCIPOpt's solution pool contained "
                  "only the incumbent -- no additional topologies to extract.")
        return []

    # Legacy fallback: no in-memory pool at all, meaning
    # collect_solution_pool wasn't used, or PySCIPOpt wasn't available and
    # solve_model() fell back to the plain ASL solve. Scan for .sol files
    # in case some other mechanism dropped them there.
    pool_dir = pool_dir or os.getcwd()
    sol_files = sorted(glob.glob(os.path.join(pool_dir, pattern)))

    if not sol_files:
        if verbose:
            print(f"extract_all_scip_solutions: no in-memory PySCIPOpt pool and no "
                  f"'{pattern}' files found in '{pool_dir}' -- was "
                  f"solve_model(..., collect_solution_pool=True) used, and is "
                  f"PySCIPOpt installed? Falling back to zero extra topologies.")
        return []

    var_dicts, labels = [], []
    for path in sol_files:
        try:
            with open(path, "r") as fh:
                text = fh.read()
        except OSError as exc:
            warnings.warn(f"extract_all_scip_solutions: could not read '{path}' "
                           f"({exc}); skipping.")
            continue

        blocks = _parse_scip_sol_blocks(text)
        multi = len(blocks) > 1
        for b_idx, var_dict in enumerate(blocks):
            var_dicts.append(var_dict)
            labels.append(os.path.basename(path) + (f"#{b_idx}" if multi else ""))

    all_results = _extract_solutions_from_var_dicts(
        m, data, var_dicts, labels, Q_thresh, dedupe, verbose)

    if verbose:
        print(f"extract_all_scip_solutions: extracted {len(all_results)} distinct "
              f"topolog{'y' if len(all_results) == 1 else 'ies'} from {len(sol_files)} "
              f"pool file(s) in '{pool_dir}'.")

    return all_results


def print_error_report(res_dict):
    """Prints a detailed per-HEX and network-wide error analysis report."""
    if res_dict.get("no_solution_found"):
        print("=" * 80)
        print("HENS NETWORK ERROR ANALYSIS REPORT")
        print("=" * 80)
        print("No feasible solution was found — nothing to report.")
        print("(Increase the time limit, tune solver heuristics, or check")
        print(" whether a feasible network exists for this problem size.)")
        return

    edges = res_dict.get("edges", [])
    summary = res_dict.get("error_summary", {})

    print("\n" + "=" * 80)
    print("                      HENS NETWORK ERROR ANALYSIS REPORT              ")
    print("=" * 80)

    # ── 1. Global Objective Error ─────────────────────────────────────────────
    milp_obj = res_dict.get("milp_obj", 0.0)
    tac_true = res_dict.get("TAC_true", 0.0)
    obj_err_abs = milp_obj - tac_true
    obj_err_pct = (
        (abs(obj_err_abs) / tac_true * 100.0) if tac_true > 0 else 0.0)

    print("\n1. OVERALL OBJECTIVE (TAC) DISCREPANCY:")
    print(f"   • Linearized MILP Objective : ${milp_obj:,.2f} / yr")
    print(f"   • True Thermodynamic TAC    : ${tac_true:,.2f} / yr")
    print(
        f"   • Absolute Difference       : ${obj_err_abs:+,.2f} / yr ({obj_err_pct:.2f}%)")

    # ── 2. Per-Exchanger Detailed Breakdown ───────────────────────────────────
    print("\n2. PER-HEX DETAILED ERROR BREAKDOWN:")
    if not edges:
        print("   (No active process heat exchangers in network)")
    else:
        for idx, e in enumerate(edges, 1):
            print(
                f"\n   [{idx}] Exchanger: {e['hot']} -> {e['cold']} (Stage {e['stage']}) | Q = {e['Q']:.2f} kW")
            print("       " + "-" * 70)
            print(
                f"       • LMTD Error : Model = {e['LMTD_model']:7.2f} K  | True = {e['LMTD_true']:7.2f} K  "
                f"| Abs: {e['err_lmtd_abs']:+6.3f} K  | Rel: {e['err_lmtd_pct']:5.2f}%")
            print(
                f"       • Area Error : Model = {e['Area_m2_model']:7.2f} m² | True = {e['Area_m2_true']:7.2f} m² "
                f"| Abs: {e['err_area_abs']:+6.3f} m² | Rel: {e['err_area_pct']:5.2f}%")
            print(
                f"       • Cost Error : Model = ${e['Cost_model_$']:<7,.0f}   | True = ${e['Cost_true_$']:<7,.0f}   "
                f"| Abs: ${e['err_cost_abs']:+6.0f}    | Rel: {e['err_cost_pct']:5.2f}%")
            # FIX: the model was redesigned to use H=(U*LMTD)^-beta (tangent-
            # plane OA envelope) and G=Q^beta (1D SOS2) in place of the old
            # LMTD-grid/Cost-grid DLOG diagnostics, and P=G*H (a McCormick
            # product) in place of the old W=A*LMTD bilinear. extract_hens_
            # results() has produced "err_H_fit_*"/"err_G_fit_*"/"err_P_*" on
            # each edge for a while now -- the old "sos2_lmtd_fit_err",
            # "sos2_cost_fit_err", "W", "W_true_AxLMTD", "err_W_abs", and
            # "err_W_pct" keys this function was still reading no longer
            # exist on the edge dict at all, so every one of these prints
            # raised a KeyError the first time real edges reached this
            # report (previously masked by the solution-loading bug, which
            # meant "edges" was always empty and this code never ran).
            print(
                f"       • H Fit (OA) : Model = {(e.get('H') or 0.0):9.6f} | True = {e.get('H_true', 0.0):9.6f} "
                f"| Abs: {e['err_H_fit_abs']:+9.6f} | Rel: {e['err_H_fit_pct']:5.2f}%")
            print(
                f"       • G Fit (SOS2): Model = {(e.get('G') or 0.0):7.4f}   | True = {e.get('G_true', 0.0):7.4f}   "
                f"| Abs: {e['err_G_fit_abs']:+6.4f}   | Rel: {e['err_G_fit_pct']:5.2f}%")
            print(
                f"       • P vs G×H   : Model = {(e.get('Abeta') or 0.0):9.6f} | G×H = {(e.get('P_true_GxH') or 0.0):9.6f} "
                f"| Abs: {e['err_P_abs']:+9.6f} | Rel: {e['err_P_pct']:5.2f}%")

    # ── 3. Summary Aggregates ────────────────────────────────────────────────
    print("\n3. NETWORK-WIDE ERROR SUMMARY AGGREGATES:")
    print("   Metric              | Max Error  | Mean Error")
    print("   ---------------------------------------------")
    print(f"   LMTD Discrepancy    | {summary.get('max_lmtd_err_pct', 0.0):6.2f}%    | {summary.get('mean_lmtd_err_pct', 0.0):6.2f}%")
    print(f"   Area Discrepancy    | {summary.get('max_area_err_pct', 0.0):6.2f}%    | {summary.get('mean_area_err_pct', 0.0):6.2f}%")
    print(f"   Cost Discrepancy    | {summary.get('max_cost_err_pct', 0.0):6.2f}%    | {summary.get('mean_cost_err_pct', 0.0):6.2f}%")
    print(f"   P vs G×H (McCormick Gap)    | {summary.get('max_P_err_pct', 0.0):6.2f}%    | {summary.get('mean_P_err_pct', 0.0):6.2f}%")
    print("   ---------------------------------------------")
    print(f"   Max H Envelope (OA) Fit Error                : {summary.get('max_H_fit_err_abs', 0.0):.6f} ({summary.get('max_H_fit_err_pct', 0.0):.2f}%, mean {summary.get('mean_H_fit_err_pct', 0.0):.2f}%)")
    print(f"   Max G=Q^beta SOS2 Fit Error                   : {summary.get('max_G_fit_err_abs', 0.0):.4f} ({summary.get('max_G_fit_err_pct', 0.0):.2f}%, mean {summary.get('mean_G_fit_err_pct', 0.0):.2f}%)")
    print(f"   Max McCormick P vs G×H Gap (Abs)              : {summary.get('max_P_err_abs', 0.0):.6f}")
    print("=" * 80 + "\n")


# ═══════════════════════════════════════════════════════════════════════════
# DRIVER
# ═══════════════════════════════════════════════════════════════════════════

def solve_hen_milp(Hsap, Csap, delta_tmin, qh, qc,
                   cu_hot    = 60.0,
                   cu_cold   = 6.0,
                   U_overall = 0.5,
                   U_matrix  = None,
                   cost_a    = 2000.0,
                   cost_b    = 70.0,
                   cost_beta = 1,
                   payback   = 1,
                   hours_per_year = 8600,
                   utility_specs = None,
                   N_G_process = 6,
                   N_G_util = 6,
                   Q_floor_frac = 0.02,
                   solver_name = 'scip',
                   time_limit = 500,
                   gap = 0.01,
                   tee = True,):
    """
    End-to-end driver: preprocess -> build -> solve -> extract -> report.

    FIX: previously called pre_process_milp(...) with a long positional
    argument list ending in `lmtd_grid_pts`, which (by position) actually
    landed on `N_G_process` -- silently correct by luck, but fragile to any
    future change in either signature. Now called with explicit keywords,
    and N_G_util / Q_floor_frac (previously only reachable via
    pre_process_milp's own defaults) are exposed too.
    """
    data = pre_process_milp(
        Hsap, Csap, delta_tmin, qh, qc,
        cu_hot=cu_hot,
        cu_cold=cu_cold,
        U_overall=U_overall,
        U_matrix=U_matrix,
        cost_a=cost_a,
        cost_b=cost_b,
        cost_beta=cost_beta,
        payback=payback,
        hours_per_year=hours_per_year,
        utility_specs=utility_specs,
        N_G_process=N_G_process,
        N_G_util=N_G_util,
        Q_floor_frac=Q_floor_frac,
    )

    print("Building Pyomo HENS model...")
    model = build_model(data)

    # ── Step 2: Solve Model ───────────────────────────────────────────
    print("Starting optimization...")
    results = solve_model(
        model, solver_name=solver_name, time_limit=time_limit, gap=gap, tee=tee)

    # ── Step 3: Check Solver Status ──────────────────────────────────
    status = results.solver.termination_condition
    print(f"\nSolver finished with status: {status}")

    if status in [
        TerminationCondition.optimal,
        TerminationCondition.locallyOptimal,]:
        print("✔ Solution status: Optimal / Locally Solved")
    elif status == TerminationCondition.maxTimeLimit:
        print(
            "⚠️ Solution status: Reached time limit — extracting best feasible point.")
    else:
        print(
            f"⚠️ Solution status: Ended with condition '{status}'. Extracting available values...")

    # ── Step 4: Post-Process Results ──────────────────────────────────
    res_dict = extract_hens_results(model, data=data, solver_results=results)

    # ── Step 5: Print Complete Error Diagnostics ─────────────────────
    print_error_report(res_dict)

    return res_dict


def solve_hen_milp_pool(Hsap, Csap, delta_tmin, qh, qc,
                        cu_hot    = 60.0,
                        cu_cold   = 6.0,
                        U_overall = 0.5,
                        U_matrix  = None,
                        cost_a    = 2000.0,
                        cost_b    = 70.0,
                        cost_beta = 1,
                        payback   = 1,
                        hours_per_year = 8600,
                        utility_specs = None,
                        N_G_process = 6,
                        N_G_util = 6,
                        Q_floor_frac = 0.02,
                        solver_name = 'scip',
                        time_limit = 500,
                        gap = 0.01,
                        tee = True,
                        collect_solution_pool = True,
                        pool_dir = None,
                        pool_size = 50,
                        Q_thresh = 1.0,):

    data = pre_process_milp(
        Hsap, Csap, delta_tmin, qh, qc,
        cu_hot=cu_hot,
        cu_cold=cu_cold,
        U_overall=U_overall,
        U_matrix=U_matrix,
        cost_a=cost_a,
        cost_b=cost_b,
        cost_beta=cost_beta,
        payback=payback,
        hours_per_year=hours_per_year,
        utility_specs=utility_specs,
        N_G_process=N_G_process,
        N_G_util=N_G_util,
        Q_floor_frac=Q_floor_frac,
    )

    if tee:
        print("Building Pyomo HENS model...")
    model = build_model(data)

    if tee:
        print("Starting optimization "
              f"({'with' if collect_solution_pool else 'without'} solution-pool collection)...")
    results = solve_model(
        model, solver_name=solver_name, time_limit=time_limit, gap=gap, tee=tee,
        collect_solution_pool=collect_solution_pool, pool_dir=pool_dir, pool_size=pool_size,)

    status = results.solver.termination_condition
    if tee:
        print(f"\nSolver finished with status: {status}")
        if status in [TerminationCondition.optimal, TerminationCondition.locallyOptimal]:
            print("✔ Solution status: Optimal / Locally Solved")
        elif status == TerminationCondition.maxTimeLimit:
            print("⚠️ Solution status: Reached time limit — extracting best feasible point.")
        else:
            print(f"⚠️ Solution status: Ended with condition '{status}'. Extracting available values...")

    res_dict = extract_hens_results(model, data=data, solver_results=results, verbose=tee)
    if tee:
        print_error_report(res_dict)

    all_candidates = []
    if collect_solution_pool and not res_dict.get("no_solution_found"):
        all_candidates = extract_all_scip_solutions(
            model, data=data, pool_dir=pool_dir, Q_thresh=Q_thresh, verbose=tee)

    # Always make sure the incumbent itself is represented in the pool --
    # whether collect_solution_pool was False, this SCIP build silently
    # ignored 'write/allsolutions', or the incumbent's own topology
    # happened to get deduped out for some reason -- so the caller can
    # always treat `all_candidates` uniformly.
    if not res_dict.get("no_solution_found"):
        incumbent_key = _topology_fingerprint(model, Q_thresh=Q_thresh)
        already_present = any(c.get("topology_key") == incumbent_key for c in all_candidates)
        if not already_present:
            incumbent_entry = dict(res_dict)
            incumbent_entry.setdefault("pool_file", "incumbent")
            incumbent_entry["topology_key"] = incumbent_key
            all_candidates.append(incumbent_entry)

    return res_dict, all_candidates, data, model