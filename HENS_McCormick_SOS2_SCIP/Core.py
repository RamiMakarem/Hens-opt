import pandas as pd
from scipy.optimize import milp, LinearConstraint, Bounds, minimize
import warnings
from types import SimpleNamespace
import numpy as np
import pyomo.environ as pyo
import math

def classify_streams(df):
    df = df.copy()

    df["Tin"] = pd.to_numeric(df["Tin"], errors='coerce')
    df["Tout"] = pd.to_numeric(df["Tout"], errors='coerce')
    df["CP"] = pd.to_numeric(df["CP"], errors='coerce')

    df = df.dropna()

    df["Type"] = df.apply(
        lambda row: "Hot" if row["Tin"] > row["Tout"] else "Cold",
        axis=1
    )

    return df


def adjust_temperatures(df, delta_tmin):
    delta = delta_tmin / 2
    df = df.copy()

    df["Tin_adj"] = df.apply(
        lambda r: r["Tin"] - delta if r["Type"] == "Hot" else r["Tin"] + delta,
        axis=1
    )

    df["Tout_adj"] = df.apply(
        lambda r: r["Tout"] - delta if r["Type"] == "Hot" else r["Tout"] + delta,
        axis=1
    )

    temps = sorted(
        set(df["Tin_adj"]).union(df["Tout_adj"]),
        reverse=True
    )

    intervals = [(temps[i], temps[i+1]) for i in range(len(temps)-1)]

    return df, intervals

def calculate_delta_h(df, intervals):
    results = []

    for high, low in intervals:
        hot_cp = 0
        cold_cp = 0

        hot_streams = []
        cold_streams = []

        for _, s in df.iterrows():
            t_high = max(s["Tin_adj"], s["Tout_adj"])
            t_low = min(s["Tin_adj"], s["Tout_adj"])

            if high <= t_high and low >= t_low:
                if s["Type"] == "Hot":
                    hot_cp += s["CP"]
                    hot_streams.append(s["Stream ID"])
                else:
                    cold_cp += s["CP"]
                    cold_streams.append(s["Stream ID"])

        delta_h = (cold_cp - hot_cp) * (high - low)

        results.append({
            "T_high": high,
            "T_low": low,
            "Hot Streams": hot_streams,
            "Cold Streams": cold_streams,
            "ΔH": delta_h
        })

    return pd.DataFrame(results)

def cascade(delta_h_df):
    cascade = []
    cum = 0

    # Initial row
    cascade.append({"Interval": "Start", "Cum ΔH": 0})

    for _, row in delta_h_df.iterrows():
        cum = cum - row["ΔH"]

        cascade.append({
            "T_high": row["T_high"],
            "T_low": row["T_low"],
            "Interval": f"{row['T_high']} → {row['T_low']}",
            "Cum ΔH": cum
        })

    cascade_df = pd.DataFrame(cascade)

    # Shift so minimum is zero
    min_val = cascade_df["Cum ΔH"].min()
    cascade_df["Adjusted ΔH"] = cascade_df["Cum ΔH"] - min_val

    # Utilities
    qh = cascade_df["Adjusted ΔH"].iloc[0]
    qc = cascade_df["Adjusted ΔH"].iloc[-1]

    # Pinch point (first zero)
    pinch_row = cascade_df.loc[cascade_df["Adjusted ΔH"].idxmin()]
    
    pinch_temp_high = pinch_row["T_high"]
    
    pinch_temp_low = pinch_row["T_low"]

    pinch_temp = pinch_temp_low

    pinch_interval=(pinch_temp_high, pinch_temp_low)

    return cascade_df, qh, qc, pinch_temp, pinch_interval

def stream_energy_table(df):
    df = df.copy()

    df["ΔT"] = abs(df["Tin"] - df["Tout"])
    df["Heat Duty"] = df["CP"] * df["ΔT"]

    return df[["Stream ID", "Type", "CP", "ΔT", "Heat Duty"]]

def _lmtd(dT1: float, dT2: float) -> float:
    """True log-mean temperature difference. Both inputs clamped to ≥1e-3 for B&B stability."""
    dT1 = max(dT1, 1e-3)
    dT2 = max(dT2, 1e-3)
    if abs(dT1 - dT2) < 1e-4:
        return max(dT1,1e-4)
    return (dT1 - dT2) / np.log(dT1 / dT2)

"""
Stage-2 NLP refinement of a solved HENS MILP, using the fixed-topology
Pyomo NLP (build_variables_nlp / build_constraints_nlp / build_objective_nlp)
in place of the old scipy/SLSQP formulation.

No solver binary (ipopt/cbc/glpk) is available in the environment this was
written in, so the actual `solver.solve(m)` call is untested here -- the
model *construction* (sets, variables, constraints, objective, warm start)
is exercised by the smoke test at the bottom of this message, but the
numerical solve itself you'll need to verify in your own environment.
"""
import warnings
from types import SimpleNamespace

import numpy as np
import pyomo.environ as pyo

from Variables_NLP import build_variables_nlp
from Constraints_NLP import build_constraints_nlp, _lmtd_true
from Objective_NLP import build_objective_nlp

# build_active_topology / build_active_utilities / build_warm_start used to
# be duplicated here with slightly weaker guards (bare `e["active"]` /
# `e["hot"]` / `int(e["utility"])` indexing that raises KeyError on a
# malformed/partial edge instead of skipping it, and no fast path for
# reusing the MILP's own `data` namespace). Build_active_topology.py is the
# single source of truth for this extraction now -- importing it here
# instead of keeping a second, drifting copy means every caller (including
# refine_all_topologies_nlp's per-candidate loop over a whole SCIP solution
# pool, where a single malformed edge previously aborted that candidate
# instead of just being skipped) gets the same, more robust behavior.
from Build_active_topology import (
    build_active_topology, build_active_utilities, build_warm_start)


def _idx_set(n):
    """0-based ordered Set, safe for n == 0 (unlike RangeSet(0, -1))."""
    return pyo.Set(initialize=list(range(n)), ordered=True)


def _assemble_data(results, Hsap, Csap, delta_tmin,
                    U_overall, U_matrix,
                    cost_a, cost_b, cost_beta, payback,
                    hours_per_year, Q_THRESH, utility_specs,
                    milp_data=None):
    """
    Build the `data` namespace expected by build_variables_nlp /
    build_constraints_nlp / build_objective_nlp, from the MILP results
    dict plus cost/utility parameters. Split out from refine_hen_nlp so
    it can be unit-tested without needing a solver.
    """
    try:
        (ActiveIJK, I, J, K, S, HID, CID, CP_H, CP_C,
         Tin_H, Tout_H, Tin_C, Tout_C) = build_active_topology(
            results, data=milp_data, Hsap=Hsap, Csap=Csap, Q_thresh=Q_THRESH)
    except AttributeError:
        # `milp_data` was supplied but doesn't carry every attribute
        # (I/J/K/S/HID/CID/CP_H/CP_C/Tin_H/Tout_H/Tin_C/Tout_C)
        # Build_active_topology.py's data-path expects -- fall back to
        # re-deriving everything from Hsap/Csap instead of failing this
        # (and, in the batch case, every other) candidate outright.
        warnings.warn(
            "_assemble_data: milp_data namespace was missing an expected "
            "attribute; falling back to Hsap/Csap re-derivation for this "
            "candidate.")
        (ActiveIJK, I, J, K, S, HID, CID, CP_H, CP_C,
         Tin_H, Tout_H, Tin_C, Tout_C) = build_active_topology(
            results, data=None, Hsap=Hsap, Csap=Csap, Q_thresh=Q_THRESH)

    ActiveHU, ActiveCU = build_active_utilities(results, Q_thresh=Q_THRESH)

    hot_utils = (utility_specs or {}).get("hot_utils") or results.get("hot_utils") or []
    cold_utils = (utility_specs or {}).get("cold_utils") or results.get("cold_utils") or []
    n_HU, n_CU = len(hot_utils), len(cold_utils)

    if not ActiveIJK and not ActiveHU and not ActiveCU:
        raise ValueError(
            "refine_hen_nlp: no active exchangers or utility duties found "
            f"above Q_THRESH={Q_THRESH}. Nothing to refine."
        )

    Q_H_total = [CP_H[i] * (Tin_H[i] - Tout_H[i]) for i in range(I)]
    Q_C_total = [CP_C[j] * (Tout_C[j] - Tin_C[j]) for j in range(J)]

    # Gamma must bound approach temperatures for utility exchangers too,
    # not just process-process ones -- utilities (e.g. steam, refrigerant)
    # routinely sit outside the process streams' own Tin/Tout range.
    hot_side_temps = list(Tin_H) + list(Tout_H) + [
        hu[k] for hu in hot_utils for k in ("T_supply", "T_return") if k in hu]
    cold_side_temps = list(Tin_C) + list(Tout_C) + [
        cu[k] for cu in cold_utils for k in ("T_supply", "T_return") if k in cu]
    Gamma = max(
        (max(hot_side_temps) - min(cold_side_temps)) if hot_side_temps and cold_side_temps else 0,
        1.0,)

    Um = np.asarray(U_matrix) if U_matrix is not None else None

    def U_process(i, j):
        if Um is not None and i < Um.shape[0] and j < Um.shape[1]:
            return float(Um[i, j])
        return U_overall

    def U_hu(u, j):
        # scipy refine_hen_nlp: row = I + n_CU + u, col = J + u
        if Um is not None:
            r, c = I + n_CU + u, J + u
            if r < Um.shape[0] and c < Um.shape[1]:
                return float(Um[r, c])
        return U_overall

    def U_cu(v, i):
        # ASSUME: row = I + v, col = J + n_HU + v
        if Um is not None:
            r, c = I + v, J + n_HU + v
            if r < Um.shape[0] and c < Um.shape[1]:
                return float(Um[r, c])
        return U_overall

    cost_fixed_g = cost_a / payback
    cost_coeff_g = cost_b / payback
    cost_exp_g = cost_beta

    active_ij = sorted({(i, j) for (i, j, k) in ActiveIJK})

    data = SimpleNamespace()
    data.K = K
    data.Gamma = Gamma
    data.delta_tmin = delta_tmin
    data.cost_a = cost_fixed_g  # per-exchanger fixed annualized capex, used in build_objective_nlp

    data.CP_H, data.CP_C = CP_H, CP_C
    data.Tin_H, data.Tout_H = Tin_H, Tout_H
    data.Tin_C, data.Tout_C = Tin_C, Tout_C
    data.Q_H_total, data.Q_C_total = Q_H_total, Q_C_total

    data.ActiveIJK, data.ActiveHU, data.ActiveCU = ActiveIJK, ActiveHU, ActiveCU

    # Bounds only -- don't need to be tight since the active set is already
    # fixed from the MILP solution; a safe upper bound is all that matters.
    data.Q_match_max = {(i, j): max(min(Q_H_total[i], Q_C_total[j]), 1e-3) for (i, j) in active_ij}
    data.A_max = {(i, j): 1e7 for (i, j) in active_ij}
    data.U = {(i, j): U_process(i, j) for (i, j) in active_ij}
    data.cost_fixed = {(i, j): cost_fixed_g for (i, j) in active_ij}
    data.cost_coeff = {(i, j): cost_coeff_g for (i, j) in active_ij}
    data.cost_exp = {(i, j): cost_exp_g for (i, j) in active_ij}
    data.dT1_hi = {(i, j): Gamma for (i, j) in active_ij}
    data.dT2_hi = {(i, j): Gamma for (i, j) in active_ij}

    data.T_HU_supply = {u: hot_utils[u]["T_supply"] for u in range(n_HU)}
    data.T_HU_return = {u: hot_utils[u]["T_return"] for u in range(n_HU)}
    data.hot_Q_per_kg = {u: hot_utils[u]["Q_per_kg"] for u in range(n_HU)}
    data.hot_max_flow = {u: hot_utils[u].get("max_flow", float("inf")) for u in range(n_HU)}
    data.hot_is_combined = {u: hot_utils[u].get("is_combined", False) for u in range(n_HU)}
    data.hot_T_phase = {u: hot_utils[u].get("T_phase", None) for u in range(n_HU)}
    data.hot_cp_vap = {u: hot_utils[u].get("cp_vap", 0.0) for u in range(n_HU)}
    data.hu_opex = {u: hot_utils[u].get("cost_per_kw", 0.0) for u in range(n_HU)}

    data.T_CU_supply = {v: cold_utils[v]["T_supply"] for v in range(n_CU)}
    data.T_CU_return = {v: cold_utils[v]["T_return"] for v in range(n_CU)}
    data.cold_Q_per_kg = {v: cold_utils[v]["Q_per_kg"] for v in range(n_CU)}
    data.cold_max_flow = {v: cold_utils[v].get("max_flow", float("inf")) for v in range(n_CU)}
    data.cold_is_combined = {v: cold_utils[v].get("is_combined", False) for v in range(n_CU)}
    data.cold_T_phase = {v: cold_utils[v].get("T_phase", None) for v in range(n_CU)}
    data.cold_cp_liq = {v: cold_utils[v].get("cp_liq", 0.0) for v in range(n_CU)}
    data.cu_opex = {v: cold_utils[v].get("cost_per_kw", 0.0) for v in range(n_CU)}

    data.A_max_hu = {(u, j): 1e7 for (u, j) in ActiveHU}
    data.A_max_cu = {(v, i): 1e7 for (v, i) in ActiveCU}
    data.U_hu = {(u, j): U_hu(u, j) for (u, j) in ActiveHU}
    data.U_cu = {(v, i): U_cu(v, i) for (v, i) in ActiveCU}
    data.cost_fixed_hu = {(u, j): cost_fixed_g for (u, j) in ActiveHU}
    data.cost_coeff_hu = {(u, j): cost_coeff_g for (u, j) in ActiveHU}
    data.cost_exp_hu = {(u, j): cost_exp_g for (u, j) in ActiveHU}
    data.cost_fixed_cu = {(v, i): cost_fixed_g for (v, i) in ActiveCU}
    data.cost_coeff_cu = {(v, i): cost_coeff_g for (v, i) in ActiveCU}
    data.cost_exp_cu = {(v, i): cost_exp_g for (v, i) in ActiveCU}

    data.M_dT2_HU = {(u, j): Gamma for (u, j) in ActiveHU}
    data.M_dT1_CU = {(v, i): Gamma for (v, i) in ActiveCU}
    data.dT1_HU = {(u, j): data.T_HU_supply[u] - data.Tout_C[j] for (u, j) in ActiveHU}
    data.dT2_CU = {(v, i): data.Tout_H[i] - data.T_CU_supply[v] for (v, i) in ActiveCU}
    #data.M_dT1_HU = {(u, j): Gamma for (u, j) in ActiveHU}
    #data.M_dT2_CU = {(v, i): Gamma for (v, i) in ActiveCU}

    meta = dict(
        I=I, J=J, K=K, S=S, HID=HID, CID=CID,
        n_HU=n_HU, n_CU=n_CU, hot_utils=hot_utils, cold_utils=cold_utils,
        Q_THRESH=Q_THRESH,
    )
    return data, meta


def _build_model(data, meta):
    m = pyo.ConcreteModel()
    m.Hi = _idx_set(meta["I"])
    m.Hj = _idx_set(meta["J"])
    m.Hs = _idx_set(meta["S"])
    m.Knodes = _idx_set(meta["K"])
    m.HU = _idx_set(meta["n_HU"])
    m.CU = _idx_set(meta["n_CU"])

    build_variables_nlp(m, data)
    build_constraints_nlp(m, data)
    build_objective_nlp(m, data)
    return m


def _warm_start(m, results, data, meta):
    """Seed the NLP from the MILP's own solution so SLSQP/IPOPT starts
    close to the optimum instead of at variable-bound defaults.

    Q/LMTD/Area/(variable) Cost are seeded from
    Build_active_topology.build_warm_start()'s error-free values wherever
    it has one for a given (i,j,k)/(u,j)/(v,i) key -- those come straight
    from the MILP's own true LMTD_true/Area_m2_true (process matches) or
    exact closed-form utility dT1/dT2 (utilities), not the DLOG/SOS2-fitted
    model variables, and Q_init in particular is the MILP's *actual* duty
    rather than an upper-bound placeholder. Any (i,j,k)/(u,j)/(v,i) key
    build_warm_start doesn't have a value for falls back to the original
    TH/TC-derived recompute below (e.g. if `results` came from a source
    that doesn't carry "LMTD_true"/"Area_m2_true" on its edges).
    """
    bw = build_warm_start(results, Q_thresh=meta.get("Q_THRESH", 1.0))
    T_hot_milp = results.get("T_hot")
    T_cold_milp = results.get("T_cold")
    I, J, K = meta["I"], meta["J"], meta["K"]
    n_HU, n_CU = meta["n_HU"], meta["n_CU"]

    if T_hot_milp is not None:
        for i in range(I):
            for k in range(K):
                m.TH[i, k].set_value(T_hot_milp[i][k])
    if T_cold_milp is not None:
        for j in range(J):
            for k in range(K):
                m.TC[j, k].set_value(T_cold_milp[j][k])

    for (i, j, k) in data.ActiveIJK:
        q0 = bw["Q_init"].get((i, j, k), data.Q_match_max[i, j])
        m.Q[i, j, k].set_value(q0)
        dt1 = max(pyo.value(m.TH[i, k]) - pyo.value(m.TC[j, k]), data.delta_tmin)
        dt2 = max(pyo.value(m.TH[i, k + 1]) - pyo.value(m.TC[j, k + 1]), data.delta_tmin)
        m.dT1[i, j, k].set_value(dt1)
        m.dT2[i, j, k].set_value(dt2)
        lmtd0 = bw["LMTD_init"].get((i, j, k)) or max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv[i, j, k].set_value(lmtd0)
        a0 = bw["Area_init"].get((i, j, k)) or (q0 / (data.U[i, j] * lmtd0))
        m.A[i, j, k].set_value(a0)
        cost_var0 = bw["Cost_init"].get((i, j, k))
        m.Cost[i, j, k].set_value(
            data.cost_fixed[i, j] + (cost_var0 if cost_var0 is not None
                                      else data.cost_coeff[i, j] * a0 ** data.cost_exp[i, j]))

    QH_milp = results.get("QH")
    QC_milp = results.get("QC")

    if QH_milp is not None:
        for j in range(J):
            m.QH[j].set_value(QH_milp[j])
    if QC_milp is not None:
        for i in range(I):
            m.QC[i].set_value(QC_milp[i])

    for (u, j) in data.ActiveHU:
        q0 = bw["QHU_init"].get((u, j), pyo.value(m.QH[j]))
        m.QHU[u, j].set_value(q0)
        dt1 = max(data.T_HU_supply[u] - data.Tout_C[j], data.delta_tmin)
        dt2 = max(data.T_HU_return[u] - pyo.value(m.TC[j, 0]), data.delta_tmin)
        m.dT2_HU[u, j].set_value(dt2)
        lmtd0 = max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv_HU[u, j].set_value(lmtd0)
        a0 = q0 / (data.U_hu[u, j] * lmtd0)
        m.A_HU[u, j].set_value(a0)
        cost_var0 = bw["CostHU_init"].get((u, j))
        m.Cost_HU[u, j].set_value(
            data.cost_fixed_hu[u, j] + (cost_var0 if cost_var0 is not None
                                         else data.cost_coeff_hu[u, j] * a0 ** data.cost_exp_hu[u, j]))

    for (v, i) in data.ActiveCU:
        q0 = bw["QCU_init"].get((v, i), pyo.value(m.QC[i]))
        m.QCU[v, i].set_value(q0)
        dt1 = max(pyo.value(m.TH[i, K - 1]) - data.T_CU_return[v], data.delta_tmin)
        dt2 = max(data.Tout_H[i] - data.T_CU_supply[v], data.delta_tmin)
        m.dT1_CU[v, i].set_value(dt1)
        lmtd0 = max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv_CU[v, i].set_value(lmtd0)
        a0 = q0 / (data.U_cu[v, i] * lmtd0)
        m.A_CU[v, i].set_value(a0)
        cost_var0 = bw["CostCU_init"].get((v, i))
        m.Cost_CU[v, i].set_value(
            data.cost_fixed_cu[v, i] + (cost_var0 if cost_var0 is not None
                                         else data.cost_coeff_cu[v, i] * a0 ** data.cost_exp_cu[v, i]))


def refine_hen_nlp(results, Hsap, Csap, delta_tmin,
                    cu_hot=80.0,
                    cu_cold=20.0,
                    U_overall=0.5,
                    U_matrix=None,
                    cost_a=32000.0,
                    cost_b=70.0,
                    cost_beta=0.6,
                    payback=1,
                    hours_per_year=8600,
                    Q_THRESH=1.0,
                    maxiter=300,
                    utility_specs=None,
                    milp_data=None):
    """
    Stage-2 NLP refinement of a solved HENS MILP: fixes the topology from
    `results` (z/yHU/yCU > 0.5, filtered by Q_THRESH), then solves the
    continuous Pyomo NLP (exact/smooth LMTD + power-law cost, no SOS2/
    binaries) to get the true optimal duties, temperatures, areas, and
    cost for that fixed structure.

    milp_data : optional SimpleNamespace, the `data` object returned by
        Pre_process.pre_process_milp for the MILP `results` was solved
        from. When supplied, Build_active_topology.build_active_topology
        pulls I/J/K/S and the stream property lists directly from it
        instead of re-deriving them from Hsap/Csap -- guaranteeing they
        exactly match what the MILP was actually built with, which
        matters most when this is called many times in a loop (e.g. from
        refine_all_topologies_nlp over a whole SCIP solution pool) with
        the same Hsap/Csap passed alongside many different `results`.

    Returns a dict shaped like `results`, updated in place (via
    dict(results)) with refined "edges", "util_hex_edges", "QH", "QC",
    "T_hot", "T_cold", "TAC", and cost breakdowns -- mirroring the old
    scipy-based refine_hen_nlp's output shape.
    """
    data, meta = _assemble_data(
        results, Hsap, Csap, delta_tmin,
        U_overall, U_matrix,
        cost_a, cost_b, cost_beta, payback,
        hours_per_year, Q_THRESH, utility_specs,
        milp_data=milp_data,)
    I, J, K = meta["I"], meta["J"], meta["K"]
    HID, CID = meta["HID"], meta["CID"]
    hot_utils, cold_utils = meta["hot_utils"], meta["cold_utils"]

    print(f"\n  Stage-2 NLP refinement: {len(data.ActiveIJK)} fixed exchangers, "
          f"{len(data.ActiveHU)} hot-utility + {len(data.ActiveCU)} cold-utility assignments.")
    print("  Building Pyomo NLP (exact LMTD, power-law cost) ...")

    m = _build_model(data, meta)
    _warm_start(m, results, data, meta)

    m.write("hen_nlp_debug.nl", io_options={"symbolic_solver_labels": True})
    print("  Model successfully written to hen_nlp_debug.nl")

    print("  Solving with IPOPT ...")
    solver = pyo.SolverFactory("ipopt")

    if not solver.available(exception_flag=False):
        raise RuntimeError(
            "IPOPT executable was not found. "
            "Make sure IPOPT is installed and C:\\Ipopt\\bin is on PATH."
        )

    solver.options["max_iter"] = maxiter
    solver.options["tol"] = 1e-8

    solve_result = solver.solve(
        m,
        tee=True,
        load_solutions=False,
    )

    term_cond = solve_result.solver.termination_condition
    solver_status = solve_result.solver.status

    # Pyomo's own `solver.solve(..., load_solutions=True)` (the default)
    # calls model.solutions.load_from(...) internally and raises an opaque,
    # low-level ValueError ("Cannot load a SolverResults object with bad
    # status: error") when the solver crashed or produced nothing usable --
    # e.g. IPOPT not fully installed (missing linear-solver DLLs on
    # Windows), or the solver process dying outright. That's a solver-level
    # failure, not a modelling infeasibility, so it's caught here and
    # re-raised as a clear, actionable error instead of the cryptic one.
    loadable = (
        solver_status in (pyo.SolverStatus.ok, pyo.SolverStatus.warning)
        and term_cond not in (pyo.TerminationCondition.error, pyo.TerminationCondition.other)
    )
    if not loadable:
        raise RuntimeError(
            "refine_hen_nlp: the solver failed before producing a usable solution "
            f"(solver_status='{solver_status}', termination_condition='{term_cond}'). "
            "This is a solver-level failure (e.g. IPOPT crashed or is not fully "
            "installed), not a modelling infeasibility -- check the solver output "
            "printed above (tee=True) and the 'hen_nlp_debug.nl' file written next "
            "to this script. On Windows, a common cause is IPOPT missing its "
            "linear-solver DLLs (MUMPS/HSL) on PATH even though the ipopt.exe "
            "itself was found."
        )

    try:
        m.solutions.load_from(solve_result)
    except Exception as exc:
        raise RuntimeError(
            f"refine_hen_nlp: solver reported status='{solver_status}' / "
            f"termination_condition='{term_cond}' but no solution could be loaded "
            f"into the model ({exc}). Check the solver output printed above."
        ) from exc

    nlp_ok = term_cond in (
        pyo.TerminationCondition.optimal,
        pyo.TerminationCondition.locallyOptimal,
    )

    if not nlp_ok:
        warnings.warn(
            f"refine_hen_nlp: IPOPT did not converge "
            f"(termination_condition={term_cond}, "
            f"solver_status={solve_result.solver.status}). "
            f"Returning best iterate found."
        )

    tac_before = results.get("TAC")

    # ── Extract refined solution ──────────────────────────────────────
    T_hot_ref = [[pyo.value(m.TH[i, k]) for k in range(K)] for i in range(I)]
    T_cold_ref = [[pyo.value(m.TC[j, k]) for k in range(K)] for j in range(J)]
    QH_ref = [pyo.value(m.QH[j]) for j in range(J)]
    QC_ref = [pyo.value(m.QC[i]) for i in range(I)]

    edges_ref = []
    for (i, j, k) in data.ActiveIJK:
        Q = pyo.value(m.Q[i, j, k])
        if Q <= Q_THRESH:
            continue
        dT1_v = pyo.value(m.dT1[i, j, k])
        dT2_v = pyo.value(m.dT2[i, j, k])
        lmtd_model = pyo.value(m.LMTDv[i, j, k])
        lmtd_true = _lmtd_true(dT1_v, dT2_v)
        area_model = pyo.value(m.A[i, j, k])
        area_true = Q / (data.U[i, j] * max(lmtd_true, 1e-4))
        cost_model = pyo.value(m.Cost[i, j, k])
        cost_true = data.cost_fixed[i, j] + data.cost_coeff[i, j] * area_true ** data.cost_exp[i, j]
        edges_ref.append({
            "hot": HID[i],
            "cold": CID[j],
            "stage": k + 1,
            "Q": round(Q, 4),
            "LMTD": round(lmtd_model, 2),
            "Area_m2": round(area_model, 2),
            "CapCost_$": round(cost_model, 0),
            "LMTD_true": round(lmtd_true, 2),
            "Area_m2_true": round(area_true, 2),
            "Cost_true_$": round(cost_true, 0),
        })

    util_hex_edges = []
    for (u, j) in data.ActiveHU:
        Q_uj = pyo.value(m.QHU[u, j])
        if Q_uj <= Q_THRESH:
            continue
        hu = hot_utils[u]
        Q_pk = hu["Q_per_kg"]
        mdot = Q_uj / Q_pk if Q_pk > 0 else None
        lmtd_true_hu = _lmtd_true(data.dT1_HU[u, j], pyo.value(m.dT2_HU[u, j]))
        area_true_hu = Q_uj / (data.U_hu[u, j] * max(lmtd_true_hu, 1e-4))
        cost_true_hu = data.cost_fixed_hu[u, j] + data.cost_coeff_hu[u, j] * area_true_hu ** data.cost_exp_hu[u, j]
        util_hex_edges.append({
            "utility": hu.get("uid", u),
            "hot": hu.get("uid", u),
            "cold": CID[j],
            "side": "hot_util",
            "Q": round(Q_uj, 4),
            "mdot_kg_s": round(mdot, 4) if mdot else None,
            "LMTD": round(pyo.value(m.LMTDv_HU[u, j]), 2),
            "Area_m2": round(pyo.value(m.A_HU[u, j]), 2),
            "CapCost_$": round(pyo.value(m.Cost_HU[u, j]), 0),
            "LMTD_true": round(lmtd_true_hu, 2),
            "Area_m2_true": round(area_true_hu, 2),
            "Cost_true_$": round(cost_true_hu, 0),
        })

    for (v, i) in data.ActiveCU:
        Q_vi = pyo.value(m.QCU[v, i])
        if Q_vi <= Q_THRESH:
            continue
        cu = cold_utils[v]
        Q_pk = cu["Q_per_kg"]
        mdot = Q_vi / Q_pk if Q_pk > 0 else None
        lmtd_true_cu = _lmtd_true(pyo.value(m.dT1_CU[v, i]), data.dT2_CU[v, i])
        area_true_cu = Q_vi / (data.U_cu[v, i] * max(lmtd_true_cu, 1e-4))
        cost_true_cu = data.cost_fixed_cu[v, i] + data.cost_coeff_cu[v, i] * area_true_cu ** data.cost_exp_cu[v, i]
        util_hex_edges.append({
            "utility": cu.get("uid", v),
            "hot": HID[i],
            "cold": cu.get("uid", v),
            "side": "cold_util",
            "Q": round(Q_vi, 4),
            "mdot_kg_s": round(mdot, 4) if mdot else None,
            "LMTD": round(pyo.value(m.LMTDv_CU[v, i]), 2),
            "Area_m2": round(pyo.value(m.A_CU[v, i]), 2),
            "CapCost_$": round(pyo.value(m.Cost_CU[v, i]), 0),
            "LMTD_true": round(lmtd_true_cu, 2),
            "Area_m2_true": round(area_true_cu, 2),
            "Cost_true_$": round(cost_true_cu, 0),
        })

    ann_util_ref = (
        sum(data.hu_opex[u] * pyo.value(m.QHU[u, j]) for (u, j) in data.ActiveHU)
        + sum(data.cu_opex[v] * pyo.value(m.QCU[v, i]) for (v, i) in data.ActiveCU))
    ann_cap_process = sum(e["CapCost_$"] for e in edges_ref)
    ann_cap_util = sum(e["CapCost_$"] for e in util_hex_edges)
    ann_cap_ref = ann_cap_process + ann_cap_util
    tac_ref = ann_util_ref + ann_cap_ref

    print("  ── NLP Refinement Result ─────────────────────────────────────")
    if tac_before is not None:
        print(f"  TAC before NLP (MILP stage): ${tac_before:,.0f}/yr")
    print(f"  TAC after  NLP (exact LMTD): ${tac_ref:,.0f}/yr")
    if tac_before:
        improvement = (tac_before - tac_ref) / tac_before * 100
        print(f"  Improvement: {improvement:+.2f}%")
    print(f"  Solver status: {term_cond}")

    out = dict(results)
    out.update({
        "edges": edges_ref,
        "util_hex_edges": util_hex_edges,
        "QH": QH_ref,
        "QC": QC_ref,
        "T_hot": T_hot_ref,
        "T_cold": T_cold_ref,
        "hex_map": {(e["hot"], e["cold"], e["stage"]): e["Q"] for e in edges_ref},
        "TAC": round(tac_ref, 0),
        "ann_util_cost": round(ann_util_ref, 0),
        "ann_cap_cost": round(ann_cap_ref, 0),
        "ann_cap_process": round(ann_cap_process, 0),
        "ann_cap_util_hex": round(ann_cap_util, 0),
        "tac_before_nlp": tac_before,
        "nlp_status": str(term_cond),
        "nlp_success": nlp_ok,
        "nlp_message": str(term_cond),
        "utility_specs": utility_specs,
        "hot_utils": hot_utils,
        "cold_utils": cold_utils,
        "U_matrix": np.asarray(U_matrix).tolist() if U_matrix is not None else None,
    })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# BATCH NLP EVALUATION OF A SOLUTION POOL
# ═══════════════════════════════════════════════════════════════════════════
#
# Companion to Solve_extract.extract_all_scip_solutions(): takes the list
# of candidate MILP-solution dicts it returns (one per distinct topology
# SCIP's solution pool visited) and refines EVERY one of them through the
# exact-equation Pyomo NLP (build_active_topology stays untouched -- this
# just calls refine_hen_nlp once per candidate), so the true minimum-TAC
# topology can be picked from among several discrete alternatives instead
# of trusting whichever one SCIP happened to report as MILP-optimal.

def refine_all_topologies_nlp(all_results, Hsap, Csap, delta_tmin,
                               cu_hot=80.0,
                               cu_cold=20.0,
                               U_overall=0.5,
                               U_matrix=None,
                               cost_a=32000.0,
                               cost_b=70.0,
                               cost_beta=0.6,
                               payback=1,
                               hours_per_year=8600,
                               Q_THRESH=1.0,
                               maxiter=300,
                               utility_specs=None,
                               milp_data=None,
                               verbose=True):
    """
    Batch NLP evaluation of every candidate topology in `all_results`
    (as produced by Solve_extract.extract_all_scip_solutions).

    For each candidate MILP-solution dict:
      1. Its topology is fixed and the exact-equation Pyomo NLP is built
         and warm-started from that candidate's own MILP values, via the
         existing `refine_hen_nlp` (no change to its model formulation or
         to build_active_topology's constraint logic -- this function
         only orchestrates calling it once per candidate).
      2. It's solved with IPOPT and the resulting true continuous TAC is
         recorded on success.
      3. Any failure for that ONE topology -- IPOPT missing/crashing,
         `refine_hen_nlp` raising because IPOPT didn't converge to a
         usable solution, a degenerate candidate with no active
         exchangers or utility duties at all, etc. -- is caught and
         logged against that candidate only; it never aborts the batch.

    milp_data : optional SimpleNamespace, the `data` object returned by
        Pre_process.pre_process_milp for the MILP `all_results` came from
        (e.g. Solve_extract.solve_hen_milp_pool's 4th return value). Passed
        straight through to every refine_hen_nlp call so every candidate's
        I/J/K/S/stream data is pulled from that single source of truth
        instead of being re-derived from Hsap/Csap on every iteration --
        see refine_hen_nlp's own docstring for why that matters more here
        than in the single-topology case.

    Parameters mirror `refine_hen_nlp`'s cost/utility/solver arguments and
    are applied identically to every candidate in the pool.

    Returns
    -------
    ranked : list of dicts, one per candidate, SORTED with every
        successfully-refined topology first (ascending true TAC, cheapest
        first, each given a 1-based "rank"), followed by any topologies
        that failed to refine ("rank": None, "TAC": None, "error" set).
        Each entry:
            {
              "rank":            1-based rank among successes, or None,
              "candidate_index": index into the input `all_results` list,
              "topology_key":    the fingerprint from extract_all_scip_solutions
                                  (or None if the candidate didn't carry one),
              "pool_file":       which pool .sol file this candidate came
                                  from, if available,
              "TAC_milp":        that candidate's own MILP-stage TAC_true
                                  estimate (for comparison),
              "success":         bool,
              "TAC":             true refined TAC ($/yr), or None on failure,
              "results":         the full refine_hen_nlp() output dict for
                                  this candidate, or None on failure,
              "error":           None on success, else "ExcType: message",
            }
    best : the rank-1 entry of `ranked` (lowest true TAC), or None if
        every candidate failed to refine.
    """
    if not all_results:
        if verbose:
            print("refine_all_topologies_nlp: nothing to refine (empty candidate list).")
        return [], None

    summary = []

    for idx, milp_res in enumerate(all_results):
        tac_milp = milp_res.get("TAC_true", milp_res.get("TAC"))

        if verbose:
            tac_str = f"${tac_milp:,.0f}/yr" if tac_milp else "n/a"
            print(f"\n[{idx + 1}/{len(all_results)}] Refining topology "
                  f"(pool_file={milp_res.get('pool_file', '?')}, "
                  f"MILP-stage TAC ≈ {tac_str}) ...")

        entry = {
            "candidate_index": idx,
            "topology_key": milp_res.get("topology_key"),
            "pool_file": milp_res.get("pool_file"),
            "TAC_milp": tac_milp,
            "success": False,
            "TAC": None,
            "results": None,
            "error": None,
        }

        try:
            refined = refine_hen_nlp(
                milp_res, Hsap, Csap, delta_tmin,
                cu_hot=cu_hot, cu_cold=cu_cold,
                U_overall=U_overall, U_matrix=U_matrix,
                cost_a=cost_a, cost_b=cost_b, cost_beta=cost_beta,
                payback=payback, hours_per_year=hours_per_year,
                Q_THRESH=Q_THRESH, maxiter=maxiter,
                utility_specs=utility_specs, milp_data=milp_data,)
            entry["success"] = True
            entry["TAC"] = refined.get("TAC")
            entry["results"] = refined
        except Exception as exc:
            # Catches, per candidate: refine_hen_nlp's own RuntimeErrors
            # (IPOPT not found / IPOPT crashed before producing a usable
            # solution), ValueError from a degenerate topology with no
            # active exchangers or utility duties, or anything else an
            # individual candidate's NLP build/solve could throw. One bad
            # topology in the pool never kills the rest of the batch.
            entry["error"] = f"{type(exc).__name__}: {exc}"
            if verbose:
                print(f"    ✗ Failed: {entry['error']}")

        summary.append(entry)

    successes = [e for e in summary if e["success"]]
    failures = [e for e in summary if not e["success"]]
    successes.sort(key=lambda e: e["TAC"])

    for rank, e in enumerate(successes, start=1):
        e["rank"] = rank
    for e in failures:
        e["rank"] = None

    ranked = successes + failures
    best = successes[0] if successes else None

    if verbose:
        print("\n" + "=" * 72)
        print(f"BATCH NLP REFINEMENT SUMMARY: {len(successes)}/{len(summary)} "
              f"topologies solved successfully")
        print("=" * 72)
        for e in successes:
            print(f"  #{e['rank']:>2}  TAC = ${e['TAC']:,.0f}/yr   "
                  f"(MILP est. ${e['TAC_milp']:,.0f}/yr)   "
                  f"pool_file={e['pool_file']}")
        if failures:
            print(f"  ({len(failures)} topology(ies) failed to refine -- "
                  f"see each entry's 'error' field)")
        if best is not None:
            print(f"\n  \u2605 BEST: TAC = ${best['TAC']:,.0f}/yr "
                  f"(candidate #{best['candidate_index']}, "
                  f"pool_file={best['pool_file']})")
        else:
            print("\n  No candidate topology refined successfully.")
        print("=" * 72)

    return ranked, best