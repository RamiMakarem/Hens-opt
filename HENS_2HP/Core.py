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

def build_active_topology(results, Hsap, Csap, Q_thresh=1.0):

    I = len(Hsap)
    J = len(Csap)
    K = I + J - 1
    S = K - 1

    HID    = [r[0] for r in Hsap];   CID    = [r[0] for r in Csap]
    CP_H   = [r[3] for r in Hsap];   CP_C   = [r[3] for r in Csap]
    Tin_H  = [r[1] for r in Hsap];   Tin_C  = [r[1] for r in Csap]
    Tout_H = [r[2] for r in Hsap];   Tout_C = [r[2] for r in Csap]

    hid_to_i = {hid: i for i, hid in enumerate(HID)}
    cid_to_j = {cid: j for j, cid in enumerate(CID)}

    ActiveIJK = []
    for e in results.get("edges", []):
        if not e["active"]:
            continue
        if e.get("Q", 0.0) <= Q_thresh:
            continue
        i = hid_to_i.get(e["hot"])
        j = cid_to_j.get(e["cold"])
        if i is None or j is None:
            continue
        k = int(e["stage"]) - 1   # "stage" is 1-based (s_idx + 1); NLP uses 0-based
        ActiveIJK.append((i, j, k))

    return (ActiveIJK, I, J, K, S, HID, CID, CP_H, CP_C,
            Tin_H, Tout_H, Tin_C, Tout_C)


def build_active_utilities(results, Q_thresh=1.0):

    ActiveHU = []
    ActiveCU = []
    for e in results.get("util_hex_edges", []):
        if e.get("Q", 0.0) <= Q_thresh:
            continue
        idx = int(e["utility"])
        s_idx = e["stream"]
        if e["type"] == "hot":
            ActiveHU.append((idx, s_idx))
        elif e["type"] == "cold":
            ActiveCU.append((idx, s_idx))

    return ActiveHU, ActiveCU


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


def _idx_set(n):
    """0-based ordered Set, safe for n == 0 (unlike RangeSet(0, -1))."""
    return pyo.Set(initialize=list(range(n)), ordered=True)


def _assemble_data(results, Hsap, Csap, delta_tmin,
                    U_overall, U_matrix,
                    cost_a, cost_b, cost_beta, payback,
                    hours_per_year, Q_THRESH, utility_specs):
    """
    Build the `data` namespace expected by build_variables_nlp /
    build_constraints_nlp / build_objective_nlp, from the MILP results
    dict plus cost/utility parameters. Split out from refine_hen_nlp so
    it can be unit-tested without needing a solver.
    """
    (ActiveIJK, I, J, K, S, HID, CID, CP_H, CP_C,
     Tin_H, Tout_H, Tin_C, Tout_C) = build_active_topology(
        results, Hsap, Csap, Q_thresh=Q_THRESH)

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
    close to the optimum instead of at variable-bound defaults."""
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
        q0 = min(data.Q_match_max[i, j], data.Q_match_max[i, j])
        m.Q[i, j, k].set_value(q0)
        dt1 = max(pyo.value(m.TH[i, k]) - pyo.value(m.TC[j, k]), data.delta_tmin)
        dt2 = max(pyo.value(m.TH[i, k + 1]) - pyo.value(m.TC[j, k + 1]), data.delta_tmin)
        m.dT1[i, j, k].set_value(dt1)
        m.dT2[i, j, k].set_value(dt2)
        lmtd0 = max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv[i, j, k].set_value(lmtd0)
        a0 = q0 / (data.U[i, j] * lmtd0)
        m.A[i, j, k].set_value(a0)
        m.Cost[i, j, k].set_value(data.cost_fixed[i, j] + data.cost_coeff[i, j] * a0 ** data.cost_exp[i, j])

    QH_milp = results.get("QH")
    QC_milp = results.get("QC")

    if QH_milp is not None:
        for j in range(J):
            m.QH[j].set_value(QH_milp[j])
    if QC_milp is not None:
        for i in range(I):
            m.QC[i].set_value(QC_milp[i])

    for (u, j) in data.ActiveHU:
        q0 = pyo.value(m.QH[j])
        m.QHU[u, j].set_value(q0)
        dt1 = max(data.T_HU_supply[u] - data.Tout_C[j], data.delta_tmin)
        dt2 = max(data.T_HU_return[u] - pyo.value(m.TC[j, 0]), data.delta_tmin)
        m.dT2_HU[u, j].set_value(dt2)
        lmtd0 = max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv_HU[u, j].set_value(lmtd0)
        a0 = q0 / (data.U_hu[u, j] * lmtd0)
        m.A_HU[u, j].set_value(a0)
        m.Cost_HU[u, j].set_value(
            data.cost_fixed_hu[u, j] + data.cost_coeff_hu[u, j] * a0 ** data.cost_exp_hu[u, j])

    for (v, i) in data.ActiveCU:
        q0 = pyo.value(m.QC[i])
        m.QCU[v, i].set_value(q0)
        dt1 = max(pyo.value(m.TH[i, K - 1]) - data.T_CU_return[v], data.delta_tmin)
        dt2 = max(data.Tout_H[i] - data.T_CU_supply[v], data.delta_tmin)
        m.dT1_CU[v, i].set_value(dt1)
        lmtd0 = max(_lmtd(dt1, dt2), 1e-3)
        m.LMTDv_CU[v, i].set_value(lmtd0)
        a0 = q0 / (data.U_cu[v, i] * lmtd0)
        m.A_CU[v, i].set_value(a0)
        m.Cost_CU[v, i].set_value(
            data.cost_fixed_cu[v, i] + data.cost_coeff_cu[v, i] * a0 ** data.cost_exp_cu[v, i])


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
                    utility_specs=None):
    """
    Stage-2 NLP refinement of a solved HENS MILP: fixes the topology from
    `results` (z/yHU/yCU > 0.5, filtered by Q_THRESH), then solves the
    continuous Pyomo NLP (exact/smooth LMTD + power-law cost, no SOS2/
    binaries) to get the true optimal duties, temperatures, areas, and
    cost for that fixed structure.

    Returns a dict shaped like `results`, updated in place (via
    dict(results)) with refined "edges", "util_hex_edges", "QH", "QC",
    "T_hot", "T_cold", "TAC", and cost breakdowns -- mirroring the old
    scipy-based refine_hen_nlp's output shape.
    """
    data, meta = _assemble_data(
        results, Hsap, Csap, delta_tmin,
        U_overall, U_matrix,
        cost_a, cost_b, cost_beta, payback,
        hours_per_year, Q_THRESH, utility_specs,)
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
    )

    term_cond = solve_result.solver.termination_condition

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