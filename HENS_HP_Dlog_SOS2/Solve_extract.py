import math
import numpy as np
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from types import SimpleNamespace

from Pre_process import _lmtd, pre_process_milp, build_model


# ═══════════════════════════════════════════════════════════════════════════
# SOLVE
# ═══════════════════════════════════════════════════════════════════════════

def solve_model(m, solver_name='scip', time_limit=200, gap=0.01, tee=True):
    """
    Solve the HENS MILP using Pyomo's 'scip' solver interface.

    `solver_name` is kept as a parameter (defaulting to 'scip') in case you
    want to point it at a different SCIP install/interface later, but no
    other solver is tried as a fallback.
    """
    solver = pyo.SolverFactory(solver_name)
    if not solver.available():
        raise RuntimeError(
            f"Solver '{solver_name}' is not available to Pyomo. "
            f"Make sure the `scip` executable is on PATH.")

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

                    # A. Physical True Values (Thermodynamic Rigorous)
                    dT1_true = max(T_hot[i_idx, s_idx] - T_cold[j_idx, s_idx], delta_tmin)
                    dT2_true = max(T_hot[i_idx, s_idx + 1] - T_cold[j_idx, s_idx + 1], delta_tmin)
                    lmtd_true = _lmtd(dT1_true, dT2_true)
                    area_true = Q / (U_ij * max(lmtd_true, 1e-4))
                    cost_true = cost_a + cost_b * (area_true ** cost_beta)

                    # B. Decision Variables from Pyomo Model
                    # (FIX: the DLOG-based model exposes LMTD, dT1, dT2 and
                    # Cost directly -- there is no ln_LMTDv / ln_A in this
                    # formulation, so those lookups are replaced below.)
                    lmtd_model = _get_val(["LMTD"], (i, j, s), default=lmtd_true)
                    dT1_model = _get_val(["dT1"], (i, j, s), default=dT1_true)
                    dT2_model = _get_val(["dT2"], (i, j, s), default=dT2_true)
                    area_model = Q / (U_ij * max(lmtd_model, 1e-4))

                    cap_fixed_per_hex = cost_a
                    cost_model_var = _get_val(["Cost", "cost"], (i, j, s), default=(cost_true - cap_fixed_per_hex))
                    cost_model = cap_fixed_per_hex + cost_model_var

                    # C. Physical Errors (model decision variables vs. rigorous thermodynamics)
                    err_lmtd_abs = lmtd_model - lmtd_true
                    err_lmtd_pct = (abs(err_lmtd_abs) / max(lmtd_true, 1e-4)) * 100.0

                    err_area_abs = area_model - area_true
                    err_area_pct = (abs(err_area_abs) / max(area_true, 1e-4)) * 100.0

                    err_cost_abs = cost_model - cost_true
                    err_cost_pct = (abs(err_cost_abs) / max(cost_true, 1e-4)) * 100.0

                    # D. Linearisation Fitting Errors
                    #    (i)  LMTD OA gap: BLOCK 9's tangent-plane cuts
                    #         over-estimate the true LMTD(dT1,dT2) surface;
                    #         this is the gap between LMTD[i,j,k] and the
                    #         exact LMTD evaluated at the model's own dT1/dT2.
                    exact_lmtd_at_model_dT = _lmtd(dT1_model, dT2_model)
                    sos2_lmtd_fit_err = abs(lmtd_model - exact_lmtd_at_model_dT)

                    #    (ii) DLOG Cost grid fit error: gap between the
                    #         DLOG-reconstructed Cost[i,j,k] and the exact
                    #         cost formula evaluated at the model's own
                    #         (Q, LMTD) -- i.e. the residual PWL
                    #         interpolation error of the 2D triangulated
                    #         DLOG grid at the chosen operating point.
                    exact_cost_at_model_QL = cost_a + cost_b * (Q / (U_ij * max(lmtd_model, 1e-4))) ** cost_beta
                    sos2_cost_fit_err = abs(cost_model - exact_cost_at_model_QL)

                    edges.append({
                        "hot": HID[i_idx],
                        "cold": CID[j_idx],
                        "stage": s_idx + 1 if isinstance(s, int) else s,
                        "hot_id": HID[i_idx],
                        "cold_id": CID[j_idx],
                        "Q": round(Q, 4),
                        "A": round(area_model, 2),
                        "active": bool(z_val > 0.5),
                        # Plain names -- as-designed (model) values, used by the UI
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
                        "err_cost_pct": round(err_cost_pct, 2),
                        "sos2_lmtd_fit_err": round(sos2_lmtd_fit_err, 4),
                        "sos2_cost_fit_err": round(sos2_cost_fit_err, 2),})

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
            "no_solution_found": True,}
    converged = status_str in ["OPTIMAL", "LOCALLY_SOLVED", "LOCALLYSOLVED"]

    # ── 8. Error Summary Calculation ─────────────────────────────────────────
    if edges:
        max_lmtd_err_pct = max(e["err_lmtd_pct"] for e in edges)
        mean_lmtd_err_pct = float(np.mean([e["err_lmtd_pct"] for e in edges]))

        max_area_err_pct = max(e["err_area_pct"] for e in edges)
        mean_area_err_pct = float(np.mean([e["err_area_pct"] for e in edges]))

        max_cost_err_pct = max(e["err_cost_pct"] for e in edges)
        mean_cost_err_pct = float(np.mean([e["err_cost_pct"] for e in edges]))

        max_sos2_lmtd_err = max(e["sos2_lmtd_fit_err"] for e in edges)
        max_sos2_cost_err = max(e["sos2_cost_fit_err"] for e in edges)
    else:
        max_lmtd_err_pct = mean_lmtd_err_pct = max_area_err_pct = mean_area_err_pct = 0.0
        max_cost_err_pct = mean_cost_err_pct = max_sos2_lmtd_err = max_sos2_cost_err = 0.0

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
        print(f"  DLOG/SOS2 Grid Fit Max Errors : LMTD = {max_sos2_lmtd_err:.4f} K | Cost = ${max_sos2_cost_err:.2f}")
        print(f"  Active Process Exchangers  : {len(edges)} | Utility Exchangers: {len(util_hex_edges)}")

        if edges:
            print("\n ── Individual Process HEX Error Breakdown ─────────────────────────")
            for e in edges:
                print(f"  HEX ({e['hot_id']} -> {e['cold_id']}, Stage {e['stage']}): Q = {e['Q']} kW")
                print(f"    ├─ LMTD (Model/True): {e['LMTD_model']} / {e['LMTD_true']} K  (Err: {e['err_lmtd_pct']}%)")
                print(f"    ├─ Area (Model/True): {e['Area_m2_model']} / {e['Area_m2_true']} m² (Err: {e['err_area_pct']}%)")
                print(f"    └─ Cost (Model/True): ${e['Cost_model_$']} / ${e['Cost_true_$']} (Err: {e['err_cost_pct']}%)")

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

        # Model Parameters
        "cu_hot":           getattr(data, "cu_hot", 80.0),
        "cu_cold":          getattr(data, "cu_cold", 20.0),
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
            "max_sos2_lmtd_err": round(max_sos2_lmtd_err, 4),
            "max_sos2_cost_err": round(max_sos2_cost_err, 2),},}


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
            print(f"       • DLOG Fit   : LMTD Grid Error = {e['sos2_lmtd_fit_err']:.4f} K | Cost Grid Error = ${e['sos2_cost_fit_err']:.2f}")

    # ── 3. Summary Aggregates ────────────────────────────────────────────────
    print("\n3. NETWORK-WIDE ERROR SUMMARY AGGREGATES:")
    print("   Metric              | Max Error  | Mean Error")
    print("   ---------------------------------------------")
    print(f"   LMTD Discrepancy    | {summary.get('max_lmtd_err_pct', 0.0):6.2f}%    | {summary.get('mean_lmtd_err_pct', 0.0):6.2f}%")
    print(f"   Area Discrepancy    | {summary.get('max_area_err_pct', 0.0):6.2f}%    | {summary.get('mean_area_err_pct', 0.0):6.2f}%")
    print(f"   Cost Discrepancy    | {summary.get('max_cost_err_pct', 0.0):6.2f}%    | {summary.get('mean_cost_err_pct', 0.0):6.2f}%")
    print("   ---------------------------------------------")
    print(f"   Max DLOG/SOS2 LMTD Grid Approximation Error : {summary.get('max_sos2_lmtd_err', 0.0):.4f} K")
    print(f"   Max DLOG/SOS2 Cost Grid Approximation Error : ${summary.get('max_sos2_cost_err', 0.0):,.2f}")
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
                   time_limit = 200,
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