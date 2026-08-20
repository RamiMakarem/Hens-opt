import pyomo.environ as pyo


def _lmtd_true(dT1, dT2):
    """
    True log-mean temperature difference -- for use on plain floats only
    (e.g. post-solve reporting from pyo.value(...) results), NOT inside a
    Pyomo constraint rule. It uses Python `if`/`max`, which can't be
    evaluated on a symbolic Var/expression, and even a branch-free version
    of this exact formula has a removable singularity at dT1==dT2 (0/0)
    that breaks automatic differentiation at exactly the point a
    stage-wise HEN model tends to sit near.
    """
    dT1 = max(dT1, 1e-3)
    dT2 = max(dT2, 1e-3)
    if abs(dT1 - dT2) < 1e-4:
        return max(dT1, 1e-4)
    import math
    return (dT1 - dT2) / math.log(dT1 / dT2)


def _lmtd_smooth(dT1, dT2):
    """
    Chen's (1987) smooth algebraic approximation to the log-mean
    temperature difference -- pure arithmetic (no log, no branch, no
    division-by-a-possibly-zero-quantity), so it's safe to use directly
    inside a Pyomo constraint rule and differentiable everywhere on the
    positive orthant, including at dT1==dT2. Typical error vs. the true
    LMTD is well under 1% over the ranges HEN approach temperatures live
    in. This is what LMTDv/LMTDv_HU/LMTDv_CU are actually sized against;
    use _lmtd_true on the solved (dT1, dT2) values afterward if you want
    the true LMTD for reporting/QA against these model values.
    """
    return (dT1 * dT2 * (dT1 + dT2) / 2.0) ** (1.0 / 3.0)



def build_constraints_nlp(m, data):
    Hi, Hj, Hs = m.Hi, m.Hj, m.Hs
    K = data.K

    # ════════════════════════════════════════════════════════════════
    # BLOCK 1: Overall energy balance (unchanged from MILP, sums now run
    # only over active matches touching each stream)
    # ════════════════════════════════════════════════════════════════
    def hot_balance_rule(m, i):
        total_duty = data.CP_H[i] * (data.Tin_H[i] - data.Tout_H[i])
        active_jk = [(j, k) for (ii, j, k) in m.ActiveIJK if ii == i]
        return sum(m.Q[i, j, k] for (j, k) in active_jk) + m.QC[i] == total_duty
    m.c_hot_balance = pyo.Constraint(Hi, rule=hot_balance_rule)

    def cold_balance_rule(m, j):
        total_duty = data.CP_C[j] * (data.Tout_C[j] - data.Tin_C[j])
        active_ik = [(i, k) for (i, jj, k) in m.ActiveIJK if jj == j]
        return sum(m.Q[i, j, k] for (i, k) in active_ik) + m.QH[j] == total_duty
    m.c_cold_balance = pyo.Constraint(Hj, rule=cold_balance_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 2: Stage energy balance
    # ════════════════════════════════════════════════════════════════
    def hot_stage_rule(m, i, k):
        active_j = [j for (ii, j, kk) in m.ActiveIJK if ii == i and kk == k]
        return data.CP_H[i] * (m.TH[i, k] - m.TH[i, k + 1]) == sum(
            m.Q[i, j, k] for j in active_j)
    m.c_hot_stage = pyo.Constraint(Hi, Hs, rule=hot_stage_rule)

    def cold_stage_rule(m, j, k):
        active_i = [i for (i, jj, kk) in m.ActiveIJK if jj == j and kk == k]
        return data.CP_C[j] * (m.TC[j, k] - m.TC[j, k + 1]) == sum(
            m.Q[i, j, k] for i in active_i)
    m.c_cold_stage = pyo.Constraint(Hj, Hs, rule=cold_stage_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 3: Inlet temperatures
    # ════════════════════════════════════════════════════════════════
    m.c_hot_inlet = pyo.Constraint(Hi, rule=lambda m, i: m.TH[i, 0] == data.Tin_H[i])
    m.c_cold_inlet = pyo.Constraint(Hj, rule=lambda m, j: m.TC[j, K - 1] == data.Tin_C[j])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 4: Temperature monotonicity
    # ════════════════════════════════════════════════════════════════
    m.c_hot_mono = pyo.Constraint(Hi, Hs, rule=lambda m, i, k: m.TH[i, k] >= m.TH[i, k + 1])
    m.c_cold_mono = pyo.Constraint(Hj, Hs, rule=lambda m, j, k: m.TC[j, k] >= m.TC[j, k + 1])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 5: Outlet temperature feasibility
    # ════════════════════════════════════════════════════════════════
    m.c_hot_outlet = pyo.Constraint(Hi, rule=lambda m, i: m.TH[i, K - 1] >= data.Tout_H[i])
    m.c_cold_outlet = pyo.Constraint(Hj, rule=lambda m, j: m.TC[j, 0] <= data.Tout_C[j])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 6: Utility duties -- no more yHU/yCU big-M, just direct sums
    # over the active assignment set
    # ════════════════════════════════════════════════════════════════
    # c_qc_def / c_qh_def (QC[i] == CP_H[i]*(TH[i,K-1]-Tout_H[i]), and the
    # cold-stream analog) are DELIBERATELY NOT included here. They're
    # algebraically redundant with c_hot_balance + c_hot_stage (telescoped
    # across stages) + c_hot_inlet -- i.e. QC[i] is already uniquely pinned
    # once those hold, and re-stating it as its own equality just adds a
    # linearly-dependent row. Harmless for an LP/MIP solver (which is why
    # the MILP version of this file keeps the analogous pair), but IPOPT's
    # variable-vs-equality-constraint count catches it and refuses to solve
    # at all (TOO_FEW_DOF), since a fixed topology here already has very
    # few genuine continuous degrees of freedom to begin with.

    def qh_agg_rule(m, j):
        active_u = [u for (u, jj) in m.ActiveHU if jj == j]
        if not active_u:
            return m.QH[j] == 0
        return m.QH[j] == sum(m.QHU[u, j] for u in active_u)
    m.c_qh_agg = pyo.Constraint(Hj, rule=qh_agg_rule)

    def qc_agg_rule(m, i):
        active_v = [v for (v, ii) in m.ActiveCU if ii == i]
        if not active_v:
            return m.QC[i] == 0
        return m.QC[i] == sum(m.QCU[v, i] for v in active_v)
    m.c_qc_agg = pyo.Constraint(Hi, rule=qc_agg_rule)

    # Global capacity constraint (unchanged -- still meaningful even with
    # a single fixed utility choice per stream, e.g. multiple cold streams
    # sharing one HU's flow budget)
    def hu_capacity_rule(m, u):
        assigned_j = [j for (uu, j) in m.ActiveHU if uu == u]
        if not assigned_j or not __import__("math").isfinite(data.hot_max_flow[u]):
            return pyo.Constraint.Skip
        Q_limit = data.hot_max_flow[u] * data.hot_Q_per_kg[u]
        return sum(m.QHU[u, j] for j in assigned_j) <= Q_limit
    m.c_hu_capacity = pyo.Constraint(m.HU, rule=hu_capacity_rule)

    def cu_capacity_rule(m, v):
        assigned_i = [i for (vv, i) in m.ActiveCU if vv == v]
        if not assigned_i or not __import__("math").isfinite(data.cold_max_flow[v]):
            return pyo.Constraint.Skip
        Q_limit = data.cold_max_flow[v] * data.cold_Q_per_kg[v]
        return sum(m.QCU[v, i] for i in assigned_i) <= Q_limit
    m.c_cu_capacity = pyo.Constraint(m.CU, rule=cu_capacity_rule)

    # Temperature feasibility & phase-change elbow: topology (and hence
    # utility choice) is fixed, so these become plain equalities/inequalities
    # on the active pairs only -- no big-M, no yHU/yCU multiplier.
    def hu_temp_feas_rule(m, u, j):
        T_sup = data.T_HU_supply[u]
        if T_sup > data.Tin_C[j] + data.delta_tmin:
            return m.TC[j, 0] <= T_sup - data.delta_tmin
        return pyo.Constraint.Skip
    m.c_hu_temp_feas = pyo.Constraint(m.ActiveHU, rule=hu_temp_feas_rule)

    def cu_temp_feas_rule(m, v, i):
        T_sup = data.T_CU_supply[v]
        if T_sup < data.Tout_H[i] - data.delta_tmin:
            return m.TH[i, K - 1] >= T_sup + data.delta_tmin
        return pyo.Constraint.Skip
    m.c_cu_temp_feas = pyo.Constraint(m.ActiveCU, rule=cu_temp_feas_rule)

    def hu_phase_rule(m, u, j):
        if not data.hot_is_combined[u]:
            return pyo.Constraint.Skip
        T_ph = data.hot_T_phase[u]
        cp_vap = data.hot_cp_vap[u]
        T_sup = data.T_HU_supply[u]
        Q_pk = data.hot_Q_per_kg[u]
        if cp_vap <= 0 or T_sup <= T_ph:
            return pyo.Constraint.Skip
        coef_m = cp_vap * (T_sup - T_ph) / max(data.CP_C[j], 1e-6)
        coef_Q = coef_m / Q_pk
        return m.TC[j, 0] + coef_Q * m.QHU[u, j] <= T_ph - data.delta_tmin
    m.c_hu_phase = pyo.Constraint(m.ActiveHU, rule=hu_phase_rule)

    def cu_phase_rule(m, v, i):
        if not data.cold_is_combined[v]:
            return pyo.Constraint.Skip
        T_ph = data.cold_T_phase[v]
        cp_liq = data.cold_cp_liq[v]
        T_sup = data.T_CU_supply[v]
        Q_pk = data.cold_Q_per_kg[v]
        if cp_liq <= 0 or T_ph <= T_sup:
            return pyo.Constraint.Skip
        coef_m = cp_liq * (T_ph - T_sup) / max(data.CP_H[i], 1e-6)
        coef_Q = coef_m / Q_pk
        return m.TH[i, K - 1] >= coef_Q * m.QCU[v, i] + T_ph + data.delta_tmin
    m.c_cu_phase = pyo.Constraint(m.ActiveCU, rule=cu_phase_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 7: dTmin enforcement -- hard constraint now, no z/Gamma gating
    # ════════════════════════════════════════════════════════════════
    def dtmin_1_rule(m, i, j, k):
        return m.TH[i, k] - m.TC[j, k] >= data.delta_tmin
    m.c_dtmin_1 = pyo.Constraint(m.ActiveIJK, rule=dtmin_1_rule)

    def dtmin_2_rule(m, i, j, k):
        return m.TH[i, k + 1] - m.TC[j, k + 1] >= data.delta_tmin
    m.c_dtmin_2 = pyo.Constraint(m.ActiveIJK, rule=dtmin_2_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 8: Approach temperatures + exact/smooth LMTD (replaces the
    # 2D SOS2 grid entirely)
    # ════════════════════════════════════════════════════════════════
    def dT1_def_rule(m, i, j, k):
        return m.dT1[i, j, k] == m.TH[i, k] - m.TC[j, k]
    m.c_dT1_def = pyo.Constraint(m.ActiveIJK, rule=dT1_def_rule)

    def dT2_def_rule(m, i, j, k):
        return m.dT2[i, j, k] == m.TH[i, k + 1] - m.TC[j, k + 1]
    m.c_dT2_def = pyo.Constraint(m.ActiveIJK, rule=dT2_def_rule)

    def LMTDv_def_rule(m, i, j, k):
        return m.LMTDv[i, j, k] == _lmtd_smooth(m.dT1[i, j, k], m.dT2[i, j, k])
    m.c_LMTDv_def = pyo.Constraint(m.ActiveIJK, rule=LMTDv_def_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 9: Area sizing + power-law capital cost (replaces the 1D SOS2
    # grid entirely). This is the relation that used to be *implicit* in
    # how sos2_A/sos2_fA were fit against the same (dT1,dT2) grid as the
    # LMTD surface -- now it must be explicit since A, Q, LMTDv are all
    # free variables solved simultaneously.
    #
    # ASSUME: data carries per-match overall heat-transfer coefficients
    # `data.U[i, j]`, and Turton-style power-law cost coefficients
    # `data.cost_fixed[i, j]`, `data.cost_coeff[i, j]`, `data.cost_exp[i, j]`
    # such that Cost = cost_fixed + cost_coeff * A**cost_exp.
    # Swap in your real field names if they differ.
    # ════════════════════════════════════════════════════════════════
    def A_size_rule(m, i, j, k):
        return m.A[i, j, k] * data.U[i, j] * m.LMTDv[i, j, k] == m.Q[i, j, k]
    m.c_A_size = pyo.Constraint(m.ActiveIJK, rule=A_size_rule)

    def Cost_def_rule(m, i, j, k):
        return m.Cost[i, j, k] == (
            data.cost_fixed[i, j]
            + data.cost_coeff[i, j] * m.A[i, j, k] ** data.cost_exp[i, j])
    m.c_Cost_def = pyo.Constraint(m.ActiveIJK, rule=Cost_def_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 10: HOT UTILITY (HU) -- approach temps, LMTD, sizing, cost
    # ════════════════════════════════════════════════════════════════
    def dT2_HU_def_rule(m, u, j):
        return m.dT2_HU[u, j] == data.T_HU_return[u] - m.TC[j, 0]
    m.c_dT2_HU_def = pyo.Constraint(m.ActiveHU, rule=dT2_HU_def_rule)

    def LMTDv_HU_def_rule(m, u, j):
        return m.LMTDv_HU[u, j] == _lmtd_smooth(data.dT1_HU[u, j], m.dT2_HU[u, j])
    m.c_LMTDv_HU_def = pyo.Constraint(m.ActiveHU, rule=LMTDv_HU_def_rule)

    def A_HU_size_rule(m, u, j):
        return m.A_HU[u, j] * data.U_hu[u, j] * m.LMTDv_HU[u, j] == m.QHU[u, j]
    m.c_A_HU_size = pyo.Constraint(m.ActiveHU, rule=A_HU_size_rule)

    def Cost_HU_def_rule(m, u, j):
        return m.Cost_HU[u, j] == (
            data.cost_fixed_hu[u, j]
            + data.cost_coeff_hu[u, j] * m.A_HU[u, j] ** data.cost_exp_hu[u, j])
    m.c_Cost_HU_def = pyo.Constraint(m.ActiveHU, rule=Cost_HU_def_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 11: COLD UTILITY (CU) -- approach temps, LMTD, sizing, cost
    # ════════════════════════════════════════════════════════════════
    last_k = m.Knodes.last()

    def dT1_CU_def_rule(m, v, i):
        return m.dT1_CU[v, i] == m.TH[i, last_k] - data.T_CU_return[v]
    m.c_dT1_CU_def = pyo.Constraint(m.ActiveCU, rule=dT1_CU_def_rule)

    def LMTDv_CU_def_rule(m, v, i):
        return m.LMTDv_CU[v, i] == _lmtd_smooth(m.dT1_CU[v, i], data.dT2_CU[v, i])
    m.c_LMTDv_CU_def = pyo.Constraint(m.ActiveCU, rule=LMTDv_CU_def_rule)

    def A_CU_size_rule(m, v, i):
        return m.A_CU[v, i] * data.U_cu[v, i] * m.LMTDv_CU[v, i] == m.QCU[v, i]
    m.c_A_CU_size = pyo.Constraint(m.ActiveCU, rule=A_CU_size_rule)

    def Cost_CU_def_rule(m, v, i):
        return m.Cost_CU[v, i] == (
            data.cost_fixed_cu[v, i]
            + data.cost_coeff_cu[v, i] * m.A_CU[v, i] ** data.cost_exp_cu[v, i])
    m.c_Cost_CU_def = pyo.Constraint(m.ActiveCU, rule=Cost_CU_def_rule)

    # ════════════════════════════════════════════════════════════════
    # Block 12: Minimum duty caps
    # ════════════════════════════════════════════════════════════════
    # 1. Cold Stream Stage-wise Minimum Duty (Fixed Topology NLP)
    def min_duty_cold_nlp_rule(m, i, j, k):
        # Sum over all active hot streams matching with cold stream j at stage k
        stage_duty_C = sum(m.Q[ip, j, k] for (ip, jj, kk) in m.ActiveIJK if jj == j and kk == k)
        return m.Q[i, j, k] >= 0.15 * stage_duty_C

    m.c_min_duty_cold = pyo.Constraint(m.ActiveIJK, rule=min_duty_cold_nlp_rule)


    # 2. Hot Stream Stage-wise Minimum Duty (Fixed Topology NLP)
    def min_duty_hot_nlp_rule(m, i, j, k):
        # Sum over all active cold streams matching with hot stream i at stage k
        stage_duty_H = sum(m.Q[i, jp, k] for (ii, jp, kk) in m.ActiveIJK if ii == i and kk == k)
        return m.Q[i, j, k] >= 0.15 * stage_duty_H

    m.c_min_duty_hot = pyo.Constraint(m.ActiveIJK, rule=min_duty_hot_nlp_rule)

    '''
    def dT1_HU_def_rule(m, u, j):
        return m.dT1_HU[u, j] == data.T_HU_supply[u] - data.Tout_C[j]
    m.c_dT1_HU_def = pyo.Constraint(m.ActiveHU, rule=dT1_HU_def_rule)
    def dT2_CU_def_rule(m, v, i):
        return m.dT2_CU[v, i] == data.Tout_H[i] - data.T_CU_supply[v]
    m.c_dT2_CU_def = pyo.Constraint(m.ActiveCU, rule=dT2_CU_def_rule)
    '''

    return m