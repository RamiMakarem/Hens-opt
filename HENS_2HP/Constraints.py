import math
import pyomo.environ as pyo


def build_constraints(m, data):
    Hi, Hj, Hs = m.Hi, m.Hj, m.Hs
    K = data.K

    # ════════════════════════════════════════════════════════════════
    # BLOCK 1: Overall energy balance
    # ════════════════════════════════════════════════════════════════
    def hot_balance_rule(m, i):
        total_duty = data.CP_H[i] * (data.Tin_H[i] - data.Tout_H[i])
        return sum(m.Q[i, j, k] for j in Hj for k in Hs) + m.QC[i] == total_duty
    m.c_hot_balance = pyo.Constraint(Hi, rule=hot_balance_rule)

    def cold_balance_rule(m, j):
        total_duty = data.CP_C[j] * (data.Tout_C[j] - data.Tin_C[j])
        return sum(m.Q[i, j, k] for i in Hi for k in Hs) + m.QH[j] == total_duty
    m.c_cold_balance = pyo.Constraint(Hj, rule=cold_balance_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 2: Stage energy balance
    # ════════════════════════════════════════════════════════════════
    def hot_stage_rule(m, i, k):
        return data.CP_H[i] * (m.TH[i, k] - m.TH[i, k + 1]) == sum(m.Q[i, j, k] for j in Hj)
    m.c_hot_stage = pyo.Constraint(Hi, Hs, rule=hot_stage_rule)

    def cold_stage_rule(m, j, k):
        return data.CP_C[j] * (m.TC[j, k] - m.TC[j, k + 1]) == sum(m.Q[i, j, k] for i in Hi)
    m.c_cold_stage = pyo.Constraint(Hj, Hs, rule=cold_stage_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 3: Inlet temperatures
    # ════════════════════════════════════════════════════════════════
    m.c_hot_inlet = pyo.Constraint(Hi, rule=lambda m, i: m.TH[i, 0] == data.Tin_H[i])
    m.c_cold_inlet = pyo.Constraint(Hj, rule=lambda m, j: m.TC[j, K-1] == data.Tin_C[j])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 4: Temperature monotonicity
    # ════════════════════════════════════════════════════════════════
    m.c_hot_mono = pyo.Constraint(Hi, Hs, rule=lambda m, i, k: m.TH[i, k] >= m.TH[i, k + 1])
    m.c_cold_mono = pyo.Constraint(Hj, Hs, rule=lambda m, j, k: m.TC[j, k] >= m.TC[j, k + 1])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 5: Outlet temperature feasibility
    # ════════════════════════════════════════════════════════════════
    m.c_hot_outlet = pyo.Constraint(Hi, rule=lambda m, i: m.TH[i, K-1] >= data.Tout_H[i])
    m.c_cold_outlet = pyo.Constraint(Hj, rule=lambda m, j: m.TC[j, 0] <= data.Tout_C[j])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 6: Utility duties (disaggregated to individual utility flowrates)
    # ════════════════════════════════════════════════════════════════
    m.c_qc_def = pyo.Constraint(Hi, rule=lambda m, i: m.QC[i] == data.CP_H[i] * (m.TH[i, K-1] - data.Tout_H[i]))
    m.c_qh_def = pyo.Constraint(Hj, rule=lambda m, j: m.QH[j] == data.CP_C[j] * (data.Tout_C[j] - m.TC[j, 0]))

    # (6b-1) Aggregate Duty Equality
    m.c_qh_agg = pyo.Constraint(Hj, rule=lambda m, j: m.QH[j] == sum(m.QHU[u, j] for u in m.HU))
    m.c_qc_agg = pyo.Constraint(Hi, rule=lambda m, i: m.QC[i] == sum(m.QCU[v, i] for v in m.CU))

    # (6b-2) At most one hot utility per cold stream; at most one cold utility per hot stream.
    if data.n_HU > 0:
        m.c_one_hu = pyo.Constraint(Hj, rule=lambda m, j: sum(m.yHU[u, j] for u in m.HU) <= 1)
    if data.n_CU > 0:
        m.c_one_cu = pyo.Constraint(Hi, rule=lambda m, i: sum(m.yCU[v, i] for v in m.CU) <= 1)

    # (6b-3) Q_HU[u,j] <= Q_max * y_HU[u,j]
    def hu_bigM_rule(m, u, j):
        Q_max = data.Q_C_total[j] + 1.0
        return m.QHU[u, j] <= Q_max * m.yHU[u, j]
    m.c_hu_bigM = pyo.Constraint(m.HU, Hj, rule=hu_bigM_rule)

    def cu_bigM_rule(m, v, i):
        Q_max = data.Q_H_total[i] + 1.0
        return m.QCU[v, i] <= Q_max * m.yCU[v, i]
    m.c_cu_bigM = pyo.Constraint(m.CU, Hi, rule=cu_bigM_rule)

    # (6b-4) Global capacity constraint
    def hu_capacity_rule(m, u):
        if not math.isfinite(data.hot_max_flow[u]):
            return pyo.Constraint.Skip
        Q_limit = data.hot_max_flow[u] * data.hot_Q_per_kg[u]
        return sum(m.QHU[u, j] for j in Hj) <= Q_limit
    m.c_hu_capacity = pyo.Constraint(m.HU, rule=hu_capacity_rule)

    def cu_capacity_rule(m, v):
        if not math.isfinite(data.cold_max_flow[v]):
            return pyo.Constraint.Skip
        Q_limit = data.cold_max_flow[v] * data.cold_Q_per_kg[v]
        return sum(m.QCU[v, i] for i in Hi) <= Q_limit
    m.c_cu_capacity = pyo.Constraint(m.CU, rule=cu_capacity_rule)

    # (6b-5) Temperature feasibility for each utility option (big-M form)
    def hu_temp_feas_rule(m, u, j):
        T_sup = data.T_HU_supply[u]
        if T_sup > data.Tin_C[j] + data.delta_tmin:
            return m.TC[j, 0] + data.dTmax_HU[u,j] * m.yHU[u, j] <= T_sup - data.delta_tmin + data.dTmax_HU[u,j]
        return pyo.Constraint.Skip
    m.c_hu_temp_feas = pyo.Constraint(m.HU, Hj, rule=hu_temp_feas_rule)

    def cu_temp_feas_rule(m, v, i):
        T_sup = data.T_CU_supply[v]
        if T_sup < data.Tout_H[i] - data.delta_tmin:
            return m.TH[i, K-1] - data.dTmax_CU[v,i] * m.yCU[v, i] >= T_sup + data.delta_tmin - data.dTmax_CU[v,i]
        return pyo.Constraint.Skip
    m.c_cu_temp_feas = pyo.Constraint(m.CU, Hi, rule=cu_temp_feas_rule)

    # (6b-6) Phase-change elbow constraint (combined utilities only)
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
        return m.TC[j, 0] + coef_Q * m.QHU[u, j] + data.dTmax_HU[u,j] * m.yHU[u, j] <= T_ph - data.delta_tmin + data.dTmax_HU[u,j]
    m.c_hu_phase = pyo.Constraint(m.HU, Hj, rule=hu_phase_rule)

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
        return m.TH[i, K-1] + data.dTmax_CU[v,i] >= coef_Q * m.QCU[v, i] + data.dTmax_CU[v,i] * m.yCU[v, i] + T_ph + data.delta_tmin
    m.c_cu_phase = pyo.Constraint(m.CU, Hi, rule=cu_phase_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 7: Big-M on Q and Minimum Flowrate Enforcement
    # ════════════════════════════════════════════════════════════════
    def Q_bigM_rule(m, i, j, k):
        return m.Q[i, j, k] <= data.Q_match_max[i, j] * m.z[i, j, k]
    m.c_Q_bigM = pyo.Constraint(Hi, Hj, Hs, rule=Q_bigM_rule)
    '''
    def Q_min_hot_stage_linear_rule(m, i, j, k):
        stage_duty_H = sum(m.Q[i, jp, k] for jp in Hj)
        return m.Q[i, j, k] >= 0.15 * stage_duty_H - data.Q_match_max[i, j] * (1 - m.z[i, j, k])
    m.c_Q_min_hot_stage = pyo.Constraint(Hi, Hj, Hs, rule=Q_min_hot_stage_linear_rule)

    def Q_min_cold_stage_linear_rule(m, i, j, k):
        stage_duty_C = sum(m.Q[ip, j, k] for ip in Hi)
        return m.Q[i, j, k] >= 0.15 * stage_duty_C - data.Q_match_max[i, j] * (1 - m.z[i, j, k])
    m.c_Q_min_cold_stage = pyo.Constraint(Hi, Hj, Hs, rule=Q_min_cold_stage_linear_rule)
    '''
    # ════════════════════════════════════════════════════════════════
    # BLOCK 8: dTmin enforcement and dT1 and dT2 
    # ════════════════════════════════════════════════════════════════
    # 8a: dTmin Enforcement   
    def dtmin_1_rule(m, i, j, k):
        return m.TH[i, k] - m.TC[j, k] + data.dT1_hi[i,j] >= data.delta_tmin + data.dT1_hi[i,j] * m.z[i, j, k]
    m.c_dtmin_1 = pyo.Constraint(Hi, Hj, Hs, rule=dtmin_1_rule)

    def dtmin_2_rule(m, i, j, k):
        return m.TH[i, k + 1] - m.TC[j, k + 1] + data.dT1_hi[i,j] >= data.delta_tmin + data.dT1_hi[i,j] * m.z[i, j, k]
    m.c_dtmin_2 = pyo.Constraint(Hi, Hj, Hs, rule=dtmin_2_rule)

    # 8b: dT1/dT2 -> TH/TC linkage
    def dT1_link_hi_rule(m, i, j, k):
        return m.dT1[i, j, k] <= m.TH[i, k] - m.TC[j, k] + data.dT1_hi[i,j] * (1 - m.z[i, j, k])
    m.c_dT1_link_hi = pyo.Constraint(Hi, Hj, Hs, rule=dT1_link_hi_rule)

    def dT1_link_lo_rule(m, i, j, k):
        return m.dT1[i, j, k] >= m.TH[i, k] - m.TC[j, k] - data.dT1_hi[i,j] * (1 - m.z[i, j, k])
    m.c_dT1_link_lo = pyo.Constraint(Hi, Hj, Hs, rule=dT1_link_lo_rule)

    def dT2_link_hi_rule(m, i, j, k):
        return m.dT2[i, j, k] <= m.TH[i, k + 1] - m.TC[j, k + 1] + data.dT1_hi[i,j] * (1 - m.z[i, j, k])
    m.c_dT2_link_hi = pyo.Constraint(Hi, Hj, Hs, rule=dT2_link_hi_rule)

    def dT2_link_lo_rule(m, i, j, k):
        return m.dT2[i, j, k] >= m.TH[i, k + 1] - m.TC[j, k + 1] - data.dT1_hi[i,j] * (1 - m.z[i, j, k])
    m.c_dT2_link_lo = pyo.Constraint(Hi, Hj, Hs, rule=dT2_link_lo_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 9: LMTD tangent-plane (outer approximation) constraints
    # ════════════════════════════════════════════════════════════════
    def LMTD_hyperplane_rule(m, i, j, k, p):
        a0, a1, a2 = data.lmtd_planes[i, j][p]
        return m.LMTD[i, j, k] <= a0 + a1 * m.dT1[i, j, k] + a2 * m.dT2[i, j, k]
    m.c_LMTD_hyperplane = pyo.Constraint(m.LMTD_cut_index, rule=LMTD_hyperplane_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 10: Hyperplane (A^beta) Linearization: (Q, LMTD) -> Cost
    # ════════════════════════════════════════════════════════════════
    # 1. Hyperplane Selection (1 plane active iff match z=1)
    # In Constraints.py around line 195
    match_planes = {
        (i, j): [p for p in m.K_planes if (i, j, p) in data.hp_data]
        for (i, j) in m.FeasiblePairs
    }
    def hp_select_plane_rule(m, i, j, k):
        return sum(m.z_plane[i, j, k, p] for p in match_planes[i, j]) == m.z[i, j, k]
    m.c_hp_select_plane = pyo.Constraint(m.FeasibleIJK, rule=hp_select_plane_rule)

    # 2. Domain Partition Constraints (Enforce cell bounds on Q & LMTD)
    def hp_cell_q_lo_rule(m, i, j, k):
        return m.Q[i, j, k] >= sum(data.hp_data[i, j, p]["q_min"] * m.z_plane[i, j, k, p] for p in match_planes[i, j])
    m.c_hp_cell_q_lo = pyo.Constraint(m.FeasibleIJK, rule=hp_cell_q_lo_rule)

    def hp_cell_q_hi_rule(m, i, j, k):
        m_q = float(data.Q_match_max[i, j])
        return m.Q[i, j, k] <= sum(data.hp_data[i, j, p]["q_max"] * m.z_plane[i, j, k, p] for p in match_planes[i, j]) + m_q * (1 - m.z[i, j, k])
    m.c_hp_cell_q_hi = pyo.Constraint(m.FeasibleIJK, rule=hp_cell_q_hi_rule)

    def hp_cell_l_lo_rule(m, i, j, k):
        return m.LMTD[i, j, k] >= sum(data.hp_data[i, j, p]["l_min"] * m.z_plane[i, j, k, p] for p in match_planes[i, j])
    m.c_hp_cell_l_lo = pyo.Constraint(m.FeasibleIJK, rule=hp_cell_l_lo_rule)

    def hp_cell_l_hi_rule(m, i, j, k):
        m_l = float(data.dT1_hi[i, j])
        return m.LMTD[i, j, k] <= sum(data.hp_data[i, j, p]["l_max"] * m.z_plane[i, j, k, p] for p in match_planes[i, j]) + m_l * (1 - m.z[i, j, k])
    m.c_hp_cell_l_hi = pyo.Constraint(m.FeasibleIJK, rule=hp_cell_l_hi_rule)

    # 3. Big-M Hyperplane Approximation Constraints for A_beta
    def hp_abeta_lo_rule(m, i, j, k, p):
        hp = data.hp_data[i, j, p]
        m_a = m.A_beta[i, j, k].ub or 1e5
        lhs = hp["a0"] * m.z_plane[i, j, k, p] + hp["a1"] * m.Q[i, j, k] + hp["a2"] * m.LMTD[i, j, k]
        return m.A_beta[i, j, k] >= lhs - m_a * (1 - m.z_plane[i, j, k, p])
    m.c_hp_abeta_lo = pyo.Constraint(m.FeasibleIJKP, rule=hp_abeta_lo_rule)

    def hp_abeta_hi_rule(m, i, j, k, p):
        hp = data.hp_data[i, j, p]
        m_a = m.A_beta[i, j, k].ub or 1e5
        lhs = hp["a0"] * m.z_plane[i, j, k, p] + hp["a1"] * m.Q[i, j, k] + hp["a2"] * m.LMTD[i, j, k]
        return m.A_beta[i, j, k] <= lhs + m_a * (1 - m.z_plane[i, j, k, p])
    m.c_hp_abeta_hi = pyo.Constraint(m.FeasibleIJKP, rule=hp_abeta_hi_rule)

    # 4. Variable Area Cost Link
    def hp_cost_link_rule(m, i, j, k):
        c_area = getattr(data, "C_area", {}).get((i, j), 1.0)
        return m.Cost[i, j, k] == c_area * m.A_beta[i, j, k]
    m.c_hp_cost_link = pyo.Constraint(m.FeasibleIJK, rule=hp_cost_link_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 11: SOS2 constraints for utility cost
    # ════════════════════════════════════════════════════════════════    
    #SOS2 variable definition
    def lam_hu_sos_rule(m, u, j):
        var_list = [m.lam_hu[u, j, k] for k in m.GU0]
        weight_list = [m.Qbp_HU[u, j, k] for k in m.GU0]
        return (var_list, weight_list)
    m.sos_HU = pyo.SOSConstraint(m.FeasibleHU, rule=lam_hu_sos_rule, sos=2)

    def lam_cu_sos_rule(m, v, i):
        var_list = [m.lam_cu[v, i, k] for k in m.GU0]
        weight_list = [m.Qbp_CU[v, i, k] for k in m.GU0]
        return (var_list, weight_list)
    m.sos_CU = pyo.SOSConstraint(m.FeasibleCU, rule=lam_cu_sos_rule, sos=2)

    m.lam_hu_sum = pyo.Constraint(
        m.FeasibleHU,
        rule=lambda m, u, j: sum(m.lam_hu[u, j, k] for k in m.GU0) == m.yHU[u, j])
    m.lam_cu_sum = pyo.Constraint(
        m.FeasibleCU,
        rule=lambda m, v, i: sum(m.lam_cu[v, i, k] for k in m.GU0) == m.yCU[v, i])

    # Q and Cost as the SOS2 convex combination of breakpoints
    m.QHU_link = pyo.Constraint(
        m.FeasibleHU,
        rule=lambda m, u, j: m.QHU[u, j] ==
        sum(m.lam_hu[u, j, k] * m.Qbp_HU[u, j, k] for k in m.GU0))
    m.CostHU_link = pyo.Constraint(
        m.FeasibleHU,
        rule=lambda m, u, j: m.Cost_HU[u, j] ==
        sum(m.lam_hu[u, j, k] * m.Cbp_HU[u, j, k] for k in m.GU0))

    m.QCU_link = pyo.Constraint(
        m.FeasibleCU,
        rule=lambda m, v, i: m.QCU[v, i] ==
        sum(m.lam_cu[v, i, k] * m.Qbp_CU[v, i, k] for k in m.GU0))
    m.CostCU_link = pyo.Constraint(
        m.FeasibleCU,
        rule=lambda m, v, i: m.Cost_CU[v, i] ==
        sum(m.lam_cu[v, i, k] * m.Cbp_CU[v, i, k] for k in m.GU0))

    return m