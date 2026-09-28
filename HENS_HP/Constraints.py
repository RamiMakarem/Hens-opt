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
    # BLOCK 4: Temperature monotonicity-Commented as redundant from Block 2
    # ════════════════════════════════════════════════════════════════
    #m.c_hot_mono = pyo.Constraint(Hi, Hs, rule=lambda m, i, k: m.TH[i, k] >= m.TH[i, k + 1])
    #m.c_cold_mono = pyo.Constraint(Hj, Hs, rule=lambda m, j, k: m.TC[j, k] >= m.TC[j, k + 1])

    # ════════════════════════════════════════════════════════════════
    # BLOCK 5: Outlet temperature feasibility-Commented as redundant from Blocks, 1 2 and 3 subbed in each other
    # ════════════════════════════════════════════════════════════════
    #m.c_hot_outlet = pyo.Constraint(Hi, rule=lambda m, i: m.TH[i, K-1] >= data.Tout_H[i])
    #m.c_cold_outlet = pyo.Constraint(Hj, rule=lambda m, j: m.TC[j, 0] <= data.Tout_C[j])

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
        return m.QHU[u, j] <= data.Q_max_hu[u,j] * m.yHU[u, j]
    m.c_hu_bigM = pyo.Constraint(m.FeasibleHU, rule=hu_bigM_rule)

    def cu_bigM_rule(m, v, i):
        return m.QCU[v, i] <= data.Q_max_cu[v,i] * m.yCU[v, i]
    m.c_cu_bigM = pyo.Constraint(m.FeasibleCU, rule=cu_bigM_rule)

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
    m.c_hu_temp_feas = pyo.Constraint(m.FeasibleHU, rule=hu_temp_feas_rule)

    def cu_temp_feas_rule(m, v, i):
        T_sup = data.T_CU_supply[v]
        if T_sup < data.Tout_H[i] - data.delta_tmin:
            return m.TH[i, K-1] - data.dTmax_CU[v,i] * m.yCU[v, i] >= T_sup + data.delta_tmin - data.dTmax_CU[v,i]
        return pyo.Constraint.Skip
    m.c_cu_temp_feas = pyo.Constraint(m.FeasibleCU, rule=cu_temp_feas_rule)

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
    m.c_hu_phase = pyo.Constraint(m.FeasibleHU, rule=hu_phase_rule)

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
    m.c_cu_phase = pyo.Constraint(m.FeasibleCU, rule=cu_phase_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 7: Big-M on Q and Minimum Flowrate Enforcement and Ensure
    # No full heat Exchange at k>0
    # ════════════════════════════════════════════════════════════════
    def Q_bigM_rule(m, i, j, k):
        return m.Q[i, j, k] <= data.Q_match_max[i, j] * m.z[i, j, k]
    m.c_Q_bigM = pyo.Constraint(m.FeasibleIJK, rule=Q_bigM_rule)

    #def disallow_shifted_inlet_matches_rule(m, i, j, k):
    #    if k == 0:
    #        return pyo.Constraint.Skip

    #    return m.z[i, j, k] <= (
    #        sum(m.z[i, jp, kp] for jp in m.Hj for kp in range(0, k) if (i, jp, kp) in m.FeasibleIJK)
    #        + sum(m.z[ip, j, kp] for ip in m.Hi for kp in range(k + 1, len(m.Hs)) if (ip, j, kp) in m.FeasibleIJK)
    #    )
    #m.c_disallow_shifted_inlet_matches_rule = pyo.Constraint(m.FeasibleIJK, rule=disallow_shifted_inlet_matches_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 8: dTmin enforcement and dT1 and dT2 
    # ════════════════════════════════════════════════════════════════
    # 8a: dT Upper bound   
    def dtmin_1_rule(m, i, j, k):
        return m.TH[i, k] - m.TC[j, k] + data.M_dt[i,j] >= data.delta_tmin + data.M_dt[i,j] * m.z[i, j, k]
    m.c_dtmin_1 = pyo.Constraint(m.FeasibleIJK, rule=dtmin_1_rule)

    def dtmin_2_rule(m, i, j, k):
        return m.TH[i, k + 1] - m.TC[j, k + 1] + data.M_dt[i,j] >= data.delta_tmin + data.M_dt[i,j] * m.z[i, j, k]
    m.c_dtmin_2 = pyo.Constraint(m.FeasibleIJK, rule=dtmin_2_rule)

    # 8b: dT1/dT2 -> TH/TC linkage
    def dT1_link_hi_rule(m, i, j, k):
        return m.dT1[i, j, k] <= m.TH[i, k] - m.TC[j, k] + data.M_dt[i,j] * (1 - m.z[i, j, k])
    m.c_dT1_link_hi = pyo.Constraint(m.FeasibleIJK, rule=dT1_link_hi_rule)

    def dT1_link_lo_rule(m, i, j, k):
        return m.dT1[i, j, k] >= m.TH[i, k] - m.TC[j, k] - data.M_dt[i,j] * (1 - m.z[i, j, k])
    m.c_dT1_link_lo = pyo.Constraint(m.FeasibleIJK, rule=dT1_link_lo_rule)

    def dT2_link_hi_rule(m, i, j, k):
        return m.dT2[i, j, k] <= m.TH[i, k + 1] - m.TC[j, k + 1] + data.M_dt[i,j] * (1 - m.z[i, j, k])
    m.c_dT2_link_hi = pyo.Constraint(m.FeasibleIJK, rule=dT2_link_hi_rule)

    def dT2_link_lo_rule(m, i, j, k):
        return m.dT2[i, j, k] >= m.TH[i, k + 1] - m.TC[j, k + 1] - data.M_dt[i,j] * (1 - m.z[i, j, k])
    m.c_dT2_link_lo = pyo.Constraint(m.FeasibleIJK, rule=dT2_link_lo_rule)

    # 8c: Upper bounds when z = 1

    def dt1_UB_rule(m, i, j, k):
        return m.dT1[i,j,k] <= data.dT1_hi[i,j] * m.z[i, j, k]
    m.c_dt1_UB = pyo.Constraint(m.FeasibleIJK, rule=dt1_UB_rule)

    def dt2_UB_rule(m, i, j, k):
        return m.dT2[i,j,k] <= data.dT2_hi[i,j] * m.z[i, j, k]
    m.c_dt2_UB = pyo.Constraint(m.FeasibleIJK, rule=dt2_UB_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 9: H = (U*LMTD)^-beta tangent-plane (supporting-hyperplane)
    # outer approximation.
    # ════════════════════════════════════════════════════════════════
    # 9a: Supporting Hyperplane Constraints (LOWER bound on H).
    # Renamed from LMTD_hyperplane_rule / m.c_LMTD_hyperplane. DIRECTION
    # FLIPPED relative to the old LMTD cuts: LMTD was concave, so its
    # tangent planes were valid global OVERestimators (LMTD <= ...). H is
    # jointly CONVEX in (dT1,dT2) instead (verified numerically), so its
    # tangent planes are valid global UNDERestimators (H >= ...), and the
    # envelope is the same max-over-planes construction, just enforced
    # from below instead of above. Reads from data.h_planes / m.H_cut_index
    # (renamed from data.lmtd_planes / m.NegBetaLnLMTD_cut_index).
    def H_hyperplane_rule(m, i, j, k, p):
        a0, a1, a2 = data.h_planes[i, j][p]
        return m.H[i, j, k] >= a0 * m.z[i, j, k] + a1 * m.dT1[i, j, k] + a2 * m.dT2[i, j, k]
    m.c_H_hyperplane = pyo.Constraint(m.H_cut_index, rule=H_hyperplane_rule)

    # 9b: No LB guardrail analog here (the old m.c_LMTD_LB is REMOVED, not
    # renamed). That guardrail existed because the old LMTD envelope only
    # bounded LMTD from above, leaving nothing to stop LMTD collapsing
    # towards its lower bound at z=1; here the H cuts already bound H from
    # BELOW directly, which is the physically-meaningful direction for a
    # convex underestimator, so no separate floor constraint is needed.
    # At z=0, dT1=dT2=0 (Block 8) makes every cut's RHS 0 (trivial H>=0),
    # and H is independently forced to 0 there by the Block 10 McCormick
    # relaxation once G=0 -- see the derivation in the chat notes.

    # ════════════════════════════════════════════════════════════════
    # BLOCK 10: G = Q^beta (1D SOS2, NonlinearIJK only) and
    # P = G*H = A^beta (single ordinary McCormick product, all matches).
    # ════════════════════════════════════════════════════════════════
    """
    Replaces the old Block 10 (W = A*lmtd McCormick + W = Q/U physical
    link) and the m.NonlinearIJK half of the old Block 11 (A -> Cost SOS2).
    G = Q^beta is now the ONLY piecewise-nonlinear link left on the Q side
    (m.LinearIJK matches skip it entirely and substitute Q for G below);
    H = (U*LMTD)^-beta is supplied whole by Block 9's tangent planes; and
    P = G*H = A^beta is produced by ONE ordinary (non-z-scaled-by-two-
    variables... it IS z-scaled, exactly like the old W McCormick) bilinear
    McCormick relaxation. Cost = cost_b*P is then already linear -- see
    the single c_Cost_eq equality that replaces ALL of the old Block 11
    (both the beta==1 fast path and the beta!=1 SOS2 path).
    """

    # ---- G = Q^beta via 1D SOS2 (NonlinearIJK only) ----
    def G_w_sum_rule(m, i, j, k):
        return sum(
            m.lam_G[i, j, k, p] for p in range(len(data.G_grid[i, j]))
        ) == m.z[i, j, k]
    m.c_G_w_sum = pyo.Constraint(m.NonlinearIJK, rule=G_w_sum_rule)

    def Q_sos_rule(m, i, j, k):
        grid = data.Q_grid_G[i, j]
        return m.Q[i, j, k] == sum(
            m.lam_G[i, j, k, p] * grid[p] for p in range(len(grid))
        )
    m.c_Q_sos = pyo.Constraint(m.NonlinearIJK, rule=Q_sos_rule)

    def G_sos_rule(m, i, j, k):
        grid = data.G_grid[i, j]
        return m.G[i, j, k] == sum(
            m.lam_G[i, j, k, p] * grid[p] for p in range(len(grid))
        )
    m.c_G_sos = pyo.Constraint(m.NonlinearIJK, rule=G_sos_rule)

    def sos2_lam_G_rule(m, i, j, k):
        return [m.lam_G[i, j, k, p] for p in range(len(data.G_grid[i, j]))]
    m.sos2_lam_G = pyo.SOSConstraint(m.NonlinearIJK, rule=sos2_lam_G_rule, sos=2)

    # ---- P = G*H McCormick (z-scaled, vanishes at z=0), all FeasibleIJK ----
    def _G_bounds(i, j):
        # (G_min, G_max) for this match -- shared across all stages k.
        return data.G_bounds[i, j]

    def _H_bounds(i, j):
        # (H_min, H_max) for this match -- shared across all stages k.
        return data.H_bounds[i, j]

    def _G_expr(m, i, j, k):
        # LinearIJK (beta==1): G = Q^1 = Q exactly -- substitute m.Q
        # directly, no separate G variable exists for these matches.
        # NonlinearIJK: use the SOS2-interpolated m.G variable above.
        if (i, j, k) in m.LinearIJK:
            return m.Q[i, j, k]
        return m.G[i, j, k]

    def p_mccormick_lo1_rule(m, i, j, k):
        G_L, G_U = _G_bounds(i, j)
        H_L, H_U = _H_bounds(i, j)
        return m.P[i, j, k] >= (
            G_L * m.H[i, j, k] + _G_expr(m, i, j, k) * H_L - G_L * H_L * m.z[i, j, k]
        )
    m.c_P_mccormick_lo1 = pyo.Constraint(m.FeasibleIJK, rule=p_mccormick_lo1_rule)

    def p_mccormick_lo2_rule(m, i, j, k):
        G_L, G_U = _G_bounds(i, j)
        H_L, H_U = _H_bounds(i, j)
        return m.P[i, j, k] >= (
            G_U * m.H[i, j, k] + _G_expr(m, i, j, k) * H_U - G_U * H_U * m.z[i, j, k]
        )
    m.c_P_mccormick_lo2 = pyo.Constraint(m.FeasibleIJK, rule=p_mccormick_lo2_rule)

    def p_mccormick_hi1_rule(m, i, j, k):
        G_L, G_U = _G_bounds(i, j)
        H_L, H_U = _H_bounds(i, j)
        return m.P[i, j, k] <= (
            G_U * m.H[i, j, k] + _G_expr(m, i, j, k) * H_L - G_U * H_L * m.z[i, j, k]
        )
    m.c_P_mccormick_hi1 = pyo.Constraint(m.FeasibleIJK, rule=p_mccormick_hi1_rule)

    def p_mccormick_hi2_rule(m, i, j, k):
        G_L, G_U = _G_bounds(i, j)
        H_L, H_U = _H_bounds(i, j)
        return m.P[i, j, k] <= (
            G_L * m.H[i, j, k] + _G_expr(m, i, j, k) * H_U - G_L * H_U * m.z[i, j, k]
        )
    m.c_P_mccormick_hi2 = pyo.Constraint(m.FeasibleIJK, rule=p_mccormick_hi2_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 11: Cost = cost_b * P.
    # Replaces BOTH halves of the old Block 11 (the m.LinearIJK
    # Cost_linear rule and the m.NonlinearIJK VarCost_sos rule) with a
    # single equality over the whole of m.FeasibleIJK: once G and H each
    # individually approximate Q^beta and (U*LMTD)^-beta, their McCormick
    # product P already equals A^beta, so Cost is linear in P for every
    # match regardless of beta -- there is no longer an A -> Cost SOS2
    # stage, linear or nonlinear, at all.
    # ════════════════════════════════════════════════════════════════
    def Cost_eq_rule(m, i, j, k):
        return m.Cost[i, j, k] == data.cost_b * m.P[i, j, k]
    m.c_Cost_eq = pyo.Constraint(m.FeasibleIJK, rule=Cost_eq_rule)

    # ════════════════════════════════════════════════════════════════
    # BLOCK 12: SOS2 constraints for utility cost
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