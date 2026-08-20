import pyomo.environ as pyo
import math


def build_variables(m, data):
    Hi, Hj, Hs = m.Hi, m.Hj, m.Hs

    # ── Temperatures & Heat Exchangers ──────────────────────────────────
    def TH_bounds(m, i, k):
        return (data.Tout_H[i], data.Tin_H[i])
    m.TH = pyo.Var(Hi, m.Knodes, bounds=TH_bounds, domain=pyo.Reals)

    def TC_bounds(m, j, k):
        return (data.Tin_C[j], data.Tout_C[j])
    m.TC = pyo.Var(Hj, m.Knodes, bounds=TC_bounds, domain=pyo.Reals)

    def Q_bounds(m, i, j, k):
        return (0, data.Q_match_max[i, j])
    m.Q = pyo.Var(Hi, Hj, Hs, bounds=Q_bounds, domain=pyo.NonNegativeReals)

    def QH_bounds(m, j):
        return (0, data.Q_C_total[j])
    m.QH = pyo.Var(Hj, bounds=QH_bounds, domain=pyo.NonNegativeReals)

    def QC_bounds(m, i):
        return (0, data.Q_H_total[i])
    m.QC = pyo.Var(Hi, bounds=QC_bounds, domain=pyo.NonNegativeReals)

    m.z = pyo.Var(Hi, Hj, Hs, domain=pyo.Binary)

    for i in Hi:
        for j in Hj:
            if not data.feasible_match[i, j]:
                for k in Hs:
                    m.z[i, j, k].fix(0)

    def dT_bounds(m, i, j, k):
        return (data.dT1_lo[i, j], data.dT1_hi[i, j])
    m.dT1 = pyo.Var(Hi, Hj, Hs, bounds=dT_bounds, domain=pyo.Reals)
    m.dT2 = pyo.Var(Hi, Hj, Hs, bounds=dT_bounds, domain=pyo.Reals)

    def LMTD_bounds(m, i, j, k):
        return (0.0, data.dT1_hi[i, j])
    m.LMTD = pyo.Var(Hi, Hj, Hs, bounds=LMTD_bounds, domain=pyo.NonNegativeReals)

    m.Cost = pyo.Var(Hi, Hj, Hs, domain=pyo.NonNegativeReals)

    # ── Utilities ────────────────────────────────────────────────────────
    m.yHU = pyo.Var(m.HU, m.Hj, domain=pyo.Binary)
    m.yCU = pyo.Var(m.CU, m.Hi, domain=pyo.Binary)
    m.QHU = pyo.Var(m.HU, m.Hj, domain=pyo.NonNegativeReals)
    m.QCU = pyo.Var(m.CU, m.Hi, domain=pyo.NonNegativeReals)

    for u in m.HU:
        Q_pk = data.hot_Q_per_kg[u]
        for j in m.Hj:
            Q_limit = data.hot_max_flow[u] * Q_pk
            Q_max = min(Q_limit, data.Q_C_total[j] + Q_pk)
            m.QHU[u, j].setub(Q_max)
            if not data.feasible_hu[u, j]:
                m.yHU[u, j].fix(0)
                m.QHU[u, j].fix(0.0)

    for v in m.CU:
        Q_pk = data.cold_Q_per_kg[v]
        for i in m.Hi:
            Q_limit = data.cold_max_flow[v] * Q_pk
            Q_max = min(Q_limit, data.Q_H_total[i] + Q_pk)
            m.QCU[v, i].setub(Q_max)
            if not data.feasible_cu[v, i]:
                m.yCU[v, i].fix(0)
                m.QCU[v, i].fix(0.0)

    m.dT2_HU = pyo.Expression(m.FeasibleHU,
        rule=lambda m, u, j: data.T_hu_out_eff[u] - data.Tout_C[j]
        + m.QHU[u, j] / data.CP_C[j])
    m.dT1_CU = pyo.Expression(m.FeasibleCU,
        rule=lambda m, v, i: data.Tout_H[i] + m.QCU[v, i] / data.CP_H[i]
        - data.T_cu_out_eff[v])

    m.Cost_HU = pyo.Var(m.HU, m.Hj, domain=pyo.NonNegativeReals)
    m.Cost_CU = pyo.Var(m.CU, m.Hi, domain=pyo.NonNegativeReals)

    for (u, j) in m.FeasibleHU:
        m.Cost_HU[u, j].setub(m.Cbp_HU[u, j, m.GU0.last()])
    for (v, i) in m.FeasibleCU:
        m.Cost_CU[v, i].setub(m.Cbp_CU[v, i, m.GU0.last()])

    m.lam_hu = pyo.Var(m.FeasibleHU, m.GU0, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m.lam_cu = pyo.Var(m.FeasibleCU, m.GU0, domain=pyo.NonNegativeReals, bounds=(0, 1))

# ── Hyperplane (A^beta) Formulation Variables ───────────────────────────

    # Physical A^beta term = (Q / (U * LMTD))^beta
    m.A_beta = pyo.Var(m.FeasibleIJK, domain=pyo.NonNegativeReals)

    # Active hyperplane indicator
    m.z_plane = pyo.Var(m.FeasibleIJKP, domain=pyo.Binary)

    # Upper bounds for purely variable area cost and A_beta
    for (i, j, k) in m.FeasibleIJK:
        q_max = data.Q_match_max[i, j]
        l_min = max(data.dT1_lo[i, j], 1e-3)
        u_val = data.U_mat[i,j]
        beta_val = data.cost_beta
        c_area = data.cost_b  # cost_b

        max_A_beta = (q_max / (u_val * l_min)) ** beta_val
        m.A_beta[i, j, k].setub(max_A_beta)

        # Strictly variable cost upper bound (c_area * max_A_beta)
        m.Cost[i, j, k].setub(c_area * max_A_beta)
    return m