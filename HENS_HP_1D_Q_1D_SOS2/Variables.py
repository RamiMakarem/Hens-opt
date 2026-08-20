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

    def NegBetaLnLMTD_bounds(m, i, j, k):
            cb = data.cost_beta[i, j] if isinstance(data.cost_beta, dict) else data.cost_beta
            lo = -cb * math.log(data.dT1_hi[i, j]) 
            hi = -cb * math.log(data.dT1_lo[i, j])
            return (lo, hi)
    m.NegBetaLnLMTD = pyo.Var(Hi, Hj, Hs, bounds=NegBetaLnLMTD_bounds, domain=pyo.Reals)

    m.Cost = pyo.Var(Hi, Hj, Hs, domain=pyo.NonNegativeReals)

    # ── (2) Q -> ln(Q): plain tangent-line envelope, no extra binaries ──

    def LnQ_bounds(m, i, j, k):
        return (math.log(data.Q_lo_lnq[i, j]), math.log(data.Q_hi_lnq[i, j]))
    m.LnQ = pyo.Var(Hi, Hj, Hs, bounds=LnQ_bounds, domain=pyo.Reals)

    # ── (3) A_beta = cost_beta*LnQ - cost_beta*ln(U) + NegBetaLnLMTD ────
    def Abeta_bounds(m, i, j, k):
        grid = data.Abeta_grid[i, j]
        return (float(grid.min()), float(grid.max()))
    m.A_beta = pyo.Var(Hi, Hj, Hs, bounds=Abeta_bounds, domain=pyo.Reals)

    # ── (4) A_beta -> Cost: 1D SOS2 convex-combination weights ──────────
    m.w_Abeta = pyo.Var(m.Abeta_Pt, domain=pyo.NonNegativeReals, bounds=(0, 1))
    for (i, j, k) in m.FeasibleIJK:
        m.Cost[i, j, k].setub(float(data.Abeta_cost_grid[i, j].max()))

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

    return m