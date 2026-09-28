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

    def Q_bounds_rule(m, i, j, k):
        lo,hi=data.Q_bounds[i, j]
        return (0,hi)
    m.Q = pyo.Var(Hi, Hj, Hs, bounds=Q_bounds_rule, domain=pyo.NonNegativeReals)

    def QH_bounds(m, j):
        return (0, data.Q_C_total[j])
    m.QH = pyo.Var(Hj, bounds=QH_bounds, domain=pyo.NonNegativeReals)

    def QC_bounds(m, i):
        return (0, data.Q_H_total[i])
    m.QC = pyo.Var(Hi, bounds=QC_bounds, domain=pyo.NonNegativeReals)

    m.z = pyo.Var(m.FeasibleIJK, domain=pyo.Binary)


    def dT_bounds(m, i, j, k):
        return (0, data.dT1_hi[i, j])
    m.dT1 = pyo.Var(m.FeasibleIJK, bounds=dT_bounds, domain=pyo.NonNegativeReals)
    m.dT2 = pyo.Var(m.FeasibleIJK, bounds=dT_bounds, domain=pyo.NonNegativeReals)

    # ── H = (U*LMTD)^-beta (convex tangent-plane envelope) ───────────────
    # Replaces the old m.lmtd variable. H is bounded above by H_bounds[i,j]
    # (achieved at lmtd_min); the lower bound is hard-coded to 0.0 rather
    # than H_bounds' own H_min, exactly like the old lmtd/A pattern, so H
    # is free to collapse to 0 when z=0 (see the McCormick zero-forcing
    # argument for m.P below -- no separate LB guardrail is needed here,
    # unlike the old m.c_LMTD_LB, because the tangent planes underestimate
    # rather than overestimate).
    def H_bounds_rule(m, i, j, k):
        lo, hi = data.H_bounds[i, j]
        return (0.0, hi)
    m.H = pyo.Var(m.FeasibleIJK, bounds=H_bounds_rule, domain=pyo.NonNegativeReals)

    # ── G = Q^beta (1D SOS2), NonlinearIJK only. LinearIJK (beta==1)
    # matches have no G variable at all -- Constraints.py substitutes
    # m.Q directly for G in the G*H McCormick product for those matches
    # (data.G_grid[i,j] is empty there, matching m.G_Pt being empty too).
    def G_bounds_rule(m, i, j, k):
        lo, hi = data.G_bounds[i, j]
        return (0.0, hi)
    m.G = pyo.Var(m.NonlinearIJK, bounds=G_bounds_rule, domain=pyo.NonNegativeReals)

    # ── SOS2 continuous weight variables for G = Q^beta ──────────────────
    m.lam_G = pyo.Var(m.G_Pt, domain=pyo.NonNegativeReals, bounds=(0, 1))

    # ── P = G*H = A^beta (single ordinary McCormick product) ────────────
    # Replaces the old m.A and m.W bilinear-product machinery. There is no
    # separate area variable anymore: A^beta is produced directly as the
    # McCormick product of G and H, and Cost = cost_b*P is linear in P
    # once G and H exist (see Constraints.py / Objective.py).
    def P_bounds_rule(m, i, j, k):
        lo, hi = data.P_bounds[i, j]
        return (0.0, hi)
    m.P = pyo.Var(m.FeasibleIJK, bounds=P_bounds_rule, domain=pyo.NonNegativeReals)

    def VarCost_bounds(m, i, j, k):
        lo, hi = data.VarCost_bounds[i, j]
        return (0.0, hi)
    m.Cost = pyo.Var(m.FeasibleIJK, bounds=VarCost_bounds, domain=pyo.NonNegativeReals)

    # ── Utilities ────────────────────────────────────────────────────────
    m.yHU = pyo.Var(m.FeasibleHU, domain=pyo.Binary)
    m.yCU = pyo.Var(m.FeasibleCU, domain=pyo.Binary)
    m.QHU = pyo.Var(m.FeasibleHU, domain=pyo.NonNegativeReals)
    m.QCU = pyo.Var(m.FeasibleCU, domain=pyo.NonNegativeReals)

    m.dT2_HU = pyo.Expression(m.FeasibleHU,
        rule=lambda m, u, j: data.T_hu_out_eff[u] - data.Tout_C[j]
        + m.QHU[u, j] / data.CP_C[j])
    m.dT1_CU = pyo.Expression(m.FeasibleCU,
        rule=lambda m, v, i: data.Tout_H[i] + m.QCU[v, i] / data.CP_H[i]
        - data.T_cu_out_eff[v])

    m.Cost_HU = pyo.Var(m.FeasibleHU, domain=pyo.NonNegativeReals)
    m.Cost_CU = pyo.Var(m.FeasibleCU, domain=pyo.NonNegativeReals)

    for (u, j) in m.FeasibleHU:
        m.Cost_HU[u, j].setub(m.Cbp_HU[u, j, m.GU0.last()])
    for (v, i) in m.FeasibleCU:
        m.Cost_CU[v, i].setub(m.Cbp_CU[v, i, m.GU0.last()])

    m.lam_hu = pyo.Var(m.FeasibleHU, m.GU0, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m.lam_cu = pyo.Var(m.FeasibleCU, m.GU0, domain=pyo.NonNegativeReals, bounds=(0, 1))

    return m