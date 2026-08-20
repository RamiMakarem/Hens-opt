import pyomo.environ as pyo

def build_variables_nlp(m, data):
    """
    Fixed-topology NLP variable block.

    Assumes `data` carries the *filtered* index sets coming out of the MILP
    solution (z > 0.5, yHU > 0.5, yCU > 0.5), exposed as plain Python
    iterables of tuples:

        data.ActiveIJK  -> [(i, j, k), ...]   matches with z[i,j,k] > 0.5
        data.ActiveHU   -> [(u, j), ...]      hot-utility assignments yHU > 0.5
        data.ActiveCU   -> [(v, i), ...]      cold-utility assignments yCU > 0.5

    Hi, Hj, Hs, Knodes, HU, CU are still the full MILP sets (streams/stages
    don't disappear, only match/assignment *combinations* do).
    """
    Hi, Hj, Hs = m.Hi, m.Hj, m.Hs

    # ── Active index sets (replace z / yHU / yCU) ───────────────────────
    m.ActiveIJK = pyo.Set(dimen=3, initialize=data.ActiveIJK)
    m.ActiveHU = pyo.Set(dimen=2, initialize=data.ActiveHU)
    m.ActiveCU = pyo.Set(dimen=2, initialize=data.ActiveCU)

    # Convenience: which (i,j) pairs are active in *some* stage, and which
    # i / j actually participate in any active match or utility at all.
    m.ActiveIJ = pyo.Set(
        dimen=2,
        initialize=sorted({(i, j) for (i, j, k) in data.ActiveIJK}),)

    # ── Temperatures ─────────────────────────────────────────────────────
    def TH_bounds(m, i, k):
        return (data.Tout_H[i], data.Tin_H[i])
    m.TH = pyo.Var(Hi, m.Knodes, bounds=TH_bounds, domain=pyo.Reals)

    def TC_bounds(m, j, k):
        return (data.Tin_C[j], data.Tout_C[j])
    m.TC = pyo.Var(Hj, m.Knodes, bounds=TC_bounds, domain=pyo.Reals)

    # ── Heat exchanger duties, area, cost, LMTD (active matches only) ───
    def Q_bounds(m, i, j, k):
        return (0, data.Q_match_max[i, j])
    m.Q = pyo.Var(m.ActiveIJK, bounds=Q_bounds, domain=pyo.NonNegativeReals)

    # Kept as free Vars per your call (needed for utility elbow/phase-change
    # constraints downstream).
    m.QH = pyo.Var(Hj, bounds=(0, None), domain=pyo.NonNegativeReals)
    m.QC = pyo.Var(Hi, bounds=(0, None), domain=pyo.NonNegativeReals)

    def A_bounds(m, i, j, k):
        return (0, data.A_max[i, j])
    m.A = pyo.Var(m.ActiveIJK, bounds=A_bounds, domain=pyo.NonNegativeReals)

    m.Cost = pyo.Var(m.ActiveIJK, domain=pyo.NonNegativeReals)

    def dT_bounds(m, i, j, k):
        return (0, data.Gamma)
    m.dT1 = pyo.Var(m.ActiveIJK, bounds=dT_bounds, domain=pyo.Reals)
    m.dT2 = pyo.Var(m.ActiveIJK, bounds=dT_bounds, domain=pyo.Reals)

    def LMTDv_bounds(m, i, j, k):
        return (0, max(data.dT1_hi[i, j], data.dT2_hi[i, j]))
    m.LMTDv = pyo.Var(m.ActiveIJK, bounds=LMTDv_bounds, domain=pyo.NonNegativeReals)

    # ── Utility duties, area, cost, LMTD (active assignments only) ──────
    def QHU_bounds(m, u, j):
        Q_pk = data.hot_Q_per_kg[u]
        Q_limit = data.hot_max_flow[u] * Q_pk
        return (0, min(Q_limit, data.Q_C_total[j] + Q_pk))
    m.QHU = pyo.Var(m.ActiveHU, bounds=QHU_bounds, domain=pyo.NonNegativeReals)

    def QCU_bounds(m, v, i):
        Q_pk = data.cold_Q_per_kg[v]
        Q_limit = data.cold_max_flow[v] * Q_pk
        return (0, min(Q_limit, data.Q_H_total[i] + Q_pk))
    m.QCU = pyo.Var(m.ActiveCU, bounds=QCU_bounds, domain=pyo.NonNegativeReals)

    def A_HU_bounds(m, u, j):
        return (0, data.A_max_hu[u, j])
    m.A_HU = pyo.Var(m.ActiveHU, bounds=A_HU_bounds, domain=pyo.NonNegativeReals)
    m.Cost_HU = pyo.Var(m.ActiveHU, domain=pyo.NonNegativeReals)

    def dT2_HU_bounds(m, u, j):
        return (0, data.M_dT2_HU[u, j])
    m.dT2_HU = pyo.Var(m.ActiveHU, bounds=dT2_HU_bounds, domain=pyo.Reals)

    def LMTDv_HU_bounds(m, u, j):
        if hasattr(data, "dT1_HU_hi") and hasattr(data, "dT2_HU_hi"):
            return (0, max(data.dT1_HU_hi[u, j], data.dT2_HU_hi[u, j]))
        return (0, data.Gamma)
    m.LMTDv_HU = pyo.Var(m.ActiveHU, bounds=LMTDv_HU_bounds, domain=pyo.NonNegativeReals)

    def A_CU_bounds(m, v, i):
        return (0, data.A_max_cu[v, i])
    m.A_CU = pyo.Var(m.ActiveCU, bounds=A_CU_bounds, domain=pyo.NonNegativeReals)
    m.Cost_CU = pyo.Var(m.ActiveCU, domain=pyo.NonNegativeReals)

    def dT1_CU_bounds(m, v, i):
        return (0, data.M_dT1_CU[v, i])

    m.dT1_CU = pyo.Var(m.ActiveCU, bounds=dT1_CU_bounds, domain=pyo.Reals)


    def LMTDv_CU_bounds(m, v, i):
        if hasattr(data, "dT1_CU_hi") and hasattr(data, "dT2_CU_hi"):
            return (0, max(data.dT1_CU_hi[v, i], data.dT2_CU_hi[v, i]))
        return (0, data.Gamma)
    m.LMTDv_CU = pyo.Var(m.ActiveCU, bounds=LMTDv_CU_bounds, domain=pyo.NonNegativeReals)


    '''
    def dT1_HU_bounds(m, u, j):
        return (0, data.M_dT1_HU[u, j])
    m.dT1_HU = pyo.Var(m.ActiveHU, bounds=dT1_HU_bounds, domain=pyo.Reals)
    def dT2_CU_bounds(m, v, i):
        return (0, data.M_dT2_CU[v, i])
    m.dT2_CU = pyo.Var(m.ActiveCU, bounds=dT2_CU_bounds, domain=pyo.Reals)
    '''
    return m