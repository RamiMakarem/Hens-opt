import pyomo.environ as pyo


def build_objective(m, data):
    Hi, Hj = m.Hi, m.Hj

    def obj_rule(m):
        # 1. Hot Utility Operating Cost (OPEX)
        # m.HU = RangeSet(0, n_HU-1) is 0-based -- index hu_opex directly.
        hu_opex_cost = sum(data.hu_opex[u] * m.QHU[u, j] for u in m.HU for j in Hj)

        # 2. Cold Utility Operating Cost (OPEX)
        cu_opex_cost = sum(data.cu_opex[v] * m.QCU[v, i] for v in m.CU for i in Hi)

        # 3. Process-Process Variable Capital Cost (Area CAPEX)
        var_capex = sum(m.Cost[i, j, k] for (i, j, k) in m.FeasibleIJK)

        # 4. Process-Process Variable Capital Cost (Area CAPEX)
        fixed_capex = sum(data.cost_a * m.z[i,j,k] for (i, j, k) in m.FeasibleIJK)

        # 5. Hot Utility Operating Cost (CAPEX)
        hu_capex_cost = sum(m.Cost_HU[u,j] for u in m.HU for j in Hj)

        # 6. Hot Utility Operating Cost (CAPEX)
        cu_capex_cost = sum(m.Cost_CU[v,i] for v in m.CU for i in Hi)

        # 7. Hot Utility Operating Cost (FIX_CAPEX)
        hu_fix_capex_cost = sum(m.yHU[u,j] * data.cost_a for u in m.HU for j in Hj)

        # 8. Hot Utility Operating Cost (FIX_CAPEX)
        cu_fix_capex_cost = sum(m.yCU[v,i] * data.cost_a  for v in m.CU for i in Hi)
        return hu_opex_cost + cu_opex_cost + var_capex + hu_capex_cost + cu_capex_cost + fixed_capex + hu_fix_capex_cost +cu_fix_capex_cost

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return m