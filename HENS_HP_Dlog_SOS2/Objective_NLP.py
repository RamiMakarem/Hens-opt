import pyomo.environ as pyo

def build_objective_nlp(m, data):
    Hi, Hj = m.Hi, m.Hj

    def obj_rule(m):

        # 1. Hot Utility Operating Cost (OPEX) -- only active (u, j) pairs
        hu_opex_cost = sum(data.hu_opex[u] * m.QHU[u, j] for (u, j) in m.ActiveHU)

        # 2. Cold Utility Operating Cost (OPEX)
        cu_opex_cost = sum(data.cu_opex[v] * m.QCU[v, i] for (v, i) in m.ActiveCU)

        # 3. Process-Process Variable Capital Cost (Area CAPEX)
        var_capex = sum(m.Cost[i, j, k] for (i, j, k) in m.ActiveIJK)

        # 4. Process-Process Fixed Capital Cost 
        #fixed_capex = data.cost_a * len(m.ActiveIJK)

        # 5. Hot Utility Variable Capital Cost (Area CAPEX)
        hu_capex_cost = sum(m.Cost_HU[u, j] for (u, j) in m.ActiveHU)

        # 6. Cold Utility Variable Capital Cost (Area CAPEX)
        cu_capex_cost = sum(m.Cost_CU[v, i] for (v, i) in m.ActiveCU)

        # 7. Hot Utility Fixed Capital Cost (yHU fixed at 1 -> constant)
        #hu_fix_capex_cost = data.cost_a * len(m.ActiveHU)

        # 8. Cold Utility Fixed Capital Cost (yCU fixed at 1 -> constant)
        #cu_fix_capex_cost = data.cost_a * len(m.ActiveCU)

        return (hu_opex_cost
            + cu_opex_cost
            + var_capex
            + hu_capex_cost
            + cu_capex_cost)

    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)
    return m