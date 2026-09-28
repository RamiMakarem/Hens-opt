# Formulations & Benchmarks

The mathematics behind HENS-Opt as currently implemented: the cost reformulation, the MILP linearizations (tangent-plane envelope, 1D SOS2, McCormick product), the utility model, the fixed-topology NLP, known error sources, the planned DLOG formulation, and the benchmark protocol. For how the design got here, see [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md).

---

## 1. The non-convexity

For a match between hot stream *i* and cold stream *j* in stage *k*:

$$A = \frac{Q}{U\,\text{LMTD}}, \qquad \text{LMTD} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1/\Delta T_2)}, \qquad \text{Cost} = a\,z + b\,A^{\beta}$$

where *z* is the binary existence variable and $0 < \beta \le 1$. Including this directly makes the synthesis problem a non-convex MINLP. The MILP replaces it with linear structure plus bounded error, and the NLP stage later restores the exact equations on a fixed topology.

## 2. Cost as a product of two one-variable kernels

Because $A^{\beta} = Q^{\beta}\,(U\,\text{LMTD})^{-\beta}$, define

$$G = Q^{\beta}, \qquad H(\Delta T_1,\Delta T_2) = (U\,\text{LMTD})^{-\beta}, \qquad P = G\,H = A^{\beta}, \qquad \text{Cost}_{\text{var}} = b\,P.$$

Each kernel depends on a different set of variables, so the only coupling left is the product $P = G\,H$ (§5).

## 3. The H envelope (supporting hyperplanes)

$H$ is a convex function of $(\Delta T_1,\Delta T_2)$, so tangent planes are valid **underestimators**:

$$H \;\ge\; a_0^{(p)}\,z + a_1^{(p)}\,\Delta T_1 + a_2^{(p)}\,\Delta T_2 \quad \forall p.$$

At $z = 0$ the driving forces are relaxed to zero, so every plane reduces to the trivial $H \ge 0$.

The planes are generated once, before the MILP runs: sample the $(\Delta T_1,\Delta T_2)$ domain with a uniform grid plus random points, evaluate tangent planes, measure error against the true $H$, and add planes where error is largest until the maximum relative error on a dense validation set falls below the target (1% by default). A warning is issued if a match still exceeds the target.

**Limitation.** Error is validated on a test grid, and the envelope is not tight at every point. Because cost increases with $H$, the optimizer is pushed toward wherever the envelope is loosest.

## 4. G = Q^β (1D SOS2)

For matches with $\beta \ne 1$, $G$ is a 1D SOS2 piecewise-linear interpolation of $Q^\beta$ over a grid of breakpoints (`N_G_process` of them):

$$\sum_p \lambda_p = z, \qquad Q = \sum_p \lambda_p\,Q_p, \qquad G = \sum_p \lambda_p\,G_p, \qquad \{\lambda_p\} \text{ SOS2}.$$

Matches with $\beta = 1$ use $G = Q$ directly, with no breakpoints. Interpolation error is bounded by the grid resolution.

## 5. P = G·H (McCormick product)

The bilinear product is relaxed with four McCormick inequalities, scaled by $z$ so that they vanish when the match is not selected. With per-match bounds $G \in [G_L, G_U]$ and $H \in [H_L, H_U]$:

$$\begin{aligned}
P &\ge G_L H + G H_L - G_L H_L z, & P &\ge G_U H + G H_U - G_U H_U z,\\
P &\le G_U H + G H_L - G_U H_L z, & P &\le G_L H + G H_U - G_L H_U z.
\end{aligned}$$

Cost is then linear: $\text{Cost}_{\text{var}} = b\,P$, for every $\beta$.

**Error source.** McCormick envelopes bound $P$ from both sides but do not equal $G\,H$ except at the corners of the $(G,H)$ box. Since cost is minimized, $P$ tends to sit at the lower envelope, which can underestimate $A^\beta$. The result is that the MILP's own objective can differ noticeably from the true cost of the network it selects. That is by design tolerated here: the NLP stage re-evaluates every candidate with the exact equations, and this formulation currently produces the best final networks.

## 6. Utilities

Utility exchangers sit at a stream's terminal end, where all but one driving force are fixed by problem data. The free driving force is written directly from duty, e.g. for a hot utility

$$\Delta T_{2,HU} = T^{\text{return}}_{HU} - T^{\text{target}}_{C,j} + Q_{HU}/CP_{C,j}.$$

Utility cost is then a single 1D SOS2 mapping $Q \to \text{Cost}$ per utility assignment, with weights summing to the assignment binary.

**Phase change.** For combined utilities (for example, superheated steam that desuperheats, then condenses), extra "elbow" constraints keep the sensible-heat section from violating $\Delta T_{\min}$ against the process stream before the phase-change temperature.

## 7. Objective

$$\text{TAC} = \sum_{u}\!\text{c}^{\text{hot}}_u Q^{\text{HU}} + \sum_{v}\!\text{c}^{\text{cold}}_v Q^{\text{CU}} + \sum_{ijk}\text{Cost}_{ijk} + a\!\sum_{ijk} z_{ijk} + \sum \text{Cost}_{\text{util}} + a\!\sum y_{\text{util}} + \varepsilon\!\sum k\,z_{ijk}$$

The final term ($\varepsilon = 0.01$) is a small stage penalty that breaks symmetry among equivalent stage placements.

## 8. Two-stage pipeline

1. **Solve the MILP with SCIP** and collect a solution pool of distinct feasible topologies via PySCIPOpt (falls back to the single incumbent if PySCIPOpt is missing).
2. **Fix each topology** (active matches and utility assignments only) and solve an NLP in IPOPT:
   - variables: node temperatures, duties, driving forces, LMTD, areas, cost;
   - $\Delta T_{\min}$ enforced at both ends of every process exchanger;
   - inside the NLP, LMTD uses the smooth Chen (1987) approximation $[\Delta T_1 \Delta T_2 (\Delta T_1+\Delta T_2)/2]^{1/3}$; the exact log-mean is used for reporting;
   - warm-started from error-free values (exact LMTD/area recomputed from the MILP's temperatures, not its interpolated values);
   - fixed charges are constant on a fixed topology, so they don't enter the NLP objective.
3. **Select** the candidate with the lowest true TAC.

Reason for the pool: a MILP whose objective is inexact will not always rank topologies correctly. Evaluating many with the exact equations is more robust than trusting the single MILP-optimal one.

## 9. Planned: 2D DLOG formulation and HiGHS

The McCormick error in §5 and the reliance on SOS2 (§4, §6) are the two things the next formulation targets.

- **2D DLOG over (G, H).** Replace the McCormick product and the 1D SOS2 on $G$ with a single 2D disaggregated-logarithmic triangulated interpolation. The $G$ and $H$ axes are split into grid cells; each cell has its own four corner weights summing to $z$; cell selection uses Gray-coded binaries ($\lceil\log_2 K\rceil$ per axis) plus one triangle-selection bit; and $Q$, $G$, $H$ and $\text{Cost}$ are all the same weighted combination of grid values. This removes the McCormick relaxation entirely, leaving only bounded interpolation error, and needs only logarithmically many binaries.
- **SOS2 → DLOG for utilities too.** Converting the remaining SOS2 sets removes the dependence on native SOS2, which is what allows a switch from SCIP to HiGHS. The solution-pool step currently relies on SCIP/PySCIPOpt and would need an equivalent.
- **Status.** The DLOG version is implemented and under evaluation, but the McCormick version currently produces better final networks on the benchmark and remains the production formulation.

## 10. Benchmarks

### Linnhoff & Ahmad (1990) 9-stream (5H / 4C)

See the table in the [README](README.md#benchmark-linnhoff--ahmad-1990-9-stream-problem-5-hot-4-cold). Comparators: Huber's doctoral thesis and Wu et al. Results are reported before the MILP has converged, so they are a lower bound on what the method can achieve. Parameter checklist for a fair comparison:

| Item | Match required |
|---|---|
| Stream data (T, CP or F·cp) | exact |
| ΔTmin / HRAT / EMAT | same value and same meaning |
| Utility temperatures | fixed vs free treatment stated |
| U values (film coefficients) | same |
| Utility costs | same |
| Capital cost law (a, b, β) | same |
| Annualization and hours | same |

### Verification of the reported network

Every headline result should come with an independent check of the final network:
- each stream reaches its target temperature and duties balance;
- $\Delta T \ge \Delta T_{\min}$ at both ends of every process **and utility** exchanger, evaluated with the exact LMTD;
- TAC recomputed from final areas with the source's cost law.

### Limits of what these results show

The reported TAC is the best topology found and refined, not a certified global optimum. Gaps and runtimes from different solvers (commercial global MINLP vs. open-source MILP + local NLP) are not directly comparable.

## 11. Solver trade-off (SCIP vs. Gurobi vs. HiGHS)

SCIP is used for reproducibility (no license required) and for native SOS2 and solution-pool support, but it is slower on this MILP than commercial solvers. HiGHS is an alternative open-source target once SOS2 is removed. A same-formulation comparison across solvers is on the roadmap.
