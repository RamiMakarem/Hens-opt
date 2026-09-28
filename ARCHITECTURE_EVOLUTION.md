# Architecture Evolution: Engineering Decision Log

How the HENS-Opt solver architecture reached its current form, including approaches that were tried and deliberately discarded. The dead ends are kept because the reasoning behind each pivot is more informative than a changelog of what "just worked." Current-state math is in [FORMULATIONS_AND_BENCHMARKS.md](FORMULATIONS_AND_BENCHMARKS.md).

---

## Phase 1: Flattened 1D vectors with Taylor/tangent OA (discarded)

**Approach.** Dynamic outer approximation with first-order Taylor series for $(Q,\text{LMTD}) \to A$ and tangent lines for $A \to \text{Cost}$, with the whole problem flattened into 1D compressed vectors for SciPy's optimization layer.

**Why dropped.** The flattening made the model unreadable and hard to change (every topology change meant re-deriving index offsets), and the linearization was too coarse. **Lesson:** validate the mathematics in a readable form before optimizing data structures.

## Phase 2: Manual SOS2 adjacency (discarded)

**Approach.** Piecewise-linear SOS2 with adjacency encoded by hand using binaries, since the solver setup then in use lacked native SOS2. Julia/JuMP was also evaluated.

**Why dropped.** Hand-coded adjacency gives a weak relaxation and, combined with inline OA re-solves, was too slow even on small cases. JuMP was set aside for maintainability (one language for solver core and UI), not technical reasons. **Lesson:** use a solver's native structures instead of hand-rolling them.

## Phase 3: Pyomo + SCIP with 6× SOS2 per match (superseded)

**Approach.** Pyomo with SCIP's native SOS2. Each match used six 1D SOS2 transformations: $\Delta T_1$, $\Delta T_2$, their difference, the difference of their logs, $Q$, and area to cost.

**Why superseded.** Correct, but six sets of breakpoint binaries per match made branch-and-bound grow too fast beyond small networks.

## Phase 4: Utility match simplification (adopted)

**Insight.** Utility exchangers occur only at a stream's terminal end, so three of the four temperatures are fixed by problem data and only one driving force is free. That free $\Delta T$ is an explicit function of duty, so the whole chain collapses to a single 1D SOS2 map $Q \to \text{Cost}$ per utility assignment, about a 6× reduction in binaries for utility matches. Still in use.

## Phase 5: Native 2D SOS2 (discarded)

**Approach.** Full 2D SOS2 grids for $(\Delta T_1,\Delta T_2)\to\text{LMTD}$ and $(\text{LMTD},Q)\to A$.

**Why dropped.** SCIP has no specialized branching for triangulated 2D piecewise-linear structures, so this degraded to brute-force branching and even small cases stalled. **Lesson:** choose formulations together with solver capability, not in the abstract.

## Phase 6: Pre-solve hyperplanes in log space (adopted, later replaced)

**Insight.** The recurring cost in earlier phases was linearizing *dynamically*, with repeated re-solves. Instead, an automated pre-solve step samples the driving-force domain (uniform grid plus random points), adds tangent hyperplanes where error is largest until it drops below tolerance, and hands SCIP one static model.

At this stage the model bounded $-\beta\ln(\text{LMTD})$ with hyperplanes, then exponentiated through more piecewise structure.

**Why replaced.** Exponentiating a small absolute envelope gap produced large *relative* cost errors, and because the objective minimizes cost, the optimizer steered toward exactly where the envelope was loosest. Reported MILP costs sat far below the true cost of the same network. An outer-approximation refinement loop (adding a tangent cut at each incumbent's own $(\Delta T_1,\Delta T_2)$ and re-solving) was built to close this gap, then retired with the formulation it was written for.

## Phase 7: Multiplicative kernel $b\,G\,H$ with McCormick (adopted, current production)

**Approach.** Rewrite cost as $b\,G\,H$ with $G=Q^\beta$ and $H=(U\,\text{LMTD})^{-\beta}$. $H$ is convex, so tangent planes are valid underestimators. $G$ uses a 1D SOS2 interpolation, and $P = G\,H$ is a $z$-scaled McCormick product. Cost is then linear in $P$ for every $\beta$, which removes the separate area variable and the area-to-cost SOS2 stage.

**Why.** This works in the original (not log) space, so there is no exponentiation blow-up, and it needs far fewer binaries than the 6× SOS2 scheme.

**Known trade-off.** The McCormick envelope leaves internal linearization error, so the MILP's own objective is a poor predictor of the true cost of the network it selects. The NLP re-evaluation (Phase 8) absorbs this, and this formulation currently gives the best final networks.

**Why not just hyperplane the cost surface too?** A dual-hyperplane scheme (hyperplanes for LMTD, then for $\text{Cost}(Q,L)$) would be the cheapest to solve, but $\text{Cost}(Q,L) = b\,(Q/(UL))^{c}$ is not convex. Its Hessian has

$$f_{QQ} = bc(c-1)U^{-c}Q^{c-2}L^{-c}, \quad f_{LL} = bc(c+1)U^{-c}Q^{c}L^{-c-2}, \quad f_{QL} = -bc^2U^{-c}Q^{c-1}L^{-c-1},$$

so

$$\det \nabla^2 f = f_{QQ}f_{LL} - f_{QL}^2 = -\,b^2c^2\,U^{-2c}\,Q^{2c-2}\,L^{-2c-2} \;<\; 0$$

for every $c \neq 0$. A negative determinant means the Hessian is indefinite everywhere, so tangent planes are not valid underestimators and can cut off feasible or optimal solutions.

## Phase 8: Solution pool + fixed-topology NLP re-ranking (adopted, current)

**Insight.** Even with a tighter MILP, its objective is inexact, so the MILP-optimal topology is not reliably the true-TAC-best. Rather than trust one answer or re-solve N times with no-good cuts, HENS-Opt collects a **pool of distinct feasible topologies from a single SCIP run**, refines each in an IPOPT NLP (warm-started from error-free values), and picks the lowest true TAC.

## Phase 9: 2D DLOG and SOS2 → DLOG for HiGHS (in progress)

**Goal.** Replace the McCormick product and the 1D SOS2 on $G$ with a single 2D disaggregated-logarithmic triangulated interpolation over $(G,H)$, using Gray-coded cell selection (logarithmic in grid size) plus one triangle bit. That removes the McCormick relaxation error. Converting the remaining SOS2 sets (utility cost curves) to DLOG as well removes the need for native SOS2, which is the prerequisite for moving from SCIP to HiGHS.

**Status.** Implemented and under evaluation. It is not yet the production formulation because the McCormick version currently produces better final networks on the benchmark; the DLOG version is being tuned toward MILP convergence. Also in progress: a shell-and-tube sizing module whose results would feed refined heat-transfer coefficients back into the optimizer.

---

## Summary

| Phase | Core idea | Outcome |
|---|---|---|
| 1 | Flattened vectors + Taylor/tangent OA | Discarded: unmaintainable, inaccurate |
| 2 | Manual SOS2 adjacency + Julia evaluation | Discarded: weak relaxation, latency |
| 3 | Pyomo + SCIP native SOS2 (6× per match) | Superseded: binary-heavy |
| 4 | Utility reduction (6× → 1×) | **Adopted** |
| 5 | Native 2D SOS2 | Discarded: no SCIP branching support |
| 6 | Pre-solve log-space hyperplanes (+ OA loop) | Replaced: relative-error blow-up |
| 7 | $b\,G\,H$ kernel: H planes + 1D SOS2 + McCormick | **Adopted (current production)** |
| 8 | Solution pool → NLP re-ranking | **Adopted (current)** |
| 9 | 2D DLOG, SOS2 → DLOG, HiGHS | In progress |

Every pivot was driven by measured performance, accuracy, or maintainability failure rather than starting over for its own sake.
