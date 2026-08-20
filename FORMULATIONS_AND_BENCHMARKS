# Formulations & Benchmarks

Mathematical detail behind the HENS-Opt MILP linearization layer: the four candidate formulations currently under benchmark, the non-convexity issue that rules one of them out for production use, the pre-solve hyperplane sampling algorithm, and the utility-match reduction that collapses a 6-variable SOS2 system down to one.

---

## 1. Why linearization is necessary

For a process-to-process match between hot stream *i* and cold stream *j*, exchanger area is:

$$A_{ij} = \frac{Q_{ij}}{U_{ij} \cdot \text{LMTD}_{ij}}, \qquad \text{LMTD}_{ij} = \frac{\Delta T_1 - \Delta T_2}{\ln(\Delta T_1 / \Delta T_2)}$$

and capital cost follows a power-law form, e.g. $\text{Cost}_{ij} = a + b \cdot A_{ij}^{c}$. Both relationships are nonlinear and, critically, **non-convex** over the relevant domain — direct inclusion makes the network synthesis problem a non-convex MINLP, which is not reliably solvable to global optimality at practical scale with open-source tools. The core engineering problem of this project is finding a linearization that is (a) tight enough to trust, (b) small enough in binary-variable count for SCIP to solve in reasonable time, and (c) numerically stable near $\Delta T \to 0$.

## 2. The four candidate formulations

### Formulation A — LMTD Hyperplanes + 2D Disaggregated Logarithmic (DLog) Grid

Dynamic pre-solve hyperplanes linearize $\text{LMTD}(\Delta T_1, \Delta T_2)$. A 2D DLog grid then maps $(Q, \text{LMTD}) \to \text{Cost}$ directly, using logarithmically many binary variables ($O(\log_2 n)$ per dimension) rather than the $O(n)$ growth of standard SOS2 grids. This sidesteps SCIP's lack of native 2D SOS2 branching while keeping binary overhead low.

### Formulation B — Dual-Hyperplane Convex Approximation

Sequential hyperplanes: first for $\text{LMTD}(\Delta T_1, \Delta T_2)$, then for $\text{Cost}(Q, \text{LMTD})$. This is the fastest of the four to solve — hyperplane cuts are cheap relative to SOS2/DLog binaries — but it is only valid as an *approximation* because $\text{Cost}(Q, \text{LMTD})$ is not convex over the operating domain.

**Non-convexity check.** Writing $\text{Cost}(Q, L) = a + b \left(\dfrac{Q}{U L}\right)^{c}$ for $L = \text{LMTD}$, the Hessian with respect to $(Q, L)$ has a mixed second partial

$$\frac{\partial^2 \text{Cost}}{\partial Q\, \partial L} = -\frac{bc(c-1)}{U^c}\, Q^{c-1} L^{-c-1}$$

which does not vanish and, combined with $\partial^2\text{Cost}/\partial Q^2$ and $\partial^2\text{Cost}/\partial L^2$, produces an indefinite Hessian (mixed-sign eigenvalues) across the feasible region for the typical exponent range $0 < c < 1$ used in exchanger cost correlations. An outer hyperplane approximation of a non-convex function is not guaranteed to be a valid underestimator everywhere — it can cut off true optimal or even feasible solutions, or admit infeasible-in-reality solutions as feasible. Formulation B is retained in the benchmark as the speed baseline, not as the production candidate.

### Formulation C — $-\ln(\text{LMTD})$ Hyperplanes + 1D Tangents + 1D SOS2 *(not yet implemented)*

The driving-force nonlinearity is isolated by working in log-space: dynamic hyperplanes bound $-\ln(\text{LMTD})$ as a function of $\Delta T_1, \Delta T_2$ (this transformation is convex, so hyperplanes are valid underestimators — no non-convexity issue here). Heat duty would be linearized via a 1D tangent-line set ($Q \to \ln Q$), with the final area-to-cost mapping using a single 1D SOS2 set ($\ln A \to \text{Cost}$). This formulation is specified but **not yet built** — it's on deck as the next candidate to implement, on the hypothesis that mixing a cheap tangent set for $Q$ with a single SOS2 set for cost will out-perform D's dual-SOS2 approach on binary count.

### Formulation D — $-\ln(\text{LMTD})$ Hyperplanes + Dual 1D SOS2

Same log-space driving-force hyperplanes as C, but both $Q \to \ln Q$ and $\ln A \to \text{Cost}$ are modeled as 1D SOS2 sets rather than mixing tangents and SOS2. More binary variables than C, but a more uniform/simpler constraint structure that may branch more predictably.

### Summary comparison

| Formulation | Convexity guarantee | Relative binary overhead | Relative solve speed | Status |
|---|---|---|---|---|
| A — Hyperplane + 2D DLog | Yes (driving force); Yes (DLog is exact grid) | Medium | Medium | Implemented, runs on 2×2; needs validation against literature benchmarks |
| B — Dual hyperplane | **No** (Cost surface non-convex) | Low | **Fastest** | Implemented, runs on 2×2; not production-safe, needs validation |
| C — Hyperplane + tangent + 1D SOS2 | Yes | Low–Medium | Medium–Fast (projected) | **Not yet developed** |
| D — Hyperplane + dual 1D SOS2 | Yes | Medium | Medium | Implemented, runs on 2×2; needs validation against literature benchmarks |

## 3. Pre-solve dynamic tangent hyperplane algorithm

Rather than the classical Outer Approximation pattern (solve MILP → check NLP violation → add a cut → resolve, repeated inline), the hyperplane cuts are generated **once, before the MILP is ever invoked**:

1. Define the operating domain for $(\Delta T_1, \Delta T_2)$ from stream temperature bounds.
2. Sample the domain with a combined **uniform grid + random perturbation** sampling scheme — uniform sampling alone under-resolves curvature near domain corners, so random points are added to catch local error spikes.
3. At each sample point, evaluate the tangent hyperplane of the true nonlinear function (LMTD, or $-\ln(\text{LMTD})$ in the log-space formulations) and compute the local approximation error against the true surface.
4. Iteratively add hyperplanes at the highest-error regions until the maximum error across the sampled domain drops below tolerance $\epsilon$.
5. Freeze the resulting hyperplane set as static linear constraints and pass the fully-linearized model to SCIP once.

This trades a small amount of solution tightness (versus true dynamic OA converging to the exact optimum) for the elimination of solver-callback latency, which dominated total runtime in earlier iterations of this project (see [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md), Phase 6).

## 4. Utility match reduction: 6 SOS2 sets → 1

Utility exchangers (matches against steam or cooling water rather than another process stream) are structurally simpler than process-to-process matches because they only ever appear at the terminal end of a stream's temperature path:

- Utility inlet temperature $T_{\text{in,util}}$ — fixed
- Utility outlet temperature $T_{\text{out,util}}$ — fixed
- Process stream target temperature — fixed by the problem's target temperatures

With three of the four temperatures fixed, only **one** driving force is a free variable, not two. Carrying the full six-transformation process-match formulation (§ below) into every utility match wastes binary variables on structure that isn't there.

**Full process-match formulation (6× 1D SOS2):**

$$\Delta T_1 \to \ln \Delta T_1, \quad \Delta T_2 \to \ln \Delta T_2, \quad d_{\Delta T} \to \ln(d_{\Delta T}), \quad d_{\ln T} \to \ln(d_{\ln T}), \quad Q \to \ln Q, \quad \ln A \to \text{Cost}$$

where $d_{\Delta T} = \Delta T_1 - \Delta T_2$ and $d_{\ln T} = \ln \Delta T_1 - \ln \Delta T_2$, giving $\ln(\text{LMTD}) = \ln(d_{\Delta T}) - \ln(d_{\ln T})$.

**Utility-match reduction:** because the single free $\Delta T$ can be expressed directly as a function of duty $Q$, heat capacity flow rate $CP$, and the three fixed temperatures, the entire chain collapses to a **single 1D SOS2 mapping**:

$$Q \;\longrightarrow\; \text{Cost}_{\text{util}}$$

This is a ~6x reduction in binary variable count per utility match, and since every network has at least one utility match per stream endpoint, this reduction has an outsized effect on total MILP size versus the process-match formulation.

## 5. Open-source vs. commercial solver trade-off (SCIP vs. Gurobi)

SCIP is used throughout for licensing reasons (fully open-source, no academic-license dependency for reproducibility), but it lacks Gurobi/CPLEX's native, specialized branching rules for higher-dimensional SOS2/piecewise structures. In practice this means formulations that would be tractable as native 2D SOS2 sets under Gurobi (see Phase 5, [ARCHITECTURE_EVOLUTION.md](ARCHITECTURE_EVOLUTION.md)) stall under SCIP, which is the direct motivation for Formulations A–D above — all four are ways of avoiding native 2D SOS2 entirely. A direct SCIP-vs-Gurobi benchmark on identical formulations is on the roadmap to quantify this gap precisely rather than anecdotally.

## 6. Current benchmark status

| System | Formulation A | Formulation B | Formulation C | Formulation D |
|---|---|---|---|---|
| 2×2 | ✅ Runs in seconds | ✅ Runs in seconds | ⏳ Not yet developed | ✅ Runs in seconds |
| 4×5 | 🔧 Timeout | 🔧 Fast but approximation-invalid | ⏳ Not yet developed | 🔧 Timeout |

All 2×2 results above confirm the formulations execute and return solutions quickly — they have **not yet been validated for correctness against literature HENS benchmark problems** (e.g. the standard Linnhoff/Colberg-style test sets used in the pinch/HENS literature). That validation pass, along with the 4×5 performance work, is the immediate next phase before any formulation can be called production-ready.
# Architecture Evolution: Engineering Decision Log

This document tracks the full evolution of the HENS-Opt solver architecture, including approaches that were tried and deliberately discarded. It's kept in the repo intentionally: the dead ends carry as much engineering signal as the current baseline, and the reasoning behind each pivot is more useful to a reader than a changelog of what "just worked."

---

## Phase 1 — Flattened 1D Compressed Vectors (Discarded)

**Approach:** Dynamic Outer Approximation using first-order Taylor series to map $(Q, \text{LMTD}) \to A$ and tangent lines to map $A \to \text{Cost}$. The full problem structure — coefficient matrices, RHS vectors, non-zero masks — was flattened into 1D compressed index vectors for solver interfacing via SciPy's optimization layer.

**Why it was dropped:** Flattening the structure this early made the model effectively unreadable and unmaintainable — every change to the network topology required re-deriving index offsets by hand, and the tight coupling between the compression scheme and the math made debugging accuracy issues nearly impossible. The Taylor/tangent linearization itself was also too coarse, producing meaningful error against the true nonlinear surface. **Lesson:** premature low-level optimization of data structures before the mathematical formulation is validated is a trap — get the model right in a readable form first.

## Phase 2 — Manual SOS2 Adjacency Logic (Discarded)

**Approach:** Moved to piecewise-linear Special Ordered Sets (SOS2) for the nonlinear mappings, but the solver setup in use at the time had no native SOS2 support, so adjacency constraints (which pairs of breakpoints can be simultaneously active) were encoded manually with binary variables. Also evaluated Julia/JuMP as an alternative modeling layer.

**Why it was dropped:** Manually-coded adjacency logic is a well-known way to blow up MIP gaps — without a solver's native SOS2 branching heuristics, the relaxation is much weaker than it needs to be, and combined with inline dynamic OA re-solve loops, latency became unworkable even on small test cases. Julia/JuMP was set aside not for technical reasons but for maintainability: keeping one language across the optimization core and the Streamlit UI was judged more valuable than JuMP's cleaner native SOS2 syntax. **Lesson:** don't hand-roll what a solver already does better — find the solver that has it natively instead.

## Phase 3 — Pyomo + SCIP, 6× SOS2 Formulation (Discarded as production baseline)

**Approach:** Standardized on Pyomo as the modeling layer and SCIP as the open-source solver with native SOS2 support. Each process-to-process stream match was modeled with six separate 1D SOS2 transformations (driving forces, their log-differences, duty, and area-to-cost — full derivation in [FORMULATIONS_AND_BENCHMARKS.md](FORMULATIONS_AND_BENCHMARKS.md)).

**Why it was reduced:** This formulation is mathematically correct and was a real step forward in tooling (native SOS2, single language, no more manual adjacency), but six SOS2 sets per match means six sets of binary breakpoint-selection variables per match — for any network with more than a handful of streams, the binary count and resulting branch-and-bound tree size grew fast enough to dominate solve time. This is what motivated the utility-match-specific reduction in Phase 4.

## Phase 4 — Utility Match Simplification (Adopted)

**Insight:** Utility exchangers only ever occur at the terminal end of a stream's temperature path, where three of the four temperatures defining the driving forces are already fixed by problem data (utility supply/return temperature, and the stream's target temperature). Only one driving force is actually a free variable.

**Evolution:** First collapsed utility matches from 6 SOS2 sets to 3. Then recognized that the single remaining free $\Delta T$ can itself be written directly as a function of duty $Q$, flow-heat-capacity $CP$, and the fixed temperatures — eliminating the need to model it as a separate variable at all. Final result: **a single 1D SOS2 mapping, $Q \to \text{Cost}$, per utility match.** This is retained in the current architecture and is a ~6x reduction in binary overhead specifically for utility matches, which are present in every feasible network.

## Phase 5 — Full 2D SOS2 (Discarded)

**Approach:** Attempted the mathematically "cleanest" formulation — full 2D SOS2 grids for $(\Delta T_1, \Delta T_2) \to \text{LMTD}$ and $(\text{LMTD}, Q) \to A$ directly, avoiding the log-space transformations altogether.

**Why it was dropped:** SCIP does not implement the specialized branching rules that commercial solvers (Gurobi, CPLEX) use for native 2D SOS2 structures — without them, 2D SOS2 in an open-source solver essentially degrades to brute-force branching on a much larger binary space, and even small test networks stalled. **Lesson:** the "textbook" formulation isn't automatically the right one for the actual solver being used — formulation choice has to be made jointly with solver capability, not in the abstract. This finding is what directly motivated the current four-formulation benchmark in [FORMULATIONS_AND_BENCHMARKS.md](FORMULATIONS_AND_BENCHMARKS.md), all of which are explicitly designed to avoid native 2D SOS2.

## Phase 6 — Pre-Solve Dynamic Tangent Hyperplanes (Current Baseline)

**Insight:** The recurring bottleneck across Phases 1–5 wasn't the linearization concept itself, it was doing the linearization *dynamically*, inline with the solver, via repeated callback-triggered re-solves (classical Outer Approximation). Every dynamic OA loop paid a latency tax per iteration.

**Mechanism:** Move all linearization work to before the MILP solver is ever invoked. An automated pre-solve engine samples the driving-force domain with a combined uniform-and-random grid, evaluates tangent hyperplanes at each sample, measures approximation error against the true nonlinear surface, and keeps adding hyperplanes at the highest-error regions until error drops below tolerance. The result is a static, fully-linearized model handed to SCIP once — no callbacks, no iterative re-solving.

This is the current production baseline and the foundation for the four candidate formulations under active benchmark (A–D in [FORMULATIONS_AND_BENCHMARKS.md](FORMULATIONS_AND_BENCHMARKS.md)), which vary in how the *post*-hyperplane structure (LMTD → duty → cost) is linearized, not in the hyperplane pre-solve mechanism itself.

---

## Summary timeline

| Phase | Core idea | Outcome |
|---|---|---|
| 1 | Flattened vectors + Taylor/tangent OA | Discarded — unmaintainable, inaccurate |
| 2 | Manual SOS2 adjacency + Julia evaluation | Discarded — weak relaxation, latency |
| 3 | Pyomo + SCIP native SOS2 (6×/match) | Superseded — correct but binary-heavy |
| 4 | Utility match reduction (6× → 1×) | **Adopted** |
| 5 | Full native 2D SOS2 | Discarded — SCIP branching gap |
| 6 | Pre-solve dynamic hyperplanes | **Adopted, current baseline** |

The throughline across all six phases: every pivot was driven by measured performance or maintainability failure, not by starting over for its own sake. The current architecture is the accumulated result of ruling out five other approaches with specific, documented reasons — not the first thing that was tried.
