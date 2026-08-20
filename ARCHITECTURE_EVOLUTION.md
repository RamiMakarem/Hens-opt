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
