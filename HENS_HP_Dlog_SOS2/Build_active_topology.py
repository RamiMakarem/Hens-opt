"""
Build_Active_Topology.py

Takes the `results` dict produced by Solve_Extract.extract_hens_results()
(after solving the DLOG-based HENS MILP) and extracts the *fixed* network
topology -- which process matches and which utility assignments are
active -- for the follow-on NLP stage, which re-optimizes temperatures,
LMTD/areas, and costs on that fixed topology (no more binaries).
"""


def build_active_topology(results, data=None, Hsap=None, Csap=None, Q_thresh=1.0):
    """
    Extract the active process-match index set (i, j, k) from a solved
    MILP's results dict, plus the stream data the NLP needs to build its
    own energy/temperature constraints.

    FIXES vs. the original version:
      - I, J, K, S and the stream property lists (HID, CID, CP_H, CP_C,
        Tin_H, Tout_H, Tin_C, Tout_C) are now pulled directly from `data`
        (the SimpleNamespace returned by Pre_Process.pre_process_milp)
        when it's supplied, instead of being re-derived from a separately
        -passed Hsap/Csap. This guarantees they're exactly the values the
        MILP was actually solved with -- re-deriving them independently
        risked a silent mismatch if a caller passed a reordered/edited
        stream list here that differs even slightly from what built the
        model. `Hsap`/`Csap` are kept as a fallback for standalone use
        (e.g. testing this function without a `data` namespace on hand),
        reproducing the original behaviour exactly in that case.
      - Dict lookups on each edge (`e["active"]`, `e["hot"]`, `e["cold"]`,
        `e["stage"]`) now use `.get(...)` with sensible defaults instead
        of bare indexing, so a malformed/partial edge entry is skipped
        rather than raising a KeyError and aborting the whole extraction.
      - `k = int(e["stage"]) - 1` is unchanged: "stage" in the results
        dict is 1-based (`s_idx + 1` from extract_hens_results), while
        this MILP's own m.Hs / the NLP's stage index are 0-based -- this
        conversion was already correct, just confirmed and documented
        here since it's an easy place to introduce an off-by-one bug.

    Returns the same 13-tuple as the original function, in the same
    order, so any existing unpacking code (`ActiveIJK, I, J, K, S, HID,
    CID, CP_H, CP_C, Tin_H, Tout_H, Tin_C, Tout_C = build_active_topology(...)`)
    keeps working unchanged.
    """
    if data is not None:
        I, J, K, S = data.I, data.J, data.K, data.S
        HID, CID = list(data.HID), list(data.CID)
        CP_H, CP_C = list(data.CP_H), list(data.CP_C)
        Tin_H, Tout_H = list(data.Tin_H), list(data.Tout_H)
        Tin_C, Tout_C = list(data.Tin_C), list(data.Tout_C)
    else:
        if Hsap is None or Csap is None:
            raise ValueError(
                "build_active_topology needs either `data` (the "
                "pre_process_milp namespace) or both `Hsap` and `Csap`.")
        I = len(Hsap)
        J = len(Csap)
        K = I + J - 1
        S = K - 1

        HID    = [r[0] for r in Hsap];   CID    = [r[0] for r in Csap]
        CP_H   = [r[3] for r in Hsap];   CP_C   = [r[3] for r in Csap]
        Tin_H  = [r[1] for r in Hsap];   Tin_C  = [r[1] for r in Csap]
        Tout_H = [r[2] for r in Hsap];   Tout_C = [r[2] for r in Csap]

    hid_to_i = {hid: i for i, hid in enumerate(HID)}
    cid_to_j = {cid: j for j, cid in enumerate(CID)}

    ActiveIJK = []
    for e in results.get("edges", []):
        if not e.get("active", False):
            continue
        Q = e.get("Q", 0.0)
        if Q is None or Q <= Q_thresh:
            continue
        i = hid_to_i.get(e.get("hot"))
        j = cid_to_j.get(e.get("cold"))
        if i is None or j is None:
            continue
        stage = e.get("stage")
        if stage is None:
            continue
        k = int(stage) - 1   # "stage" is 1-based (s_idx + 1); NLP uses 0-based
        if not (0 <= k < S):
            continue
        ActiveIJK.append((i, j, k))

    return (ActiveIJK, I, J, K, S, HID, CID, CP_H, CP_C,
            Tin_H, Tout_H, Tin_C, Tout_C)


def build_active_utilities(results, Q_thresh=1.0):
    """
    Extract the active utility assignments: (utility_index, stream_index)
    for hot utilities (heating a cold stream) and cold utilities (cooling
    a hot stream).

    FIXES vs. the original version:
      - `int(e["utility"])` is now guarded: extract_hens_results() only
        populates the "utility" key when the model defines HU/CU utility
        -option sets (m.HU / m.CU), which is the normal case for this
        codebase -- but its fallback branch (direct-stream q_hu/q_cu
        reporting, used only if both sets are empty) does NOT include a
        "utility" key at all, which would previously raise a KeyError
        here. Those entries are now skipped instead of crashing.
      - `e["type"]` also switched to `.get(...)` for the same reason.
    """
    ActiveHU = []
    ActiveCU = []
    for e in results.get("util_hex_edges", []):
        Q = e.get("Q", 0.0)
        if Q is None or Q <= Q_thresh:
            continue
        util_id = e.get("utility")
        if util_id is None:
            continue
        try:
            idx = int(util_id)
        except (TypeError, ValueError):
            continue
        s_idx = e.get("stream")
        if s_idx is None:
            continue

        etype = e.get("type")
        if etype == "hot":
            ActiveHU.append((idx, s_idx))
        elif etype == "cold":
            ActiveCU.append((idx, s_idx))

    return ActiveHU, ActiveCU


def build_warm_start(results, Q_thresh=1.0):
    """
    Optional companion to build_active_topology/build_active_utilities:
    collects clean, error-free initial values for Q, LMTD, Area, and
    (variable) Cost at every active process match and utility duty, keyed
    exactly like the index tuples those two functions produce -- (i, j, k)
    for process matches, (utility_index, stream_index) for utilities --
    so the NLP can be warm-started from the MILP's solution instead of a
    cold start.

    FIX: this previously copied the MILP's own LMTD_model / Area_m2_model
    values and the Cost_HU/Cost_CU model variables directly. All of those
    carry the small DLOG/SOS2 piecewise-linear fit error documented in the
    error report (extract_hens_results' "sos2_*_fit_err" fields) -- exactly
    the kind of linearization artifact the NLP stage exists to eliminate,
    so seeding it from those values re-introduces the error you're trying
    to get rid of.

    Instead, this now recomputes LMTD, Area, and variable Cost from
    error-free primitives:
      - process matches: uses "LMTD_true"/"Area_m2_true" (already computed
        in extract_hens_results directly from the model's own TH/TC
        temperature profile via the exact LMTD formula -- dT1/dT2 equal
        TH-TC exactly per BLOCK 8b, so this has no OA/DLOG approximation
        in it at all), then Cost_var = cost_b * Area_true**cost_beta;
      - utilities: uses "Area_m2" (already computed directly from the
        exact closed-form dT1/dT2 formulas used in preprocessing, with no
        SOS2 interpolation involved), then Cost_var = cost_b *
        Area**cost_beta.
    Q itself is left as-is in both cases: it's a genuine MILP decision
    variable, not a derived/approximated quantity, so it carries no fit
    error to avoid in the first place.

    Cost_init/CostHU_init/CostCU_init are the *variable* CAPEX only (no
    cost_a fixed term), matching the m.Cost / m.Cost_HU / m.Cost_CU
    variables in this codebase's MILP.
    """
    cost_b = results.get("cost_b", 150.0)
    cost_beta = results.get("cost_beta", 1.0)

    def _process_key(e, hid_to_i, cid_to_j):
        i = hid_to_i.get(e.get("hot"))
        j = cid_to_j.get(e.get("cold"))
        stage = e.get("stage")
        if i is None or j is None or stage is None:
            return None
        return (i, j, int(stage) - 1)

    Q_init, LMTD_init, Area_init, Cost_init = {}, {}, {}, {}
    QHU_init, QCU_init = {}, {}
    CostHU_init, CostCU_init = {}, {}

    HID = results.get("HIDs", [])
    CID = results.get("CIDs", [])
    hid_to_i = {hid: i for i, hid in enumerate(HID)}
    cid_to_j = {cid: j for j, cid in enumerate(CID)}

    for e in results.get("edges", []):
        if not e.get("active", False):
            continue
        Q = e.get("Q", 0.0)
        if Q is None or Q <= Q_thresh:
            continue
        key = _process_key(e, hid_to_i, cid_to_j)
        if key is None:
            continue
        Q_init[key] = Q
        lmtd_true = e.get("LMTD_true")
        area_true = e.get("Area_m2_true")
        if lmtd_true is not None:
            LMTD_init[key] = lmtd_true
        if area_true is not None:
            Area_init[key] = area_true
            Cost_init[key] = cost_b * (area_true ** cost_beta)

    for e in results.get("util_hex_edges", []):
        Q = e.get("Q", 0.0)
        if Q is None or Q <= Q_thresh:
            continue
        util_id = e.get("utility")
        s_idx = e.get("stream")
        if util_id is None or s_idx is None:
            continue
        try:
            idx = int(util_id)
        except (TypeError, ValueError):
            continue

        area_u = e.get("Area_m2")
        cost_var_u = cost_b * (area_u ** cost_beta) if area_u is not None else None

        if e.get("type") == "hot":
            QHU_init[(idx, s_idx)] = Q
            if area_u is not None:
                CostHU_init[(idx, s_idx)] = cost_var_u
        elif e.get("type") == "cold":
            QCU_init[(idx, s_idx)] = Q
            if area_u is not None:
                CostCU_init[(idx, s_idx)] = cost_var_u

    return {
        "Q_init": Q_init,
        "LMTD_init": LMTD_init,
        "Area_init": Area_init,
        "Cost_init": Cost_init,
        "QHU_init": QHU_init,
        "QCU_init": QCU_init,
        "CostHU_init": CostHU_init,
        "CostCU_init": CostCU_init,
    }