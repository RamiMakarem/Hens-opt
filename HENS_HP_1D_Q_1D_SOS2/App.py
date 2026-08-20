# === STREAMLIT APP ===

import streamlit as st
import pandas as pd
import numpy as np
import streamlit.components.v1 as components
from Core import (
    classify_streams, adjust_temperatures, calculate_delta_h,
    cascade, stream_energy_table, refine_hen_nlp,)
from Generate_hen_svg import generate_hen_svg
from Solve_extract import solve_hen_milp
from Pre_process import _normalise_utility_list

st.set_page_config(page_title="HENS Optimization Tool", layout="wide")
st.title("Pinch Analysis Tool")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 ─ Process streams
# ─────────────────────────────────────────────────────────────────────────────
st.subheader("Process Streams")

'''
df_input = pd.DataFrame({
    "Stream ID": ["H1", "H2", "C1", "C2"],
    "Tin":  [650.0, 590.0, 410.0, 350.0],
    "Tout": [370.0, 370.0, 650.0, 500.0],
    "CP":   [10.0, 20.0, 15.0, 13.0],
    "h (kW/m²·°C)": [1.0, 1.0, 1.0, 1.0]
})

'''

df_input = pd.DataFrame(
    {
        "Stream ID": [
            "H1",
            "H2",
            "H3",
            "H4",
            "C1",
            "C2",
            "C3",
            "C4",
            "C5",
        ],
        "Tin": [
            327,
            220,
            220,
            160,
            100,
            35,
            85,
            60,
            140,
        ],
        "Tout": [
            40,
            160,
            60,
            45,
            300,
            164,
            138,
            170,
            300,
        ],
        "CP": [100.0, 160.0, 60.0, 400.0, 100.0, 70.0, 350.0, 60.0, 200.0],
        "h (kW/m²·°C)": [0.5, 0.4, 0.14, 0.3, 0.35, 0.7, 0.5, 0.14, 0.6],
    }
)


edited_df = st.data_editor(df_input, num_rows="dynamic", key="proc_streams")

hot_streams  = edited_df[edited_df["Tin"] > edited_df["Tout"]]["Stream ID"].tolist()
cold_streams = edited_df[edited_df["Tin"] < edited_df["Tout"]]["Stream ID"].tolist()
h_hot  = edited_df[edited_df["Tin"] > edited_df["Tout"]]["h (kW/m²·°C)"].tolist()
h_cold = edited_df[edited_df["Tin"] < edited_df["Tout"]]["h (kW/m²·°C)"].tolist()
I = len(hot_streams)
J = len(cold_streams)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 ─ Utility streams
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔥❄️ Utility Streams")
st.markdown(
    """
Add as many utility streams as needed.  
**Role is inferred automatically** from temperatures:
- `Tin > Tout` → **Hot utility** (heats cold process streams at their outlet).
- `Tin < Tout` → **Cold utility** (cools hot process streams at their outlet).
- `Tin = Tout` (pure latent) → you must choose *Heating* or *Cooling*.

**Placement:** utilities are applied at the network **ends only** (Yee–Grossmann).  
The solver selects **at most one utility per process stream** and respects each utility's
maximum flowrate.  All utility HEXs are sized and their capital cost included in TAC.
"""
)

# ── Default utility table ────────────────────────────────────────────────────
UTIL_COLS = {
    "ID":             "HU",
    "Tin (°C)":       330.0,
    "Tout (°C)":      250.0,
    "cp (kJ/kg·°C)":  100.0,        # 0 → latent-only (steam)
    "λ (kJ/kg)":      0.0,
    "cp_vap (kJ/kg·°C)": 0.0,
    "cp_liq (kJ/kg·°C)": 0.0,
    "T_phase (°C)":   None,        # None → no phase change elbow
    "Max flowrate (kg/s)": None,   # None → ∞
    "Cost ($/kW·yr)": 60.0,
}

default_utils = pd.DataFrame([
    {"ID": "HU", "Tin (°C)": 330.0, "Tout (°C)": 250.0,
     "cp (kJ/kg·°C)": 100.0, "λ (kJ/kg)": 0.0,
     "cp_vap (kJ/kg·°C)": 0.0, "cp_liq (kJ/kg·°C)": 0.0,
     "T_phase (°C)": None, "Max flowrate (kg/s)": None, "Cost ($/kW·yr)": 60.0,
     "h (kW/m²·°C)": 0.5},   # NEW — condensing steam, typically high h
    {"ID": "CU",       "Tin (°C)": 15.0,  "Tout (°C)": 30.0,
     "cp (kJ/kg·°C)": 160.0, "λ (kJ/kg)": 0.0,
     "cp_vap (kJ/kg·°C)": 0.0, "cp_liq (kJ/kg·°C)": 0.0,
     "T_phase (°C)": None, "Max flowrate (kg/s)": None, "Cost ($/kW·yr)": 6.0,
     "h (kW/m²·°C)": 0.5},   # NEW — sensible cooling water
])

st.markdown(
    "**Column guide:** `cp`=sensible heat, `λ`=latent heat (steam), "
    "`cp_vap`/`cp_liq`/`T_phase` only for combined (superheated+condensing) utilities. "
    "Leave `T_phase`, `Max flowrate` blank for no limit / pure sensible or pure latent."
)

util_df = st.data_editor(
    default_utils,
    num_rows="dynamic",
    width='stretch',
    key="util_table",
    column_config={
        "Tin (°C)":            st.column_config.NumberColumn(format="%.3f"),
        "Tout (°C)":           st.column_config.NumberColumn(format="%.3f"),
        "cp (kJ/kg·°C)":       st.column_config.NumberColumn(format="%.3f"),
        "λ (kJ/kg)":           st.column_config.NumberColumn(format="%.1f"),
        "cp_vap (kJ/kg·°C)":   st.column_config.NumberColumn(format="%.3f"),
        "cp_liq (kJ/kg·°C)":   st.column_config.NumberColumn(format="%.3f"),
        "T_phase (°C)":        st.column_config.NumberColumn(format="%.1f"),
        "Max flowrate (kg/s)": st.column_config.NumberColumn(format="%.2f"),
        "Cost ($/kW·yr)":      st.column_config.NumberColumn(format="%.2f"),
        "h (kW/m²·°C)": st.column_config.NumberColumn(format="%.3f"),
    },
)

# ── Parse utility table → utility_specs dicts ────────────────────────────────
def _safe_float(val, default=None):
    """Convert val to float, returning default for None/NaN/empty/non-numeric."""
    if val is None:
        return default
    try:
        f = float(val)
        return default if (f != f) else f   # NaN check: NaN != NaN
    except (TypeError, ValueError):
        return default


def parse_util_row(row):
    """Convert a data_editor row into a utility spec dict for core.py."""
    Tin  = _safe_float(row.get("Tin (°C)"),  0.0)
    Tout = _safe_float(row.get("Tout (°C)"), 0.0)
    cp   = _safe_float(row.get("cp (kJ/kg·°C)"),  0.0) or 0.0
    lam  = _safe_float(row.get("λ (kJ/kg)"),       0.0) or 0.0
    cp_vap  = _safe_float(row.get("cp_vap (kJ/kg·°C)"), 0.0) or 0.0
    cp_liq  = _safe_float(row.get("cp_liq (kJ/kg·°C)"), 0.0) or 0.0
    T_phase = _safe_float(row.get("T_phase (°C)"),  None)
    max_flow = _safe_float(row.get("Max flowrate (kg/s)"), None)
    cost = _safe_float(row.get("Cost ($/kW·yr)"), 80.0) or 80.0
    uid  = str(row.get("ID") or "util").strip() or "util"
    h_val = _safe_float(row.get("h (kW/m²·°C)"), 2.0) or 2.0
    if Tin is None or Tout is None:
        raise ValueError(f"Utility '{uid}': Tin and Tout are required.")

    spec = {"id": uid, "cost_per_kw": cost,"h": h_val}
    if max_flow is not None and max_flow > 0:
        spec["max_flowrate"] = max_flow

    if abs(Tin - Tout) < 0.01:
        spec["type"]       = "steam"
        spec["T_steam"]    = Tin
        spec["lambda_vap"] = lam if lam > 0 else 20000
    elif T_phase is not None and cp_vap > 0:
        spec["type"]       = "combined"
        spec["Tin"]        = Tin
        spec["Tout"]       = Tout
        spec["T_phase"]    = T_phase
        spec["lambda_vap"] = lam
        spec["cp_vap"]     = cp_vap
        spec["cp_liq"]     = cp_liq
    else:
        spec["type"] = "sensible"
        spec["Tin"]  = Tin
        spec["Tout"] = Tout
        spec["cp"]   = cp if cp > 0 else 500

    return spec, Tin, Tout

hot_util_specs  = []
cold_util_specs = []
util_parse_errors = []

# Track role ambiguity (Tin == Tout) — let user choose
latent_roles = {}
for idx, row in util_df.iterrows():
    Tin_raw  = row.get("Tin (°C)")
    Tout_raw = row.get("Tout (°C)")
    # Skip rows where both Tin and Tout are missing/NaN (blank editor rows)
    if (Tin_raw is None or (isinstance(Tin_raw, float) and Tin_raw != Tin_raw)) and \
       (Tout_raw is None or (isinstance(Tout_raw, float) and Tout_raw != Tout_raw)):
        continue
    Tin  = _safe_float(Tin_raw,  None)
    Tout = _safe_float(Tout_raw, None)
    if Tin is None or Tout is None:
        continue
    uid  = str(row.get("ID", f"util_{idx}") or f"util_{idx}").strip()
    if abs(Tin - Tout) < 0.01:
        role_key = f"latent_role_{idx}"
        role = st.selectbox(
            f"Utility **{uid}** (Tin = Tout = {Tin}°C): is it a *Heating* or *Cooling* utility?",
            ["Heating (hot utility)", "Cooling (cold utility)"],
            key=role_key,
        )
        latent_roles[idx] = "hot" if "Heating" in role else "cold"

for idx, row in util_df.iterrows():
    Tin_raw  = row.get("Tin (°C)")
    Tout_raw = row.get("Tout (°C)")
    if _safe_float(Tin_raw, None) is None or _safe_float(Tout_raw, None) is None:
        continue
    try:
        spec, Tin, Tout = parse_util_row(row)
        uid = spec["id"]
        if abs(Tin - Tout) < 0.01:
            role = latent_roles.get(idx, "hot")
        elif Tin > Tout:
            role = "hot"
        else:
            role = "cold"
        if role == "hot":
            hot_util_specs.append(spec)
        else:
            cold_util_specs.append(spec)
    except Exception as e:
        util_parse_errors.append(f"Row {idx}: {e}")

if util_parse_errors:
    for err in util_parse_errors:
        st.warning(f"Utility parse warning — {err}")

utility_specs = {"hot": hot_util_specs, "cold": cold_util_specs}
n_HU = len(hot_util_specs)
n_CU = len(cold_util_specs)
h_hu = [u.get("h", 2.0) for u in hot_util_specs]
h_cu = [u.get("h", 2.0) for u in cold_util_specs]

# ── Show inferred roles ──────────────────────────────────────────────────────
if hot_util_specs or cold_util_specs:
    role_rows = []
    for u in hot_util_specs:
        role_rows.append({"ID": u["id"], "Role": "🔥 Hot utility",
                          "T_supply (°C)": u.get("T_steam", u.get("Tin", "—")),
                          "Type": u["type"]})
    for u in cold_util_specs:
        role_rows.append({"ID": u["id"], "Role": "❄️ Cold utility",
                          "T_supply (°C)": u.get("Tin", "—"),
                          "Type": u["type"]})
    st.dataframe(pd.DataFrame(role_rows), width='stretch')

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 ─ U matrix
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔁 Heat Transfer Coefficients — U Matrix")
st.markdown(
    f"Matrix is **(I + n_HU) × (J + n_CU)** = **({I + n_HU}) × ({J + n_CU})**:  \n"
    f"Rows 0…{I-1} = hot process streams; rows {I}…{I+n_HU-1} = hot utility streams.  \n"
    f"Cols 0…{J-1} = cold process streams; cols {J}…{J+n_CU-1} = cold utility streams.")

hu_ids = [u["id"] for u in hot_util_specs]
cu_ids = [u["id"] for u in cold_util_specs]
row_labels = hot_streams + hu_ids
col_labels = cold_streams + cu_ids

u_mode = st.radio("U specification", ["Single U for all", "Per-pair matrix", "Calculate from h (film coefficients)"], horizontal=True)
total_rows = I + n_HU
total_cols = J + n_CU

if u_mode == "Single U for all":
    u_scalar = st.number_input("Overall U [kW/(m²·°C)]", value=0.5, format="%.3f", step=0.05)
    U_matrix_val = [[u_scalar] * total_cols for _ in range(total_rows)]
    with st.expander("View implied U matrix"):
        st.dataframe(pd.DataFrame(U_matrix_val, index=row_labels or list(range(total_rows)),
                                  columns=col_labels or list(range(total_cols))).style.format("{:.3f}"))
elif u_mode=="Per-pair matrix": 
    df_u_default = pd.DataFrame(
        [[0.5] * total_cols for _ in range(total_rows)],
        index=row_labels or list(range(total_rows)),
        columns=col_labels or list(range(total_cols)),
    )
    df_u_edited = st.data_editor(df_u_default, width='stretch', key="u_matrix_editor")
    U_matrix_val = df_u_edited.values.tolist()
    if (np.array(U_matrix_val) <= 0).any():
        st.error("All U values must be positive.")
else:
    row_h = h_hot + h_hu     # aligned with row_labels = hot_streams + hu_ids
    col_h = h_cold + h_cu    # aligned with col_labels = cold_streams + cu_ids

    if any(h <= 0 for h in row_h) or any(h <= 0 for h in col_h):
        st.error("All film heat transfer coefficients (h) must be positive.")
        U_matrix_val = [[0.5] * total_cols for _ in range(total_rows)]
    else:
        U_matrix_val = [
            [1.0 / (1.0 / row_h[r] + 1.0 / col_h[c]) for c in range(total_cols)]
            for r in range(total_rows)
        ]

    with st.expander("View U matrix calculated from h"):
        st.dataframe(
            pd.DataFrame(
                U_matrix_val,
                index=row_labels or list(range(total_rows)),
                columns=col_labels or list(range(total_cols)),
            ).style.format("{:.4f}")
        )
        st.caption("U_ij = 1 / (1/h_hot,i + 1/h_cold,j)")    

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 ─ TAC cost parameters
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("## 💰 TAC Model")
st.markdown(
    "**TAC = Utility operating cost + Process HEX capital + Utility HEX capital**  \n"
    "Utility operating cost per stream = `Cost ($/kW·yr)` × Q [kW] (set in the utility table above).  \n"
    "Capital cost per HEX = `(cost_a + cost_b × A^β) / payback`."
)
col_c1, col_c2 = st.columns(2)
with col_c1:
    cost_a    = st.number_input("Fixed cost per HEX ($)",  value=2000.0, step=100.0)
    cost_b    = st.number_input("Area coeff ($/m^beta)",   value=70.0,    step=5.0)
with col_c2:
    cost_beta = st.number_input("Area exponent beta",      value=1.0,     step=0.05, format="%.2f")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 ─ Solver settings
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
delta_tmin = st.number_input("ΔTmin (°C)", value=10.0)
run_nlp = st.checkbox("Run NLP refinement after MILP", value=True,
    help="IPOPT re-optimises heat loads and temperatures with exact LMTD + utility HEX sizing.")
run_button = st.button("▶ Run Analysis")

# ─────────────────────────────────────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────────────────────────────────────
if run_button:
    if not hot_util_specs and not cold_util_specs:
        st.error("Add at least one hot and one cold utility before running.")
        st.stop()

    st.subheader("🔄 Processing...")

    df = classify_streams(edited_df)
    st.write("### Classified Streams"); st.dataframe(df)

    energy_df = stream_energy_table(df)
    st.subheader("Stream Heat Contributions"); st.dataframe(energy_df)

    df_adj, intervals = adjust_temperatures(df, delta_tmin)
    st.write("### Adjusted Streams"); st.dataframe(df_adj)

    delta_h_df = calculate_delta_h(df_adj, intervals)
    st.write("### ΔH Table"); st.dataframe(delta_h_df)

    cascade_df, qh, qc, pinch_temp, pinch_interval = cascade(delta_h_df)
    st.write("### Cascade Table"); st.dataframe(cascade_df)

    st.subheader("Heat Integration Summary")
    c1, c2 = st.columns(2)
    c1.metric("Min Hot Utility (QH)", f"{qh:.2f} kW")
    c1.metric("Pinch Temperature",    f"{pinch_temp:.2f} °C")
    c2.metric("Min Cold Utility (QC)", f"{qc:.2f} kW")

    Hsap, Csap = [], []
    for _, row in edited_df.iterrows():
        if row["Tin"] > row["Tout"]:
            Hsap.append([row["Stream ID"], row["Tin"], row["Tout"], row["CP"]])
        else:
            Csap.append([row["Stream ID"], row["Tin"], row["Tout"], row["CP"]])

    # Show U matrix
    st.subheader("🔁 U Matrix in Use")
    if row_labels and col_labels:
        st.dataframe(pd.DataFrame(U_matrix_val, index=row_labels, columns=col_labels).style.format("{:.3f}"))
    else:
        st.dataframe(pd.DataFrame(U_matrix_val).style.format("{:.3f}"))

    # ── MILP ─────────────────────────────────────────────────────────────────
    with st.spinner("Running MILP..."):
        results = solve_hen_milp(
            Hsap, Csap, delta_tmin, qh, qc,
            U_matrix=U_matrix_val,
            cost_a=cost_a, cost_b=cost_b, cost_beta=cost_beta,
            utility_specs=utility_specs,)

    milp_tac = results.get("TAC", 0)

    # ── NLP ──────────────────────────────────────────────────────────────────
    if run_nlp:
        with st.spinner("Running NLP refinement..."):
            results = refine_hen_nlp(
                results, Hsap, Csap, delta_tmin,
                U_matrix=U_matrix_val,
                cost_a=cost_a, cost_b=cost_b, cost_beta=cost_beta,
                utility_specs=utility_specs,
            )
        nlp_tac = results.get("TAC", 0)
        improvement = (milp_tac - nlp_tac) / milp_tac * 100 if milp_tac else 0
        st.subheader("🔁 MILP → NLP")
        c1, c2, c3 = st.columns(3)
        c1.metric("TAC (MILP)", f"${milp_tac:,.0f}")
        c2.metric("TAC (NLP)",  f"${nlp_tac:,.0f}", delta=f"{improvement:+.2f}%", delta_color="inverse")
        c3.metric("NLP status",
                  "✅ Converged" if results.get("nlp_success") else "⚠️ Best-effort",
                  delta=results.get("nlp_message", "")[:40])

    # ── TAC summary ──────────────────────────────────────────────────────────
    st.subheader("💰 TAC Summary")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TAC ($/yr)",           f"${results.get('TAC', 0):,.0f}")
    c2.metric("Utility OpEx ($/yr)",  f"${results.get('ann_util_cost', 0):,.0f}")
    c3.metric("Process HEX CapEx",    f"${results.get('ann_cap_process', 0):,.0f}")
    c4.metric("Utility HEX CapEx",    f"${results.get('ann_cap_util_hex', 0):,.0f}")
    c5.metric("Total CapEx ($/yr)",   f"${results.get('ann_cap_cost', 0):,.0f}")

    c6, c7 = st.columns(2)
    c6.metric("Hot Utility Total",  f"{sum(results.get('QH', [0])):.2f} kW")
    c7.metric("Cold Utility Total", f"{sum(results.get('QC', [0])):.2f} kW")

    # ── Process HEX table ────────────────────────────────────────────────────
    st.subheader("🧱 Process HEXs")
    edges_df = pd.DataFrame(results.get("edges", []))
    if not edges_df.empty:
        edges_df["HX"] = edges_df["hot"].astype(str) + " → " + edges_df["cold"].astype(str)
        st.dataframe(edges_df[["HX","stage","Q","LMTD","Area_m2","CapCost_$"]].rename(columns={
            "stage":"Stage","Q":"Q (kW)","LMTD":"LMTD (°C)","Area_m2":"Area (m²)","CapCost_$":"CapEx ($/yr)"
        }).style.format({"Q (kW)":"{:.3f}","LMTD (°C)":"{:.2f}","Area (m²)":"{:.2f}","CapEx ($/yr)":"{:,.0f}"}),
        width='stretch')

    # ── Utility HEX table ────────────────────────────────────────────────────
    st.subheader("⚙️ Utility HEXs — Sizing, Cost & Flowrate")
    util_edges = results.get("util_hex_edges", [])
    if util_edges:
        udf = pd.DataFrame(util_edges)
        udf["Exchanger"] = udf["hot"].astype(str) + " ↔ " + udf["cold"].astype(str)
        cols_show = ["Exchanger","utility","Q","mdot_kg_s","LMTD","Area_m2","CapCost_$"]
        cols_show = [c for c in cols_show if c in udf.columns]
        udf_display = udf[cols_show].rename(columns={
            "utility":"Utility ID","Q":"Q (kW)","mdot_kg_s":"Flowrate (kg/s)",
            "LMTD":"LMTD (°C)","Area_m2":"Area (m²)","CapCost_$":"CapEx ($/yr)"
        })
        fmt = {"Q (kW)":"{:.3f}","LMTD (°C)":"{:.2f}","Area (m²)":"{:.2f}","CapEx ($/yr)":"{:,.0f}"}
        if "Flowrate (kg/s)" in udf_display.columns:
            fmt["Flowrate (kg/s)"] = "{:.4f}"
        st.dataframe(udf_display.style.format(fmt, na_rep="—"), width='stretch')
        st.caption("Flowrate = Q / q_per_kg  where q_per_kg = cp·|ΔT| (sensible) or λ (steam).")
    else:
        st.info("No utility HEXs active.")

    # ── Temperature profiles ─────────────────────────────────────────────────
    st.subheader("🌡️ Stream Temperature Profiles")
    for i, temps in enumerate(results.get("T_hot", [])):
        st.write(f"Hot stream {i+1}: {[round(t,1) for t in temps]}")
    for j, temps in enumerate(results.get("T_cold", [])):
        st.write(f"Cold stream {j+1}: {[round(t,1) for t in temps]}")

    # ── HEN diagram ──────────────────────────────────────────────────────────
    st.subheader("🔥 HEN Diagram")
    svg_hen = results.get("svg") or generate_hen_svg(results) or "<p>No diagram</p>"
    components.html(
        f'<div style="overflow:auto;width:100%;height:700px;background:#1a1a2e;border-radius:8px;padding:10px;">'
        f'{svg_hen}</div>',
        height=720, scrolling=True,)
