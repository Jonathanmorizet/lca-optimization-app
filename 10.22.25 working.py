# app.py (fixed)
import re
import sys
import random
import subprocess
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# --- Ensure DEAP is present ---
try:
    from deap import base, creator, tools
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ===============================
# Constants & helpers
# ===============================
DEFAULT_TREES = 1900.0
DEFAULT_COST_HEADER = "Unit Cost ($)"  # preferred display name

# Canonical impact names we want to detect (normalized forms below)
CANON_IMPACTS = [
    "kg co2-eq/unit",       # GWP
    "kg so2-eq/unit",       # Acidification
    "ctue/unit",            # Freshwater ecotox
    "kg n-eq/unit",         # Eutrophication (N)
    "ctuh/unit",            # Human toxicity (carcinogenic)
    "ctuh/unit.1",          # Human toxicity (non-carcinogenic) often appears as .1 duplicate
    "kg cfc-11-eq/unit",    # Ozone depletion
    "kg pm2.5-eq/unit",     # Particulate matter
    "kg o3-eq/unit",        # Smog/photochemical oxidants
]

# Extra keys to recognize GWP by name
GWP_ALIASES = [
    "gwp", "climate change", "kg co2-eq/unit", "kg co2 eq/unit", "kg co₂-eq/unit",
    "kgco2eq/unit", "kgco2-eq/unit", "kg co2e/unit"
]

def normalize(s: str) -> str:
    """
    Lowercase, replace unicode subscripts, remove punctuation -> spaces,
    collapse spaces. Used for header matching.
    """
    if s is None:
        return ""
    s = str(s)
    # unify unicode CO₂ and 2.5 etc.
    s = s.replace("₂", "2").replace("₅", "5")
    # slashes and dashes treated as separators
    s = re.sub(r"[^a-zA-Z0-9]+", " ", s.lower()).strip()
    return re.sub(r"\s+", " ", s)

def to_numeric_strict(series: pd.Series) -> pd.Series:
    """
    Convert text-y numeric columns to float:
    - handles commas, spaces
    - handles parentheses for negatives "(123)" -> -123
    - coerces errors to NaN, then fills with 0
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    s = series.astype(str).str.strip()
    # Handle "(123)" negatives
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    # Remove thousands separators and spaces
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("\u00A0", " ", regex=False)  # non-breaking space
    return pd.to_numeric(s, errors="coerce").fillna(0.0)

def first_present(cols, candidates_norm):
    """Return the first column name from cols whose NORMALIZED form appears in candidates_norm."""
    by_norm = {normalize(c): c for c in cols}
    for key in candidates_norm:
        if key in by_norm:
            return by_norm[key]
    return None

def detect_cost_col(df: pd.DataFrame) -> str | None:
    # Prefer exact display header, then anything containing "cost"
    if DEFAULT_COST_HEADER in df.columns:
        return DEFAULT_COST_HEADER
    # try normalized match
    col = first_present(df.columns, ["unit cost", "unit cost $", "cost", "unit price", "price"])
    if col:
        return col
    # fallback: any header with 'cost'
    for c in df.columns:
        if "cost" in c.lower():
            return c
    return None

def detect_gwp_col(df: pd.DataFrame) -> str | None:
    # Try explicit normalized aliases first
    col = first_present(df.columns, [normalize(x) for x in GWP_ALIASES])
    if col:
        return col
    # Any column containing those tokens
    for c in df.columns:
        lc = normalize(c)
        if any(alias in lc for alias in [normalize(x) for x in GWP_ALIASES]):
            return c
    return None

def detect_traci_cols(df: pd.DataFrame) -> list[str]:
    """
    Return list of columns that look like TRACI/unit impacts,
    based on our canonical names (normalized) & numeric-ness after coercion.
    """
    cols = []
    norm_to_real = {normalize(c): c for c in df.columns}
    for key in CANON_IMPACTS:
        real = norm_to_real.get(key)
        if real is not None:
            cols.append(real)
    # Also include any other numeric '.../Unit' style columns the user may have added
    for c in df.columns:
        n = normalize(c)
        if n.endswith(" unit") and ("eq" in n or "ctu" in n):
            if c not in cols:
                cols.append(c)
    return cols

def _clip(ind, lows, highs):
    for i in range(len(ind)):
        if ind[i] < lows[i]:
            ind[i] = lows[i]
        elif ind[i] > highs[i]:
            ind[i] = highs[i]
    return ind

# ===============================
# Data loader
# ===============================
@st.cache_data
def load_any(uploaded):
    """
    Reads csv/xlsx/xlsm.
    Accepts either:
      • single merged sheet with [Material, Unit, Amount, <cost>, <impacts>]
      • or 3-sheet workbook: [Inputs, Costs, Impacts] merged on [Material, Unit]
    Returns: raw_df, rolled_df, warn_msg
    """
    if uploaded is None:
        return None, None, "No file uploaded"

    name = uploaded.name.lower()
    warn = []

    # --- Read file
    if name.endswith(".csv"):
        book = None
        merged = pd.read_csv(uploaded)
        raw = merged.copy()
    else:
        # openpyxl handles xlsx/xlsm; sheet_name=None -> dict of sheets
        book = pd.read_excel(uploaded, sheet_name=None, engine="openpyxl")
        # Try single merged sheet first
        merged_candidates = [
            d for _, d in book.items()
            if set(["Material", "Unit", "Amount"]).issubset(set(d.columns))
        ]
        if merged_candidates:
            merged = merged_candidates[0].copy()
            raw = merged.copy()
        else:
            # assume 3-sheet order: Inputs, Costs, Impacts
            keys = list(book.keys())
            if len(keys) < 3:
                return None, None, ("Workbook must have either a merged sheet "
                                    "OR 3 sheets (Inputs/Costs/Impacts) with Material/Unit keys.")
            inputs, costs, impacts = book[keys[0]].copy(), book[keys[1]].copy(), book[keys[2]].copy()

            for need in ["Material", "Unit", "Amount"]:
                if need not in inputs.columns:
                    return None, None, f"Missing '{need}' in first sheet."

            raw = inputs.copy()
            merged = inputs.merge(costs, on=["Material", "Unit"], how="left")
            merged = merged.merge(impacts, on=["Material", "Unit"], how="left")

    # --- Basic sanity
    for need in ["Material", "Unit", "Amount"]:
        if need not in merged.columns:
            return None, None, "File must include columns: Material, Unit, Amount."

    # Normalize Amount numeric
    merged["Amount"] = to_numeric_strict(merged["Amount"])

    # If Year is missing, add a placeholder
    if "Year" not in merged.columns:
        merged["Year"] = 0

    # Roll up by Material+Unit
    rolled = merged.groupby(["Material", "Unit"], as_index=False)["Amount"].sum()

    # Carry first non-null for all the other columns
    other_cols = [c for c in merged.columns if c not in ["Year", "Amount"]]
    for c in other_cols:
        rolled[c] = (
            merged.groupby(["Material", "Unit"])[c]
            .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
            .reset_index(drop=True)
        )

    # === Detect cost & impacts (by header, case-insensitive, tolerant) ===
    cost_col = detect_cost_col(rolled)
    gwp_col = detect_gwp_col(rolled)

    # If missing, keep columns but warn (we’ll let the user map in UI)
    if not cost_col:
        warn.append("No unit cost column detected automatically.")
        # Create a placeholder so UI can still proceed
        rolled[DEFAULT_COST_HEADER] = 0.0
        cost_col = DEFAULT_COST_HEADER

    # Detect TRACI columns (including GWP if present)
    traci_cols = detect_traci_cols(rolled)

    # Coerce cost & all TRACI-like columns to numeric
    rolled[cost_col] = to_numeric_strict(rolled[cost_col])
    for c in traci_cols:
        rolled[c] = to_numeric_strict(rolled[c])

    # If no explicit GWP detected yet but a CO2-like column exists inside traci_cols,
    # pick the first that contains "co2"
    if not gwp_col:
        for c in traci_cols:
            if "co2" in normalize(c):
                gwp_col = c
                break

    # If still none, add a numeric column for GWP so UI shows a slot
    if not gwp_col:
        rolled["kg CO2-Eq/Unit"] = 0.0
        gwp_col = "kg CO2-Eq/Unit"
        warn.append("No explicit GWP column found; added 'kg CO2-Eq/Unit' filled with 0s.")

    return (raw.fillna(0), rolled.fillna(0), cost_col, gwp_col, traci_cols, "; ".join(warn) if warn else None)

# ===============================
# GA evaluators & engines
# ===============================
def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, df, cost_col, impact_cols):
    # Clean old DEAP classes if re-run
    for attr in ("FitnessMin", "Individual"):
        try:
            delattr(creator, attr)
        except Exception:
            pass

    n_objectives = 1 + len(impact_cols)  # cost + each impact
    creator.create("FitnessMin", base.Fitness, weights=tuple([-1.0] * n_objectives))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    impact_matrix = df[impact_cols].to_numpy(dtype=float)
    costs = df[cost_col].to_numpy(dtype=float)

    def evaluate_multi(ind):
        x = np.maximum(0.0, np.array(ind, dtype=float))
        total_cost = float(np.dot(x, costs))
        totals = [float(np.dot(x, impact_matrix[:, i])) for i in range(impact_matrix.shape[1])]
        return (total_cost, *totals)

    toolbox.register("evaluate", evaluate_multi)

    def _clip_local(individual):
        for i in range(len(individual)):
            if individual[i] < lows[i]:
                individual[i] = lows[i]
            elif individual[i] > highs[i]:
                individual[i] = highs[i]

    def make_children(pop):
        selected = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i + 1], alpha=0.5)
            _clip_local(offspring[i])
            if i + 1 < len(offspring):
                _clip_local(offspring[i + 1])
        for ind in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.2)
            _clip_local(ind)
        return offspring

    pop = toolbox.population(n=popsize)
    for ind in pop:
        _clip_local(ind)
        ind.fitness.values = toolbox.evaluate(ind)

    for _ in range(ngen):
        off = make_children(pop)
        for ind in off:
            ind.fitness.values = toolbox.evaluate(ind)
        pop = tools.selNSGA2(pop + off, popsize)

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pareto

def run_single(obj, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    for attr in ("FitnessMin", "Individual"):
        try:
            delattr(creator, attr)
        except Exception:
            pass

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return obj(ind, *args, lows, highs)

    def make_children(pop):
        selected = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i+1], alpha=0.5)
            _clip(offspring[i], lows, highs)
            if i + 1 < len(offspring):
                _clip(offspring[i+1], lows, highs)
        for ind in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.2)
            _clip(ind, lows, highs)
        return offspring

    pop = toolbox.population(n=popsize)
    for ind in pop:
        _clip(ind, lows, highs)
        ind.fitness.values = evaluate(ind)

    for _ in range(ngen):
        off = make_children(pop)
        for ind in off:
            ind.fitness.values = evaluate(ind)
        pop = tools.selBest(pop + off, popsize)

    return tools.selBest(pop, 1)[0]

# Objective helpers
def eval_cost_only(ind, df, cost_col, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    return (float(np.dot(x, df[cost_col].to_numpy(dtype=float))),)

def eval_single_impact(ind, df, impact_col, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    return (float(np.dot(x, df[impact_col].to_numpy(dtype=float))),)

def eval_cost_plus_all(ind, df, impact_cols, cost_col, alpha_cost, alpha_imp, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    total_cost = float(np.dot(x, df[cost_col].to_numpy(dtype=float)))
    total_imp  = float(np.dot(x, df[impact_cols].to_numpy(dtype=float)).sum())
    return (alpha_cost * total_cost + alpha_imp * total_imp,)

# ===============================
# UI
# ===============================
st.title("LCA Optimization: Cost vs GWP")

uploaded = st.file_uploader("Upload merged table (.csv / .xlsm / .xlsx)", type=["csv", "xlsm", "xlsx"])
raw_df, rolled_df, detected_cost, detected_gwp, detected_traci, warnings = load_any(uploaded)

if uploaded and warnings:
    st.warning(warnings)

if rolled_df is None:
    st.info("Upload a file with Material, Unit, Amount. Costs & impacts can be on separate sheets.")
    st.stop()

# Column mapping with robust defaults
st.markdown("#### Column mapping (auto-detect + manual override)")
numeric_like = [c for c in rolled_df.columns]  # allow mapping even if dtype is object; we'll coerce

# Cost
cost_col_default = detected_cost if detected_cost in rolled_df.columns else (DEFAULT_COST_HEADER if DEFAULT_COST_HEADER in rolled_df.columns else numeric_like[0])
cost_col = st.selectbox("Select Unit Cost column", options=numeric_like, index=numeric_like.index(cost_col_default))

# GWP
gwp_default = detected_gwp if detected_gwp in rolled_df.columns else numeric_like[0]
gwp_col = st.selectbox("Select GWP column", options=numeric_like, index=numeric_like.index(gwp_default))

# Other impacts (multi-select, pre-populate with detected TRACI columns except the chosen GWP)
preselect_impacts = [c for c in detected_traci if c != gwp_col]
other_impacts = st.multiselect(
    "Select additional impact columns (TRACI, etc.)",
    options=[c for c in numeric_like if c not in ["Material", "Unit", "Amount"]],
    default=preselect_impacts
)

# Coerce chosen columns to numeric now (safe even if already numeric)
rolled_df["Amount"] = to_numeric_strict(rolled_df["Amount"])
rolled_df[cost_col] = to_numeric_strict(rolled_df[cost_col])
rolled_df[gwp_col]  = to_numeric_strict(rolled_df[gwp_col])
for c in other_impacts:
    rolled_df[c] = to_numeric_strict(rolled_df[c])

# Final impact list (GWP first)
impact_cols = [gwp_col] + [c for c in other_impacts if c != gwp_col]

st.success("File uploaded, mapped, and parsed successfully.")
st.dataframe(rolled_df, use_container_width=True)

# --- Baseline QA ---
baseline_cost = float((rolled_df["Amount"] * rolled_df[cost_col]).sum())
baseline_gwp  = float((rolled_df["Amount"] * rolled_df[gwp_col]).sum())
trees = DEFAULT_TREES
b1, b2, b3, b4 = st.columns(4)
b1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
b2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
b3.metric("Cost / tree ($/tree)", f"{baseline_cost/trees:,.2f}")
b4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/trees:,.2f}")

# --- Sidebar controls ---
scenario = st.sidebar.selectbox("Optimization Scenario", [
    "Optimize Cost vs GWP (Tradeoff)",
    "Optimize Cost + Combined Impact",
    "Optimize Single Impact",
    "Optimize Cost Only",
])

global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom = st.sidebar.checkbox("Set per-material bounds")

base_amounts = rolled_df["Amount"].to_numpy(dtype=float)
materials    = rolled_df["Material"].astype(str).tolist()

lows = np.maximum(0.0, base_amounts * (1 - global_dev/100.0))
highs = base_amounts * (1 + global_dev/100.0)

if use_custom:
    st.sidebar.markdown("**Per-material ±%**")
    for i, m in enumerate(materials):
        dev = st.sidebar.number_input(m, min_value=0, max_value=100, value=global_dev, step=5, key=f"dev_{i}")
        lows[i]  = max(0.0, base_amounts[i] * (1 - dev/100.0))
        highs[i] = base_amounts[i] * (1 + dev/100.0)

# Hard minimum floors (e.g., Diesel)
st.sidebar.markdown("### Hard minimum floors (optional)")
protect_mat = st.sidebar.multiselect(
    "Choose materials to floor",
    materials,
    default=[m for m in materials if "diesel" in m.lower()]
)
for m in protect_mat:
    i = materials.index(m)
    pct = st.sidebar.number_input(f"Min % of baseline for {m}", min_value=0, max_value=100, value=80, step=5, key=f"floor_{i}")
    lows[i] = max(lows[i], base_amounts[i] * pct / 100.0)

# Bounds QA
btab = rolled_df[["Material","Unit"]].copy()
btab["Baseline"] = base_amounts
btab["Low"] = lows
btab["High"] = highs
st.markdown("#### Bounds QA")
st.dataframe(btab, use_container_width=True)

# GA params
popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.60)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.30)

# Single-impact selection
sel_impact = None
if scenario == "Optimize Single Impact":
    sel_impact = st.selectbox(
        "Select impact column",
        options=impact_cols,
        index=max(0, impact_cols.index(gwp_col) if gwp_col in impact_cols else 0)
    )

# ===============================
# Run optimization
# ===============================
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col, impact_cols)

        # Build Pareto table
        pareto_rows = []
        for ind in pareto:
            vec = _clip(list(ind), list(lows), list(highs))
            x = np.asarray(vec, float)
            row = {"Total Cost ($)": float(np.dot(x, rolled_df[cost_col].to_numpy(dtype=float)))}
            for imp in impact_cols:
                row[imp] = float(np.dot(x, rolled_df[imp].to_numpy(dtype=float)))
            pareto_rows.append(row)

        cols = ["Total Cost ($)"] + impact_cols
        df_pf = pd.DataFrame(pareto_rows, columns=cols) if pareto_rows else pd.DataFrame(columns=cols)

        st.markdown("#### Pareto front (Total Cost + all impacts)")
        st.dataframe(df_pf, use_container_width=True)

        # Download
        st.download_button(
            "Download Pareto table (CSV)",
            data=df_pf.to_csv(index=False).encode(),
            file_name="pareto_table.csv",
            mime="text/csv"
        )

        # Cost vs GWP quick plot
        if gwp_col in df_pf.columns:
            fig, ax = plt.subplots()
            ax.scatter(df_pf["Total Cost ($)"], df_pf[gwp_col])
            ax.set_xlabel("Total Cost ($)")
            ax.set_ylabel(gwp_col)
            ax.set_title("Pareto Front: Cost vs GWP")
            st.pyplot(fig)

        # Example optimized inventory: min-cost point
        best = min(pareto, key=lambda x: x.fitness.values[0])
        opt = np.array(best, float)
        inv = rolled_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt
        st.markdown("#### Example optimized inventory (min-cost on Pareto)")
        st.dataframe(inv, use_container_width=True)

        tot_cost = float(best.fitness.values[0])
        tot_gwp = float(np.dot(opt, rolled_df[gwp_col].to_numpy(dtype=float)))
        m1, m2 = st.columns(2)
        m1.metric("Cost / tree ($/tree)", f"{tot_cost/DEFAULT_TREES:,.2f}")
        m2.metric("GWP / tree (kg CO₂e/tree)", f"{tot_gwp/DEFAULT_TREES:,.2f}")

        # Per-year scaled CSV if Year present
        if (raw_df is not None) and ("Year" in raw_df.columns):
            ratio = np.divide(opt, base_amounts, out=np.ones_like(opt), where=(base_amounts>0))
            key = list(zip(rolled_df["Material"], rolled_df["Unit"]))
            scale = {k: r for k, r in zip(key, ratio)}
            out = raw_df.copy()
            out["Amount_Optimized"] = out.apply(
                lambda r: r["Amount"] * scale.get((r["Material"], r["Unit"]), 1.0), axis=1
            )
            st.download_button(
                "Download optimized per-year inventory (CSV)",
                out.to_csv(index=False).encode(),
                file_name="optimized_yearly_inventory.csv",
                mime="text/csv"
            )

    elif scenario == "Optimize Cost + Combined Impact":
        best = run_single(
            eval_cost_plus_all, popsize, ngen, cxpb, mutpb, lows, highs,
            rolled_df, impact_cols, cost_col, 1.0, 1.0
        )
        opt = np.array(best, float)
        inv = rolled_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt
        st.metric("Objective (cost + sum impacts)", f"{best.fitness.values[0]:,.2f}")
        st.dataframe(inv, use_container_width=True)

    elif scenario == "Optimize Single Impact" and sel_impact:
        best = run_single(
            eval_single_impact, popsize, ngen, cxpb, mutpb, lows, highs,
            rolled_df, sel_impact
        )
        opt = np.array(best, float)
        inv = rolled_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt
        st.metric(sel_impact, f"{best.fitness.values[0]:,.4f}")
        st.dataframe(inv, use_container_width=True)

    elif scenario == "Optimize Cost Only":
        best = run_single(
            eval_cost_only, popsize, ngen, cxpb, mutpb, lows, highs,
            rolled_df, cost_col
        )
        opt = np.array(best, float)
        inv = rolled_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt
        st.metric("Total Cost ($)", f"{best.fitness.values[0]:,.2f}")
        st.dataframe(inv, use_container_width=True)

# ===============================
# Downloads (merged totals)
# ===============================
st.markdown("#### Downloads")
excel_buf = BytesIO()
with pd.ExcelWriter(excel_buf, engine="xlsxwriter") as writer:
    rolled_df.to_excel(writer, index=False, sheet_name="Merged Totals")
excel_buf.seek(0)
st.download_button(
    "Download merged totals (Excel)",
    data=excel_buf,
    file_name="merged_totals_for_optimization.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)
