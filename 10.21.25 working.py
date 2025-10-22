# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import sys
from io import BytesIO

# --- Ensure DEAP is present ---
try:
    from deap import base, creator, tools, algorithms
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ===============================
# Helpers & constants
# ===============================
DEFAULT_TREES = 1900.0
DEFAULT_COST_HEADER = "Unit Cost ($)"
GWP_KEYS = [
    "kg co2-eq/unit", "kg co2 eq/unit", "kg co₂-eq/unit",
    "kg co2-eq per unit", "gwp", "climate change"
]

def _to_num(s):
    return pd.to_numeric(s, errors="coerce")

def _clip(ind, lows, highs):
    for i in range(len(ind)):
        if ind[i] < lows[i]:
            ind[i] = lows[i]
        elif ind[i] > highs[i]:
            ind[i] = highs[i]
    return ind

def _find_case_insensitive(df, targets):
    lowmap = {c.lower(): c for c in df.columns}
    for t in targets:
        if t.lower() in lowmap:
            return lowmap[t.lower()]
    return None

def _guess_gwp_col(df):
    # Prefer explicit name
    c = _find_case_insensitive(df, GWP_KEYS)
    if c: return c
    # Otherwise any column containing a key
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in GWP_KEYS):
            return c
    return None

def _impact_numeric_cols(df, cost_col):
    exclude = set(["Year", "Material", "Unit", "Amount", cost_col])
    cols = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            cols.append(c)
    return cols

# ===============================
# Data loader (CSV or XLSM/XLSX)
# - Single merged sheet OR 3-sheet (inputs/costs/impacts)
# ===============================
@st.cache_data
def load_any(uploaded):
    if uploaded is None:
        return None, None, "No file uploaded"

    name = uploaded.name.lower()
    warn = []

    # --- Read file
    if name.endswith(".csv"):
        book = None
        raw = pd.read_csv(uploaded)
        merged = raw.copy()
    else:
        book = pd.read_excel(uploaded, sheet_name=None)
        # single merged candidate
        merged_candidates = [d for _, d in book.items()
                             if set(["Material", "Unit", "Amount"]).issubset(d.columns)]
        if merged_candidates:
            merged = merged_candidates[0].copy()
            raw = merged.copy()
        else:
            # assume classic 3-sheet
            keys = list(book.keys())
            if len(keys) < 3:
                return None, None, "Workbook must have a merged sheet OR 3 sheets (Inputs/Costs/Impacts)."
            inputs, costs, impacts = book[keys[0]].copy(), book[keys[1]].copy(), book[keys[2]].copy()
            for need in ["Material", "Unit", "Amount"]:
                if need not in inputs.columns:
                    return None, None, f"Missing '{need}' in first sheet."
            raw = inputs.copy()
            merged = inputs.merge(costs, on=["Material", "Unit"], how="left")
            merged = merged.merge(impacts, on=["Material", "Unit"], how="left")

    # --- Sanity
    for need in ["Material", "Unit", "Amount"]:
        if need not in merged.columns:
            return None, None, "File must include columns: Material, Unit, Amount."

    # fill Year if missing
    if "Year" not in merged.columns:
        merged["Year"] = 0

    # numeric
    merged["Amount"] = _to_num(merged["Amount"]).fillna(0.0)

    # roll totals by Material+Unit (sum Amount), carry first non-null of other cols
    rolled = merged.groupby(["Material", "Unit"], as_index=False)["Amount"].sum()
    other_cols = [c for c in merged.columns if c not in ["Year", "Amount"]]
    for c in other_cols:
        rolled[c] = (merged.groupby(["Material", "Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))

    # cost detection (might fail; user will map below)
    cost_col = DEFAULT_COST_HEADER if DEFAULT_COST_HEADER in rolled.columns else None
    if cost_col is None:
        # find any col with "cost" in its name
        for c in rolled.columns:
            if "cost" in c.lower():
                cost_col = c
                break
    if cost_col is None:
        rolled[DEFAULT_COST_HEADER] = 0.0
        cost_col = DEFAULT_COST_HEADER
        warn.append("No unit cost column found; created zero-cost column.")

    rolled[cost_col] = _to_num(rolled[cost_col]).fillna(0.0)

    # gwp detection (might fail; user will map below)
    gwp_col = _guess_gwp_col(rolled)
    if gwp_col is None:
        rolled["kg CO2-Eq/Unit"] = 0.0
        gwp_col = "kg CO2-Eq/Unit"
        warn.append("No explicit GWP column found; added 'kg CO2-Eq/Unit' filled with 0s.")

    # impact set
    impact_cols = _impact_numeric_cols(rolled, cost_col)
    if gwp_col not in impact_cols:
        impact_cols = [gwp_col] + impact_cols

    # numeric all impacts
    for c in [cost_col] + impact_cols:
        rolled[c] = _to_num(rolled[c]).fillna(0.0)

    return (raw.fillna(0), rolled.fillna(0), "; ".join(warn) if warn else None)

# ===============================
# Evaluators (with hard clipping)
# ===============================
def eval_cost_gwp(ind, df, cost_col, gwp_col, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    cost = float(np.dot(x, df[cost_col].to_numpy(float)))
    gwp  = float(np.dot(x, df[gwp_col].to_numpy(float)))
    return cost, gwp

def eval_cost_only(ind, df, cost_col, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    return (float(np.dot(x, df[cost_col].to_numpy(float))),)

def eval_single_impact(ind, df, impact_col, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    return (float(np.dot(x, df[impact_col].to_numpy(float))),)

def eval_cost_plus_all(ind, df, impact_cols, cost_col, alpha_cost, alpha_imp, lows, highs):
    _clip(ind, lows, highs)
    x = np.asarray(ind, float)
    total_cost = float(np.dot(x, df[cost_col].to_numpy(float)))
    total_imp  = float(np.dot(x, df[impact_cols].to_numpy(float)).sum())
    return (alpha_cost * total_cost + alpha_imp * total_imp,)

# ===============================
# NSGA-II (strict bounds)
# ===============================
def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, df, cost_col, impact_cols):
    """
    Multi-objective NSGA-II: minimize Total Cost and each impact in impact_cols.

    Args:
        popsize, ngen, cxpb, mutpb, lows, highs: GA params and bounds
        df: rolled dataframe providing cost & impacts
        cost_col: column name for unit cost
        impact_cols: list of impact column names (order preserved; GWP should be first if desired)

    Returns:
        list: Pareto front (first front) of nondominated individuals
    """
    # Clean up any previously created DEAP classes to avoid redefinition errors
    try:
        del creator.FitnessMin
    except Exception:
        pass
    try:
        del creator.Individual
    except Exception:
        pass

    # number of objectives = 1 (cost) + number of impact columns
    n_objectives = 1 + len(impact_cols)
    weights = tuple([-1.0] * n_objectives)  # minimize all objectives

    creator.create("FitnessMin", base.Fitness, weights=weights)
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    impact_matrix = df[impact_cols].to_numpy(float)
    costs = df[cost_col].to_numpy(float)

    def evaluate_multi(ind):
        x = np.maximum(0.0, np.array(ind, dtype=float))
        total_cost = float(np.dot(x, costs))
        total_impacts = []
        # preserve order of impact_cols
        for i in range(impact_matrix.shape[1]):
            total_impacts.append(float(np.dot(x, impact_matrix[:, i])))
        return (total_cost,) + tuple(total_impacts)

    toolbox.register("evaluate", evaluate_multi)

    # Simple genetic operators with hard clipping
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

# ===============================
# Single-objective runner
# ===============================
def run_single(obj, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    try: del creator.FitnessMin; del creator.Individual
    except Exception: pass

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
            if i + 1 < len(offspring): _clip(offspring[i+1], lows, highs)
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

# ===============================
# UI
# ===============================
st.title("LCA Optimization: Cost vs GWP")

uploaded = st.file_uploader("Upload merged table (.csv / .xlsm / .xlsx)", type=["csv", "xlsm", "xlsx"])
raw_df, rolled_df, warnings = load_any(uploaded)

if uploaded and warnings:
    st.warning(warnings)

if rolled_df is None:
    st.info("Upload a file with Material, Unit, Amount. Costs & GWP can be on separate sheets.")
    st.stop()

# --- Column mapping (fix for 'zeros in table') ---
st.markdown("#### Column mapping (if auto-detection failed)")
numeric_cols = [c for c in rolled_df.columns if pd.api.types.is_numeric_dtype(rolled_df[c])]

# Guess cost/gwp again, but let user pick
guessed_cost = DEFAULT_COST_HEADER if DEFAULT_COST_HEADER in rolled_df.columns else None
if not guessed_cost:
    for c in rolled_df.columns:
        if "cost" in c.lower():
            guessed_cost = c; break
if not guessed_cost and numeric_cols:
    guessed_cost = numeric_cols[0]

guessed_gwp = _guess_gwp_col(rolled_df) or "kg CO2-Eq/Unit"
if guessed_gwp not in rolled_df.columns:
    rolled_df[guessed_gwp] = 0.0

col1, col2 = st.columns(2)
cost_col = col1.selectbox("Select Unit Cost column", options=numeric_cols, index=max(numeric_cols.index(guessed_cost) if guessed_cost in numeric_cols else 0, 0))
gwp_col  = col2.selectbox("Select GWP column", options=numeric_cols, index=max(numeric_cols.index(guessed_gwp) if guessed_gwp in numeric_cols else 0, 0))

# Ensure numeric
rolled_df[cost_col] = _to_num(rolled_df[cost_col]).fillna(0.0)
rolled_df[gwp_col]  = _to_num(rolled_df[gwp_col]).fillna(0.0)

# Impact list (include gwp)
impact_cols = _impact_numeric_cols(rolled_df, cost_col)
if gwp_col not in impact_cols:
    impact_cols = [gwp_col] + impact_cols

st.success("File uploaded and merged successfully.")
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

base_amounts = rolled_df["Amount"].to_numpy(float)
materials    = rolled_df["Material"].tolist()

lows = np.maximum(0.0, base_amounts * (1 - global_dev/100.0))
highs = base_amounts * (1 + global_dev/100.0)

if use_custom:
    st.sidebar.markdown("**Per-material ±%**")
    for i, m in enumerate(materials):
        dev = st.sidebar.number_input(m, min_value=0, max_value=100, value=global_dev, step=5, key=f"dev_{i}")
        lows[i]  = max(0.0, base_amounts[i] * (1 - dev/100.0))
        highs[i] = base_amounts[i] * (1 + dev/100.0)

# Hard minimum floors (e.g., Diesel) — choose any material and set floor
st.sidebar.markdown("### Hard minimum floors (optional)")
protect_mat = st.sidebar.multiselect("Choose materials to floor", materials, default=[m for m in materials if "diesel" in m.lower()])
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
    sel_impact = st.selectbox("Select impact column", impact_cols, index=max(0, impact_cols.index(gwp_col) if gwp_col in impact_cols else 0))

# ===============================
# Run optimization
# ===============================
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        # Run NSGA-II (cost vs GWP) to get the Pareto set
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col, gwp_col)

        # Build a multi-objective Pareto table: Total Cost + all impact columns
        pareto_rows = []
        for ind in pareto:
            vec = _clip(list(ind), list(lows), list(highs))
            x = np.asarray(vec, float)
            row = {"Total Cost ($)": float(np.dot(x, rolled_df[cost_col].to_numpy(float)))}
            for imp in impact_cols:
                row[imp] = float(np.dot(x, rolled_df[imp].to_numpy(float)))
            pareto_rows.append(row)

        cols = ["Total Cost ($)"] + impact_cols
        df_pf = pd.DataFrame(pareto_rows, columns=cols) if pareto_rows else pd.DataFrame(columns=cols)

        st.markdown("#### Pareto front (Total Cost + all impacts)")
        st.dataframe(df_pf, use_container_width=True)

        # Download Pareto table (CSV)
        csv_bytes = df_pf.to_csv(index=False).encode()
        st.download_button(
            "Download Pareto table (CSV)",
            data=csv_bytes,
            file_name="pareto_table.csv",
            mime="text/csv"
        )

        # Quick visualization: Cost vs GWP (if GWP column present)
        if gwp_col in df_pf.columns:
            fig, ax = plt.subplots()
            ax.scatter(df_pf["Total Cost ($)"], df_pf[gwp_col])
            ax.set_xlabel("Total Cost ($)")
            ax.set_ylabel(gwp_col)
            ax.set_title("Pareto Front: Cost vs GWP")
            st.pyplot(fig)
        else:
            st.info("GWP column not present in Pareto table for plotting.")

        # Show a concrete optimized inventory: min-cost point on Pareto
        best = min(pareto, key=lambda x: x.fitness.values[0])
        opt = np.array(best, float)
        inv = rolled_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt
        st.markdown("#### Example optimized inventory (min-cost on Pareto)")
        st.dataframe(inv, use_container_width=True)

        tot_cost = float(best.fitness.values[0])
        # compute GWP from optimized vector to remain consistent with the displayed table
        tot_gwp = float(np.dot(np.asarray(best, float), rolled_df[gwp_col].to_numpy(float))) if gwp_col in rolled_df.columns else 0.0
        st.metric("Cost / tree ($/tree)", f"{tot_cost/DEFAULT_TREES:,.2f}")
        st.metric("GWP / tree (kg CO₂e/tree)", f"{tot_gwp/DEFAULT_TREES:,.2f}")

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
if st.button("Download merged totals (Excel)"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        rolled_df.to_excel(writer, index=False, sheet_name="Merged Totals")
    output.seek(0)
    st.download_button(
        "Save file",
        data=output,
        file_name="merged_totals_for_optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
