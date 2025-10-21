# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import random
import subprocess
import sys

# ============= Ensure DEAP =============
try:
    from deap import base, creator, tools, algorithms
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ------------------------------
# Constants / helpers
# ------------------------------
CORE_ID = ["Material", "Unit", "Amount"]
DEFAULT_COST_COL = "Unit Cost ($)"              # your merged table cost header
GWP_KEYS = ["kg co2-eq/unit", "kg co2 eq/unit", "kg co₂-eq/unit", "gwp", "climate change"]
PROTECT_DEFAULTS = ["diesel"]                   # materials that often need hard floors
DEFAULT_TREES = 1900.0

def to_num(s):
    return pd.to_numeric(s, errors="coerce")

def find_column_caseinsensitive(df, names):
    cols_lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in cols_lower:
            return cols_lower[n.lower()]
    return None

def find_gwp_col(df):
    for key in GWP_KEYS:
        c = find_column_caseinsensitive(df, [key])
        if c: return c
    # also scan for any column that contains the key
    for c in df.columns:
        lc = c.lower()
        if any(k in lc for k in GWP_KEYS):
            return c
    return None

def detect_impact_cols(df, cost_col):
    exclude = set(["Year", "Amount", "Material", "Unit", cost_col])
    nc = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            nc.append(c)
    return nc

def clip_to_bounds(ind, lows, highs):
    for i in range(len(ind)):
        if ind[i] < lows[i]:
            ind[i] = lows[i]
        elif ind[i] > highs[i]:
            ind[i] = highs[i]
    return ind

# ------------------------------
# Loading (CSV or XLSM/XLSX)
# Accepts: one merged sheet; or 3-sheet (Inputs/Costs/Impacts)
# ------------------------------
@st.cache_data
def load_data(uploaded):
    """
    Returns:
      raw_df      : original (first sheet or CSV) for per-year scaling if 'Year' exists
      rolled_df   : totals merged by Material+Unit with costs/impacts
      cost_col    : chosen cost column (or created if missing)
      gwp_col     : chosen gwp column (created if missing)
      impact_cols : numeric impacts including gwp
      msg         : None or string with warnings
    """
    if uploaded is None:
        return None, None, None, None, None, "No file"

    name = uploaded.name.lower()
    msg = []

    if name.endswith(".csv"):
        book = None
        raw = pd.read_csv(uploaded)
        one_sheet = raw
    else:
        book = pd.read_excel(uploaded, sheet_name=None)
        # try to find any sheet holding Material/Unit/Amount (merged)
        candidates = [df for _, df in book.items() if set(["Material", "Unit", "Amount"]).issubset(df.columns)]
        if candidates:
            one_sheet = candidates[0].copy()
            raw = one_sheet.copy()
        else:
            # assume classic 3-sheet: inputs, costs, impacts
            keys = list(book.keys())
            if len(keys) < 3:
                return None, None, None, None, None, "Workbook needs merged table or 3 sheets (Inputs/Costs/Impacts)."
            inputs = book[keys[0]].copy()
            costs  = book[keys[1]].copy()
            imps   = book[keys[2]].copy()
            for col in ["Material", "Unit", "Amount"]:
                if col not in inputs.columns:
                    return None, None, None, None, None, f"Missing '{col}' in first sheet."
            # stash raw
            raw = inputs.copy()
            one_sheet = inputs.merge(costs, on=["Material", "Unit"], how="left")
            one_sheet = one_sheet.merge(imps,    on=["Material", "Unit"], how="left")

    # sanitize
    for need in ["Material", "Unit", "Amount"]:
        if need not in one_sheet.columns:
            return None, None, None, None, None, "Upload must include 'Material','Unit','Amount'."

    # numeric
    if "Year" not in one_sheet.columns:
        one_sheet["Year"] = 0
    one_sheet["Amount"] = to_num(one_sheet["Amount"]).fillna(0)

    # costs
    cost_col = DEFAULT_COST_COL if DEFAULT_COST_COL in one_sheet.columns else None
    if cost_col is None:
        # try any column containing "cost"
        for c in one_sheet.columns:
            if "cost" in c.lower():
                cost_col = c
                break
    if cost_col is None:
        one_sheet[DEFAULT_COST_COL] = 0.0
        cost_col = DEFAULT_COST_COL
        msg.append("No unit cost column found; created zero-cost column.")

    one_sheet[cost_col] = to_num(one_sheet[cost_col]).fillna(0.0)

    # roll totals by Material+Unit
    rolled = one_sheet.groupby(["Material", "Unit"], as_index=False)["Amount"].sum()
    # carry first non-null values of other fields (costs + impacts)
    other_cols = [c for c in one_sheet.columns if c not in ["Year", "Amount"]]
    for c in other_cols:
        rolled[c] = (one_sheet.groupby(["Material", "Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))

    # GWP column
    gwp_col = find_gwp_col(rolled)
    if gwp_col is None:
        rolled["kg CO2-Eq/Unit"] = 0.0
        gwp_col = "kg CO2-Eq/Unit"
        msg.append("No explicit GWP column found; added 'kg CO2-Eq/Unit' filled with 0s.")

    # impacts
    impact_cols = detect_impact_cols(rolled, cost_col)
    if gwp_col not in impact_cols:
        impact_cols = [gwp_col] + impact_cols

    # finalize numeric
    for c in [cost_col] + impact_cols:
        rolled[c] = to_num(rolled[c]).fillna(0.0)

    return raw, rolled.fillna(0), cost_col, gwp_col, impact_cols, ("; ".join(msg) if msg else None)

# ------------------------------
# Evaluators (cost/gwp/impacts)
# ------------------------------
def eval_cost_gwp(ind, df, cost_col, gwp_col, lows, highs):
    clip_to_bounds(ind, lows, highs)
    x = np.asarray(ind, dtype=float)
    cost = float(np.dot(x, df[cost_col].to_numpy(float)))
    gwp  = float(np.dot(x, df[gwp_col].to_numpy(float)))
    return cost, gwp

def eval_cost_only(ind, df, cost_col, lows, highs):
    clip_to_bounds(ind, lows, highs)
    x = np.asarray(ind, dtype=float)
    return (float(np.dot(x, df[cost_col].to_numpy(float))),)

def eval_single_impact(ind, df, impact_col, lows, highs):
    clip_to_bounds(ind, lows, highs)
    x = np.asarray(ind, dtype=float)
    return (float(np.dot(x, df[impact_col].to_numpy(float))),)

def eval_cost_plus_all_impacts(ind, df, impact_cols, cost_col, alpha_cost, alpha_imp, lows, highs):
    clip_to_bounds(ind, lows, highs)
    x = np.asarray(ind, dtype=float)
    total_cost = float(np.dot(x, df[cost_col].to_numpy(float)))
    total_imp  = float(np.dot(x, df[impact_cols].to_numpy(float)).sum())
    return (alpha_cost * total_cost + alpha_imp * total_imp,)

# ------------------------------
# NSGA-II with strict bounds
# ------------------------------
def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, df, cost_col, gwp_col):
    try:
        del creator.FitnessMin; del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def make_children(inds):
        # tournament selection
        selected = tools.selTournament(inds, len(inds), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]

        # crossover + clip
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i+1], alpha=0.5)
            clip_to_bounds(offspring[i], lows, highs)
            if i + 1 < len(offspring):
                clip_to_bounds(offspring[i+1], lows, highs)

        # mutation + clip
        for ind in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.2)
            clip_to_bounds(ind, lows, highs)
        return offspring

    def evaluate(ind):
        return eval_cost_gwp(ind, df, cost_col, gwp_col, lows, highs)

    # initialize
    pop = toolbox.population(n=popsize)
    for ind in pop:
        clip_to_bounds(ind, lows, highs)
        ind.fitness.values = evaluate(ind)

    for _ in range(ngen):
        offspring = make_children(pop)
        for ind in offspring:
            ind.fitness.values = evaluate(ind)
        pop = tools.selNSGA2(pop + offspring, popsize)

    pareto = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]
    return pareto

# ------------------------------
# Single-objective driver
# ------------------------------
def run_single(obj_fn, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    try:
        del creator.FitnessMin; del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(ind):
        return obj_fn(ind, *args, lows, highs)

    def make_children(inds):
        selected = tools.selTournament(inds, len(inds), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]
        for i in range(0, len(offspring), 2):
            if i + 1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i+1], alpha=0.5)
            clip_to_bounds(offspring[i], lows, highs)
            if i + 1 < len(offspring):
                clip_to_bounds(offspring[i+1], lows, highs)
        for ind in offspring:
            if random.random() < mutpb:
                tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.2)
            clip_to_bounds(ind, lows, highs)
        return offspring

    pop = toolbox.population(n=popsize)
    for ind in pop:
        clip_to_bounds(ind, lows, highs)
        ind.fitness.values = evaluate(ind)

    for _ in range(ngen):
        offspring = make_children(pop)
        for ind in offspring:
            ind.fitness.values = evaluate(ind)
        pop = tools.selBest(pop + offspring, popsize)

    best = tools.selBest(pop, 1)[0]
    return best

# ------------------------------
# UI
# ------------------------------
st.title("LCA Optimization: Cost vs GWP")

uploaded = st.file_uploader("Upload merged table (.csv / .xlsm / .xlsx)", type=["csv", "xlsm", "xlsx"])

raw_df, merged_df, cost_col, gwp_col, impact_cols, warn = load_data(uploaded)
if uploaded and warn:
    st.warning(warn)

if merged_df is None:
    st.info("Upload a file with at least Material, Unit, Amount. Costs & impacts may be on separate sheets.")
    st.stop()

st.success("File uploaded and merged successfully.")
st.dataframe(merged_df, use_container_width=True)

# Baseline QA
baseline_cost = float((merged_df["Amount"] * merged_df[cost_col]).sum())
baseline_gwp  = float((merged_df["Amount"] * merged_df[gwp_col]).sum())
trees = DEFAULT_TREES
st.markdown("### QA Baseline (from uploaded totals)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
c2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
c3.metric("Cost / tree ($/tree)", f"{baseline_cost/trees:,.2f}")
c4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/trees:,.2f}")

# Sidebar
scenario = st.sidebar.selectbox("Optimization Scenario", [
    "Optimize Cost vs GWP (Tradeoff)",
    "Optimize Cost + Combined Impact",
    "Optimize Single Impact",
    "Optimize Cost Only",
])

global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom = st.sidebar.checkbox("Set per-material bounds")

base_amounts = merged_df["Amount"].to_numpy(float)
materials    = merged_df["Material"].tolist()

lows = np.maximum(0.0, base_amounts * (1 - global_dev/100.0))
highs = base_amounts * (1 + global_dev/100.0)

if use_custom:
    st.sidebar.markdown("**Per-material ±%**")
    for i, m in enumerate(materials):
        dev = st.sidebar.number_input(m, min_value=0, max_value=100, value=global_dev, step=5, key=f"dev_{i}")
        lows[i]  = max(0.0, base_amounts[i] * (1 - dev/100.0))
        highs[i] = base_amounts[i] * (1 + dev/100.0)

# Hard floors (keep certain items from dropping unrealistically)
st.sidebar.markdown("---")
st.sidebar.markdown("**Hard minimum floors (optional)**")
protect_mask = merged_df["Material"].str.lower().apply(lambda s: any(k in s for k in PROTECT_DEFAULTS))
for i, m in enumerate(materials):
    if protect_mask.iloc[i]:
        pct = st.sidebar.number_input(f"Min % of baseline for {m}", min_value=0, max_value=100, value=80, step=5, key=f"floor_{i}")
        lows[i] = max(lows[i], base_amounts[i] * pct / 100.0)

# Bounds QA table
bdf = merged_df[["Material","Unit"]].copy()
bdf["Baseline"] = base_amounts
bdf["Low"] = lows
bdf["High"] = highs
st.markdown("### Bounds QA")
st.dataframe(bdf, use_container_width=True)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

# Single-impact selector
selected_impact = None
if scenario == "Optimize Single Impact":
    selected_impact = st.selectbox("Choose impact column", impact_cols, index=max(0, impact_cols.index(gwp_col) if gwp_col in impact_cols else 0))

# Run
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, merged_df, cost_col, gwp_col)
        df_out = pd.DataFrame([[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
                              columns=["Total Cost ($)", "Total GWP (kg CO₂e)"])
        st.dataframe(df_out, use_container_width=True)

        fig, ax = plt.subplots()
        ax.scatter(df_out["Total Cost ($)"], df_out["Total GWP (kg CO₂e)"])
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Total GWP (kg CO₂e)")
        ax.set_title("Pareto front: Cost vs GWP")
        st.pyplot(fig)

        # Example optimized inventory: pick min-cost point on Pareto
        best_cost_ind = min(pareto, key=lambda x: x.fitness.values[0])
        opt_amounts = np.array(best_cost_ind, float)
        inv = merged_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt_amounts
        st.markdown("### Example optimized inventory (min-cost on Pareto)")
        st.dataframe(inv, use_container_width=True)

        # per-tree for that solution
        tot_cost = float(best_cost_ind.fitness.values[0])
        tot_gwp  = float(best_cost_ind.fitness.values[1])
        st.metric("Cost / tree ($/tree)", f"{tot_cost/trees:,.2f}")
        st.metric("GWP / tree (kg CO₂e/tree)", f"{tot_gwp/trees:,.2f}")

        # Download optimized year-by-year if Year present in raw
        if (raw_df is not None) and ("Year" in raw_df.columns):
            # scale by ratio optimized/base per Material+Unit
            ratio = np.divide(opt_amounts, base_amounts, out=np.ones_like(opt_amounts), where=(base_amounts>0))
            key = list(zip(merged_df["Material"], merged_df["Unit"]))
            scale = {k: r for k, r in zip(key, ratio)}
            out_raw = raw_df.copy()
            out_raw["Amount_Optimized"] = out_raw.apply(
                lambda r: r["Amount"] * scale.get((r["Material"], r["Unit"]), 1.0), axis=1
            )
            csv = out_raw.to_csv(index=False).encode()
            st.download_button("Download optimized per-year inventory (CSV)", csv, file_name="optimized_yearly_inventory.csv", mime="text/csv")

    elif scenario == "Optimize Cost + Combined Impact":
        best = run_single(
            eval_cost_plus_all_impacts, popsize, ngen, cxpb, mutpb, lows, highs,
            merged_df, impact_cols, cost_col, 1.0, 1.0
        )
        opt_amounts = np.array(best, float)
        inv = merged_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt_amounts
        st.metric("Objective (cost + sum impacts)", f"{best.fitness.values[0]:,.2f}")
        st.dataframe(inv, use_container_width=True)

    elif scenario == "Optimize Single Impact" and selected_impact:
        best = run_single(
            eval_single_impact, popsize, ngen, cxpb, mutpb, lows, highs,
            merged_df, selected_impact
        )
        opt_amounts = np.array(best, float)
        inv = merged_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt_amounts
        st.metric(selected_impact, f"{best.fitness.values[0]:,.4f}")
        st.dataframe(inv, use_container_width=True)

    elif scenario == "Optimize Cost Only":
        best = run_single(
            eval_cost_only, popsize, ngen, cxpb, mutpb, lows, highs,
            merged_df, cost_col
        )
        opt_amounts = np.array(best, float)
        inv = merged_df[["Material","Unit"]].copy()
        inv["Base Amount"] = base_amounts
        inv["Optimized Amount"] = opt_amounts
        st.metric("Total Cost ($)", f"{best.fitness.values[0]:,.2f}")
        st.dataframe(inv, use_container_width=True)

# Downloads (merged totals)
st.markdown("### Downloads")
if st.button("Download merged totals (Excel)"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name="Merged Totals")
    output.seek(0)
    st.download_button(
        "Save file",
        data=output,
        file_name="merged_totals_for_optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
