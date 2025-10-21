# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import random
import subprocess
import sys

# Ensure DEAP is installed
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization (Cost vs GWP)", layout="wide")
random.seed(42)

# ===============================
# Helpers
# ===============================

CORE_ID_COLS = ["Year", "Material", "Unit", "Amount"]
COST_COL_NAME = "Unit Cost ($)"  # expected name in your merged table

def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def find_gwp_col(cols):
    """Find a GWP-like column name on the final merged dataframe."""
    cl = [c.lower() for c in cols]
    keys = ["kg co2-eq/unit", "kg co2-eq per unit", "gwp", "climate change"]
    for key in keys:
        for i, c in enumerate(cl):
            if key in c:
                return cols[i]
    return None

def detect_impact_cols(df):
    """Return list of impact columns (all numeric 'impact-ish' cols incl. GWP)."""
    exclude = set(CORE_ID_COLS + [COST_COL_NAME])
    numeric_cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return numeric_cols

def merge_if_three_sheets(book):
    """
    Support the legacy 3-sheet format:
      Sheet 0: inputs, Sheet 1: cost, Sheet 2: impacts
    Merge on ['Material','Unit'] and return merged frame.
    """
    keys = list(book.keys())
    if len(keys) < 3:
        st.error("Workbook doesn't have 3 sheets; please upload the merged table or a 3-sheet file.")
        return None

    inputs_df = book[keys[0]].copy()
    cost_df   = book[keys[1]].copy()
    impact_df = book[keys[2]].copy()

    for col in ["Material", "Unit"]:
        if col not in inputs_df.columns:
            st.error(f"Missing column '{col}' in first sheet.")
            return None

    merged = inputs_df.merge(cost_df, on=["Material", "Unit"], how="left")
    merged = merged.merge(impact_df, on=["Material", "Unit"], how="left")

    return merged

# ===============================
# Load & Prepare Data (robust)
# ===============================

@st.cache_data
def load_data(uploaded_file):
    """
    Returns:
      merged_df (final rolled totals)
      materials, base_amounts, costs
      impact_df (final numeric impacts columns)
      impact_col_names (list[str])
      gwp_colname (str)
    """
    if uploaded_file is None:
        return (None, None, None, None, None, None)

    # Read file
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        # xls/xlsx/xlsm
        book = pd.read_excel(uploaded_file, sheet_name=None)
        # If the file already has the merged table (has the key columns), use it
        merged_candidates = [s for s, d in book.items() if set(CORE_ID_COLS).issubset(d.columns)]
        if merged_candidates:
            df = book[merged_candidates[0]].copy()
        else:
            df = merge_if_three_sheets(book)
            if df is None:
                return (None, None, None, None, None, None)

    # Basic cleanup
    if "Material" not in df.columns or "Amount" not in df.columns or "Unit" not in df.columns:
        st.error("The uploaded file must include columns: 'Material', 'Amount', 'Unit' (and preferably 'Year').")
        return (None, None, None, None, None, None)

    # Ensure numeric
    if "Year" not in df.columns:
        df["Year"] = 0
    df["Amount"] = _to_numeric(df["Amount"]).fillna(0)
    if COST_COL_NAME not in df.columns:
        df[COST_COL_NAME] = 0.0
    df[COST_COL_NAME] = _to_numeric(df[COST_COL_NAME]).fillna(0)

    # Roll up all years to full-rotation totals by Material + Unit
    rolled = (df.groupby(["Material", "Unit"], as_index=False)["Amount"].sum())

    # Bring over cost and impact columns by taking first non-null per Material+Unit
    other_cols = [c for c in df.columns if c not in ["Year", "Amount"]]
    for c in other_cols:
        rolled[c] = (df.groupby(["Material", "Unit"])[c]
                       .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                       .reset_index(drop=True))

    # Convert 19-19-19 → 15-15-15 proxy (to match your openLCA setup)
    m_npk = rolled["Material"].str.contains("19-19-19", case=False, na=False)
    if m_npk.any():
        rolled.loc[m_npk, "Amount"] = rolled.loc[m_npk, "Amount"] * (19.0 / 15.0)
        rolled.loc[m_npk, "Material"] = "NPK (15-15-15) fertiliser"

    # Identify GWP column on final frame
    gwp_col = find_gwp_col(rolled.columns)
    if gwp_col is None:
        st.warning("No explicit GWP column found. Add a 'kg CO2-Eq/Unit' (or similar) column.")
        # Create empty gwp column
        rolled["kg CO2-Eq/Unit"] = 0.0
        gwp_col = "kg CO2-Eq/Unit"

    # Identify all numeric impact columns (including the GWP one)
    impact_cols = detect_impact_cols(rolled)
    impact_df = rolled[impact_cols].copy().fillna(0.0)

    # Vectors for optimization (strictly aligned to rolled)
    materials = rolled["Material"].tolist()
    base_amounts = rolled["Amount"].to_numpy(dtype=float)
    costs = rolled[COST_COL_NAME].to_numpy(dtype=float)

    # Final merged df is what the UI shows
    merged_df = rolled.fillna(0)

    return (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)

# ===============================
# DEAP Evaluation Functions (safe to column names)
# ===============================

def evaluate_cost_gwp(ind, df, cost_col, gwp_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    cost = np.dot(x, df[cost_col].to_numpy(dtype=float))
    gwp  = np.dot(x, df[gwp_col].to_numpy(dtype=float))
    return cost, gwp

def evaluate_cost_only(ind, df, cost_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (np.dot(x, df[cost_col].to_numpy(dtype=float)),)

def evaluate_combined(ind, df, impact_cols, cost_col, alpha_cost=1.0, alpha_imp=1.0):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    impacts = df[impact_cols].to_numpy(dtype=float)  # [n, m]
    total_imp = np.dot(x, impacts).sum()
    total_cost = np.dot(x, df[cost_col].to_numpy(dtype=float))
    return (alpha_cost * total_cost + alpha_imp * total_imp,)

def evaluate_single_impact(ind, df, impact_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (np.dot(x, df[impact_col].to_numpy(dtype=float)),)

# ===============================
# Optimization Runners
# ===============================

def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, df, cost_col, gwp_col):
    try:
        del creator.FitnessMin
        del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp, df=df, cost_col=COST_COL_NAME, gwp_col=gwp_col)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    algorithms.eaMuPlusLambda(pop, toolbox, mu=popsize, lambda_=popsize,
                              cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single(obj, popsize, ngen, cxpb, mutpb, lows, highs, *args, **kwargs):
    try:
        del creator.FitnessMin
        del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj, *args, **kwargs)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popsize)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen,
                        halloffame=hof, verbose=False)
    return hof[0]

# ===============================
# UI
# ===============================

st.title("LCA Optimization: Cost vs GWP")

uploaded_file = st.file_uploader("Upload your merged table (.csv or .xlsm/.xlsx)", type=["csv", "xlsm", "xlsx"])

(merged_df,
 materials,
 base_amounts,
 costs,
 impact_df,
 impact_cols,
 gwp_col) = load_data(uploaded_file)

if merged_df is None:
    st.info("Upload a valid file to begin.")
    st.stop()

st.success("File uploaded and data processed successfully!")
st.dataframe(merged_df, use_container_width=True)

# === QA baseline from uploaded data (this prevents silent under-count) ===
baseline_cost = float((merged_df["Amount"] * merged_df[COST_COL_NAME]).sum())
baseline_gwp  = float((merged_df["Amount"] * merged_df[gwp_col]).sum())
st.markdown("### QA Baseline (from uploaded file)")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
col2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
per_tree = 1900.0
col3.metric("Cost / tree ($/tree)", f"{baseline_cost/per_tree:,.2f}")
col4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/per_tree:,.2f}")

# === Sidebar controls ===
scenario = st.sidebar.selectbox(
    "Optimization Scenario",
    ["Optimize Cost vs GWP (Tradeoff)",
     "Optimize Cost + Combined Impact",
     "Optimize Single Impact",
     "Optimize Cost Only"]
)

global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom_bounds = st.sidebar.checkbox("Set per-material bounds")

lows = np.copy(base_amounts)
highs = np.copy(base_amounts)
if use_custom_bounds:
    for i, mat in enumerate(materials):
        dev = st.sidebar.number_input(f"{mat}", min_value=0, max_value=100,
                                      value=global_dev, key=f"bound_{i}")
        lows[i] = max(0.0, base_amounts[i] * (1 - dev/100))
        highs[i] = base_amounts[i] * (1 + dev/100)
else:
    lows = np.maximum(0.0, base_amounts * (1 - global_dev/100))
    highs = base_amounts * (1 + global_dev/100)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

selected_impact = None
if scenario == "Optimize Single Impact":
    # Let the user pick any numeric impact column (including GWP)
    selected_impact = st.selectbox("Select impact column", impact_cols)

# ===============================
# Run Optimization
# ===============================
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb,
                           lows, highs, merged_df, COST_COL_NAME, gwp_col)
        df_out = pd.DataFrame(
            [[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
            columns=["Total Cost", "Total GWP"]
        )
        st.dataframe(df_out, use_container_width=True)

        fig, ax = plt.subplots()
        ax.scatter(df_out["Total Cost"], df_out["Total GWP"])
        ax.set_xlabel("Total Cost ($)")
        ax.set_ylabel("Total GWP (kg CO₂e)")
        st.pyplot(fig)

        # Per-tree view
        per_tree_df = df_out.copy()
        per_tree_df["Cost / tree"] = per_tree_df["Total Cost"] / per_tree
        per_tree_df["GWP / tree"]  = per_tree_df["Total GWP"]  / per_tree
        st.dataframe(per_tree_df[["Cost / tree", "GWP / tree"]])

        st.session_state.setdefault("history", []).append(
            {"scenario": scenario, "results": df_out}
        )

    elif scenario == "Optimize Cost + Combined Impact":
        best = run_single(evaluate_combined, popsize, ngen, cxpb, mutpb,
                          lows, highs, merged_df, impact_cols, COST_COL_NAME)
        df_out = pd.DataFrame({"Material": materials,
                               "Base Amount": base_amounts,
                               "Optimized": best})
        st.metric("Objective", f"{best.fitness.values[0]:.2f}")
        st.dataframe(df_out, use_container_width=True)
        st.session_state.setdefault("history", []).append(
            {"scenario": scenario, "results": df_out}
        )

    elif scenario == "Optimize Cost Only":
        best = run_single(evaluate_cost_only, popsize, ngen, cxpb, mutpb,
                          lows, highs, merged_df, COST_COL_NAME)
        df_out = pd.DataFrame({"Material": materials,
                               "Base Amount": base_amounts,
                               "Optimized": best})
        st.metric("Cost", f"{best.fitness.values[0]:.2f}")
        st.dataframe(df_out, use_container_width=True)
        st.session_state.setdefault("history", []).append(
            {"scenario": scenario, "results": df_out}
        )

    elif scenario == "Optimize Single Impact" and selected_impact:
        best = run_single(evaluate_single_impact, popsize, ngen, cxpb, mutpb,
                          lows, highs, merged_df, selected_impact)
        df_out = pd.DataFrame({"Material": materials,
                               "Base Amount": base_amounts,
                               "Optimized": best})
        st.metric(f"{selected_impact}", f"{best.fitness.values[0]:.2f}")
        st.dataframe(df_out, use_container_width=True)
        st.session_state.setdefault("history", []).append(
            {"scenario": f"Single Impact - {selected_impact}", "results": df_out}
        )

# ===============================
# Download merged data
# ===============================
if st.button("Download Merged Data as Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
    output.seek(0)
    st.download_button(
        label="Download Excel File",
        data=output,
        file_name="merged_totals_for_optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

# ===============================
# History viewer
# ===============================
st.markdown("### 📈 View Optimization History")
if 'history' in st.session_state and st.session_state['history']:
    for i, record in enumerate(st.session_state['history'], 1):
        st.write(f"**Run {i}: {record['scenario']}**")
        st.dataframe(record['results'], use_container_width=True)
else:
    st.info("No optimization runs recorded yet.")
