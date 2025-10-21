import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import random
import subprocess
import sys
from typing import List, Optional

# Ensure DEAP is installed
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization (Cost vs GWP)", layout="wide")
random.seed(42)

# ---------------------------
# Helpers / schema
# ---------------------------
CORE_MIN = {"Material", "Unit", "Amount"}        # Year optional
COST_CANDIDATES = [
    "unit cost ($)", "unit cost", "cost/unit", "cost per unit", "unit_cost", "cost"
]
# normalize helper
def norm(s: str) -> str:
    return "".join(c for c in s.lower().strip() if c.isalnum())

def pick_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    norm_map = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        for c in df.columns:
            if norm(c) == n:
                return c
    # fuzzy contains
    for c in df.columns:
        if any(n in norm(c) for n in [norm(x) for x in candidates]):
            return c
    return None

def find_gwp_col(cols: List[str]) -> Optional[str]:
    cand = [
        "kg co2-eq/unit", "kgco2eq/unit", "kg co2 eq/unit",
        "kg co2-eq per unit", "kgco2eqperunit", "gwp", "climate change"
    ]
    ncols = {norm(c): c for c in cols}
    for k in cand:
        if norm(k) in ncols:
            return ncols[norm(k)]
    # fuzzy contains
    for c in cols:
        nc = norm(c)
        if "kgco2" in nc or "co2eq" in nc or "gwp" in nc:
            return c
    return None

def detect_impact_cols(df: pd.DataFrame, cost_col: str) -> List[str]:
    exclude = set(["Year", "Material", "Unit", "Amount", cost_col])
    imps = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            imps.append(c)
    return imps

def merge_if_three_sheets(book: dict) -> Optional[pd.DataFrame]:
    # Try to find sheets by likely names
    def find_sheet(keys, terms):
        for k in keys:
            nk = norm(k)
            if any(t in nk for t in terms):
                return k
        return None

    keys = list(book.keys())
    if len(keys) < 3:
        return None

    s_inputs  = find_sheet(keys, ["input", "amount", "material"]) or keys[0]
    s_costs   = find_sheet(keys, ["cost", "price"]) or keys[1]
    s_impacts = find_sheet(keys, ["impact", "traci", "factors"]) or keys[2]

    inputs_df  = book[s_inputs].copy()
    cost_df    = book[s_costs].copy()
    impact_df  = book[s_impacts].copy()

    if not CORE_MIN.issubset(inputs_df.columns):
        return None

    merged = inputs_df.merge(cost_df,   on=["Material", "Unit"], how="left")
    merged = merged.merge(impact_df,    on=["Material", "Unit"], how="left")
    return merged

# ---------------------------
# Load & prepare data
# ---------------------------
@st.cache_data
def load_data(uploaded_file):
    """
    Returns 7-tuple ALWAYS:
      merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col
    """
    if uploaded_file is None:
        return (None, None, None, None, None, None, None)

    name = uploaded_file.name.lower()
    # 1) Read
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        book = pd.read_excel(uploaded_file, sheet_name=None)
        # Try to detect a single merged sheet (Material,Unit,Amount present; Year optional)
        merged_candidates = []
        for s, d in book.items():
            if CORE_MIN.issubset(set(d.columns)):
                merged_candidates.append(s)
        if merged_candidates:
            # prefer a sheet literally named 'Merged' when present
            preferred = [s for s in merged_candidates if "merged" in s.lower()]
            sheet = preferred[0] if preferred else merged_candidates[0]
            df = book[sheet].copy()
        else:
            # Try legacy 3-sheet merge
            df = merge_if_three_sheets(book)
            if df is None:
                return (None, None, None, None, None, None, None)

    # 2) Basic cleanup / coercion
    if not CORE_MIN.issubset(set(df.columns)):
        return (None, None, None, None, None, None, None)
    if "Year" not in df.columns:
        df["Year"] = 0
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)

    # 3) Locate cost & gwp columns robustly
    cost_col = pick_column(df, COST_CANDIDATES) or "Unit Cost ($)"
    if cost_col not in df.columns:
        df[cost_col] = 0.0
    df[cost_col] = pd.to_numeric(df[cost_col], errors="coerce")

    gwp_col = find_gwp_col(df.columns)
    if gwp_col is None:
        # keep a named GWP column (zeros) so downstream code runs
        st.warning("No explicit GWP column found. Add a 'kg CO2-Eq/Unit' (or similar) column.")
        gwp_col = "kg CO2-Eq/Unit"
        if gwp_col not in df.columns:
            df[gwp_col] = 0.0
    df[gwp_col] = pd.to_numeric(df[gwp_col], errors="coerce")

    # 4) Roll up all years to full-rotation totals by Material+Unit
    #    Keep FIRST non-null for all non-amount numeric columns so factors persist.
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    keep_first_cols = [c for c in numeric_cols if c not in ["Amount"]]

    rolled = df.groupby(["Material", "Unit"], as_index=False).agg(
        Amount=("Amount", "sum"),
        **{c: (c, lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan) for c in keep_first_cols}
    )

    # 5) 19-19-19 → 15-15-15 conversion (mass only, keep factors)
    m_npk = rolled["Material"].str.contains("19-19-19", case=False, na=False)
    if m_npk.any():
        rolled.loc[m_npk, "Amount"] = rolled.loc[m_npk, "Amount"] * (19.0 / 15.0)
        rolled.loc[m_npk, "Material"] = "NPK (15-15-15) fertiliser"

    # 6) Final vectors
    costs = pd.to_numeric(rolled[cost_col], errors="coerce").fillna(0.0).to_numpy(float)
    materials = rolled["Material"].tolist()
    base_amounts = rolled["Amount"].to_numpy(float)

    # Impacts (include gwp_col)
    impact_cols = detect_impact_cols(rolled, cost_col)
    impact_df = rolled[impact_cols].copy().fillna(0.0)

    merged_df = rolled.copy()
    return (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)

# ---------------------------
# Evaluators
# ---------------------------
def evaluate_cost_gwp(ind, df, cost_col, gwp_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (
        float(np.dot(x, df[cost_col].to_numpy(float))),
        float(np.dot(x, df[gwp_col].to_numpy(float))),
    )

def evaluate_cost_only(ind, df, cost_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (float(np.dot(x, df[cost_col].to_numpy(float))),)

def evaluate_combined(ind, df, impact_cols, cost_col, alpha_cost=1.0, alpha_imp=1.0):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    impacts = df[impact_cols].to_numpy(float)
    total_imp = float(np.dot(x, impacts).sum())
    total_cost = float(np.dot(x, df[cost_col].to_numpy(float)))
    return (alpha_cost * total_cost + alpha_imp * total_imp,)

def evaluate_single_impact(ind, df, impact_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (float(np.dot(x, df[impact_col].to_numpy(float))),)

# ---------------------------
# Optimizers
# ---------------------------
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
    toolbox.register("evaluate", evaluate_cost_gwp, df=df, cost_col=cost_col, gwp_col=gwp_col)
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

# ---------------------------
# UI
# ---------------------------
st.title("LCA Optimization: Cost vs GWP")

uploaded_file = st.file_uploader(
    "Upload your merged table (.csv or .xlsm/.xlsx)",
    type=["csv", "xlsm", "xlsx"]
)

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

# Baseline QA
cost_col = pick_column(merged_df, COST_CANDIDATES) or "Unit Cost ($)"
baseline_cost = float((merged_df["Amount"] * merged_df[cost_col].fillna(0)).sum())
baseline_gwp  = float((merged_df["Amount"] * merged_df[gwp_col].fillna(0)).sum())
per_tree = 1900.0

st.markdown("### QA Baseline (from uploaded file)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
c2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
c3.metric("Cost / tree ($/tree)", f"{baseline_cost/per_tree:,.2f}")
c4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/per_tree:,.2f}")

# Top contributors
tmp = merged_df.copy()
tmp["_GWP"] = tmp["Amount"] * tmp[gwp_col].fillna(0)
st.markdown("#### Top GWP contributors")
st.dataframe(
    tmp.groupby(["Material","Unit"], as_index=False)
       .agg(Amount=("Amount","sum"), GWP=("_GWP","sum"))
       .sort_values("GWP", ascending=False)
       .head(12),
    use_container_width=True
)

# Sidebar controls
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
    selected_impact = st.selectbox("Select impact column", impact_cols)

# Run
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb,
                           lows, highs, merged_df, cost_col, gwp_col)
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

        per_tree_df = df_out.copy()
        per_tree_df["Cost / tree"] = per_tree_df["Total Cost"] / per_tree
        per_tree_df["GWP / tree"]  = per_tree_df["Total GWP"]  / per_tree
        st.dataframe(per_tree_df[["Cost / tree", "GWP / tree"]], use_container_width=True)

        st.session_state.setdefault("history", []).append(
            {"scenario": scenario, "results": df_out}
        )

    elif scenario == "Optimize Cost + Combined Impact":
        best = run_single(evaluate_combined, popsize, ngen, cxpb, mutpb,
                          lows, highs, merged_df, impact_cols, cost_col)
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
                          lows, highs, merged_df, cost_col)
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

# Download
if st.button("Download Merged Data as Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
    output.seek(0)
    st.download_button(
        label="Download Excel File",
        data=output,
        file_name="merged_totals_for_optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.s
