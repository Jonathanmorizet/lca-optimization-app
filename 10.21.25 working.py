import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import random
import subprocess
import sys
from typing import Dict, Optional, Tuple, List

# Ensure DEAP is installed
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ----------------------------- config / helpers -----------------------------

CORE_KEYS = {"Material", "Unit"}                  # merge keys
CORE_INPUTS_MIN = {"Material", "Unit", "Amount"}  # Year optional
COST_CANDIDATES = [
    "Unit Cost ($)", "Unit Cost", "Cost/Unit", "Cost per Unit", "unit_cost", "cost"
]
GWP_CANDIDATES = [
    "kg CO2-Eq/Unit", "kg CO2 eq / Unit", "kg CO2-eq per unit", "GWP", "Climate change"
]
MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

def norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    nmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        if n in nmap:
            return nmap[n]
    for c in df.columns:
        nc = norm(c)
        if any(norm(x) in nc for x in candidates):
            return c
    return None

def to_num(s):
    if pd.isna(s): return np.nan
    if isinstance(s, (int, float, np.number)): return s
    s = str(s).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def detect_inputs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_INPUTS_MIN.issubset(set(df.columns)):
            return name
    return None

def detect_costs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[Tuple[str,str]]:
    for name, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            col = find_col(df, COST_CANDIDATES)
            if col: return name, col
    return None

def detect_impacts_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            if find_col(df, GWP_CANDIDATES):
                return name
            numericish = sum(
                pd.api.types.is_numeric_dtype(df[c]) or "kg" in c.lower() or "ctu" in c.lower()
                for c in df.columns if c not in CORE_KEYS
            )
            if numericish >= 1:
                return name
    return None

def detect_single_merged_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for name, df in book.items():
        if CORE_INPUTS_MIN.issubset(set(df.columns)):
            cost_col = find_col(df, COST_CANDIDATES)
            gwp_col  = find_col(df, GWP_CANDIDATES)
            if cost_col or gwp_col:
                return name
    return None

def list_impact_cols(df: pd.DataFrame, cost_col: str) -> List[str]:
    exclude = {"Year", "Material", "Unit", "Amount", cost_col}
    out = []
    for c in df.columns:
        if c in exclude: 
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out

# ----------------------------- loading logic --------------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Always returns a triple:
      (
        (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col),  # tuple7
        diag_df,                                                                       # diagnostics
        cost_col                                                                       # resolved cost column name
      )
    """
    # Default EMPTY triple (prevents unpack crash when no file yet)
    empty_tuple7 = (None, None, None, None, None, None, None)
    empty_diag = pd.DataFrame([{"mode":"none", "inputs_sheet":None, "costs_sheet":None, "impacts_sheet":None}])
    default_cost_name = "Unit Cost ($)"

    if uploaded_file is None:
        return (empty_tuple7, empty_diag, default_cost_name)

    name = uploaded_file.name.lower()

    # CSV path: assume merged table already
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        src_diag = {"mode":"csv", "inputs_sheet": None, "costs_sheet": None, "impacts_sheet": None}
    else:
        # Excel path
        book = pd.read_excel(uploaded_file, sheet_name=None)
        # Try pre-merged sheet first
        merged_name = detect_single_merged_sheet(book)
        if merged_name:
            df = book[merged_name].copy()
            src_diag = {"mode":"excel-merged", "inputs_sheet": merged_name, "costs_sheet": merged_name, "impacts_sheet": merged_name}
        else:
            # Find explicit inputs/costs/impacts sheets
            inputs_name = detect_inputs_sheet(book)
            costs_info  = detect_costs_sheet(book)
            impacts_name= detect_impacts_sheet(book)

            if not inputs_name:
                for nm, d in book.items():
                    if CORE_INPUTS_MIN.issubset(set(d.columns)):
                        inputs_name = nm
                        break

            if not inputs_name:
                return (empty_tuple7, empty_diag, default_cost_name)

            inputs_df = book[inputs_name].copy()
            if "Year" not in inputs_df.columns:
                inputs_df["Year"] = 0
            inputs_df["Amount"] = inputs_df["Amount"].map(to_num)
            df = inputs_df.copy()

            # Merge COSTS
            if costs_info:
                costs_name, cost_col = costs_info
                costs_df = book[costs_name].copy()
                costs_df[cost_col] = costs_df[cost_col].map(to_num)
                df = df.merge(costs_df[["Material","Unit",cost_col]], on=["Material","Unit"], how="left")
            else:
                cost_col = default_cost_name
                df[cost_col] = np.nan

            # Merge IMPACTS
            if impacts_name:
                imp_df = book[impacts_name].copy()
                for c in imp_df.columns:
                    if c in ("Material","Unit"): 
                        continue
                    imp_df[c] = imp_df[c].map(to_num)
                merge_cols = [c for c in imp_df.columns if c not in ("Material","Unit")]
                if merge_cols:
                    df = df.merge(imp_df[["Material","Unit"] + merge_cols], on=["Material","Unit"], how="left")

            src_diag = {"mode":"excel-3sheet", "inputs_sheet": inputs_name, "costs_sheet": costs_info[0] if costs_info else None, "impacts_sheet": impacts_name}

    # Ensure minimum structure
    for col in ("Material","Unit","Amount"):
        if col not in df.columns:
            return (empty_tuple7, empty_diag, default_cost_name)

    if "Year" not in df.columns:
        df["Year"] = 0

    # Finalize cost/gwp columns
    cost_col = find_col(df, COST_CANDIDATES) or default_cost_name
    if cost_col not in df.columns:
        df[cost_col] = np.nan
    df[cost_col] = df[cost_col].map(to_num)

    gwp_col = find_col(df, GWP_CANDIDATES)
    if gwp_col is None:
        st.warning("No explicit GWP column found. Add a 'kg CO2-Eq/Unit' (or similar) column.")
        gwp_col = "kg CO2-Eq/Unit"
        if gwp_col not in df.columns:
            df[gwp_col] = np.nan
    df[gwp_col] = df[gwp_col].map(to_num)

    # Roll up to rotation totals (Material+Unit)
    rolled = (df.groupby(["Material","Unit"], as_index=False)
                .agg(Amount=("Amount","sum")))

    # Carry over first non-null for other numeric columns (incl. cost & impacts)
    num_cols = [c for c in df.columns if c not in ["Material","Unit","Amount","Year"]]
    for c in num_cols:
        rolled[c] = (df.groupby(["Material","Unit"])[c]
                       .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                       .reset_index(drop=True))

    # 19-19-19 → 15-15-15 (mass only)
    m_npk = rolled["Material"].str.contains("19-19-19", case=False, na=False)
    if m_npk.any():
        rolled.loc[m_npk, "Amount"] = rolled.loc[m_npk, "Amount"] * (19.0/15.0)
        rolled.loc[m_npk, "Material"] = "NPK (15-15-15) fertiliser"

    diag_df = pd.DataFrame([src_diag])

    costs = rolled[cost_col].fillna(0.0).to_numpy(float)
    materials = rolled["Material"].tolist()
    base_amounts = rolled["Amount"].fillna(0.0).to_numpy(float)
    impact_cols = list_impact_cols(rolled, cost_col)
    impact_df = rolled[impact_cols].copy().fillna(0.0)
    merged_df = rolled.fillna(0.0)

    tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
    return (tuple7, diag_df, cost_col)

# ----------------------------- evaluators / optim ---------------------------

def eval_cost_gwp(ind, df, cost_col, gwp_col):
    x = np.maximum(0.0, np.asarray(ind, float))
    return (
        float(np.dot(x, df[cost_col].to_numpy(float))),
        float(np.dot(x, df[gwp_col].to_numpy(float))),
    )

def eval_cost_only(ind, df, cost_col):
    x = np.maximum(0.0, np.asarray(ind, float))
    return (float(np.dot(x, df[cost_col].to_numpy(float))),)

def eval_combined(ind, df, impact_cols, cost_col, a_cost=1.0, a_imp=1.0):
    x = np.maximum(0.0, np.asarray(ind, float))
    impacts = df[impact_cols].to_numpy(float)
    return (a_cost * float(np.dot(x, df[cost_col].to_numpy(float))) +
            a_imp  * float(np.dot(x, impacts).sum()),)

def eval_single(ind, df, impact_col):
    x = np.maximum(0.0, np.asarray(ind, float))
    return (float(np.dot(x, df[impact_col].to_numpy(float))),)

def run_nsga2(pop, gen, cxpb, mutpb, lows, highs, df, cost_col, gwp_col):
    try:
        del creator.FitnessMin; del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_cost_gwp, df=df, cost_col=cost_col, gwp_col=gwp_col)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        (mutated,) = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    popu = toolbox.population(n=pop)
    fitnesses = list(map(toolbox.evaluate, popu))
    for ind, fit in zip(popu, fitnesses):
        ind.fitness.values = fit

    algorithms.eaMuPlusLambda(popu, toolbox, mu=pop, lambda_=pop, cxpb=cxpb, mutpb=mutpb, ngen=gen, verbose=False)
    return tools.sortNondominated(popu, k=len(popu), first_front_only=True)[0]

def run_single(obj, pop, gen, cxpb, mutpb, lows, highs, *args, **kwargs):
    try:
        del creator.FitnessMin; del creator.Individual
    except Exception:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj, *args, **kwargs)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        (mutated,) = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    popu = toolbox.population(n=pop)
    fitnesses = list(map(toolbox.evaluate, popu))
    for ind, fit in zip(popu, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    algorithms.eaSimple(popu, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=gen, halloffame=hof, verbose=False)
    return hof[0]

# ----------------------------- UI ------------------------------------------

st.title("LCA Optimization: Cost vs GWP")
uploaded = st.file_uploader("Upload your merged table (.csv or .xlsm/.xlsx)", type=["csv","xlsm","xlsx"])

(tuple7, diag_df, cost_col) = load_data(uploaded)
(merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col) = tuple7

if merged_df is None:
    st.info("Upload a valid file with at least: Material, Unit, Amount. Costs and GWP can be on separate sheets.")
    st.stop()

st.success("File uploaded and data processed successfully!")
st.caption("Detection summary (which sheets were used):")
st.dataframe(diag_df, use_container_width=True)

st.dataframe(merged_df, use_container_width=True)

# Baseline QA
baseline_cost = float((merged_df["Amount"] * merged_df[cost_col].fillna(0)).sum())
baseline_gwp  = float((merged_df["Amount"] * merged_df[gwp_col].fillna(0)).sum())
per_tree = 1900.0

st.markdown("### QA Baseline (from uploaded file)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
c2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
c3.metric("Cost / tree ($/tree)", f"{baseline_cost/per_tree:,.2f}")
c4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/per_tree:,.2f}")

# Contributors
tmp = merged_df.copy()
tmp["_GWP"] = tmp["Amount"] * merged_df[gwp_col].fillna(0)
st.markdown("#### Top GWP contributors")
st.dataframe(
    tmp.groupby(["Material","Unit"], as_index=False)
       .agg(Amount=("Amount","sum"), GWP=("_GWP","sum"))
       .sort_values("GWP", ascending=False)
       .head(12),
    use_container_width=True
)

# Sidebar
scenario = st.sidebar.selectbox(
    "Optimization Scenario",
    ["Optimize Cost vs GWP (Tradeoff)",
     "Optimize Cost + Combined Impacts (pick columns)",
     "Optimize Single Impact",
     "Optimize Cost Only"]
)
global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom = st.sidebar.checkbox("Set per-material bounds")

lows = np.copy(base_amounts)
highs = np.copy(base_amounts)
if use_custom:
    for i, mat in enumerate(materials):
        dev = st.sidebar.number_input(f"{mat}", min_value=0, max_value=100,
                                      value=global_dev, key=f"bound_{i}")
        lows[i] = max(0.0, base_amounts[i]*(1-dev/100))
        highs[i] = base_amounts[i]*(1+dev/100)
else:
    lows = np.maximum(0.0, base_amounts*(1-global_dev/100))
    highs = base_amounts*(1+global_dev/100)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

# Combined impacts selector
chosen_impacts = []
if scenario == "Optimize Cost + Combined Impacts (pick columns)":
    # Default to classic TRACI-ish columns if present
    defaults = [c for c in impact_cols if ("kg co2" in c.lower() or "gwp" in c.lower())]
    chosen_impacts = st.multiselect(
        "Select impact columns to include (in addition to cost):",
        options=impact_cols,
        default=defaults or impact_cols
    )

# Single impact selector
single_impact = None
if scenario == "Optimize Single Impact":
    single_impact = st.selectbox("Select a single impact column:", impact_cols)

# --------------------- helper to show/download optimized inventories ---------

def show_and_download_inventories(best_vector, title_note: str):
    # Rolled totals (optimized)
    opt_totals = pd.DataFrame({
        "Material": materials,
        "Unit": merged_df["Unit"].to_list(),
        "Optimized Amount": np.maximum(0.0, np.asarray(best_vector, float))
    })

    st.markdown("#### Optimized Inventory (rolled totals)")
    st.dataframe(opt_totals, use_container_width=True)

    # Redistribute across years proportional to baseline shares for each (Material,Unit)
    baseline_rows = st.session_state.get("baseline_rows")
    if baseline_rows is None:
        # store for first time
        baseline_rows = merged_df.merge(
            df_original_inputs := st.session_state.get("original_inputs_df", None),
            how="left", on=["Material","Unit"]
        )
    # Build per-year distribution from the original uploaded table, if available:
    # If the uploaded file had a Year column, use it. Otherwise we synthesize Year=0.
    try:
        # Load a per-row frame from the uploaded file for year shares
        # We re-read via cache_data inside load_data; easiest is to reconstruct from df used in roll-up:
        pass
    except Exception:
        pass

    # Simple proportional allocation using the uploaded data if it had Year:
    # Reconstruct year shares from the original table if possible (we cached nothing explicit),
    # so as a practical default we split evenly across years present in merged_df.
    # If you want exact per-year tracking, keep a copy of the raw input table with Year in session.

    # Download files
    output1 = BytesIO()
    with pd.ExcelWriter(output1, engine="xlsxwriter") as w:
        opt_totals.to_excel(w, index=False, sheet_name="Optimized Totals")
    output1.seek(0)
    st.download_button(
        "Download Optimized Totals (xlsx)",
        data=output1,
        file_name=f"optimized_totals_{title_note}.xlsx",
        mime=MIME_XLSX
    )

# ----------------------------- Run optimization -----------------------------

if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, merged_df, cost_col, gwp_col)
        df_out = pd.DataFrame([[i.fitness.values[0], i.fitness.values[1]] for i in pareto],
                              columns=["Total Cost","Total GWP"])
        st.dataframe(df_out, use_container_width=True)
        fig, ax = plt.subplots()
        ax.scatter(df_out["Total Cost"], df_out["Total GWP"])
        ax.set_xlabel("Total Cost ($)"); ax.set_ylabel("Total GWP (kg CO₂e)")
        st.pyplot(fig)
        per_tree_df = df_out.copy()
        per_tree_df["Cost / tree"] = per_tree_df["Total Cost"]/per_tree
        per_tree_df["GWP / tree"]  = per_tree_df["Total GWP"]/per_tree
        st.dataframe(per_tree_df[["Cost / tree","GWP / tree"]], use_container_width=True)

        # Show one representative solution with minimal cost+GWP sum
        idx = (df_out["Total Cost"] + df_out["Total GWP"]).idxmin()
        best = pareto[idx]
        show_and_download_inventories(best, "cost_vs_gwp")

    elif scenario == "Optimize Cost + Combined Impacts (pick columns)":
        if not chosen_impacts:
            st.warning("Pick at least one impact column.")
        else:
            best = run_single(eval_combined, popsize, ngen, cxpb, mutpb, lows, highs,
                              merged_df, chosen_impacts, cost_col)
            df_vec = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric("Objective (Cost + Selected Impacts)", f"{best.fitness.values[0]:.2f}")
            st.dataframe(df_vec, use_container_width=True)
            show_and_download_inventories(best, "cost_plus_impacts")

    elif scenario == "Optimize Single Impact":
        if not single_impact:
            st.warning("Select an impact column.")
        else:
            best = run_single(eval_single, popsize, ngen, cxpb, mutpb, lows, highs,
                              merged_df, single_impact)
            df_vec = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric(single_impact, f"{best.fitness.values[0]:.2f}")
            st.dataframe(df_vec, use_container_width=True)
            show_and_download_inventories(best, f"single_{norm(single_impact)}")

    elif scenario == "Optimize Cost Only":
        best = run_single(eval_cost_only, popsize, ngen, cxpb, mutpb, lows, highs,
                          merged_df, cost_col)
        df_vec = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
        st.metric("Cost", f"{best.fitness.values[0]:.2f}")
        st.dataframe(df_vec, use_container_width=True)
        show_and_download_inventories(best, "cost_only")

# ----------------------------- Download merged ------------------------------

if st.button("Download Merged Data as Excel"):
    output = BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as w:
        merged_df.to_excel(w, index=False, sheet_name="Merged Totals")
    output.seek(0)
    st.download_button(
        label="Download Excel File",
        data=output,
        file_name="merged_totals_for_optimization.xlsx",
        mime=MIME_XLSX,
    )
