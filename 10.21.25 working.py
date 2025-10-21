# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import sys
from io import BytesIO

# --- Make sure DEAP is present ---
try:
    from deap import base, creator, tools, algorithms
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

# ------------------------------
# Helpers & column detection
# ------------------------------
CORE_ID_COLS = ["Year", "Material", "Unit", "Amount"]

def _num(s):
    return pd.to_numeric(s, errors="coerce")

def find_cost_col(cols):
    keys = ["unit cost", "cost per unit", "unit $", "price"]
    lower = [c.lower() for c in cols]
    for k in keys:
        for i, c in enumerate(lower):
            if k in c:
                return cols[i]
    return None

def find_gwp_col(cols):
    keys = ["kg co2-eq/unit", "kg co2 eq/unit", "climate change", "gwp"]
    lower = [c.lower() for c in cols]
    for k in keys:
        for i, c in enumerate(lower):
            if k in c:
                return cols[i]
    return None

def detect_impact_cols(df, cost_col):
    exclude = set(CORE_ID_COLS + ([cost_col] if cost_col else []))
    # numeric columns only, excluding the core
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    return cols

def coerce_numeric(df, extra_numeric_cols=None):
    extra_numeric_cols = extra_numeric_cols or []
    for c in ["Amount"] + extra_numeric_cols:
        if c in df.columns:
            df[c] = _num(df[c]).fillna(0.0)
    # try to coerce all non-keys in costs/impacts later
    return df

def merge_by_headers(workbook):
    """Support legacy 3-sheet style:
       Sheet A (Inputs): has Material, Unit, Amount, Year (Year optional)
       Sheet B (Costs):  has Material, Unit, a cost-like column
       Sheet C (Impacts): has Material, Unit, one or more impact columns (incl. GWP)
    """
    inputs = None
    costs  = None
    impacts = None

    for name, df in workbook.items():
        cols = [c.lower() for c in df.columns]
        if {"material","unit","amount"}.issubset(set(cols)):
            inputs = df.copy()
        elif {"material","unit"}.issubset(set(cols)) and any(("cost" in c) or ("price" in c) for c in cols):
            costs = df.copy()
        elif {"material","unit"}.issubset(set(cols)) and any(("co2" in c) or ("gwp" in c) or ("climate" in c) for c in cols):
            impacts = df.copy()

    if inputs is None:
        return None, "Could not find an Inputs sheet with at least Material / Unit / Amount."

    # normalize
    if "Year" not in inputs.columns:
        inputs["Year"] = 0
    inputs = coerce_numeric(inputs, extra_numeric_cols=["Year"])

    # costs
    if costs is None:
        costs = inputs[["Material","Unit"]].copy()
        costs["Unit Cost ($)"] = 0.0
    # create a consistent cost column name
    ccol = find_cost_col(costs.columns) or "Unit Cost ($)"
    if ccol not in costs.columns:
        costs[ccol] = 0.0
    costs[ccol] = _num(costs[ccol]).fillna(0.0)

    # impacts
    if impacts is None:
        impacts = inputs[["Material","Unit"]].copy()
        # create blank GWP column at minimum
        impacts["kg CO2-Eq/Unit"] = 0.0
    # coerce all non-id cols to numeric
    for c in impacts.columns:
        if c not in ["Material","Unit"]:
            impacts[c] = _num(impacts[c]).fillna(0.0)

    # Merge
    merged = inputs.merge(costs, on=["Material","Unit"], how="left")
    merged = merged.merge(impacts, on=["Material","Unit"], how="left")

    return merged.fillna(0.0), None

# ------------------------------
# Load & prepare
# ------------------------------
@st.cache_data
def load_data(file):
    """
    Returns:
        raw_df (as uploaded or merged across sheets)
        rolled_df (totals by Material+Unit)
        cost_col (name)
        gwp_col (name)
        impact_cols (list[str])
    """
    if file is None:
        return None, None, None, None, None, "No file uploaded."

    name = file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(file)
        if not {"Material","Unit","Amount"}.issubset(df.columns):
            return None, None, None, None, None, "CSV must include at least Material, Unit, Amount (and optional Year)."
        if "Year" not in df.columns:
            df["Year"] = 0
        df = coerce_numeric(df, extra_numeric_cols=["Year"])
    else:
        # excel
        book = pd.read_excel(file, sheet_name=None)
        # If any sheet is already "merged" (has Material,Unit,Amount and also any impact or cost), pick it
        merged_candidates = []
        for nm, d in book.items():
            if {"Material","Unit","Amount"}.issubset(d.columns):
                merged_candidates.append(d)
        if len(merged_candidates) == 1 and any(
            any(("cost" in c.lower()) or ("co2" in c.lower()) or ("gwp" in c.lower()) for c in merged_candidates[0].columns)
        ):
            df = merged_candidates[0].copy()
            if "Year" not in df.columns:
                df["Year"] = 0
            df = coerce_numeric(df, extra_numeric_cols=["Year"])
            # coerce other numeric cols
            for c in df.columns:
                if c not in ["Material","Unit","Year"]:
                    df[c] = _num(df[c]).fillna(df[c] if pd.api.types.is_numeric_dtype(df[c]) else 0.0)
        else:
            df, err = merge_by_headers(book)
            if err:
                return None, None, None, None, None, err

    # find cost column (or create)
    cost_col = find_cost_col(df.columns)
    if cost_col is None:
        cost_col = "Unit Cost ($)"
        df[cost_col] = 0.0
    else:
        df[cost_col] = _num(df[cost_col]).fillna(0.0)

    # find gwp column (or create)
    gwp_col = find_gwp_col(df.columns)
    if gwp_col is None:
        gwp_col = "kg CO2-Eq/Unit"
        st.warning("No explicit GWP column found. A blank 'kg CO2-Eq/Unit' column was created.")
        df[gwp_col] = 0.0
    else:
        df[gwp_col] = _num(df[gwp_col]).fillna(0.0)

    # roll totals by Material+Unit (keep one row per Material for optimization)
    rolled = df.groupby(["Material","Unit"], as_index=False).agg({"Amount":"sum"})
    # bring over first non-null cost/impact values per material+unit
    for c in df.columns:
        if c not in ["Year","Amount"]:
            rolled[c] = (df.groupby(["Material","Unit"])[c]
                         .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                         .reset_index(drop=True))
    rolled = rolled.fillna(0.0)

    # detect impact columns
    impact_cols = detect_impact_cols(rolled, cost_col)
    return df, rolled, cost_col, gwp_col, impact_cols, None

# ------------------------------
# DEAP evals & runners
# ------------------------------
def eval_cost_gwp(ind, rolled_df, cost_col, gwp_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    cost = np.dot(x, rolled_df[cost_col].to_numpy(float))
    gwp  = np.dot(x, rolled_df[gwp_col].to_numpy(float))
    return cost, gwp

def eval_cost_only(ind, rolled_df, cost_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (np.dot(x, rolled_df[cost_col].to_numpy(float)),)

def eval_single_impact(ind, rolled_df, impact_col):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    return (np.dot(x, rolled_df[impact_col].to_numpy(float)),)

def eval_cost_plus_sum_impacts(ind, rolled_df, impact_cols, cost_col, a_cost=1.0, a_imp=1.0):
    x = np.maximum(0.0, np.asarray(ind, dtype=float))
    total_cost = np.dot(x, rolled_df[cost_col].to_numpy(float))
    impacts = rolled_df[impact_cols].to_numpy(float)  # (n, m)
    total_imp = float(np.dot(x, impacts).sum())
    return (a_cost * total_cost + a_imp * total_imp,)

def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col, gwp_col):
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
    toolbox.register("evaluate", eval_cost_gwp, rolled_df=rolled_df, cost_col=cost_col, gwp_col=gwp_col)
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

    algorithms.eaMuPlusLambda(pop, toolbox, mu=popsize, lambda_=popsize, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
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

    hof = tools.HallOfFame(1)   # correct name
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    return hof[0] if len(hof) else None

# ------------------------------
# UI
# ------------------------------
st.title("LCA Optimization: Cost vs GWP")

uploaded = st.file_uploader("Upload merged table or 3-sheet workbook (.csv / .xlsm / .xlsx)", type=["csv","xlsm","xlsx"])
raw_df, rolled_df, cost_col, gwp_col, impact_cols, err = load_data(uploaded)

if err:
    st.info(err)
    st.stop()
if rolled_df is None:
    st.info("Upload a valid file to begin.")
    st.stop()

# Show merged totals table
st.success("File uploaded and data processed successfully!")
st.dataframe(rolled_df, use_container_width=True)

# Baseline QA
baseline_cost = float((rolled_df["Amount"] * rolled_df[cost_col]).sum())
baseline_gwp  = float((rolled_df["Amount"] * rolled_df[gwp_col]).sum())
PER_TREE = 1900.0

st.markdown("### QA Baseline (from uploaded file)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
c2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
c3.metric("Cost / tree ($/tree)", f"{baseline_cost/PER_TREE:,.2f}")
c4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/PER_TREE:,.2f}")

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

base_amounts = rolled_df["Amount"].to_numpy(float)
materials = rolled_df["Material"].tolist()

lows = np.copy(base_amounts)
highs = np.copy(base_amounts)
if use_custom_bounds:
    for i, mat in enumerate(materials):
        dev = st.sidebar.number_input(f"{mat}", min_value=0, max_value=100, value=global_dev, key=f"bound_{i}")
        lows[i]  = max(0.0, base_amounts[i] * (1 - dev/100))
        highs[i] =           base_amounts[i] * (1 + dev/100)
else:
    dev = global_dev
    lows = np.maximum(0.0, base_amounts * (1 - dev/100))
    highs =             base_amounts * (1 + dev/100)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

selected_impact = None
if scenario == "Optimize Single Impact":
    selectable_impacts = [c for c in impact_cols if c != cost_col]
    if not selectable_impacts:
        st.warning("No numeric impact columns detected. Add impacts (e.g., kg CO2-Eq/Unit).")
    else:
        selected_impact = st.selectbox("Select impact column", selectable_impacts)

# ------------------------------
# RUN
# ------------------------------
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col, gwp_col)
        if not pareto:
            st.error("No non-dominated solutions found.")
        else:
            df_out = pd.DataFrame([[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
                                  columns=["Total Cost", "Total GWP"])
            st.dataframe(df_out, use_container_width=True)

            fig, ax = plt.subplots()
            ax.scatter(df_out["Total Cost"], df_out["Total GWP"])
            ax.set_xlabel("Total Cost ($)")
            ax.set_ylabel("Total GWP (kg CO₂e)")
            ax.set_title("Pareto Front")
            st.pyplot(fig)

            # per-tree view
            per_tree_df = df_out.copy()
            per_tree_df["Cost / tree"] = per_tree_df["Total Cost"] / PER_TREE
            per_tree_df["GWP / tree"]  = per_tree_df["Total GWP"]  / PER_TREE
            st.dataframe(per_tree_df[["Cost / tree","GWP / tree"]], use_container_width=True)

            # also expose one concrete optimized inventory (pick the min-cost point)
            best = min(pareto, key=lambda x: x.fitness.values[0])
            opt_amounts = np.maximum(0.0, np.array(best, dtype=float))
            per_mat = rolled_df[["Material","Unit"]].copy()
            per_mat["Base Amount"] = base_amounts
            per_mat["Optimized Amount"] = opt_amounts
            st.markdown("#### Example optimized inventory (min-cost point on Pareto)")
            st.dataframe(per_mat, use_container_width=True)

            # per-year scaling export
            scale = np.divide(opt_amounts, base_amounts, out=np.ones_like(opt_amounts), where=base_amounts>0)
            scale_map = dict(zip(rolled_df["Material"]+"||"+rolled_df["Unit"], scale))
            tmp = raw_df.copy()
            keys = tmp["Material"]+"||"+tmp["Unit"]
            tmp["Optimized Amount"] = tmp["Amount"] * keys.map(scale_map).fillna(1.0)
            st.download_button(
                "Download optimized per-year inventory (CSV)",
                data=tmp.to_csv(index=False).encode("utf-8"),
                file_name="optimized_inventory_per_year.csv",
                mime="text/csv"
            )

    elif scenario == "Optimize Cost + Combined Impact":
        best = run_single(eval_cost_plus_sum_impacts, popsize, ngen, cxpb, mutpb,
                          lows, highs, rolled_df, impact_cols, cost_col)
        if best is None:
            st.error("No solution returned.")
        else:
            opt = np.maximum(0.0, np.array(best, dtype=float))
            out = rolled_df[["Material","Unit"]].copy()
            out["Base Amount"] = base_amounts
            out["Optimized Amount"] = opt
            st.metric("Objective (Cost + ΣImpacts)", f"{best.fitness.values[0]:,.2f}")
            st.dataframe(out, use_container_width=True)

            # export per-year
            scale = np.divide(opt, base_amounts, out=np.ones_like(opt), where=base_amounts>0)
            scale_map = dict(zip(rolled_df["Material"]+"||"+rolled_df["Unit"], scale))
            tmp = raw_df.copy()
            keys = tmp["Material"]+"||"+tmp["Unit"]
            tmp["Optimized Amount"] = tmp["Amount"] * keys.map(scale_map).fillna(1.0)
            st.download_button(
                "Download optimized per-year inventory (CSV)",
                data=tmp.to_csv(index=False).encode("utf-8"),
                file_name="optimized_inventory_per_year.csv",
                mime="text/csv"
            )

    elif scenario == "Optimize Single Impact":
        if not selected_impact:
            st.warning("Please choose an impact column.")
        else:
            best = run_single(eval_single_impact, popsize, ngen, cxpb, mutpb,
                              lows, highs, rolled_df, selected_impact)
            if best is None:
                st.error("No solution returned.")
            else:
                opt = np.maximum(0.0, np.array(best, dtype=float))
                out = rolled_df[["Material","Unit"]].copy()
                out["Base Amount"] = base_amounts
                out["Optimized Amount"] = opt
                st.metric(selected_impact, f"{best.fitness.values[0]:,.4f}")
                st.dataframe(out, use_container_width=True)

                # export per-year
                scale = np.divide(opt, base_amounts, out=np.ones_like(opt), where=base_amounts>0)
                scale_map = dict(zip(rolled_df["Material"]+"||"+rolled_df["Unit"], scale))
                tmp = raw_df.copy()
                keys = tmp["Material"]+"||"+tmp["Unit"]
                tmp["Optimized Amount"] = tmp["Amount"] * keys.map(scale_map).fillna(1.0)
                st.download_button(
                    "Download optimized per-year inventory (CSV)",
                    data=tmp.to_csv(index=False).encode("utf-8"),
                    file_name="optimized_inventory_per_year.csv",
                    mime="text/csv"
                )

    elif scenario == "Optimize Cost Only":
        best = run_single(eval_cost_only, popsize, ngen, cxpb, mutpb,
                          lows, highs, rolled_df, cost_col)
        if best is None:
            st.error("No solution returned.")
        else:
            opt = np.maximum(0.0, np.array(best, dtype=float))
            out = rolled_df[["Material","Unit"]].copy()
            out["Base Amount"] = base_amounts
            out["Optimized Amount"] = opt
            st.metric("Min Cost ($)", f"{best.fitness.values[0]:,.2f}")
            st.dataframe(out, use_container_width=True)

            # export per-year
            scale = np.divide(opt, base_amounts, out=np.ones_like(opt), where=base_amounts>0)
            scale_map = dict(zip(rolled_df["Material"]+"||"+rolled_df["Unit"], scale))
            tmp = raw_df.copy()
            keys = tmp["Material"]+"||"+tmp["Unit"]
            tmp["Optimized Amount"] = tmp["Amount"] * keys.map(scale_map).fillna(1.0)
            st.download_button(
                "Download optimized per-year inventory (CSV)",
                data=tmp.to_csv(index=False).encode("utf-8"),
                file_name="optimized_inventory_per_year.csv",
                mime="text/csv"
            )

# Download the rolled/merged table you’re optimizing
if st.button("Download merged totals (Excel)"):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        rolled_df.to_excel(w, index=False, sheet_name="Merged Totals")
    bio.seek(0)
    st.download_button(
        "Save file",
        data=bio.getvalue(),
        file_name="merged_totals_for_optimization.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
