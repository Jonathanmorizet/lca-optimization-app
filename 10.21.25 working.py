# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO, StringIO
import random
import subprocess
import sys
from typing import Dict, Optional, Tuple, List

# --- Ensure DEAP is installed ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools, algorithms

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

MIME_XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"

# ----------------------------- helpers --------------------------------------

CORE_KEYS = {"Material", "Unit"}
CORE_INPUT_MIN = {"Material", "Unit", "Amount"}

COST_CANDIDATES = [
    "Unit Cost ($)", "Unit Cost", "Cost/Unit", "Cost per Unit", "unit_cost", "cost"
]
GWP_CANDIDATES = [
    "kg CO2-Eq/Unit", "kg CO2 eq / Unit", "kg CO2-eq per unit", "GWP", "Climate change"
]

def norm(s: str) -> str:
    return "".join(c for c in str(s).lower() if c.isalnum())

def find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    # exact after normalization
    nmap = {norm(c): c for c in df.columns}
    for cand in candidates:
        n = norm(cand)
        if n in nmap:
            return nmap[n]
    # fuzzy contains
    for c in df.columns:
        nc = norm(c)
        if any(norm(x) in nc for x in candidates):
            return c
    return None

def to_num(x):
    if pd.isna(x): return np.nan
    if isinstance(x, (int, float, np.number)): return float(x)
    s = str(x).strip().replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [c.strip() for c in out.columns]
    return out

def list_impact_cols(df: pd.DataFrame, cost_col: str) -> List[str]:
    exclude = {"Material", "Unit", "Amount", "Year", cost_col}
    return [c for c in df.columns
            if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]

# -------------- sheet detection (auto) --------------------------------------

def detect_inputs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for nm, df in book.items():
        if CORE_INPUT_MIN.issubset(set(df.columns)):
            return nm
    return None

def detect_costs_sheet(book: Dict[str, pd.DataFrame]) -> Optional[Tuple[str, str]]:
    for nm, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            col = find_col(df, COST_CANDIDATES)
            if col:
                return nm, col
    return None

def detect_impacts_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for nm, df in book.items():
        if CORE_KEYS.issubset(set(df.columns)):
            if find_col(df, GWP_CANDIDATES):
                return nm
            # or any numeric impact-like column
            numish = sum(pd.api.types.is_numeric_dtype(df[c]) for c in df.columns if c not in CORE_KEYS)
            if numish >= 1:
                return nm
    return None

def detect_merged_sheet(book: Dict[str, pd.DataFrame]) -> Optional[str]:
    for nm, df in book.items():
        if CORE_INPUT_MIN.issubset(set(df.columns)):
            if find_col(df, COST_CANDIDATES) or find_col(df, GWP_CANDIDATES):
                return nm
    return None

# ----------------------------- loader ---------------------------------------

@st.cache_data
def load_data(uploaded_file):
    """
    Returns FOUR values:
      tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
      diag_df (what was detected/used)
      cost_col (resolved name)
      original_df (row-level table we read, with Year if present)
    If no file yet, returns (all Nones/empties) but keeps shapes, so no crash.
    """
    empty_tuple7 = (None, None, None, None, None, None, None)
    empty_diag = pd.DataFrame([{"mode": "none", "inputs": None, "costs": None, "impacts": None}])
    default_cost = "Unit Cost ($)"

    if uploaded_file is None:
        return (empty_tuple7, empty_diag, default_cost, None)

    fname = uploaded_file.name.lower()

    # ---- CSV: assume merged already
    if fname.endswith(".csv"):
        original_df = sanitize_columns(pd.read_csv(uploaded_file))
        mode = "csv-merged"
        s_inputs = s_costs = s_impacts = "(csv)"

    else:
        book = {nm: sanitize_columns(df) for nm, df in pd.read_excel(uploaded_file, sheet_name=None).items()}
        merged_nm = detect_merged_sheet(book)
        if merged_nm:
            original_df = book[merged_nm].copy()
            mode = "excel-merged"
            s_inputs = s_costs = s_impacts = merged_nm
        else:
            inp_nm = detect_inputs_sheet(book)
            if not inp_nm:
                # nothing recognizable; let caller offer manual mapping
                return (empty_tuple7, empty_diag, default_cost, None)
            original_df = book[inp_nm].copy()
            original_df["Year"] = original_df["Year"] if "Year" in original_df.columns else 0
            original_df["Amount"] = original_df["Amount"].map(to_num)

            # costs
            cost_info = detect_costs_sheet(book)
            if cost_info:
                c_nm, c_col = cost_info
                cdf = book[c_nm].copy()
                cdf[c_col] = cdf[c_col].map(to_num)
                original_df = original_df.merge(cdf[["Material", "Unit", c_col]], on=["Material", "Unit"], how="left")
                resolved_cost = c_col
            else:
                resolved_cost = default_cost
                original_df[resolved_cost] = np.nan

            # impacts
            imp_nm = detect_impacts_sheet(book)
            if imp_nm:
                idf = book[imp_nm].copy()
                for c in idf.columns:
                    if c not in ("Material", "Unit"):
                        idf[c] = idf[c].map(to_num)
                merge_cols = [c for c in idf.columns if c not in ("Material", "Unit")]
                if merge_cols:
                    original_df = original_df.merge(idf[["Material", "Unit"] + merge_cols], on=["Material", "Unit"], how="left")

            mode = "excel-3sheet"
            s_inputs, s_costs, s_impacts = inp_nm, (cost_info[0] if cost_info else None), (imp_nm if imp_nm else None)

    # ---- ensure minimum structure
    for col in ("Material", "Unit", "Amount"):
        if col not in original_df.columns:
            # signal to caller to attempt manual mapping
            return (empty_tuple7, empty_diag, default_cost, None)
    if "Year" not in original_df.columns:
        original_df["Year"] = 0

    # ensure numeric cost & impacts
    cost_col = find_col(original_df, COST_CANDIDATES) or default_cost
    if cost_col not in original_df.columns:
        original_df[cost_col] = np.nan
    original_df[cost_col] = original_df[cost_col].map(to_num)

    gwp_col = find_col(original_df, GWP_CANDIDATES)
    if gwp_col is None:
        gwp_col = "kg CO2-Eq/Unit"
        if gwp_col not in original_df.columns:
            original_df[gwp_col] = np.nan
    original_df[gwp_col] = original_df[gwp_col].map(to_num)

    # ---- roll up to totals by Material+Unit
    rolled = (original_df.groupby(["Material", "Unit"], as_index=False)
              .agg(Amount=("Amount", "sum")))

    # carry first non-null for other numeric cols
    other_cols = [c for c in original_df.columns if c not in ("Material", "Unit", "Amount", "Year")]
    for c in other_cols:
        rolled[c] = (original_df.groupby(["Material", "Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))

    # 19-19-19 → 15-15-15 proxy (mass only)
    m_npk = rolled["Material"].str.contains("19-19-19", case=False, na=False)
    if m_npk.any():
        rolled.loc[m_npk, "Amount"] = rolled.loc[m_npk, "Amount"] * (19.0 / 15.0)
        rolled.loc[m_npk, "Material"] = "NPK (15-15-15) fertiliser"

    diag_df = pd.DataFrame([{"mode": mode, "inputs": s_inputs, "costs": s_costs, "impacts": s_impacts}])

    # build tuple7
    costs = rolled[cost_col].fillna(0.0).to_numpy(float)
    materials = rolled["Material"].tolist()
    base_amounts = rolled["Amount"].fillna(0.0).to_numpy(float)
    impact_cols = list_impact_cols(rolled, cost_col)
    impact_df = rolled[impact_cols].copy().fillna(0.0)
    merged_df = rolled.fillna(0.0)

    tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
    return (tuple7, diag_df, cost_col, original_df)

# --------------------- manual mapping (UI path if needed) -------------------

def manual_map_ui(uploaded_file):
    book = {nm: sanitize_columns(df) for nm, df in pd.read_excel(uploaded_file, sheet_name=None).items()}
    sheet_names = list(book.keys())
    st.info("Auto-detect couldn’t find `Material`, `Unit`, `Amount`. Map your sheets/columns below.")

    s_inputs  = st.selectbox("Inputs sheet (must include the rows per year)", sheet_names)
    s_costs   = st.selectbox("Costs sheet (optional)", ["(none)"] + sheet_names)
    s_impacts = st.selectbox("Impacts sheet (optional)", ["(none)"] + sheet_names)

    df_in = book[s_inputs].copy()
    st.caption("Preview: Inputs sheet")
    st.dataframe(df_in.head(), use_container_width=True)

    col_mat  = st.selectbox("Inputs → Material", df_in.columns)
    col_unit = st.selectbox("Inputs → Unit", df_in.columns, index=min(1, len(df_in.columns)-1))
    col_amt  = st.selectbox("Inputs → Amount", df_in.columns, index=min(2, len(df_in.columns)-1))
    col_year = st.selectbox("Inputs → Year (optional)", ["(none)"] + df_in.columns.tolist())

    inputs_df = pd.DataFrame({
        "Material": df_in[col_mat],
        "Unit": df_in[col_unit],
        "Amount": pd.to_numeric(df_in[col_amt], errors="coerce")
    })
    inputs_df["Year"] = pd.to_numeric(df_in[col_year], errors="coerce").fillna(0).astype(int) if col_year != "(none)" else 0

    cost_col_final = "Unit Cost ($)"
    if s_costs != "(none)":
        df_c = book[s_costs].copy()
        st.caption("Preview: Costs sheet")
        st.dataframe(df_c.head(), use_container_width=True)
        c_mat  = st.selectbox("Costs → Material", df_c.columns, key="c_mat")
        c_unit = st.selectbox("Costs → Unit", df_c.columns, key="c_unit")
        c_val  = st.selectbox("Costs → Cost column", df_c.columns, key="c_val")
        costs_df = pd.DataFrame({
            "Material": df_c[c_mat],
            "Unit": df_c[c_unit],
            cost_col_final: pd.to_numeric(df_c[c_val], errors="coerce")
        })
        inputs_df = inputs_df.merge(costs_df, on=["Material", "Unit"], how="left")

    if s_impacts != "(none)":
        df_i = book[s_impacts].copy()
        st.caption("Preview: Impacts sheet")
        st.dataframe(df_i.head(), use_container_width=True)
        i_mat  = st.selectbox("Impacts → Material", df_i.columns, key="i_mat")
        i_unit = st.selectbox("Impacts → Unit", df_i.columns, key="i_unit")
        candidates = [c for c in df_i.columns if c not in (i_mat, i_unit)]
        pick_cols = st.multiselect("Impacts → choose columns (include your GWP column)", candidates)
        if pick_cols:
            idf = df_i[[i_mat, i_unit] + pick_cols].copy()
            idf.columns = ["Material", "Unit"] + pick_cols
            for c in pick_cols:
                idf[c] = pd.to_numeric(idf[c], errors="coerce")
            inputs_df = inputs_df.merge(idf, on=["Material", "Unit"], how="left")

    # Now run through the same roll-up pipeline in-memory (no re-upload needed)
    csv_buf = inputs_df.to_csv(index=False)
    uploaded_like = StringIO(csv_buf)
    merged = sanitize_columns(pd.read_csv(uploaded_like))

    # resolve cost/gwp + roll-up
    if "Year" not in merged.columns:
        merged["Year"] = 0
    cost_col = find_col(merged, COST_CANDIDATES) or cost_col_final
    if cost_col not in merged.columns:
        merged[cost_col] = 0.0
    gwp_col = find_col(merged, GWP_CANDIDATES) or "kg CO2-Eq/Unit"
    if gwp_col not in merged.columns:
        merged[gwp_col] = 0.0

    rolled = (merged.groupby(["Material", "Unit"], as_index=False)
              .agg(Amount=("Amount", "sum")))
    other_cols = [c for c in merged.columns if c not in ("Material", "Unit", "Amount", "Year")]
    for c in other_cols:
        rolled[c] = (merged.groupby(["Material", "Unit"])[c]
                     .apply(lambda s: s.dropna().iloc[0] if s.dropna().size else np.nan)
                     .reset_index(drop=True))
    costs = rolled[cost_col].fillna(0.0).to_numpy(float)
    materials = rolled["Material"].tolist()
    base_amounts = rolled["Amount"].fillna(0.0).to_numpy(float)
    impact_cols = list_impact_cols(rolled, cost_col)
    impact_df = rolled[impact_cols].copy().fillna(0.0)
    merged_df = rolled.fillna(0.0)

    tuple7 = (merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col)
    diag_df = pd.DataFrame([{"mode": "manual-map", "inputs": s_inputs, "costs": s_costs, "impacts": s_impacts}])
    return tuple7, diag_df, cost_col, inputs_df  # keep row-level inputs (with Year) for per-year allocation

# ------------------------- evaluators / optimizers --------------------------

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

# ------------------------------- UI -----------------------------------------

st.title("LCA Optimization: Cost vs GWP")
uploaded = st.file_uploader("Upload your merged table (.csv or .xlsm/.xlsx)", type=["csv", "xlsm", "xlsx"])

tuple7, diag_df, cost_col, original_df = load_data(uploaded)

# If loader couldn’t find the minimum columns, offer manual mapping UI
if uploaded and tuple7[0] is None:
    tuple7, diag_df, cost_col, original_df = manual_map_ui(uploaded)

(merged_df, materials, base_amounts, costs, impact_df, impact_cols, gwp_col) = tuple7

if merged_df is None:
    st.info("Upload or map a file that contains at least: Material, Unit, Amount (Year optional).")
    st.stop()

# Keep original row-level table (with Year) for per-year allocation
st.session_state["original_rows"] = original_df.copy()

st.success("File uploaded and data processed successfully!")
st.caption("Detection summary")
st.dataframe(diag_df, use_container_width=True)
st.dataframe(merged_df, use_container_width=True)

# --- Baseline QA ---
baseline_cost = float((merged_df["Amount"] * merged_df[cost_col].fillna(0)).sum())
baseline_gwp  = float((merged_df["Amount"] * merged_df[gwp_col].fillna(0)).sum())
per_tree = 1900.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
c2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
c3.metric("Cost / tree ($/tree)", f"{baseline_cost/per_tree:,.2f}")
c4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/per_tree:,.2f}")

# Top GWP contributors
tmp = merged_df.copy()
tmp["_GWP"] = tmp["Amount"] * merged_df[gwp_col].fillna(0)
st.markdown("#### Top GWP contributors")
st.dataframe(
    tmp.groupby(["Material", "Unit"], as_index=False)
       .agg(Amount=("Amount","sum"), GWP=("_GWP","sum"))
       .sort_values("GWP", ascending=False)
       .head(12),
    use_container_width=True
)

# --- Sidebar controls ---
scenario = st.sidebar.selectbox(
    "Optimization Scenario",
    [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Cost + Combined Impacts (pick columns)",
        "Optimize Single Impact",
        "Optimize Cost Only"
    ]
)

global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom = st.sidebar.checkbox("Set per-material bounds")

lows = np.copy(base_amounts)
highs = np.copy(base_amounts)
if use_custom:
    for i, mat in enumerate(materials):
        dev = st.sidebar.number_input(f"{mat}", min_value=0, max_value=100, value=global_dev, key=f"b_{i}")
        lows[i] = max(0.0, base_amounts[i]*(1-dev/100))
        highs[i] = base_amounts[i]*(1+dev/100)
else:
    lows = np.maximum(0.0, base_amounts*(1-global_dev/100))
    highs = base_amounts*(1+global_dev/100)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

# Combined-impacts selector
chosen_impacts = []
if scenario == "Optimize Cost + Combined Impacts (pick columns)":
    defaults = [c for c in impact_cols if ("kg co2" in c.lower() or "gwp" in c.lower())]
    chosen_impacts = st.multiselect("Select impact columns (in addition to cost):", impact_cols, default=defaults or impact_cols)

# Single impact selector
single_impact = None
if scenario == "Optimize Single Impact":
    single_impact = st.selectbox("Select a single impact column:", impact_cols)

# -------------- inventory outputs (totals + per-year reallocation) ----------

def show_and_download_inventories(best_vector, title_note: str):
    # Totals
    opt_totals = pd.DataFrame({
        "Material": materials,
        "Unit": merged_df["Unit"].tolist(),
        "Optimized Amount": np.maximum(0.0, np.asarray(best_vector, float))
    })

    st.markdown("#### Optimized Inventory (rolled totals)")
    st.dataframe(opt_totals, use_container_width=True)

    # Per-year: proportional to baseline shares by (Material, Unit, Year)
    orig = st.session_state.get("original_rows")
    per_year_out = None
    if orig is not None and "Year" in orig.columns:
        # sums by MUY
        mu = orig.groupby(["Material","Unit"], as_index=False)["Amount"].sum().rename(columns={"Amount":"base_total"})
        muy = orig.groupby(["Material","Unit","Year"], as_index=False)["Amount"].sum().rename(columns={"Amount":"base_year"})
        shares = muy.merge(mu, on=["Material","Unit"], how="left")
        shares["share"] = np.where(shares["base_total"]>0, shares["base_year"]/shares["base_total"], 0.0)

        per_year_out = shares.merge(opt_totals, on=["Material","Unit"], how="right")
        per_year_out["Year"] = per_year_out["Year"].fillna(0).astype(int)
        per_year_out["Optimized Amount (Year)"] = per_year_out["Optimized Amount"] * per_year_out["share"].fillna(0.0)
        per_year_out = per_year_out[["Year","Material","Unit","Optimized Amount (Year)"]].sort_values(["Year","Material"])

        st.markdown("#### Optimized Inventory by Year (proportional allocation)")
        st.dataframe(per_year_out, use_container_width=True)

    # Downloads
    out1 = BytesIO()
    with pd.ExcelWriter(out1, engine="xlsxwriter") as w:
        opt_totals.to_excel(w, index=False, sheet_name="Optimized Totals")
        if per_year_out is not None:
            per_year_out.to_excel(w, index=False, sheet_name="Optimized by Year")
    out1.seek(0)
    st.download_button(
        "Download Optimized Inventory (xlsx)",
        data=out1,
        file_name=f"optimized_inventory_{title_note}.xlsx",
        mime=MIME_XLSX
    )

# --------------------------- Run optimization -------------------------------

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

        # Per-tree view
        per_tree_df = df_out.copy()
        per_tree_df["Cost / tree"] = per_tree_df["Total Cost"] / per_tree
        per_tree_df["GWP / tree"]  = per_tree_df["Total GWP"]  / per_tree
        st.dataframe(per_tree_df[["Cost / tree","GWP / tree"]], use_container_width=True)

        # choose a representative point: min (cost + gwp)
        pick = (df_out["Total Cost"] + df_out["Total GWP"]).idxmin()
        best = pareto[pick]
        show_and_download_inventories(best, "cost_vs_gwp")

    elif scenario == "Optimize Cost + Combined Impacts (pick columns)":
        if not chosen_impacts:
            st.warning("Pick at least one impact column.")
        else:
            best = run_single(eval_combined, popsize, ngen, cxpb, mutpb, lows, highs,
                              merged_df, chosen_impacts, cost_col)
            df_vec = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric("Objective (Cost + selected impacts)", f"{best.fitness.values[0]:.2f}")
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
