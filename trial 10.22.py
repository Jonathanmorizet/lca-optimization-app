# app.py — robust loader with sheet/column mapping + diagnostics
import re, sys, random, subprocess
from io import BytesIO
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Ensure DEAP
try:
    from deap import base, creator, tools
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap", "-q"])
    from deap import base, creator, tools

st.set_page_config(page_title="LCA Optimization: Cost vs GWP", layout="wide")
random.seed(42)

DEFAULT_TREES = 1900.0

# ---------- utilities ----------
def normalize_key(s: pd.Series) -> pd.Series:
    """Trim, lower, collapse spaces, remove NBSP."""
    x = s.astype(str).str.replace("\u00A0"," ", regex=False).str.strip().str.lower()
    x = x.str.replace(r"\s+", " ", regex=True)
    return x

def to_numeric(series: pd.Series) -> pd.Series:
    """Robust numeric coercion (handles commas and (123) negatives)."""
    if pd.api.types.is_numeric_dtype(series): return series.astype(float)
    s = series.astype(str).str.strip()
    s = s.str.replace(r"^\((.*)\)$", r"-\1", regex=True)
    s = s.str.replace(",", "", regex=False)
    s = s.str.replace("\u00A0"," ", regex=False)
    return pd.to_numeric(s, errors="coerce")

def pick_numeric_cols(df: pd.DataFrame):
    out = []
    for c in df.columns:
        if c in ("Material","Unit","Amount","Year"): continue
        # Try coercion on a sample to decide
        v = to_numeric(df[c]).dropna()
        if len(v)>0:
            out.append(c)
    return out

# ---------- data loader with explicit mapping ----------
@st.cache_data
def read_workbook(uploaded):
    """Return dict of sheets (as DataFrames)."""
    if uploaded.name.lower().endswith(".csv"):
        return {"CSV": pd.read_csv(uploaded)}
    return pd.read_excel(uploaded, sheet_name=None, engine="openpyxl")

def merge_inputs_costs_impacts(inputs, costs, impacts,
                               inputs_keys, costs_key_cols, impacts_key_cols,
                               cost_col, impact_cols):
    """Normalize keys and do robust left-joins; return merged df and diagnostics."""
    # Normalize keys
    for df, (mat_col, unit_col) in [
        (inputs, inputs_keys), (costs, costs_key_cols), (impacts, impacts_key_cols)
    ]:
        df["_mat_key"]  = normalize_key(df[mat_col])
        df["_unit_key"] = normalize_key(df[unit_col])

    # Keep original visible names
    base = inputs.copy()
    # coerce Amount numeric
    base["Amount"] = to_numeric(base[inputs_keys[2]]).fillna(0.0)

    # Select cost columns
    costs_use = costs[["_mat_key","_unit_key", cost_col]].copy()
    costs_use[cost_col] = to_numeric(costs_use[cost_col]).fillna(0.0)

    # Select impact columns
    imp_use = impacts[["_mat_key","_unit_key"] + impact_cols].copy()
    for c in impact_cols:
        imp_use[c] = to_numeric(imp_use[c]).fillna(0.0)

    # Merge
    m1 = base.merge(costs_use, on=["_mat_key","_unit_key"], how="left", indicator=True)
    m2 = m1.merge(imp_use, on=["_mat_key","_unit_key"], how="left", suffixes=("",""), indicator=True)

    # Diagnostics
    diag = {}
    diag["inputs_n"] = len(inputs)
    diag["costs_n"]  = len(costs)
    diag["impacts_n"]= len(impacts)
    diag["matched_to_costs"]   = int((m1["_merge"]=="both").sum())
    diag["unmatched_inputs_to_costs"] = m1.loc[m1["_merge"]=="left_only", ["_mat_key","_unit_key"]].drop_duplicates()
    diag["unmatched_costs_to_inputs"] = costs_use.merge(
        base[["_mat_key","_unit_key"]], on=["_mat_key","_unit_key"], how="left", indicator=True
    ).loc[lambda d: d["_merge"]=="left_only", ["_mat_key","_unit_key"]].drop_duplicates()

    diag["matched_to_impacts"] = int((m2["_merge"]=="both").sum())
    diag["unmatched_inputs_to_impacts"] = m2.loc[m2["_merge"]=="left_only", ["_mat_key","_unit_key"]].drop_duplicates()
    diag["unmatched_impacts_to_inputs"] = imp_use.merge(
        base[["_mat_key","_unit_key"]], on=["_mat_key","_unit_key"], how="left", indicator=True
    ).loc[lambda d: d["_merge"]=="left_only", ["_mat_key","_unit_key"]].drop_duplicates()

    m2.drop(columns=["_merge"], inplace=True, errors="ignore")

    # Presentable columns
    out = base.copy()
    # rename standard names
    out.rename(columns={inputs_keys[0]:"Material", inputs_keys[1]:"Unit", inputs_keys[2]:"Amount"}, inplace=True)
    # carry cost & impacts
    out[cost_col] = m2[cost_col].fillna(0.0)
    for c in impact_cols:
        out[c] = m2[c].fillna(0.0)

    # Roll up duplicates (if any)
    rolled = out.groupby(["Material","Unit"], as_index=False).agg(
        Amount=("Amount","sum"),
        **{cost_col:(cost_col,"first")},
        **{c:(c,"first") for c in impact_cols}
    )
    return rolled.fillna(0.0), diag

# ---------- GA helpers (unchanged behavior) ----------
def _clip(ind, lows, highs):
    for i in range(len(ind)):
        if ind[i] < lows[i]: ind[i] = lows[i]
        elif ind[i] > highs[i]: ind[i] = highs[i]
    return ind

def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, df, cost_col, impact_cols):
    for attr in ("FitnessMin","Individual"):
        try: delattr(creator, attr)
        except Exception: pass

    n_obj = 1 + len(impact_cols)
    creator.create("FitnessMin", base.Fitness, weights=tuple([-1.0]*n_obj))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    impact_matrix = df[impact_cols].to_numpy(dtype=float)
    costs = df[cost_col].to_numpy(dtype=float)

    def evaluate(ind):
        x = np.maximum(0.0, np.array(ind, dtype=float))
        total_cost = float(np.dot(x, costs))
        totals = [float(np.dot(x, impact_matrix[:, i])) for i in range(impact_matrix.shape[1])]
        return (total_cost, *totals)

    toolbox.register("evaluate", evaluate)

    def _clip_local(individual):
        for i in range(len(individual)):
            if individual[i] < lows[i]: individual[i] = lows[i]
            elif individual[i] > highs[i]: individual[i] = highs[i]

    def make_children(pop):
        selected = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]
        for i in range(0, len(offspring), 2):
            if i+1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i+1], alpha=0.5)
            _clip_local(offspring[i])
            if i+1 < len(offspring): _clip_local(offspring[i+1])
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

    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single(obj, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    for attr in ("FitnessMin","Individual"):
        try: delattr(creator, attr)
        except Exception: pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    def evaluate(ind): return obj(ind, *args, lows, highs)
    def make_children(pop):
        selected = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [creator.Individual(ind[:]) for ind in selected]
        for i in range(0, len(offspring), 2):
            if i+1 < len(offspring) and random.random() < cxpb:
                tools.cxBlend(offspring[i], offspring[i+1], alpha=0.5)
            _clip(offspring[i], lows, highs)
            if i+1 < len(offspring): _clip(offspring[i+1], lows, highs)
        for ind in offspring:
            if random.random() < mutpb: tools.mutGaussian(ind, mu=0, sigma=0.1, indpb=0.2)
            _clip(ind, lows, highs)
        return offspring
    pop = toolbox.population(n=popsize)
    for ind in pop:
        _clip(ind, lows, highs); ind.fitness.values = evaluate(ind)
    for _ in range(ngen):
        off = make_children(pop)
        for ind in off: ind.fitness.values = evaluate(ind)
        pop = tools.selBest(pop + off, popsize)
    return tools.selBest(pop, 1)[0]

def eval_cost_only(ind, df, cost_col, lows, highs):
    _clip(ind, lows, highs); x = np.asarray(ind, float)
    return (float(np.dot(x, df[cost_col].to_numpy(dtype=float))),)

def eval_single_impact(ind, df, impact_col, lows, highs):
    _clip(ind, lows, highs); x = np.asarray(ind, float)
    return (float(np.dot(x, df[impact_col].to_numpy(dtype=float))),)

def eval_cost_plus_all(ind, df, impact_cols, cost_col, alpha_cost, alpha_imp, lows, highs):
    _clip(ind, lows, highs); x = np.asarray(ind, float)
    total_cost = float(np.dot(x, df[cost_col].to_numpy(dtype=float)))
    total_imp  = float(np.dot(x, df[impact_cols].to_numpy(dtype=float)).sum())
    return (alpha_cost*total_cost + alpha_imp*total_imp,)

# ---------- UI ----------
st.title("LCA Optimization: Cost vs GWP")

uploaded = st.file_uploader("Upload (.csv / .xlsx / .xlsm)", type=["csv","xlsx","xlsm"])
if not uploaded:
    st.stop()

sheets = read_workbook(uploaded)
sheet_names = list(sheets.keys())

st.info("Select sheets & map columns. Then the app will robustly merge (case/space-insensitive) and show any mismatches.")

# Choose mode
mode = st.radio("Workbook layout", ["Single merged sheet", "Three-sheet (Inputs / Costs / Impacts)"], horizontal=True)

if mode == "Single merged sheet":
    sname = st.selectbox("Select sheet", sheet_names, index=0)
    df = sheets[sname].copy()

    # Map columns
    cols = df.columns.tolist()
    c1, c2, c3 = st.columns(3)
    mat_col = c1.selectbox("Material column", cols)
    unit_col= c2.selectbox("Unit column", cols)
    amt_col = c3.selectbox("Amount column", cols)

    # Cost + impacts
    numeric_cands = [c for c in cols if c not in (mat_col, unit_col, amt_col)]
    cost_col = st.selectbox("Unit Cost column", numeric_cands)
    impact_cols = st.multiselect("Impact columns (TRACI etc.)", numeric_cands, default=[c for c in numeric_cands if "/Unit" in c or "Eq" in c])

    # Build rolled
    df["_mat_key"]  = normalize_key(df[mat_col])
    df["_unit_key"] = normalize_key(df[unit_col])
    out = pd.DataFrame({
        "Material": df[mat_col],
        "Unit": df[unit_col],
        "Amount": to_numeric(df[amt_col]).fillna(0.0),
        cost_col: to_numeric(df[cost_col]).fillna(0.0),
    })
    for c in impact_cols:
        out[c] = to_numeric(df[c]).fillna(0.0)

    rolled_df = out.groupby(["Material","Unit"], as_index=False).agg(
        Amount=("Amount","sum"),
        **{cost_col:(cost_col,"first")},
        **{c:(c,"first") for c in impact_cols}
    )

    diag = {"note":"Single sheet; no cross-sheet mismatches."}

else:
    # Three-sheet explicit
    s_inputs  = st.selectbox("Inputs sheet", sheet_names, index=0)
    s_costs   = st.selectbox("Costs sheet", sheet_names, index=min(1,len(sheet_names)-1))
    s_impacts = st.selectbox("Impacts sheet", sheet_names, index=min(2,len(sheet_names)-1))

    df_in  = sheets[s_inputs].copy()
    df_co  = sheets[s_costs].copy()
    df_imp = sheets[s_impacts].copy()

    cols_in  = df_in.columns.tolist()
    cols_co  = df_co.columns.tolist()
    cols_imp = df_imp.columns.tolist()

    st.markdown("**Inputs mapping**")
    c1, c2, c3 = st.columns(3)
    mat_in = c1.selectbox("Material (inputs)", cols_in)
    unit_in= c2.selectbox("Unit (inputs)", cols_in)
    amt_in = c3.selectbox("Amount (inputs)", cols_in)

    st.markdown("**Costs mapping**")
    c1, c2 = st.columns(2)
    mat_co = c1.selectbox("Material (costs)", cols_co)
    unit_co= c2.selectbox("Unit (costs)", cols_co)
    cost_col = st.selectbox("Unit Cost column", [c for c in cols_co if c not in (mat_co, unit_co)])

    st.markdown("**Impacts mapping**")
    c1, c2 = st.columns(2)
    mat_im = c1.selectbox("Material (impacts)", cols_imp)
    unit_im= c2.selectbox("Unit (impacts)", cols_imp)
    impact_cols = st.multiselect("Impact columns (TRACI etc.)", [c for c in cols_imp if c not in (mat_im, unit_im)], default=[c for c in cols_imp if "/Unit" in c or "Eq" in c or "CTU" in c])

    rolled_df, diag = merge_inputs_costs_impacts(
        inputs=df_in, costs=df_co, impacts=df_imp,
        inputs_keys=(mat_in, unit_in, amt_in),
        costs_key_cols=(mat_co, unit_co),
        impacts_key_cols=(mat_im, unit_im),
        cost_col=cost_col, impact_cols=impact_cols
    )

# ---------- show table + diagnostics ----------
st.success("File parsed & merged.")
st.dataframe(rolled_df, use_container_width=True)

with st.expander("Merge diagnostics"):
    if "note" in diag:
        st.write(diag["note"])
    else:
        st.write(f"Inputs rows: {diag['inputs_n']} | Costs rows: {diag['costs_n']} | Impacts rows: {diag['impacts_n']}")
        st.write(f"Matched Inputs↔Costs: {diag['matched_to_costs']}")
        st.write("Unmatched Inputs→Costs (check spelling/case/whitespace):")
        st.dataframe(diag["unmatched_inputs_to_costs"])
        st.write("Unmatched Costs→Inputs (extra rows in costs not found in inputs):")
        st.dataframe(diag["unmatched_costs_to_inputs"])
        st.write(f"Matched Inputs↔Impacts: {diag['matched_to_impacts']}")
        st.write("Unmatched Inputs→Impacts:")
        st.dataframe(diag["unmatched_inputs_to_impacts"])
        st.write("Unmatched Impacts→Inputs:")
        st.dataframe(diag["unmatched_impacts_to_inputs"])

# ---------- Baseline metrics ----------
cost_col_name = cost_col  # after mapping above
gwp_guess = None
for c in rolled_df.columns:
    c_norm = c.lower()
    if "co2" in c_norm and "unit" in c_norm:
        gwp_guess = c; break
gwp_col = st.selectbox("Pick GWP column for charts/metrics", [c for c in rolled_df.columns if c not in ("Material","Unit","Amount")], index=[c for c in rolled_df.columns].index(gwp_guess) if gwp_guess else 0)

rolled_df["Amount"] = to_numeric(rolled_df["Amount"]).fillna(0.0)
rolled_df[cost_col_name] = to_numeric(rolled_df[cost_col_name]).fillna(0.0)
rolled_df[gwp_col] = to_numeric(rolled_df[gwp_col]).fillna(0.0)

baseline_cost = float((rolled_df["Amount"] * rolled_df[cost_col_name]).sum())
baseline_gwp  = float((rolled_df["Amount"] * rolled_df[gwp_col]).sum())
b1,b2,b3,b4 = st.columns(4)
b1.metric("Baseline Cost ($)", f"{baseline_cost:,.2f}")
b2.metric("Baseline GWP (kg CO₂e)", f"{baseline_gwp:,.2f}")
b3.metric("Cost / tree ($/tree)", f"{baseline_cost/DEFAULT_TREES:,.2f}")
b4.metric("GWP / tree (kg CO₂e/tree)", f"{baseline_gwp/DEFAULT_TREES:,.2f}")

# ---------- Bounds & GA controls ----------
materials = rolled_df["Material"].astype(str).tolist()
base_amounts = rolled_df["Amount"].to_numpy(dtype=float)

scenario = st.sidebar.selectbox("Optimization Scenario", [
    "Optimize Cost vs GWP (Tradeoff)",
    "Optimize Cost + Combined Impact",
    "Optimize Single Impact",
    "Optimize Cost Only",
])
global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
use_custom = st.sidebar.checkbox("Set per-material bounds")

lows = np.maximum(0.0, base_amounts*(1 - global_dev/100))
highs = base_amounts*(1 + global_dev/100)

if use_custom:
    st.sidebar.markdown("**Per-material ±%**")
    for i, m in enumerate(materials):
        dev = st.sidebar.number_input(m, min_value=0, max_value=100, value=global_dev, step=5, key=f"dev_{i}")
        lows[i]  = max(0.0, base_amounts[i]*(1 - dev/100))
        highs[i] = base_amounts[i]*(1 + dev/100)

st.sidebar.markdown("### Hard minimum floors (optional)")
protect_mat = st.sidebar.multiselect("Choose materials to floor", materials, default=[m for m in materials if "diesel" in m.lower()])
for m in protect_mat:
    i = materials.index(m)
    pct = st.sidebar.number_input(f"Min % of baseline for {m}", min_value=0, max_value=100, value=80, step=5, key=f"floor_{i}")
    lows[i] = max(lows[i], base_amounts[i]*pct/100)

popsize = st.sidebar.slider("Population Size", 10, 200, 50)
ngen    = st.sidebar.slider("Generations", 10, 200, 40)
cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.60)
mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.30)

# Choose which impacts to include in multi-objective run (GWP is recommended first)
candidate_impacts = [c for c in rolled_df.columns if c not in ("Material","Unit","Amount",cost_col_name)]
impact_cols = st.multiselect("Impacts to include in optimization", candidate_impacts, default=[gwp_col])

sel_impact = None
if scenario == "Optimize Single Impact":
    sel_impact = st.selectbox("Select impact column", impact_cols)

# ---------- Run optimization ----------
if st.button("Run Optimization", type="primary"):
    st.subheader(f"Running: {scenario}")

    if scenario == "Optimize Cost vs GWP (Tradeoff)":
        pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col_name, impact_cols)

        rows = []
        for ind in pareto:
            x = np.asarray(_clip(list(ind), list(lows), list(highs)), float)
            row = {"Total Cost ($)": float(np.dot(x, rolled_df[cost_col_name].to_numpy(float)))}
            for imp in impact_cols:
                row[imp] = float(np.dot(x, rolled_df[imp].to_numpy(float)))
            rows.append(row)

        df_pf = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["Total Cost ($)"]+impact_cols)
        st.dataframe(df_pf, use_container_width=True)
        st.download_button("Download Pareto (CSV)", df_pf.to_csv(index=False).encode(), "pareto_table.csv", "text/csv")

        if impact_cols:
            fig, ax = plt.subplots()
            ax.scatter(df_pf["Total Cost ($)"], df_pf[impact_cols[0]])
            ax.set_xlabel("Total Cost ($)"); ax.set_ylabel(impact_cols[0])
            ax.set_title("Pareto front")
            st.pyplot(fig)

    elif scenario == "Optimize Cost + Combined Impact":
        # Simple scalarization
        def obj(ind, df, imps, costc, lows, highs):
            _clip(ind, lows, highs)
            x = np.asarray(ind, float)
            return (float(np.dot(x, df[costc].to_numpy(float)) +
                          np.dot(x, df[imps].to_numpy(float)).sum()),)
        best = run_single(obj, popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, impact_cols, cost_col_name)
        st.write("Objective value:", f"{best.fitness.values[0]:,.2f}")

    elif scenario == "Optimize Single Impact" and sel_impact:
        best = run_single(lambda ind, df, imp, lows, highs: eval_single_impact(ind, df, imp, lows, highs),
                          popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, sel_impact)
        st.write(sel_impact, f"{best.fitness.values[0]:,.4f}")

    elif scenario == "Optimize Cost Only":
        best = run_single(lambda ind, df, ccol, lows, highs: eval_cost_only(ind, df, ccol, lows, highs),
                          popsize, ngen, cxpb, mutpb, lows, highs, rolled_df, cost_col_name)
        st.write("Total Cost ($):", f"{best.fitness.values[0]:,.2f}")

# ---------- Download merged ----------
st.markdown("#### Download merged totals")
buf = BytesIO()
with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
    rolled_df.to_excel(w, index=False, sheet_name="Merged Totals")
buf.seek(0)
st.download_button("Save Excel", buf, "merged_totals_for_optimization.xlsx",
                   "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
