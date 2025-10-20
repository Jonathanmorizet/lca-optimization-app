import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import subprocess
import sys
from io import BytesIO

# --- Ensure DEAP is installed ---
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap"])
    from deap import base, creator, tools, algorithms

# Determinism
random.seed(42)
np.random.seed(42)

st.set_page_config(page_title="NSGA-II LCA/TEA Optimizer", layout="wide")

# File uploader
uploaded_file = st.file_uploader(
    "Upload the Excel file (e.g., 'DEAP NSGA Readable file.xlsm')",
    type=["xlsm", "xlsx"]
)

# ---- Init session state early (before any appends/reads) ----
if "history" not in st.session_state:
    st.session_state["history"] = []

# ---------- Data Loading ----------
@st.cache_data
def load_data(uploaded):
    if uploaded is None:
        return (None,)*6
    try:
        # Read all sheets
        sheets = pd.read_excel(uploaded, sheet_name=None)

        # Prefer name-based sheets; fallback to first three in order
        def get_sheet(candidates, default_idx):
            for c in candidates:
                for key in sheets.keys():
                    if str(key).strip().lower() == c:
                        return sheets[key]
            # fallback by position
            if len(sheets) > default_idx:
                return sheets[list(sheets.keys())[default_idx]]
            return None

        inputs_df  = get_sheet(["inputs", "input", "materials", "sheet1"], 0)
        cost_df    = get_sheet(["costs", "cost", "prices", "sheet2"], 1)
        impact_df  = get_sheet(["impacts", "impact", "traci", "sheet3"], 2)

        if inputs_df is None or cost_df is None or impact_df is None:
            raise ValueError("Could not identify the Inputs/Costs/Impacts sheets by name or position.")

        # Normalize keys
        for df in (inputs_df, cost_df, impact_df):
            df.columns = [str(c).strip() for c in df.columns]

        # Expected join keys
        key_cols = ["Material", "Unit"]
        for key in key_cols:
            if key not in inputs_df.columns:  raise ValueError(f"Missing '{key}' in Inputs sheet.")
            if key not in cost_df.columns:    raise ValueError(f"Missing '{key}' in Costs sheet.")
            if key not in impact_df.columns:  raise ValueError(f"Missing '{key}' in Impacts sheet.")

        merged_df = inputs_df.merge(cost_df, on=key_cols, how="left")
        merged_df = merged_df.merge(impact_df, on=key_cols, how="left")
        merged_df = merged_df.fillna(0)

        # Required numeric columns (with flexible aliases for Amount and Cost)
        amount_col = None
        for cand in ["Amount", "Base Amount", "Qty", "Quantity"]:
            if cand in merged_df.columns:
                amount_col = cand
                break
        if amount_col is None:
            raise ValueError("Could not find an 'Amount' column (tried: Amount, Base Amount, Qty, Quantity).")

        cost_col = None
        for cand in ["Unit Cost ($)", "Unit Cost", "Cost/Unit", "Cost per Unit", "Price"]:
            if cand in merged_df.columns:
                cost_col = cand
                break
        if cost_col is None:
            raise ValueError("Could not find a Unit Cost column (tried: Unit Cost ($), Unit Cost, Cost/Unit, ...).")

        # Impact columns = all numeric columns in impact_df except keys & 'Year'
        non_impact = set(["Material", "Unit", "Year"])
        impact_columns = [c for c in impact_df.columns if c not in non_impact]
        # Keep only those that exist post-merge (and are numeric)
        impact_columns = [c for c in impact_columns if c in merged_df.columns and pd.api.types.is_numeric_dtype(merged_df[c])]

        if len(impact_columns) == 0:
            st.warning("No numeric impact columns were detected. Single-impact or combined-impact scenarios may not work.")

        materials     = merged_df["Material"].tolist()
        base_amounts  = merged_df[amount_col].astype(float).values
        costs         = merged_df[cost_col].astype(float).values
        impact_matrix = merged_df[impact_columns].astype(float).values

        # Try to identify GWP column by common aliases
        gwp_aliases = [
            "kg CO2-Eq/Unit", "kg CO2e/Unit", "GWP", "GWP100", "Climate Change", "kg CO2e"
        ]
        gwp_name = next((c for c in gwp_aliases if c in impact_columns), None)
        if gwp_name is None:
            # If a column contains 'co2' heuristic
            co2_candidates = [c for c in impact_columns if "co2" in c.lower()]
            gwp_name = co2_candidates[0] if co2_candidates else None
            if gwp_name is None:
                st.warning("Could not find a GWP column. Cost vs. GWP scenario will still run but show 0 for GWP.")

        return merged_df, materials, base_amounts, costs, impact_matrix, impact_columns, gwp_name, amount_col

    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return (None,)*8

# ---------- Objective functions ----------
def evaluate_cost_gwp(ind, costs, impact_matrix, impact_cols, gwp_name):
    x = np.maximum(0, np.array(ind))
    cost = float(np.dot(x, costs))
    if gwp_name and gwp_name in impact_cols:
        gwp_idx = impact_cols.index(gwp_name)
        gwp = float(np.dot(x, impact_matrix[:, gwp_idx]))
    else:
        gwp = 0.0
    return cost, gwp

def evaluate_cost_only(ind, costs):
    x = np.maximum(0, np.array(ind))
    return (float(np.dot(x, costs)),)

def evaluate_combined(ind, costs, impact_matrix):
    x = np.maximum(0, np.array(ind))
    # Sum of all impacts + cost (simple scalarization)
    return (float(np.dot(x, costs) + np.sum(np.dot(x, impact_matrix))),)

def evaluate_single_impact(ind, matrix, colname, cols):
    x = np.maximum(0, np.array(ind))
    if colname in cols:
        idx = cols.index(colname)
        return (float(np.dot(x, matrix[:, idx])),)
    return (0.0,)

# --- Utility: safe (re)create DEAP classes once per arity ---
def ensure_deap_classes(name_suffix, weights):
    fit_name = f"FitnessMin_{name_suffix}"
    ind_name = f"Individual_{name_suffix}"
    if not hasattr(creator, fit_name):
        creator.create(fit_name, base.Fitness, weights=weights)
    if not hasattr(creator, ind_name):
        creator.create(ind_name, list, fitness=getattr(creator, fit_name))
    return fit_name, ind_name

# ---------- Optimization Routines ----------
def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, matrix, impact_cols, gwp_name):
    fit_name, ind_name = ensure_deap_classes("2obj", (-1.0, -1.0))

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, getattr(creator, ind_name), toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evaluate_cost_gwp,
                     costs=costs, impact_matrix=matrix, impact_cols=impact_cols, gwp_name=gwp_name)

    # Blend crossover then clamp
    def cx_and_clip(ind1, ind2, alpha=0.5):
        tools.cxBlend(ind1, ind2, alpha)
        for i in range(len(ind1)):
            ind1[i] = float(np.clip(ind1[i], lows[i], highs[i]))
            ind2[i] = float(np.clip(ind2[i], lows[i], highs[i]))
        return ind1, ind2

    def mut_and_clip(ind, mu=0, sigma=0.1, indpb=0.2):
        (mutated,) = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = float(np.clip(mutated[i], lows[i], highs[i]))
        return (mutated,)

    toolbox.register("mate", cx_and_clip)
    toolbox.register("mutate", mut_and_clip)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)

    # Evaluate initial pop
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Must assign crowding distance before first select
    pop = toolbox.select(pop, len(pop))

    algorithms.eaMuPlusLambda(
        pop, toolbox, mu=popsize, lambda_=popsize,
        cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False
    )

    # Return first nondominated front
    fronts = tools.sortNondominated(pop, k=len(pop), first_front_only=True)
    return fronts[0] if fronts else []

def run_single(obj_func, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    fit_name, ind_name = ensure_deap_classes("1obj", (-1.0,))

    toolbox = base.Toolbox()
    toolbox.register("attr_vec", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, getattr(creator, ind_name), toolbox.attr_vec)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_func, *args)

    def cx_and_clip(ind1, ind2, alpha=0.5):
        tools.cxBlend(ind1, ind2, alpha)
        for i in range(len(ind1)):
            ind1[i] = float(np.clip(ind1[i], lows[i], highs[i]))
            ind2[i] = float(np.clip(ind2[i], lows[i], highs[i]))
        return ind1, ind2

    def mut_and_clip(ind, mu=0, sigma=0.1, indpb=0.2):
        (mutated,) = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = float(np.clip(mutated[i], lows[i], highs[i]))
        return (mutated,)

    toolbox.register("mate", cx_and_clip)
    toolbox.register("mutate", mut_and_clip)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popsize)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    return hof[0]

# ---------- App Body ----------
merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols, gwp_name, amount_col = load_data(uploaded_file)

if merged_df is not None:
    st.success("File uploaded and data processed successfully!")
    st.dataframe(merged_df, use_container_width=True)

    # Sidebar controls
    scenario = st.sidebar.selectbox(
        "Optimization Scenario",
        [
            "Optimize Cost vs GWP (Tradeoff)",
            "Optimize Cost + Combined Impact",
            "Optimize Single Impact",
            "Optimize Cost Only",
        ]
    )

    global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
    use_custom_bounds = st.sidebar.checkbox("Set per-material bounds")

    # Bounds
    lows = np.copy(base_amounts)
    highs = np.copy(base_amounts)
    if use_custom_bounds:
        for i, mat in enumerate(materials):
            dev = st.sidebar.number_input(
                f"{mat}", min_value=0, max_value=100, value=global_dev, key=f"dev_{mat}"
            )
            lows[i] = max(0.0, float(base_amounts[i] * (1 - dev/100)))
            highs[i] = float(base_amounts[i] * (1 + dev/100))
    else:
        lows = np.maximum(0.0, base_amounts * (1 - global_dev / 100))
        highs = base_amounts * (1 + global_dev / 100)

    popsize = st.sidebar.slider("Population Size", 10, 300, 80, step=10)
    ngen    = st.sidebar.slider("Generations", 10, 400, 60, step=10)
    cxpb    = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
    mutpb   = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

    selected_impact = None
    if scenario == "Optimize Single Impact":
        sel_options = traci_impact_cols if traci_impact_cols else ["<no impacts available>"]
        selected_impact = st.selectbox("Select TRACI Impact", sel_options)

    run_clicked = st.button("Run Optimization")

    if run_clicked:
        st.subheader(f"Running: {scenario}")

        if scenario == "Optimize Cost vs GWP (Tradeoff)":
            pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix, traci_impact_cols, gwp_name)
            if not pareto:
                st.warning("No nondominated solutions found.")
            else:
                df = pd.DataFrame(
                    [[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
                    columns=["Total Cost", "Total GWP"]
                )
                st.dataframe(df, use_container_width=True)

                fig, ax = plt.subplots()
                ax.scatter(df["Total Cost"], df["Total GWP"])
                ax.set_xlabel("Total Cost")
                ax.set_ylabel("Total GWP")
                ax.set_title("Pareto Front: Cost vs GWP")
                st.pyplot(fig, clear_figure=True)

                # Attach decisions table too
                dv = pd.DataFrame(pareto, columns=materials)
                with st.expander("Decision Variables (nondominated solutions)"):
                    st.dataframe(dv, use_container_width=True)

                # Download CSV
                csv_buf = BytesIO()
                pd.concat([df, dv], axis=1).to_csv(csv_buf, index=False)
                st.download_button("Download Pareto (CSV)", csv_buf.getvalue(), "pareto.csv", "text/csv")

                st.session_state.history.append({"scenario": scenario, "results": pd.concat([df, dv], axis=1)})

        elif scenario == "Optimize Cost + Combined Impact":
            best = run_single(evaluate_combined, popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": list(best)})
            st.metric("Combined Objective (Cost + All Impacts)", f"{best.fitness.values[0]:,.4f}")
            st.dataframe(df, use_container_width=True)

            xlsx = BytesIO()
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Optimized")
            st.download_button("Download Optimized (Excel)", xlsx.getvalue(),
                               "optimized_combined.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.session_state.history.append({"scenario": scenario, "results": df})

        elif scenario == "Optimize Cost Only":
            best = run_single(evaluate_cost_only, popsize, ngen, cxpb, mutpb, lows, highs, costs)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": list(best)})
            st.metric("Minimum Cost", f"{best.fitness.values[0]:,.4f}")
            st.dataframe(df, use_container_width=True)

            xlsx = BytesIO()
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Optimized")
            st.download_button("Download Optimized (Excel)", xlsx.getvalue(),
                               "optimized_cost_only.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.session_state.history.append({"scenario": scenario, "results": df})

        elif scenario == "Optimize Single Impact" and selected_impact and selected_impact != "<no impacts available>":
            best = run_single(evaluate_single_impact, popsize, ngen, cxpb, mutpb, lows, highs,
                              impact_matrix, selected_impact, traci_impact_cols)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": list(best)})
            st.metric(f"Minimized Impact — {selected_impact}", f"{best.fitness.values[0]:,.4f}")
            st.dataframe(df, use_container_width=True)

            xlsx = BytesIO()
            with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
                df.to_excel(writer, index=False, sheet_name="Optimized")
            st.download_button("Download Optimized (Excel)", xlsx.getvalue(),
                               f"optimized_{selected_impact.replace(' ', '_')}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            st.session_state.history.append({"scenario": f"Single Impact - {selected_impact}", "results": df})
        else:
            st.info("No impact columns available to optimize against.")

    # Separate download of merged data
    st.markdown("---")
    if st.button("Download Merged Data (Excel)"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
        output.seek(0)
        st.download_button(
            label="Download Excel File",
            data=output,
            file_name="merged_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    with st.expander("📈 View Optimization History"):
        if st.session_state['history']:
            for i, record in enumerate(st.session_state['history'], 1):
                st.write(f"**Run {i}: {record['scenario']}**")
                st.dataframe(record['results'], use_container_width=True)
        else:
            st.info("No optimization runs recorded yet.")
else:
    st.info("Upload a valid Excel file to begin.")
