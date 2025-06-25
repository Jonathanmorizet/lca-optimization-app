import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import subprocess
import sys
from io import BytesIO

# Ensure DEAP is installed
try:
    from deap import base, creator, tools, algorithms
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap"])
    from deap import base, creator, tools, algorithms

random.seed(42)

# File uploader
uploaded_file = st.file_uploader("Upload the 'DEAP NSGA Readable file.xlsm' Excel file", type=["xlsm"])

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=None)
            inputs_df = df[list(df.keys())[0]]
            cost_df = df[list(df.keys())[1]]
            impact_df = df[list(df.keys())[2]]

            merged_df = inputs_df.merge(cost_df, on=["Material", "Unit"], how="left")
            merged_df = merged_df.merge(impact_df, on=["Material", "Unit"], how="left")
            merged_df = merged_df.fillna(0)

            materials = merged_df['Material'].tolist()
            base_amounts = merged_df['Amount'].values
            costs = merged_df['Unit Cost ($)'].values
            impact_columns = [col for col in impact_df.columns if col in merged_df.columns and col not in ['Material', 'Unit', 'Year']]
            impact_matrix = merged_df[impact_columns].values
            traci_impact_cols = impact_columns

            return merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols

        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

# DEAP Evaluation Functions
def evaluate_cost_gwp(ind, costs, impact_matrix, impact_cols):
    x = np.maximum(0, np.array(ind))
    cost = np.dot(x, costs)
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(x, impact_matrix[:, gwp_idx])
    except ValueError:
        st.warning("GWP column not found.")
    return cost, gwp

def evaluate_cost_only(ind, costs):
    x = np.maximum(0, np.array(ind))
    return (np.dot(x, costs),)

def evaluate_combined(ind, costs, impact_matrix):
    x = np.maximum(0, np.array(ind))
    return (np.dot(x, costs) + np.sum(np.dot(x, impact_matrix)),)

def evaluate_single_impact(ind, matrix, colname, cols):
    x = np.maximum(0, np.array(ind))
    try:
        idx = cols.index(colname)
        return (np.dot(x, matrix[:, idx]),)
    except ValueError:
        return (0.0,)

# Optimization Logic
def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, matrix, impact_cols):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp, costs=costs, impact_matrix=matrix, impact_cols=impact_cols)
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

def run_single(obj_func, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_func, *args)
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
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    return hof[0]

# Load data
merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols = load_data(uploaded_file)

if merged_df is not None:
    st.success("File uploaded and data processed successfully!")
    st.dataframe(merged_df)

    # Sidebar controls
    scenario = st.sidebar.selectbox("Optimization Scenario", [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Cost + Combined Impact",
        "Optimize Single Impact",
        "Optimize Cost Only"
    ])

    global_dev = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
    use_custom_bounds = st.sidebar.checkbox("Set per-material bounds")

    lows = np.copy(base_amounts)
    highs = np.copy(base_amounts)
    if use_custom_bounds:
        for i, mat in enumerate(materials):
            dev = st.sidebar.number_input(f"{mat}", min_value=0, max_value=100, value=global_dev, key=mat)
            lows[i] = max(0, base_amounts[i] * (1 - dev/100))
            highs[i] = base_amounts[i] * (1 + dev/100)
    else:
        lows = np.maximum(0, base_amounts * (1 - global_dev / 100))
        highs = base_amounts * (1 + global_dev / 100)

    popsize = st.sidebar.slider("Population Size", 10, 200, 50)
    ngen = st.sidebar.slider("Generations", 10, 200, 40)
    cxpb = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
    mutpb = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

    selected_impact = None
    if scenario == "Optimize Single Impact":
        selected_impact = st.selectbox("Select TRACI Impact", traci_impact_cols)

    if st.button("Run Optimization"):
        st.subheader(f"Running: {scenario}")

        if scenario == "Optimize Cost vs GWP (Tradeoff)":
            pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix, traci_impact_cols)
            df = pd.DataFrame([[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
                              columns=["Total Cost", "Total GWP"])
            st.dataframe(df)
            fig, ax = plt.subplots()
            ax.scatter(df["Total Cost"], df["Total GWP"])
            st.pyplot(fig)
            st.session_state.history.append({"scenario": scenario, "results": df})

        elif scenario == "Optimize Cost + Combined Impact":
            best = run_single(evaluate_combined, popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric("Objective", f"{best.fitness.values[0]:.2f}")
            st.dataframe(df)
            st.session_state.history.append({"scenario": scenario, "results": df})

        elif scenario == "Optimize Cost Only":
            best = run_single(evaluate_cost_only, popsize, ngen, cxpb, mutpb, lows, highs, costs)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric("Cost", f"{best.fitness.values[0]:.2f}")
            st.dataframe(df)
            st.session_state.history.append({"scenario": scenario, "results": df})

        elif scenario == "Optimize Single Impact" and selected_impact:
            best = run_single(evaluate_single_impact, popsize, ngen, cxpb, mutpb, lows, highs, impact_matrix, selected_impact, traci_impact_cols)
            df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized": best})
            st.metric(f"{selected_impact}", f"{best.fitness.values[0]:.2f}")
            st.dataframe(df)
            st.session_state.history.append({"scenario": f"Single Impact - {selected_impact}", "results": df})

    if st.button("Download Merged Data as Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
        output.seek(0)
        st.download_button(
            label="Download Excel File",
            data=output,
            file_name="optimized_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    with st.expander("\U0001F4C8 View Optimization History"):
        if st.session_state['history']:
            for i, record in enumerate(st.session_state['history'], 1):
                st.write(f"**Run {i}: {record['scenario']}**")
                st.dataframe(record['results'])
        else:
            st.info("No optimization runs recorded yet.")
else:
    st.info("Upload a valid Excel file to begin.")
