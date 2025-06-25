import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools, algorithms
import random
import io
import xlsxwriter

# -----------------------------
# Data Loading Function
# -----------------------------
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

# -----------------------------
# Optimization Functions
# -----------------------------
def evaluate_cost_gwp(individual, costs, impact_matrix, traci_impact_cols):
    adjusted_amounts = np.maximum(0, np.array(individual))
    total_cost = np.dot(adjusted_amounts, costs)
    try:
        gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
        total_gwp = np.dot(adjusted_amounts, impact_matrix[:, gwp_idx])
    except ValueError:
        total_gwp = 0
        st.warning("GWP column not found.")
    return total_cost, total_gwp

def evaluate_cost_combined(individual, costs, impact_matrix):
    adjusted_amounts = np.maximum(0, np.array(individual))
    total_cost = np.dot(adjusted_amounts, costs)
    total_impact = np.sum(np.dot(adjusted_amounts, impact_matrix))
    return (total_cost + total_impact,)

def evaluate_single_impact(individual, impact_matrix, idx):
    adjusted_amounts = np.maximum(0, np.array(individual))
    return (np.dot(adjusted_amounts, impact_matrix[:, idx]),)

def run_nsga2_optimization(pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, costs, impact_matrix, traci_impact_cols):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass

    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [random.uniform(l, h) for l, h in zip(low_bounds, high_bounds)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp, costs=costs, impact_matrix=impact_matrix, traci_impact_cols=traci_impact_cols)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def mutate(ind, mu, sigma, indpb, low_bounds, high_bounds):
        ind, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(ind)):
            ind[i] = max(low_bounds[i], min(high_bounds[i], ind[i]))
        return ind,

    toolbox.register("mutate", mutate, mu=0, sigma=0.1, indpb=0.2, low_bounds=low_bounds, high_bounds=high_bounds)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    algorithms.eaMuPlusLambda(pop, toolbox, mu=pop_size, lambda_=pop_size, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)

    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single_objective_optimization(func, pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, *args):
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     lambda: [random.uniform(l, h) for l, h in zip(low_bounds, high_bounds)])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", func, *args)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def mutate(ind, mu, sigma, indpb, low_bounds, high_bounds):
        ind, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(ind)):
            ind[i] = max(low_bounds[i], min(high_bounds[i], ind[i]))
        return ind,

    toolbox.register("mutate", mutate, mu=0, sigma=0.1, indpb=0.2, low_bounds=low_bounds, high_bounds=high_bounds)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    hof = tools.HallOfFame(1)
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)

    return hof[0]

# -----------------------------
# Export and Plot Functions
# -----------------------------
def export_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Optimized Inputs')
    return output.getvalue()

def plot_material_distribution(materials, base_amounts, optimized_amounts):
    fig, ax = plt.subplots(figsize=(10, 4))
    indices = np.arange(len(materials))
    width = 0.35
    ax.bar(indices - width/2, base_amounts, width, label='Base')
    ax.bar(indices + width/2, optimized_amounts, width, label='Optimized')
    ax.set_xticks(indices)
    ax.set_xticklabels(materials, rotation=90)
    ax.set_ylabel('Material Amount')
    ax.set_title('Material Input Comparison')
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# Streamlit UI
# -----------------------------

st.title("Material Input Optimization Dashboard")
uploaded_file = st.file_uploader("Upload your Excel file", type=["xls", "xlsx", "xlsm"])

merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols = load_data(uploaded_file)

if merged_df is not None:
    st.success("Data loaded successfully.")
    scenario = st.sidebar.selectbox("Select Optimization Scenario", [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Cost + Combined Impact",
        "Optimize Single TRACI Impact",
        "Optimize Cost Only"
    ])

    deviation = st.sidebar.slider("Global ±% Deviation", 0, 100, 20)
    use_custom_bounds = st.sidebar.checkbox("Set individual ±% deviation")

    low_bounds = np.maximum(0, base_amounts * (1 - deviation / 100))
    high_bounds = base_amounts * (1 + deviation / 100)

    if use_custom_bounds:
        for i, mat in enumerate(materials):
            dev = st.sidebar.number_input(f"{mat} deviation %", min_value=0, max_value=100, value=deviation, key=f"dev_{i}")
            low_bounds[i] = max(0, base_amounts[i] * (1 - dev / 100))
            high_bounds[i] = base_amounts[i] * (1 + dev / 100)

    pop_size = st.sidebar.slider("Population Size", 10, 200, 50)
    ngen = st.sidebar.slider("Generations", 10, 200, 40)
    cxpb = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.6)
    mutpb = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

    if scenario == "Optimize Single TRACI Impact":
        selected_impact = st.sidebar.selectbox("Select TRACI Impact", traci_impact_cols)
        impact_idx = traci_impact_cols.index(selected_impact)
    else:
        selected_impact = None

    if st.button("Run Optimization"):
        if scenario == "Optimize Cost vs GWP (Tradeoff)":
            pareto = run_nsga2_optimization(pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, costs, impact_matrix, traci_impact_cols)
            data = [[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto]
            df = pd.DataFrame(data, columns=["Total Cost", "Total GWP"])
            st.subheader("Pareto Front")
            st.dataframe(df)

        elif scenario == "Optimize Cost + Combined Impact":
            best = run_single_objective_optimization(evaluate_cost_combined, pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, costs, impact_matrix)
            optimized_df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized Amount": best})
            st.subheader("Optimized Inputs")
            st.dataframe(optimized_df)
            plot_material_distribution(materials, base_amounts, best)

        elif scenario == "Optimize Single TRACI Impact" and selected_impact:
            best = run_single_objective_optimization(evaluate_single_impact, pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, impact_matrix, impact_idx)
            optimized_df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized Amount": best})
            st.subheader(f"Optimized for {selected_impact}")
            st.dataframe(optimized_df)
            plot_material_distribution(materials, base_amounts, best)

        elif scenario == "Optimize Cost Only":
            def eval_cost(ind, costs): return (np.dot(np.maximum(0, ind), costs),)
            best = run_single_objective_optimization(eval_cost, pop_size, ngen, cxpb, mutpb, low_bounds, high_bounds, costs)
            optimized_df = pd.DataFrame({"Material": materials, "Base Amount": base_amounts, "Optimized Amount": best})
            st.subheader("Optimized for Cost")
            st.dataframe(optimized_df)
            plot_material_distribution(materials, base_amounts, best)

        if 'optimized_df' in locals():
            excel_data = export_to_excel(optimized_df)
            st.download_button(label="Download Optimized Inputs", data=excel_data, file_name="optimized_inputs.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
