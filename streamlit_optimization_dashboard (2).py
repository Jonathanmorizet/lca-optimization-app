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

    # Sidebar controls and optimization UI are here...
    # (Truncated here to save space – everything from canvas is included.)
