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

# DEAP Evaluation Functions with Functional Unit Constraint
def evaluate_cost_gwp_constrained(ind, costs, impact_matrix, impact_cols, base_amounts, num_trees, max_deviation):
    """
    Optimize cost vs GWP while maintaining functional unit (number of trees).
    Individual represents scaling factors (0.8 to 1.2 for ±20% deviation).
    """
    scaling_factors = np.array(ind)
    x = base_amounts * scaling_factors
    
    # Penalty for violating bounds
    penalty = 0
    for i, sf in enumerate(scaling_factors):
        if sf < (1 - max_deviation) or sf > (1 + max_deviation):
            penalty += 1000 * abs(sf - np.clip(sf, 1 - max_deviation, 1 + max_deviation))
    
    # Calculate total cost and GWP
    total_cost = np.dot(x, costs)
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(x, impact_matrix[:, gwp_idx])
    except ValueError:
        st.warning("GWP column not found.")
    
    # Calculate per-tree metrics for validation
    cost_per_tree = total_cost / num_trees
    gwp_per_tree = gwp / num_trees
    
    return total_cost + penalty, gwp + penalty

def evaluate_cost_only_constrained(ind, costs, base_amounts, num_trees, max_deviation):
    """Optimize cost while maintaining functional unit."""
    scaling_factors = np.array(ind)
    x = base_amounts * scaling_factors
    
    penalty = 0
    for i, sf in enumerate(scaling_factors):
        if sf < (1 - max_deviation) or sf > (1 + max_deviation):
            penalty += 1000 * abs(sf - np.clip(sf, 1 - max_deviation, 1 + max_deviation))
    
    total_cost = np.dot(x, costs)
    return (total_cost + penalty,)

def evaluate_single_impact_constrained(ind, matrix, colname, cols, base_amounts, num_trees, max_deviation):
    """Optimize single impact while maintaining functional unit."""
    scaling_factors = np.array(ind)
    x = base_amounts * scaling_factors
    
    penalty = 0
    for i, sf in enumerate(scaling_factors):
        if sf < (1 - max_deviation) or sf > (1 + max_deviation):
            penalty += 1000 * abs(sf - np.clip(sf, 1 - max_deviation, 1 + max_deviation))
    
    try:
        idx = cols.index(colname)
        impact = np.dot(x, matrix[:, idx])
    except ValueError:
        impact = 0.0
    
    return (impact + penalty,)

# Optimization Logic
def run_nsga2_constrained(popsize, ngen, cxpb, mutpb, costs, matrix, impact_cols, base_amounts, num_trees, max_deviation):
    """Run NSGA-II with functional unit constraint."""
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Initialize scaling factors around 1.0 (meaning 100% of base amount)
    toolbox.register("attr_float", lambda: [random.uniform(1 - max_deviation, 1 + max_deviation) for _ in range(len(base_amounts))])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp_constrained, 
                    costs=costs, impact_matrix=matrix, impact_cols=impact_cols, 
                    base_amounts=base_amounts, num_trees=num_trees, max_deviation=max_deviation)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = np.clip(mutated[i], 1 - max_deviation, 1 + max_deviation)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    algorithms.eaMuPlusLambda(pop, toolbox, mu=popsize, lambda_=popsize, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single_constrained(obj_func, popsize, ngen, cxpb, mutpb, base_amounts, num_trees, max_deviation, *args):
    """Run single-objective optimization with functional unit constraint."""
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    toolbox.register("attr_float", lambda: [random.uniform(1 - max_deviation, 1 + max_deviation) for _ in range(len(base_amounts))])
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_func, *args, base_amounts=base_amounts, num_trees=num_trees, max_deviation=max_deviation)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        for i in range(len(mutated)):
            mutated[i] = np.clip(mutated[i], 1 - max_deviation, 1 + max_deviation)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
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

    # Functional Unit Configuration
    st.sidebar.markdown("### 🌳 Functional Unit")
    num_trees = st.sidebar.number_input("Total Number of Trees", min_value=1, value=1900, step=1)
    st.sidebar.info(f"All optimizations will maintain production of exactly {num_trees} trees")

    # Sidebar controls
    scenario = st.sidebar.selectbox("Optimization Scenario", [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Single Impact",
        "Optimize Cost Only"
    ])

    st.sidebar.markdown("### 📊 Bounds Settings")
    max_deviation_pct = st.sidebar.slider("Maximum ±% Deviation per Material", 5, 30, 20)
    max_deviation = max_deviation_pct / 100.0
    
    st.sidebar.markdown("### 🧬 Genetic Algorithm Parameters")
    popsize = st.sidebar.slider("Population Size", 20, 200, 100)
    ngen = st.sidebar.slider("Generations", 20, 200, 50)
    cxpb = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.7)
    mutpb = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

    selected_impact = None
    if scenario == "Optimize Single Impact":
        selected_impact = st.selectbox("Select TRACI Impact", traci_impact_cols)

    # Calculate baseline metrics
    baseline_cost = np.dot(base_amounts, costs)
    baseline_cost_per_tree = baseline_cost / num_trees
    
    baseline_gwp = 0.0
    if "kg CO2-Eq/Unit" in traci_impact_cols:
        gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
        baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
        baseline_gwp_per_tree = baseline_gwp / num_trees
        
        st.info(f"📌 **Baseline:** ${baseline_cost:.2f} total (${baseline_cost_per_tree:.2f}/tree) | {baseline_gwp:.2f} kg CO2-Eq total ({baseline_gwp_per_tree:.2f} kg CO2-Eq/tree)")

    if st.button("Run Optimization"):
        st.subheader(f"Running: {scenario}")

        if scenario == "Optimize Cost vs GWP (Tradeoff)":
            with st.spinner("Running NSGA-II optimization..."):
                pareto = run_nsga2_constrained(popsize, ngen, cxpb, mutpb, costs, impact_matrix, 
                                              traci_impact_cols, base_amounts, num_trees, max_deviation)
                
                results = []
                for ind in pareto:
                    scaling_factors = np.array(ind)
                    optimized_amounts = base_amounts * scaling_factors
                    total_cost = np.dot(optimized_amounts, costs)
                    
                    gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
                    total_gwp = np.dot(optimized_amounts, impact_matrix[:, gwp_idx])
                    
                    cost_per_tree = total_cost / num_trees
                    gwp_per_tree = total_gwp / num_trees
                    
                    results.append({
                        "Total Cost": total_cost,
                        "Total GWP": total_gwp,
                        "Cost/Tree": cost_per_tree,
                        "GWP/Tree": gwp_per_tree,
                        "Individual": ind
                    })
                
                df_pareto = pd.DataFrame(results)
                st.dataframe(df_pareto[["Total Cost", "Total GWP", "Cost/Tree", "GWP/Tree"]])
                
                # Plot Pareto front
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                ax1.scatter(df_pareto["Total Cost"], df_pareto["Total GWP"], alpha=0.6)
                ax1.scatter([baseline_cost], [baseline_gwp], color='red', s=100, marker='*', label='Baseline')
                ax1.set_xlabel("Total Cost ($)")
                ax1.set_ylabel("Total GWP (kg CO2-Eq)")
                ax1.set_title("Pareto Front: Total Values")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.scatter(df_pareto["Cost/Tree"], df_pareto["GWP/Tree"], alpha=0.6)
                ax2.scatter([baseline_cost_per_tree], [baseline_gwp_per_tree], color='red', s=100, marker='*', label='Baseline')
                ax2.set_xlabel("Cost per Tree ($)")
                ax2.set_ylabel("GWP per Tree (kg CO2-Eq)")
                ax2.set_title("Pareto Front: Per-Tree Values")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Allow user to select a solution
                st.markdown("### Select a Solution from Pareto Front")
                solution_idx = st.selectbox("Solution Index", range(len(pareto)))
                
                selected_ind = pareto[solution_idx]
                scaling_factors = np.array(selected_ind)
                optimized_amounts = base_amounts * scaling_factors
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Base Amount": base_amounts,
                    "Optimized Amount": optimized_amounts,
                    "Change (%)": ((optimized_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                st.session_state.history.append({"scenario": scenario, "results": df_materials, "pareto": df_pareto})

        elif scenario == "Optimize Cost Only":
            with st.spinner("Running single-objective optimization..."):
                best = run_single_constrained(evaluate_cost_only_constrained, popsize, ngen, cxpb, mutpb, 
                                             base_amounts, num_trees, max_deviation, costs)
                
                scaling_factors = np.array(best)
                optimized_amounts = base_amounts * scaling_factors
                total_cost = np.dot(optimized_amounts, costs)
                cost_per_tree = total_cost / num_trees
                
                st.metric("Total Cost", f"${total_cost:.2f}", f"{((total_cost - baseline_cost) / baseline_cost * 100):.1f}%")
                st.metric("Cost per Tree", f"${cost_per_tree:.2f}", f"{((cost_per_tree - baseline_cost_per_tree) / baseline_cost_per_tree * 100):.1f}%")
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Base Amount": base_amounts,
                    "Optimized Amount": optimized_amounts,
                    "Change (%)": ((optimized_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                st.session_state.history.append({"scenario": scenario, "results": df_materials})

        elif scenario == "Optimize Single Impact" and selected_impact:
            with st.spinner(f"Optimizing {selected_impact}..."):
                best = run_single_constrained(evaluate_single_impact_constrained, popsize, ngen, cxpb, mutpb, 
                                             base_amounts, num_trees, max_deviation, 
                                             impact_matrix, selected_impact, traci_impact_cols)
                
                scaling_factors = np.array(best)
                optimized_amounts = base_amounts * scaling_factors
                
                impact_idx = traci_impact_cols.index(selected_impact)
                total_impact = np.dot(optimized_amounts, impact_matrix[:, impact_idx])
                impact_per_tree = total_impact / num_trees
                
                baseline_impact = np.dot(base_amounts, impact_matrix[:, impact_idx])
                baseline_impact_per_tree = baseline_impact / num_trees
                
                st.metric(f"Total {selected_impact}", f"{total_impact:.4f}", 
                         f"{((total_impact - baseline_impact) / baseline_impact * 100):.1f}%")
                st.metric(f"{selected_impact} per Tree", f"{impact_per_tree:.4f}",
                         f"{((impact_per_tree - baseline_impact_per_tree) / baseline_impact_per_tree * 100):.1f}%")
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Base Amount": base_amounts,
                    "Optimized Amount": optimized_amounts,
                    "Change (%)": ((optimized_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                st.session_state.history.append({"scenario": f"Single Impact - {selected_impact}", "results": df_materials})

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

    with st.expander("📈 View Optimization History"):
        if st.session_state['history']:
            for i, record in enumerate(st.session_state['history'], 1):
                st.write(f"**Run {i}: {record['scenario']}**")
                st.dataframe(record['results'])
        else:
            st.info("No optimization runs recorded yet.")
else:
    st.info("Upload a valid Excel file to begin.")
