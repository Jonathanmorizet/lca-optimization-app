import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
import random
import subprocess
import sys
from io import BytesIO

# Ensure DEAP is installed
try:
    from deap import base, creator, tools
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "deap"])
    from deap import base, creator, tools

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

# DEAP Evaluation Functions - Fixed Production, Variable Efficiency
def evaluate_cost_gwp_fixed_production(ind, costs, impact_matrix, impact_cols, base_amounts, 
                                       fixed_trees, efficiency_materials_mask, max_efficiency_deviation):
    """
    Optimize cost vs GWP with FIXED tree production.
    Only efficiency materials can vary (±10-20%).
    Scale materials stay at baseline to maintain fixed production.
    
    Individual encoding: [efficiency_factor_1, efficiency_factor_2, ...]
    """
    efficiency_factors = np.array(ind)
    
    # Calculate actual amounts - only efficiency materials change
    final_amounts = np.copy(base_amounts)
    efficiency_idx = 0
    for i in range(len(base_amounts)):
        if efficiency_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
            efficiency_idx += 1
    
    # Penalties for violating bounds
    penalty = 0
    for ef in efficiency_factors:
        if ef < (1 - max_efficiency_deviation) or ef > (1 + max_efficiency_deviation):
            penalty += 10000 * abs(ef - np.clip(ef, 1 - max_efficiency_deviation, 1 + max_efficiency_deviation))
    
    # Calculate metrics
    total_cost = np.dot(final_amounts, costs)
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
    except ValueError:
        pass
    
    return total_cost + penalty, gwp + penalty

def evaluate_cost_only_fixed_production(ind, costs, base_amounts, fixed_trees,
                                        efficiency_materials_mask, max_efficiency_deviation):
    """Optimize cost with fixed production."""
    efficiency_factors = np.array(ind)
    
    final_amounts = np.copy(base_amounts)
    efficiency_idx = 0
    for i in range(len(base_amounts)):
        if efficiency_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
            efficiency_idx += 1
    
    penalty = 0
    for ef in efficiency_factors:
        if ef < (1 - max_efficiency_deviation) or ef > (1 + max_efficiency_deviation):
            penalty += 10000 * abs(ef - np.clip(ef, 1 - max_efficiency_deviation, 1 + max_efficiency_deviation))
    
    total_cost = np.dot(final_amounts, costs)
    return (total_cost + penalty,)

def evaluate_single_impact_fixed_production(ind, matrix, colname, cols, base_amounts, fixed_trees,
                                            efficiency_materials_mask, max_efficiency_deviation):
    """Optimize single impact with fixed production."""
    efficiency_factors = np.array(ind)
    
    final_amounts = np.copy(base_amounts)
    efficiency_idx = 0
    for i in range(len(base_amounts)):
        if efficiency_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
            efficiency_idx += 1
    
    penalty = 0
    for ef in efficiency_factors:
        if ef < (1 - max_efficiency_deviation) or ef > (1 + max_efficiency_deviation):
            penalty += 10000 * abs(ef - np.clip(ef, 1 - max_efficiency_deviation, 1 + max_efficiency_deviation))
    
    try:
        idx = cols.index(colname)
        impact = np.dot(final_amounts, matrix[:, idx])
    except ValueError:
        impact = 0.0
    
    return (impact + penalty,)

# Optimization Logic
def run_nsga2_fixed_production(popsize, ngen, cxpb, mutpb, costs, matrix, impact_cols, base_amounts, 
                               fixed_trees, efficiency_mask, max_eff_dev):
    """Run NSGA-II with fixed production - optimize efficiency only."""
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    num_efficiency_materials = np.sum(efficiency_mask)
    
    toolbox = base.Toolbox()
    def create_individual():
        # Only efficiency factors - no production scale
        ind = []
        for _ in range(num_efficiency_materials):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp_fixed_production, 
                    costs=costs, impact_matrix=matrix, impact_cols=impact_cols, 
                    base_amounts=base_amounts, fixed_trees=fixed_trees,
                    efficiency_materials_mask=efficiency_mask, max_efficiency_deviation=max_eff_dev)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated = list(ind)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                mutated[i] = np.clip(mutated[i], 1 - max_eff_dev, 1 + max_eff_dev)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)
    
    # Initial evaluation
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    # Custom NSGA-II evolution loop
    for gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        
        # Apply crossover and mutation
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb and i < len(offspring):
                child1, child2 = toolbox.mate(offspring[i-1], offspring[i])
                offspring[i-1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        
        # Evaluate all offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Combine and select
        pop = toolbox.select(pop + offspring, popsize)
    
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single_fixed_production(obj_func, popsize, ngen, cxpb, mutpb, base_amounts, fixed_trees, 
                                efficiency_mask, max_eff_dev, *args):
    """Run single-objective optimization with fixed production."""
    try:
        del creator.FitnessMin
        del creator.Individual
    except:
        pass
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    num_efficiency_materials = np.sum(efficiency_mask)
    
    toolbox = base.Toolbox()
    def create_individual():
        ind = []
        for _ in range(num_efficiency_materials):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_func, *args, base_amounts=base_amounts, fixed_trees=fixed_trees,
                    efficiency_materials_mask=efficiency_mask, max_efficiency_deviation=max_eff_dev)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated = list(ind)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                mutated[i] = np.clip(mutated[i], 1 - max_eff_dev, 1 + max_eff_dev)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popsize)
    
    # Initial evaluation
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)
    
    # Custom evolution loop
    for gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        
        # Apply crossover and mutation
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb and i < len(offspring):
                child1, child2 = toolbox.mate(offspring[i-1], offspring[i])
                offspring[i-1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        
        # Evaluate all offspring
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        
        # Replace population
        pop[:] = offspring
        hof.update(pop)
    
    return hof[0]

# Load data
merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols = load_data(uploaded_file)

if merged_df is not None:
    st.success("File uploaded and data processed successfully!")
    st.dataframe(merged_df)

    # Material Classification - Simplified for Fixed Production
    st.sidebar.markdown("### 🔧 Material Classification")
    st.sidebar.info("**Fixed Production Mode:** Tree production is locked at baseline. Only operational materials (fertilizers, pesticides, fuel) can be optimized for efficiency.")
    
    # Auto-detect scale materials (transplants, seedlings) - these will be FIXED
    default_fixed = []
    scale_keywords = ['transplant', 'seedling', 'plant', 'tree']
    for i, mat in enumerate(materials):
        if any(keyword in mat.lower() for keyword in scale_keywords):
            default_fixed.append(i)
    
    with st.sidebar.expander("Customize Optimizable Materials", expanded=False):
        st.markdown("**Efficiency Materials** can be optimized (fertilizers, pesticides, fuel, etc.)")
        st.markdown("**Fixed Materials** stay at baseline (transplants, infrastructure, etc.)")
        
        efficiency_material_indices = st.multiselect(
            "Select materials that CAN be optimized",
            options=[i for i in range(len(materials)) if i not in default_fixed],
            default=[i for i in range(len(materials)) if i not in default_fixed],
            format_func=lambda x: f"{materials[x]}"
        )
    
    # Create masks
    efficiency_materials_mask = np.zeros(len(materials), dtype=bool)
    efficiency_materials_mask[efficiency_material_indices] = True
    
    fixed_materials_mask = ~efficiency_materials_mask
    
    # Display material status
    st.sidebar.markdown("**Material Status:**")
    st.sidebar.markdown(f"⚡ **Optimizable:** {sum(efficiency_materials_mask)} materials")
    st.sidebar.markdown(f"🔒 **Fixed:** {sum(fixed_materials_mask)} materials")

    # Production Configuration
    st.sidebar.markdown("### 🌳 Production Settings")
    fixed_trees = st.sidebar.number_input("Fixed Tree Production", min_value=1, value=1900, step=1, 
                                          help="Production is LOCKED at this value. Optimization focuses on efficiency.")
    st.sidebar.success(f"✓ Producing exactly {fixed_trees} trees")
    
    st.sidebar.markdown("### 📊 Optimization Bounds")
    max_efficiency_deviation_pct = st.sidebar.slider("Material Efficiency ±%", 5, 30, 20, 
                                                      help="How much can optimizable materials vary?")
    max_efficiency_deviation = max_efficiency_deviation_pct / 100.0

    # Sidebar controls
    scenario = st.sidebar.selectbox("Optimization Scenario", [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Single Impact",
        "Optimize Cost Only"
    ])
    
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
    baseline_cost_per_tree = baseline_cost / fixed_trees
    
    baseline_gwp = 0.0
    if "kg CO2-Eq/Unit" in traci_impact_cols:
        gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
        baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
        baseline_gwp_per_tree = baseline_gwp / fixed_trees
        
        st.info(f"📌 **Baseline ({fixed_trees} trees):** ${baseline_cost:.2f} total (${baseline_cost_per_tree:.2f}/tree) | {baseline_gwp:.2f} kg CO2-Eq total ({baseline_gwp_per_tree:.2f} kg/tree)")

    if st.button("Run Optimization"):
        st.subheader(f"Running: {scenario}")

        if scenario == "Optimize Cost vs GWP (Tradeoff)":
            with st.spinner("Running NSGA-II optimization..."):
                pareto = run_nsga2_fixed_production(popsize, ngen, cxpb, mutpb, costs, impact_matrix, 
                                                    traci_impact_cols, base_amounts, fixed_trees, 
                                                    efficiency_materials_mask, max_efficiency_deviation)
                
                results = []
                for ind in pareto:
                    efficiency_factors = np.array(ind)
                    
                    final_amounts = np.copy(base_amounts)
                    efficiency_idx = 0
                    for i in range(len(base_amounts)):
                        if efficiency_materials_mask[i]:
                            final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
                            efficiency_idx += 1
                    
                    total_cost = np.dot(final_amounts, costs)
                    
                    gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
                    total_gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                    
                    cost_per_tree = total_cost / fixed_trees
                    gwp_per_tree = total_gwp / fixed_trees
                    
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
                
                ax1.scatter(df_pareto["Total Cost"], df_pareto["Total GWP"], alpha=0.6, s=50)
                ax1.scatter([baseline_cost], [baseline_gwp], color='red', s=150, marker='*', label='Baseline', zorder=5)
                ax1.set_xlabel("Total Cost ($)")
                ax1.set_ylabel("Total GWP (kg CO2-Eq)")
                ax1.set_title(f"Pareto Front: Total Values ({fixed_trees} trees)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                ax2.scatter(df_pareto["Cost/Tree"], df_pareto["GWP/Tree"], alpha=0.6, s=50)
                ax2.scatter([baseline_cost_per_tree], [baseline_gwp_per_tree], color='red', s=150, marker='*', label='Baseline', zorder=5)
                ax2.set_xlabel("Cost per Tree ($)")
                ax2.set_ylabel("GWP per Tree (kg CO2-Eq)")
                ax2.set_title(f"Pareto Front: Per-Tree Values ({fixed_trees} trees)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Allow user to select a solution
                st.markdown("### Select a Solution from Pareto Front")
                solution_idx = st.selectbox("Solution Index", range(len(pareto)))
                
                selected_ind = pareto[solution_idx]
                efficiency_factors = np.array(selected_ind)
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                efficiency_idx = 0
                for i in range(len(base_amounts)):
                    if efficiency_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
                        material_type.append("OPTIMIZED")
                        efficiency_idx += 1
                    else:
                        material_type.append("FIXED")
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Type": material_type,
                    "Base Amount": base_amounts,
                    "Optimized Amount": final_amounts,
                    "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                
                # Summary
                total_cost_opt = np.dot(final_amounts, costs)
                total_gwp_opt = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                cost_savings = baseline_cost - total_cost_opt
                gwp_reduction = baseline_gwp - total_gwp_opt
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Cost", f"${total_cost_opt:.2f}", f"-${cost_savings:.2f}")
                with col2:
                    st.metric("Total GWP", f"{total_gwp_opt:.2f} kg", f"-{gwp_reduction:.2f} kg")
                with col3:
                    st.metric("Trees Produced", f"{fixed_trees}", "Fixed ✓")
                
                st.success(f"**Per-Tree Metrics:** ${total_cost_opt/fixed_trees:.2f}/tree | {total_gwp_opt/fixed_trees:.2f} kg CO2-Eq/tree")
                
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                st.session_state.history.append({"scenario": scenario, "results": df_materials, "pareto": df_pareto})

        elif scenario == "Optimize Cost Only":
            with st.spinner("Running single-objective optimization..."):
                best = run_single_fixed_production(evaluate_cost_only_fixed_production, popsize, ngen, cxpb, mutpb, 
                                                   base_amounts, fixed_trees, efficiency_materials_mask, 
                                                   max_efficiency_deviation, costs)
                
                efficiency_factors = np.array(best)
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                efficiency_idx = 0
                for i in range(len(base_amounts)):
                    if efficiency_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
                        material_type.append("OPTIMIZED")
                        efficiency_idx += 1
                    else:
                        material_type.append("FIXED")
                
                total_cost = np.dot(final_amounts, costs)
                cost_per_tree = total_cost / fixed_trees
                cost_savings = baseline_cost - total_cost
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Cost", f"${total_cost:.2f}", f"-${cost_savings:.2f}")
                with col2:
                    st.metric("Cost per Tree", f"${cost_per_tree:.2f}", f"-${(baseline_cost_per_tree - cost_per_tree):.2f}")
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Type": material_type,
                    "Base Amount": base_amounts,
                    "Optimized Amount": final_amounts,
                    "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                st.session_state.history.append({"scenario": scenario, "results": df_materials})

        elif scenario == "Optimize Single Impact" and selected_impact:
            with st.spinner(f"Optimizing {selected_impact}..."):
                best = run_single_fixed_production(evaluate_single_impact_fixed_production, popsize, ngen, cxpb, mutpb, 
                                                   base_amounts, fixed_trees, efficiency_materials_mask,
                                                   max_efficiency_deviation, 
                                                   impact_matrix, selected_impact, traci_impact_cols)
                
                efficiency_factors = np.array(best)
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                efficiency_idx = 0
                for i in range(len(base_amounts)):
                    if efficiency_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * efficiency_factors[efficiency_idx]
                        material_type.append("OPTIMIZED")
                        efficiency_idx += 1
                    else:
                        material_type.append("FIXED")
                
                impact_idx = traci_impact_cols.index(selected_impact)
                total_impact = np.dot(final_amounts, impact_matrix[:, impact_idx])
                impact_per_tree = total_impact / fixed_trees
                
                baseline_impact = np.dot(base_amounts, impact_matrix[:, impact_idx])
                baseline_impact_per_tree = baseline_impact / fixed_trees
                impact_reduction = baseline_impact - total_impact
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(f"Total {selected_impact}", f"{total_impact:.4f}", f"-{impact_reduction:.4f}")
                with col2:
                    st.metric(f"{selected_impact} per Tree", f"{impact_per_tree:.4f}",
                             f"-{(baseline_impact_per_tree - impact_per_tree):.4f}")
                
                df_materials = pd.DataFrame({
                    "Material": materials,
                    "Type": material_type,
                    "Base Amount": base_amounts,
                    "Optimized Amount": final_amounts,
                    "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                })
                
                st.dataframe(df_materials)
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
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
