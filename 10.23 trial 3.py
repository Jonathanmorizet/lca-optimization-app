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

# File uploader - ALLOW MULTIPLE FILES
uploaded_files = st.file_uploader(
    "Upload DEAP NSGA Readable Excel files (upload multiple farms/strategies to compare)", 
    type=["xlsm", "xlsx"],
    accept_multiple_files=True
)

@st.cache_data
def load_multiple_farms(uploaded_files):
    """Load data from multiple farm Excel files."""
    farms_data = []
    
    if not uploaded_files or len(uploaded_files) == 0:
        return None
    
    for idx, uploaded_file in enumerate(uploaded_files):
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
            
            # Calculate baseline metrics
            total_cost = np.dot(base_amounts, costs)
            gwp = 0.0
            if "kg CO2-Eq/Unit" in impact_columns:
                gwp_idx = impact_columns.index("kg CO2-Eq/Unit")
                gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
            
            farms_data.append({
                'name': uploaded_file.name,
                'merged_df': merged_df,
                'materials': materials,
                'base_amounts': base_amounts,
                'costs': costs,
                'impact_matrix': impact_matrix,
                'impact_columns': impact_columns,
                'total_cost': total_cost,
                'total_gwp': gwp
            })
            
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            continue
    
    return farms_data if len(farms_data) > 0 else None

# DEAP Evaluation Functions with Production Scale and Operational Efficiency
def evaluate_cost_gwp_constrained(ind, costs, impact_matrix, impact_cols, base_amounts, 
                                  baseline_trees, scale_materials_mask, efficiency_materials_mask, 
                                  max_scale_deviation, max_efficiency_deviation):
    """
    Optimize cost vs GWP where:
    - Scale materials (transplants) vary proportionally with production volume
    - Efficiency materials (fertilizer, pesticides) vary independently (±10-20%)
    
    Individual encoding: [production_scale_factor, efficiency_factor_1, efficiency_factor_2, ...]
    """
    production_scale = ind[0]  # First gene: production scale (0.9 to 1.1 for ±10%)
    efficiency_factors = np.array(ind[1:])  # Rest: efficiency factors for each material
    
    # Calculate actual amounts
    final_amounts = np.copy(base_amounts)
    
    # Apply production scale to scale materials
    for i in range(len(base_amounts)):
        if scale_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * production_scale
        elif efficiency_materials_mask[i]:
            # Get the efficiency index for this material
            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
    
    # Penalties for violating bounds
    penalty = 0
    if production_scale < (1 - max_scale_deviation) or production_scale > (1 + max_scale_deviation):
        penalty += 10000 * abs(production_scale - np.clip(production_scale, 1 - max_scale_deviation, 1 + max_scale_deviation))
    
    for ef in efficiency_factors:
        if ef < (1 - max_efficiency_deviation) or ef > (1 + max_efficiency_deviation):
            penalty += 10000 * abs(ef - np.clip(ef, 1 - max_efficiency_deviation, 1 + max_efficiency_deviation))
    
    # Calculate metrics
    total_cost = np.dot(final_amounts, costs)
    actual_trees = baseline_trees * production_scale
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
    except ValueError:
        pass
    
    return total_cost + penalty, gwp + penalty

def evaluate_budget_constrained(ind, costs, impact_matrix, impact_cols, base_amounts, 
                                baseline_trees, scale_materials_mask, efficiency_materials_mask,
                                max_scale_deviation, max_efficiency_deviation, budget_limit):
    """
    Maximize trees while minimizing GWP, subject to budget constraint.
    Returns: (-trees, gwp) to maximize trees and minimize gwp
    """
    production_scale = float(ind[0])
    efficiency_factors = np.array(ind[1:], dtype=float)
    
    # Ensure base_amounts is 1D
    base_amounts_flat = np.asarray(base_amounts).flatten()
    final_amounts = np.copy(base_amounts_flat).astype(float)
    
    # Count efficiency materials seen so far
    efficiency_count = 0
    for i in range(len(final_amounts)):
        if scale_materials_mask[i]:
            final_amounts[i] = float(base_amounts_flat[i] * production_scale)
        elif efficiency_materials_mask[i]:
            if efficiency_count < len(efficiency_factors):
                final_amounts[i] = float(base_amounts_flat[i] * production_scale * efficiency_factors[efficiency_count])
                efficiency_count += 1
    
    # Ensure costs is flat and matches length
    costs_flat = np.asarray(costs).flatten()
    if len(costs_flat) != len(final_amounts):
        costs_flat = costs_flat[:len(final_amounts)]
    
    # Calculate cost with proper scalar extraction
    total_cost_result = np.dot(final_amounts, costs_flat)
    total_cost = float(total_cost_result.item() if hasattr(total_cost_result, 'item') else total_cost_result)
    
    actual_trees = float(baseline_trees * production_scale)
    
    # Heavy penalty for exceeding budget
    penalty = 0.0
    if total_cost > budget_limit:
        penalty += 100000.0 * float(total_cost - budget_limit)
    
    # Production scale penalty
    if production_scale < (1 - max_scale_deviation):
        penalty += 10000.0 * float(abs(production_scale - (1 - max_scale_deviation)))
    elif production_scale > (1 + max_scale_deviation):
        penalty += 10000.0 * float(abs(production_scale - (1 + max_scale_deviation)))
    
    # Efficiency factors penalty
    for i in range(len(efficiency_factors)):
        ef = float(efficiency_factors[i])
        if ef < (1 - max_efficiency_deviation):
            penalty += 10000.0 * float(abs(ef - (1 - max_efficiency_deviation)))
        elif ef > (1 + max_efficiency_deviation):
            penalty += 10000.0 * float(abs(ef - (1 + max_efficiency_deviation)))
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        impact_col = impact_matrix[:, gwp_idx].flatten()
        # Ensure impact_col matches length
        if len(impact_col) != len(final_amounts):
            impact_col = impact_col[:len(final_amounts)]
        gwp_result = np.dot(final_amounts, impact_col)
        gwp = float(gwp_result.item() if hasattr(gwp_result, 'item') else gwp_result)
    except (ValueError, IndexError):
        gwp = 0.0
    
    # Return negative trees (to maximize) and GWP (to minimize)
    return float(-actual_trees + penalty), float(gwp + penalty)

def evaluate_compliance_constrained(ind, costs, impact_matrix, impact_cols, base_amounts,
                                   baseline_trees, scale_materials_mask, efficiency_materials_mask,
                                   max_scale_deviation, max_efficiency_deviation, gwp_target):
    """
    Minimize cost while meeting GWP reduction target.
    Returns: (cost,) with penalty for exceeding GWP target
    """
    production_scale = float(ind[0])
    efficiency_factors = np.array(ind[1:], dtype=float)
    
    # Ensure base_amounts is 1D
    base_amounts_flat = np.asarray(base_amounts).flatten()
    final_amounts = np.copy(base_amounts_flat).astype(float)
    
    # Count efficiency materials seen so far
    efficiency_count = 0
    for i in range(len(final_amounts)):
        if scale_materials_mask[i]:
            final_amounts[i] = float(base_amounts_flat[i] * production_scale)
        elif efficiency_materials_mask[i]:
            if efficiency_count < len(efficiency_factors):
                final_amounts[i] = float(base_amounts_flat[i] * production_scale * efficiency_factors[efficiency_count])
                efficiency_count += 1
    
    # Ensure costs is 1D and matches length
    costs_flat = np.asarray(costs).flatten()
    if len(costs_flat) != len(final_amounts):
        # Take only the first N elements to match
        costs_flat = costs_flat[:len(final_amounts)]
    
    # Calculate cost - use item() to extract scalar
    total_cost_result = np.dot(final_amounts, costs_flat)
    total_cost = float(total_cost_result.item() if hasattr(total_cost_result, 'item') else total_cost_result)
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        impact_col = impact_matrix[:, gwp_idx].flatten()
        # Ensure impact_col matches length
        if len(impact_col) != len(final_amounts):
            impact_col = impact_col[:len(final_amounts)]
        gwp_result = np.dot(final_amounts, impact_col)
        gwp = float(gwp_result.item() if hasattr(gwp_result, 'item') else gwp_result)
    except (ValueError, IndexError):
        gwp = 0.0
    
    # Heavy penalty for exceeding GWP target (must stay below target)
    penalty = 0.0
    if gwp > gwp_target:
        penalty += 1000000.0 * float(gwp - gwp_target)
    
    # Production scale penalty
    if production_scale < (1 - max_scale_deviation):
        penalty += 100000.0 * float(abs(production_scale - (1 - max_scale_deviation)))
    elif production_scale > (1 + max_scale_deviation):
        penalty += 100000.0 * float(abs(production_scale - (1 + max_scale_deviation)))
    
    # Efficiency factors penalty
    for i in range(len(efficiency_factors)):
        ef = float(efficiency_factors[i])
        if ef < (1 - max_efficiency_deviation):
            penalty += 100000.0 * float(abs(ef - (1 - max_efficiency_deviation)))
        elif ef > (1 + max_efficiency_deviation):
            penalty += 100000.0 * float(abs(ef - (1 + max_efficiency_deviation)))
    
    return (float(total_cost + penalty),)

def evaluate_cost_only_constrained(ind, costs, base_amounts, baseline_trees, 
                                   scale_materials_mask, efficiency_materials_mask,
                                   max_scale_deviation, max_efficiency_deviation):
    """Optimize cost with production scale and efficiency."""
    production_scale = ind[0]
    efficiency_factors = np.array(ind[1:])
    
    final_amounts = np.copy(base_amounts)
    for i in range(len(base_amounts)):
        if scale_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * production_scale
        elif efficiency_materials_mask[i]:
            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
    
    penalty = 0
    if production_scale < (1 - max_scale_deviation) or production_scale > (1 + max_scale_deviation):
        penalty += 10000 * abs(production_scale - np.clip(production_scale, 1 - max_scale_deviation, 1 + max_scale_deviation))
    
    for ef in efficiency_factors:
        if ef < (1 - max_efficiency_deviation) or ef > (1 + max_efficiency_deviation):
            penalty += 10000 * abs(ef - np.clip(ef, 1 - max_efficiency_deviation, 1 + max_efficiency_deviation))
    
    total_cost = np.dot(final_amounts, costs)
    return (total_cost + penalty,)

def evaluate_single_impact_constrained(ind, matrix, colname, cols, base_amounts, baseline_trees,
                                       scale_materials_mask, efficiency_materials_mask,
                                       max_scale_deviation, max_efficiency_deviation):
    """Optimize single impact with production scale and efficiency."""
    production_scale = ind[0]
    efficiency_factors = np.array(ind[1:])
    
    final_amounts = np.copy(base_amounts)
    for i in range(len(base_amounts)):
        if scale_materials_mask[i]:
            final_amounts[i] = base_amounts[i] * production_scale
        elif efficiency_materials_mask[i]:
            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
    
    penalty = 0
    if production_scale < (1 - max_scale_deviation) or production_scale > (1 + max_scale_deviation):
        penalty += 10000 * abs(production_scale - np.clip(production_scale, 1 - max_scale_deviation, 1 + max_scale_deviation))
    
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
def run_nsga2_constrained(popsize, ngen, cxpb, mutpb, costs, matrix, impact_cols, base_amounts, 
                         baseline_trees, scale_mask, efficiency_mask, max_scale_dev, max_eff_dev,
                         eval_func=None, **eval_kwargs):
    """Run NSGA-II with production scale and efficiency optimization."""
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
        # [production_scale, efficiency_1, efficiency_2, ...]
        ind = [random.uniform(1 - max_scale_dev, 1 + max_scale_dev)]
        for _ in range(num_efficiency_materials):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    # Use custom eval function if provided, otherwise use default
    if eval_func is None:
        eval_func = evaluate_cost_gwp_constrained
    
    toolbox.register("evaluate", eval_func, 
                    costs=costs, impact_matrix=matrix, impact_cols=impact_cols, 
                    base_amounts=base_amounts, baseline_trees=baseline_trees,
                    scale_materials_mask=scale_mask, efficiency_materials_mask=efficiency_mask,
                    max_scale_deviation=max_scale_dev, max_efficiency_deviation=max_eff_dev,
                    **eval_kwargs)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated = list(ind)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                if i == 0:  # Production scale
                    mutated[i] = np.clip(mutated[i], 1 - max_scale_dev, 1 + max_scale_dev)
                else:  # Efficiency factors
                    mutated[i] = np.clip(mutated[i], 1 - max_eff_dev, 1 + max_eff_dev)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)
    
    # Initial evaluation with error handling
    try:
        for ind in pop:
            fitness_result = toolbox.evaluate(ind)
            if not isinstance(fitness_result, tuple) or len(fitness_result) != 2:
                st.error(f"Evaluation returned unexpected type: {type(fitness_result)}, value: {fitness_result}")
            ind.fitness.values = fitness_result
    except Exception as e:
        st.error(f"Error during initial evaluation: {str(e)}")
        st.error(f"Error type: {type(e).__name__}")
        raise

    # Custom NSGA-II evolution loop
    for gen in range(ngen):
        # Select the next generation individuals
        offspring = toolbox.select(pop, popsize)
        # Clone the selected individuals - this creates new individuals with fresh fitness
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
        
        # Evaluate all offspring with error handling
        try:
            for ind in offspring:
                fitness_result = toolbox.evaluate(ind)
                ind.fitness.values = fitness_result
        except Exception as e:
            st.error(f"Error during generation {gen} evaluation: {str(e)}")
            st.error(f"Error type: {type(e).__name__}")
            st.error(f"Individual causing error: {ind}")
            raise
        
        # Combine parent and offspring populations and select best
        pop = toolbox.select(pop + offspring, popsize)
    
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single_constrained(obj_func, popsize, ngen, cxpb, mutpb, base_amounts, baseline_trees, 
                          scale_mask, efficiency_mask, max_scale_dev, max_eff_dev, *args, **kwargs):
    """Run single-objective optimization."""
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
        ind = [random.uniform(1 - max_scale_dev, 1 + max_scale_dev)]
        for _ in range(num_efficiency_materials):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", obj_func, *args, 
                    base_amounts=base_amounts, baseline_trees=baseline_trees,
                    scale_materials_mask=scale_mask, efficiency_materials_mask=efficiency_mask,
                    max_scale_deviation=max_scale_dev, max_efficiency_deviation=max_eff_dev,
                    **kwargs)
    toolbox.register("mate", tools.cxBlend, alpha=0.3)

    def bounded_mutate(ind, mu, sigma, indpb):
        mutated = list(ind)
        for i in range(len(mutated)):
            if random.random() < indpb:
                mutated[i] += random.gauss(mu, sigma)
                if i == 0:
                    mutated[i] = np.clip(mutated[i], 1 - max_scale_dev, 1 + max_scale_dev)
                else:
                    mutated[i] = np.clip(mutated[i], 1 - max_eff_dev, 1 + max_eff_dev)
        return mutated,

    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.05, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popsize)
    
    # Initial evaluation with error handling
    for ind in pop:
        try:
            fitness_result = toolbox.evaluate(ind)
            if not isinstance(fitness_result, tuple):
                fitness_result = (fitness_result,)
            # Check for invalid values
            if any(not np.isfinite(v) for v in fitness_result):
                ind.fitness.values = (1e10,)  # Assign very bad fitness
            else:
                ind.fitness.values = fitness_result
        except Exception as e:
            import traceback
            st.error(f"Error evaluating individual: {e}")
            st.error(f"Individual: {ind}")
            st.error(f"Individual types: {[type(x) for x in ind]}")
            st.error(f"Traceback:\n{traceback.format_exc()}")
            ind.fitness.values = (1e10,)  # Assign very bad fitness

    hof = tools.HallOfFame(1)
    
    # Custom evolution loop
    for gen in range(ngen):
        # Select the next generation individuals
        offspring = toolbox.select(pop, popsize)
        # Clone the selected individuals - creates new individuals with fresh fitness
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
        
        # Evaluate all offspring with error handling
        for ind in offspring:
            try:
                fitness_result = toolbox.evaluate(ind)
                if not isinstance(fitness_result, tuple):
                    fitness_result = (fitness_result,)
                # Check for invalid values
                if any(not np.isfinite(v) for v in fitness_result):
                    ind.fitness.values = (1e10,)
                else:
                    ind.fitness.values = fitness_result
            except Exception as e:
                import traceback
                st.error(f"Error evaluating individual in gen {gen}: {e}")
                st.error(f"Individual: {ind}")
                st.error(f"Individual types: {[type(x) for x in ind]}")
                st.error(f"Traceback:\n{traceback.format_exc()}")
                ind.fitness.values = (1e10,)
        
        # Replace population
        pop[:] = offspring
        hof.update(pop)
    
    return hof[0]

# Load data
farms_data = load_multiple_farms(uploaded_files)

if farms_data is not None and len(farms_data) > 0:
    st.success(f"✅ Loaded {len(farms_data)} farm management strategies!")
    
    # Display comparison table
    st.markdown("## 📊 Farm Strategy Comparison")
    
    comparison_data = []
    for farm in farms_data:
        comparison_data.append({
            'Strategy': farm['name'],
            'Total Cost ($)': f"${farm['total_cost']:.2f}",
            'Total GWP (kg CO2-Eq)': f"{farm['total_gwp']:.2f}",
            'Number of Materials': len(farm['materials'])
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    st.dataframe(df_comparison, use_container_width=True)
    
    # Visualize farms on Cost vs GWP plot
    st.markdown("### 🎯 Cost vs Environmental Impact")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for idx, farm in enumerate(farms_data):
        ax.scatter(farm['total_cost'], farm['total_gwp'], s=200, alpha=0.7, 
                  label=farm['name'], marker='o')
        ax.annotate(f"Farm {idx+1}", (farm['total_cost'], farm['total_gwp']),
                   xytext=(10, 10), textcoords='offset points', fontsize=9)
    
    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('Total GWP (kg CO2-Eq)', fontsize=12)
    ax.set_title('Farm Management Strategies: Cost vs Environmental Impact', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    
    # Identify Pareto-optimal farms
    st.markdown("### 🏆 Pareto Analysis")
    
    # Simple Pareto dominance check
    pareto_farms = []
    for i, farm_i in enumerate(farms_data):
        is_dominated = False
        for j, farm_j in enumerate(farms_data):
            if i != j:
                # farm_j dominates farm_i if it's better in both objectives
                if (farm_j['total_cost'] <= farm_i['total_cost'] and 
                    farm_j['total_gwp'] < farm_i['total_gwp']) or \
                   (farm_j['total_cost'] < farm_i['total_cost'] and 
                    farm_j['total_gwp'] <= farm_i['total_gwp']):
                    is_dominated = True
                    break
        if not is_dominated:
            pareto_farms.append(farm_i)
    
    if len(pareto_farms) == len(farms_data):
        st.info("✨ All strategies are Pareto-optimal! No single strategy dominates all others.")
    else:
        st.success(f"🎯 {len(pareto_farms)} out of {len(farms_data)} strategies are Pareto-optimal:")
        for farm in pareto_farms:
            st.markdown(f"- **{farm['name']}** - ${farm['total_cost']:.2f}, {farm['total_gwp']:.2f} kg CO2-Eq")
    
    # Select farm for detailed analysis
    st.markdown("---")
    st.markdown("## 🔍 Detailed Strategy Analysis")
    
    selected_farm_name = st.selectbox(
        "Select a farm strategy to analyze in detail:",
        [farm['name'] for farm in farms_data]
    )
    
    selected_farm = next(farm for farm in farms_data if farm['name'] == selected_farm_name)
    
    # Use selected farm's data for the rest of the analysis
    merged_df = selected_farm['merged_df']
    materials = selected_farm['materials']
    base_amounts = selected_farm['base_amounts']
    costs = selected_farm['costs']
    impact_matrix = selected_farm['impact_matrix']
    traci_impact_cols = selected_farm['impact_columns']
    
    st.dataframe(merged_df)
    
    # Show material breakdown
    st.markdown("### 💰 Cost Breakdown (Top 10 Materials)")
    material_costs = base_amounts * costs
    cost_breakdown = pd.DataFrame({
        'Material': materials,
        'Cost': material_costs
    }).sort_values('Cost', ascending=False).head(10)
    
    fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
    ax_cost.barh(cost_breakdown['Material'], cost_breakdown['Cost'])
    ax_cost.set_xlabel('Cost ($)')
    ax_cost.set_title(f'Top 10 Cost Contributors - {selected_farm_name}')
    plt.tight_layout()
    st.pyplot(fig_cost)
    
    if "kg CO2-Eq/Unit" in traci_impact_cols:
        st.markdown("### 🌍 GWP Breakdown (Top 10 Materials)")
        gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
        material_gwp = base_amounts * impact_matrix[:, gwp_idx]
        gwp_breakdown = pd.DataFrame({
            'Material': materials,
            'GWP': material_gwp
        }).sort_values('GWP', ascending=False).head(10)
        
        fig_gwp, ax_gwp = plt.subplots(figsize=(10, 6))
        ax_gwp.barh(gwp_breakdown['Material'], gwp_breakdown['GWP'], color='green', alpha=0.7)
        ax_gwp.set_xlabel('GWP (kg CO2-Eq)')
        ax_gwp.set_title(f'Top 10 GWP Contributors - {selected_farm_name}')
        plt.tight_layout()
        st.pyplot(fig_gwp)

    # Material Classification
    st.sidebar.markdown("### 🔧 Material Classification")
    
    # Auto-detect scale materials (transplants, seedlings)
    default_scale = []
    scale_keywords = ['transplant', 'seedling', 'plant', 'tree']
    for i, mat in enumerate(materials):
        if any(keyword in mat.lower() for keyword in scale_keywords):
            default_scale.append(i)
    
    with st.sidebar.expander("Customize Material Types", expanded=False):
        st.markdown("**Scale Materials** scale proportionally with production (e.g., transplants)")
        scale_material_indices = st.multiselect(
            "Select SCALE materials",
            options=list(range(len(materials))),
            default=default_scale,
            format_func=lambda x: f"{materials[x]}"
        )
        
        st.markdown("**Efficiency Materials** can be optimized independently (e.g., fertilizer, pesticides)")
        efficiency_material_indices = st.multiselect(
            "Select EFFICIENCY materials",
            options=[i for i in range(len(materials)) if i not in scale_material_indices],
            default=[i for i in range(len(materials)) if i not in scale_material_indices],
            format_func=lambda x: f"{materials[x]}"
        )
    
    # Create masks
    scale_materials_mask = np.zeros(len(materials), dtype=bool)
    scale_materials_mask[scale_material_indices] = True
    
    efficiency_materials_mask = np.zeros(len(materials), dtype=bool)
    efficiency_materials_mask[efficiency_material_indices] = True
    
    # Display material status
    st.sidebar.markdown("**Material Status:**")
    st.sidebar.markdown(f"📏 **Scale:** {sum(scale_materials_mask)} materials (vary with production)")
    st.sidebar.markdown(f"⚡ **Efficiency:** {sum(efficiency_materials_mask)} materials (optimizable)")
    st.sidebar.markdown(f"🔒 **Fixed:** {len(materials) - sum(scale_materials_mask) - sum(efficiency_materials_mask)} materials")

    # Production Configuration
    st.sidebar.markdown("### 🌳 Production Settings")
    baseline_trees = st.sidebar.number_input("Baseline Trees", min_value=1, value=1900, step=1)
    
    st.sidebar.markdown("### 📊 Optimization Bounds")
    max_scale_deviation_pct = st.sidebar.slider("Production Scale ±%", 0, 30, 10)
    max_scale_deviation = max_scale_deviation_pct / 100.0
    
    max_efficiency_deviation_pct = st.sidebar.slider("Efficiency ±%", 5, 30, 20)
    max_efficiency_deviation = max_efficiency_deviation_pct / 100.0
    
    st.sidebar.info(f"Trees range: {int(baseline_trees * (1-max_scale_deviation))} to {int(baseline_trees * (1+max_scale_deviation))}")

    # Sidebar controls
    st.sidebar.markdown("### 🎛️ Analysis Mode")
    
    if len(farms_data) > 1:
        analysis_mode = st.sidebar.selectbox("Choose Analysis", [
            "Single Farm Optimization",
            "Multi-Farm Comparison & Hybrid"
        ])
    else:
        analysis_mode = "Single Farm Optimization"
    
    if analysis_mode == "Multi-Farm Comparison & Hybrid" and len(farms_data) > 1:
        st.markdown("---")
        st.markdown("## 🔀 Create Hybrid Strategy")
        st.markdown("Combine the best materials from different farms to find an optimal hybrid approach.")
        
        # Allow user to select which materials to take from which farm
        st.markdown("### Material Selection Strategy")
        
        strategy_type = st.radio(
            "How should we create the hybrid?",
            ["Take lowest cost material for each", 
             "Take lowest GWP material for each",
             "Optimize cost-GWP tradeoff for each material"]
        )
        
        if st.button("🚀 Generate Hybrid Strategy"):
            with st.spinner("Creating hybrid strategy..."):
                # Create hybrid by selecting best material source for each unique material
                unique_materials = set()
                for farm in farms_data:
                    unique_materials.update(farm['materials'])
                
                hybrid_materials = []
                hybrid_amounts = []
                hybrid_costs = []
                hybrid_gwp_values = []
                
                for material in unique_materials:
                    # Find this material in each farm
                    candidates = []
                    for farm in farms_data:
                        if material in farm['materials']:
                            idx = farm['materials'].index(material)
                            amount = farm['base_amounts'][idx]
                            cost = farm['costs'][idx]
                            
                            if "kg CO2-Eq/Unit" in farm['impact_columns']:
                                gwp_idx = farm['impact_columns'].index("kg CO2-Eq/Unit")
                                gwp_per_unit = farm['impact_matrix'][idx, gwp_idx]
                            else:
                                gwp_per_unit = 0
                            
                            candidates.append({
                                'farm': farm['name'],
                                'amount': amount,
                                'cost': cost,
                                'gwp_per_unit': gwp_per_unit,
                                'total_cost': amount * cost,
                                'total_gwp': amount * gwp_per_unit
                            })
                    
                    if candidates:
                        # Select best candidate based on strategy
                        if strategy_type == "Take lowest cost material for each":
                            best = min(candidates, key=lambda x: x['total_cost'])
                        elif strategy_type == "Take lowest GWP material for each":
                            best = min(candidates, key=lambda x: x['total_gwp'])
                        else:  # Optimize tradeoff
                            # Normalize and find best compromise
                            max_cost = max(c['total_cost'] for c in candidates)
                            max_gwp = max(c['total_gwp'] for c in candidates) if max(c['total_gwp'] for c in candidates) > 0 else 1
                            best = min(candidates, key=lambda x: (x['total_cost']/max_cost + x['total_gwp']/max_gwp))
                        
                        hybrid_materials.append(material)
                        hybrid_amounts.append(best['amount'])
                        hybrid_costs.append(best['cost'])
                        hybrid_gwp_values.append(best['total_gwp'])
                
                # Calculate hybrid totals
                hybrid_total_cost = sum(hybrid_amounts[i] * hybrid_costs[i] for i in range(len(hybrid_materials)))
                hybrid_total_gwp = sum(hybrid_gwp_values)
                
                st.success(f"🎉 Hybrid Strategy Created!")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Hybrid Total Cost", f"${hybrid_total_cost:.2f}")
                    
                    # Show savings vs each farm
                    for farm in farms_data:
                        savings = farm['total_cost'] - hybrid_total_cost
                        st.markdown(f"💰 vs {farm['name']}: ${savings:+.2f} ({savings/farm['total_cost']*100:+.1f}%)")
                
                with col2:
                    st.metric("Hybrid Total GWP", f"{hybrid_total_gwp:.2f} kg CO2-Eq")
                    
                    # Show reduction vs each farm
                    for farm in farms_data:
                        reduction = farm['total_gwp'] - hybrid_total_gwp
                        st.markdown(f"🌱 vs {farm['name']}: {reduction:+.2f} kg ({reduction/farm['total_gwp']*100:+.1f}%)")
                
                # Visualize hybrid vs original farms
                st.markdown("### 📈 Hybrid vs Original Strategies")
                fig_hybrid, ax_hybrid = plt.subplots(figsize=(12, 7))
                
                for idx, farm in enumerate(farms_data):
                    ax_hybrid.scatter(farm['total_cost'], farm['total_gwp'], s=200, alpha=0.7,
                                    label=farm['name'], marker='o')
                
                ax_hybrid.scatter(hybrid_total_cost, hybrid_total_gwp, s=300, alpha=0.9,
                                label='🔀 Hybrid Strategy', marker='*', color='red',
                                edgecolors='black', linewidths=2)
                
                ax_hybrid.set_xlabel('Total Cost ($)', fontsize=12)
                ax_hybrid.set_ylabel('Total GWP (kg CO2-Eq)', fontsize=12)
                ax_hybrid.set_title('Hybrid Strategy vs Original Farms', fontsize=14, fontweight='bold')
                ax_hybrid.legend()
                ax_hybrid.grid(True, alpha=0.3)
                st.pyplot(fig_hybrid)
        
        st.markdown("---")
    
    scenario = st.sidebar.selectbox("Optimization Scenario", [
        "Optimize Cost vs GWP (Tradeoff)",
        "Budget-Constrained: Maximize Trees, Minimize GWP",
        "Compliance: Meet GWP Target at Lowest Cost",
        "Optimize Single Impact",
        "Optimize Cost Only"
    ])
    
    st.sidebar.markdown("### 🧬 Genetic Algorithm Parameters")
    popsize = st.sidebar.slider("Population Size", 20, 200, 100)
    ngen = st.sidebar.slider("Generations", 20, 200, 50)
    cxpb = st.sidebar.slider("Crossover Probability", 0.0, 1.0, 0.7)
    mutpb = st.sidebar.slider("Mutation Probability", 0.0, 1.0, 0.3)

    # Scenario-specific inputs
    budget_limit = None
    gwp_reduction_pct = None
    selected_impact = None
    
    if scenario == "Budget-Constrained: Maximize Trees, Minimize GWP":
        baseline_cost = np.dot(base_amounts, costs)
        st.sidebar.markdown("### 💰 Budget Constraint")
        budget_limit = st.sidebar.number_input(
            "Budget Limit ($)", 
            min_value=float(baseline_cost * 0.5), 
            max_value=float(baseline_cost * 2.0),
            value=float(baseline_cost),
            step=100.0
        )
        st.sidebar.info(f"Baseline cost: ${baseline_cost:.2f}")
        
    elif scenario == "Compliance: Meet GWP Target at Lowest Cost":
        if "kg CO2-Eq/Unit" in traci_impact_cols:
            gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
            baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
            
            st.sidebar.markdown("### 🌍 GWP Reduction Target")
            gwp_reduction_pct = st.sidebar.slider(
                "GWP Reduction (%)",
                min_value=0,
                max_value=50,
                value=15,
                step=5
            )
            gwp_target = baseline_gwp * (1 - gwp_reduction_pct / 100.0)
            st.sidebar.info(f"Baseline GWP: {baseline_gwp:.2f} kg CO2-Eq")
            st.sidebar.info(f"Target GWP: {gwp_target:.2f} kg CO2-Eq")
        else:
            st.sidebar.error("GWP data not available in the dataset")
            
    elif scenario == "Optimize Single Impact":
        selected_impact = st.selectbox("Select TRACI Impact", traci_impact_cols)

    # Calculate baseline metrics
    baseline_cost = np.dot(base_amounts, costs)
    baseline_cost_per_tree = baseline_cost / baseline_trees
    
    baseline_gwp = 0.0
    if "kg CO2-Eq/Unit" in traci_impact_cols:
        gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
        baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
        baseline_gwp_per_tree = baseline_gwp / baseline_trees
        
        st.info(f"📌 **Baseline ({baseline_trees} trees):** ${baseline_cost:.2f} total (${baseline_cost_per_tree:.2f}/tree) | {baseline_gwp:.2f} kg CO2-Eq total ({baseline_gwp_per_tree:.2f} kg/tree)")

    if st.button("Run Optimization"):
        st.subheader(f"Running: {scenario}")

        if scenario == "Budget-Constrained: Maximize Trees, Minimize GWP":
            with st.spinner("Running budget-constrained optimization..."):
                pareto = run_nsga2_constrained(
                    popsize, ngen, cxpb, mutpb, costs, impact_matrix, 
                    traci_impact_cols, base_amounts, baseline_trees, 
                    scale_materials_mask, efficiency_materials_mask, 
                    max_scale_deviation, max_efficiency_deviation,
                    eval_func=evaluate_budget_constrained,
                    budget_limit=budget_limit
                )
                
                results = []
                for ind in pareto:
                    production_scale = ind[0]
                    efficiency_factors = np.array(ind[1:])
                    
                    final_amounts = np.copy(base_amounts)
                    for i in range(len(base_amounts)):
                        if scale_materials_mask[i]:
                            final_amounts[i] = base_amounts[i] * production_scale
                        elif efficiency_materials_mask[i]:
                            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                    
                    total_cost = np.dot(final_amounts, costs)
                    actual_trees = baseline_trees * production_scale
                    
                    gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
                    total_gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                    
                    # Only include feasible solutions (within budget)
                    if total_cost <= budget_limit:
                        cost_per_tree = total_cost / actual_trees
                        gwp_per_tree = total_gwp / actual_trees
                        
                        results.append({
                            "Trees": int(actual_trees),
                            "Total Cost": total_cost,
                            "Budget Used (%)": (total_cost / budget_limit * 100),
                            "Total GWP": total_gwp,
                            "Cost/Tree": cost_per_tree,
                            "GWP/Tree": gwp_per_tree,
                            "Production Scale": production_scale,
                            "Individual": ind
                        })
                
                if len(results) == 0:
                    st.error("No feasible solutions found within budget constraint. Try increasing the budget or relaxing constraints.")
                else:
                    df_pareto = pd.DataFrame(results)
                    df_pareto = df_pareto.sort_values("Trees", ascending=False)
                    st.dataframe(df_pareto[["Trees", "Total Cost", "Budget Used (%)", "Total GWP", "Cost/Tree", "GWP/Tree"]])
                    
                    # Highlight best solution
                    best_trees_idx = df_pareto["Trees"].idxmax()
                    st.success(f"🎯 **Best Solution:** {df_pareto.loc[best_trees_idx, 'Trees']} trees using ${df_pareto.loc[best_trees_idx, 'Total Cost']:.2f} ({df_pareto.loc[best_trees_idx, 'Budget Used (%)']:.1f}% of budget)")
                    
                    # Plot results
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    scatter1 = ax1.scatter(df_pareto["Total Cost"], df_pareto["Trees"], 
                                          alpha=0.6, c=df_pareto["Total GWP"], cmap='RdYlGn_r', s=100)
                    ax1.axvline(budget_limit, color='red', linestyle='--', label=f'Budget: ${budget_limit:.0f}')
                    ax1.scatter([baseline_cost], [baseline_trees], color='red', s=200, marker='*', 
                              label='Baseline', zorder=5, edgecolors='black', linewidths=2)
                    ax1.set_xlabel("Total Cost ($)")
                    ax1.set_ylabel("Number of Trees")
                    ax1.set_title("Budget-Constrained Solutions (color=GWP)")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    plt.colorbar(scatter1, ax=ax1, label='Total GWP (kg CO2-Eq)')
                    
                    scatter2 = ax2.scatter(df_pareto["Trees"], df_pareto["Total GWP"], 
                                          alpha=0.6, c=df_pareto["Total Cost"], cmap='viridis', s=100)
                    ax2.scatter([baseline_trees], [baseline_gwp], color='red', s=200, marker='*', 
                              label='Baseline', zorder=5, edgecolors='black', linewidths=2)
                    ax2.set_xlabel("Number of Trees")
                    ax2.set_ylabel("Total GWP (kg CO2-Eq)")
                    ax2.set_title("Trees vs Environmental Impact (color=cost)")
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    plt.colorbar(scatter2, ax=ax2, label='Total Cost ($)')
                    
                    st.pyplot(fig)
                    
                    # Allow user to select a solution
                    st.markdown("### 📋 Select a Solution to View Details")
                    solution_idx = st.selectbox("Solution (sorted by trees)", 
                                               range(len(df_pareto)), 
                                               format_func=lambda x: f"Solution {x+1}: {df_pareto.iloc[x]['Trees']} trees, ${df_pareto.iloc[x]['Total Cost']:.2f}, {df_pareto.iloc[x]['Total GWP']:.2f} kg CO2-Eq")
                    
                    selected_row = df_pareto.iloc[solution_idx]
                    selected_ind = selected_row["Individual"]
                    production_scale = selected_ind[0]
                    efficiency_factors = np.array(selected_ind[1:])
                    
                    final_amounts = np.copy(base_amounts)
                    material_type = []
                    for i in range(len(base_amounts)):
                        if scale_materials_mask[i]:
                            final_amounts[i] = base_amounts[i] * production_scale
                            material_type.append("SCALE")
                        elif efficiency_materials_mask[i]:
                            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                            material_type.append("EFFICIENCY")
                        else:
                            material_type.append("FIXED")
                    
                    df_materials = pd.DataFrame({
                        "Material": materials,
                        "Type": material_type,
                        "Base Amount": base_amounts,
                        "Optimized Amount": final_amounts,
                        "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100),
                        "Base Cost": base_amounts * costs,
                        "Optimized Cost": final_amounts * costs
                    })
                    
                    st.dataframe(df_materials)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Trees", f"{int(selected_row['Trees'])}", 
                                f"{((selected_row['Trees'] - baseline_trees) / baseline_trees * 100):+.1f}%")
                    with col2:
                        st.metric("Total Cost", f"${selected_row['Total Cost']:.2f}",
                                f"{selected_row['Budget Used (%)']:.1f}% of budget")
                    with col3:
                        st.metric("Total GWP", f"{selected_row['Total GWP']:.2f} kg CO2-Eq",
                                f"{((selected_row['Total GWP'] - baseline_gwp) / baseline_gwp * 100):+.1f}%")
                    
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []
                    st.session_state.history.append({
                        "scenario": f"Budget-Constrained (${budget_limit:.0f})", 
                        "results": df_materials, 
                        "pareto": df_pareto
                    })

        elif scenario == "Compliance: Meet GWP Target at Lowest Cost":
            if "kg CO2-Eq/Unit" not in traci_impact_cols:
                st.error("GWP data not available. Cannot run compliance optimization.")
            else:
                gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
                baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
                gwp_target = baseline_gwp * (1 - gwp_reduction_pct / 100.0)
                
                with st.spinner(f"Finding cheapest way to reduce GWP by {gwp_reduction_pct}%..."):
                    best = run_single_constrained(
                        evaluate_compliance_constrained, popsize, ngen, cxpb, mutpb, 
                        base_amounts, baseline_trees, scale_materials_mask, 
                        efficiency_materials_mask, max_scale_deviation, 
                        max_efficiency_deviation,
                        costs, impact_matrix, traci_impact_cols,
                        gwp_target=gwp_target
                    )
                    
                    production_scale = best[0]
                    efficiency_factors = np.array(best[1:])
                    
                    final_amounts = np.copy(base_amounts)
                    material_type = []
                    for i in range(len(base_amounts)):
                        if scale_materials_mask[i]:
                            final_amounts[i] = base_amounts[i] * production_scale
                            material_type.append("SCALE")
                        elif efficiency_materials_mask[i]:
                            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                            material_type.append("EFFICIENCY")
                        else:
                            material_type.append("FIXED")
                    
                    actual_trees = baseline_trees * production_scale
                    total_cost = np.dot(final_amounts, costs)
                    total_gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                    
                    # Create materials dataframe (needed for history regardless of success)
                    df_materials = pd.DataFrame({
                        "Material": materials,
                        "Type": material_type,
                        "Base Amount": base_amounts,
                        "Optimized Amount": final_amounts,
                        "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100),
                        "Cost Impact": (final_amounts - base_amounts) * costs,
                        "GWP Impact": (final_amounts - base_amounts) * impact_matrix[:, gwp_idx]
                    })
                    
                    # Check if target was met
                    if total_gwp <= gwp_target * 1.01:  # Allow 1% tolerance
                        st.success(f"✅ **Compliance achieved!** GWP reduced to {total_gwp:.2f} kg CO2-Eq (target: {gwp_target:.2f})")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Trees", f"{int(actual_trees)}", 
                                    f"{((actual_trees - baseline_trees) / baseline_trees * 100):+.1f}%")
                        with col2:
                            st.metric("Total Cost", f"${total_cost:.2f}",
                                    f"{((total_cost - baseline_cost) / baseline_cost * 100):+.1f}%")
                        with col3:
                            st.metric("Total GWP", f"{total_gwp:.2f} kg CO2-Eq",
                                    f"-{gwp_reduction_pct}%")
                        with col4:
                            st.metric("GWP/Tree", f"{total_gwp/actual_trees:.2f} kg",
                                    f"{((total_gwp/actual_trees - baseline_gwp_per_tree) / baseline_gwp_per_tree * 100):+.1f}%")
                        
                        # Show material changes (already created above)
                        # Highlight biggest contributors to reduction
                        df_materials = df_materials.sort_values("GWP Impact")
                        st.markdown("### 📊 Material Changes (sorted by GWP impact)")
                        st.dataframe(df_materials)
                        
                        # Show key insights
                        st.markdown("### 💡 Key Insights")
                        biggest_gwp_reducers = df_materials[df_materials["GWP Impact"] < 0].head(3)
                        if len(biggest_gwp_reducers) > 0:
                            st.markdown("**Top GWP Reductions:**")
                            for idx, row in biggest_gwp_reducers.iterrows():
                                st.markdown(f"- **{row['Material']}**: {row['Change (%)']:.1f}% reduction → {abs(row['GWP Impact']):.2f} kg CO2-Eq saved (Cost: ${row['Cost Impact']:+.2f})")
                        
                        cost_increase = total_cost - baseline_cost
                        gwp_decrease = baseline_gwp - total_gwp
                        cost_per_kg_co2 = cost_increase / gwp_decrease if gwp_decrease > 0 else 0
                        st.info(f"💰 **Cost of Compliance:** ${abs(cost_increase):.2f} {('increase' if cost_increase > 0 else 'savings')} = ${cost_per_kg_co2:.2f} per kg CO2-Eq reduced")
                        
                    else:
                        st.error(f"❌ **Target not fully met.** Achieved {total_gwp:.2f} kg CO2-Eq (target: {gwp_target:.2f}). Try relaxing constraints or increasing generations.")
                        
                        # Still show the best attempt
                        st.markdown("### 📊 Best Attempt - Material Changes")
                        df_materials = df_materials.sort_values("GWP Impact")
                        st.dataframe(df_materials)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Trees", f"{int(actual_trees)}")
                        with col2:
                            st.metric("Total Cost", f"${total_cost:.2f}")
                        with col3:
                            st.metric("Total GWP", f"{total_gwp:.2f} kg CO2-Eq",
                                    f"{((total_gwp - baseline_gwp) / baseline_gwp * 100):+.1f}%")
                    
                    if 'history' not in st.session_state:
                        st.session_state['history'] = []
                    st.session_state.history.append({
                        "scenario": f"Compliance (-{gwp_reduction_pct}% GWP)", 
                        "results": df_materials
                    })

        elif scenario == "Optimize Cost vs GWP (Tradeoff)":
            with st.spinner("Running NSGA-II optimization..."):
                pareto = run_nsga2_constrained(popsize, ngen, cxpb, mutpb, costs, impact_matrix, 
                                              traci_impact_cols, base_amounts, baseline_trees, 
                                              scale_materials_mask, efficiency_materials_mask, 
                                              max_scale_deviation, max_efficiency_deviation)
                
                results = []
                for ind in pareto:
                    production_scale = ind[0]
                    efficiency_factors = np.array(ind[1:])
                    
                    final_amounts = np.copy(base_amounts)
                    for i in range(len(base_amounts)):
                        if scale_materials_mask[i]:
                            final_amounts[i] = base_amounts[i] * production_scale
                        elif efficiency_materials_mask[i]:
                            efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                            final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                    
                    total_cost = np.dot(final_amounts, costs)
                    actual_trees = baseline_trees * production_scale
                    
                    gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
                    total_gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                    
                    cost_per_tree = total_cost / actual_trees
                    gwp_per_tree = total_gwp / actual_trees
                    
                    results.append({
                        "Trees": int(actual_trees),
                        "Total Cost": total_cost,
                        "Total GWP": total_gwp,
                        "Cost/Tree": cost_per_tree,
                        "GWP/Tree": gwp_per_tree,
                        "Production Scale": production_scale,
                        "Individual": ind
                    })
                
                df_pareto = pd.DataFrame(results)
                st.dataframe(df_pareto[["Trees", "Total Cost", "Total GWP", "Cost/Tree", "GWP/Tree", "Production Scale"]])
                
                # Plot Pareto front
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                ax1.scatter(df_pareto["Total Cost"], df_pareto["Total GWP"], alpha=0.6, c=df_pareto["Trees"], cmap='viridis')
                ax1.scatter([baseline_cost], [baseline_gwp], color='red', s=100, marker='*', label='Baseline', zorder=5)
                ax1.set_xlabel("Total Cost ($)")
                ax1.set_ylabel("Total GWP (kg CO2-Eq)")
                ax1.set_title("Pareto Front: Total Values (color=trees)")
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                scatter = ax2.scatter(df_pareto["Cost/Tree"], df_pareto["GWP/Tree"], alpha=0.6, c=df_pareto["Trees"], cmap='viridis')
                ax2.scatter([baseline_cost_per_tree], [baseline_gwp_per_tree], color='red', s=100, marker='*', label='Baseline', zorder=5)
                ax2.set_xlabel("Cost per Tree ($)")
                ax2.set_ylabel("GWP per Tree (kg CO2-Eq)")
                ax2.set_title("Pareto Front: Per-Tree Values (color=trees)")
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax2, label='Number of Trees')
                
                st.pyplot(fig)
                
                # Allow user to select a solution
                st.markdown("### Select a Solution from Pareto Front")
                solution_idx = st.selectbox("Solution Index", range(len(pareto)))
                
                selected_ind = pareto[solution_idx]
                production_scale = selected_ind[0]
                efficiency_factors = np.array(selected_ind[1:])
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                for i in range(len(base_amounts)):
                    if scale_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * production_scale
                        material_type.append("SCALE")
                    elif efficiency_materials_mask[i]:
                        efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                        final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                        material_type.append("EFFICIENCY")
                    else:
                        material_type.append("FIXED")
                
                actual_trees = baseline_trees * production_scale
                
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
                st.success(f"**Selected Solution ({int(actual_trees)} trees):** ${total_cost_opt:.2f} total (${total_cost_opt/actual_trees:.2f}/tree) | {total_gwp_opt:.2f} kg CO2-Eq ({total_gwp_opt/actual_trees:.2f} kg/tree)")
                
                if 'history' not in st.session_state:
                    st.session_state['history'] = []
                st.session_state.history.append({"scenario": scenario, "results": df_materials, "pareto": df_pareto})

        elif scenario == "Optimize Cost Only":
            with st.spinner("Running single-objective optimization..."):
                best = run_single_constrained(evaluate_cost_only_constrained, popsize, ngen, cxpb, mutpb, 
                                             base_amounts, baseline_trees, scale_materials_mask, 
                                             efficiency_materials_mask, max_scale_deviation, 
                                             max_efficiency_deviation, costs)
                
                production_scale = best[0]
                efficiency_factors = np.array(best[1:])
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                for i in range(len(base_amounts)):
                    if scale_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * production_scale
                        material_type.append("SCALE")
                    elif efficiency_materials_mask[i]:
                        efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                        final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                        material_type.append("EFFICIENCY")
                    else:
                        material_type.append("FIXED")
                
                actual_trees = baseline_trees * production_scale
                total_cost = np.dot(final_amounts, costs)
                cost_per_tree = total_cost / actual_trees
                
                st.metric("Production Scale", f"{production_scale:.2%}", f"{int(actual_trees)} trees")
                st.metric("Total Cost", f"${total_cost:.2f}", f"{((total_cost - baseline_cost) / baseline_cost * 100):.1f}%")
                st.metric("Cost per Tree", f"${cost_per_tree:.2f}", f"{((cost_per_tree - baseline_cost_per_tree) / baseline_cost_per_tree * 100):.1f}%")
                
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
                best = run_single_constrained(evaluate_single_impact_constrained, popsize, ngen, cxpb, mutpb, 
                                             base_amounts, baseline_trees, scale_materials_mask,
                                             efficiency_materials_mask, max_scale_deviation,
                                             max_efficiency_deviation, 
                                             impact_matrix, selected_impact, traci_impact_cols)
                
                production_scale = best[0]
                efficiency_factors = np.array(best[1:])
                
                final_amounts = np.copy(base_amounts)
                material_type = []
                for i in range(len(base_amounts)):
                    if scale_materials_mask[i]:
                        final_amounts[i] = base_amounts[i] * production_scale
                        material_type.append("SCALE")
                    elif efficiency_materials_mask[i]:
                        efficiency_idx = np.where(efficiency_materials_mask[:i])[0].size
                        final_amounts[i] = base_amounts[i] * production_scale * efficiency_factors[efficiency_idx]
                        material_type.append("EFFICIENCY")
                    else:
                        material_type.append("FIXED")
                
                actual_trees = baseline_trees * production_scale
                
                impact_idx = traci_impact_cols.index(selected_impact)
                total_impact = np.dot(final_amounts, impact_matrix[:, impact_idx])
                impact_per_tree = total_impact / actual_trees
                
                baseline_impact = np.dot(base_amounts, impact_matrix[:, impact_idx])
                baseline_impact_per_tree = baseline_impact / baseline_trees
                
                st.metric("Production Scale", f"{production_scale:.2%}", f"{int(actual_trees)} trees")
                st.metric(f"Total {selected_impact}", f"{total_impact:.4f}", 
                         f"{((total_impact - baseline_impact) / baseline_impact * 100):.1f}%")
                st.metric(f"{selected_impact} per Tree", f"{impact_per_tree:.4f}",
                         f"{((impact_per_tree - baseline_impact_per_tree) / baseline_impact_per_tree * 100):.1f}%")
                
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
    st.info("📂 Upload one or more Excel files to begin analysis.\n\n**Tips:**\n- Upload multiple files to compare different farm management strategies\n- Each file should contain material inputs, costs, and environmental impacts\n- The app will identify which strategy is most efficient")
