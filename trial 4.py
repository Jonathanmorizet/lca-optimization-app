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

# ========== MODE SELECTION ==========
st.title("🌳 Farm Management Optimization System")
st.markdown("---")

mode = st.radio(
    "**Select Analysis Mode:**",
    ["📊 Multi-Farm Comparison & Hybrid Strategy", "🔬 Single Farm Optimization"],
    horizontal=True
)

st.markdown("---")

# File uploader
if mode == "📊 Multi-Farm Comparison & Hybrid Strategy":
    uploaded_files = st.file_uploader(
        "Upload DEAP NSGA Readable Excel files (upload multiple farms/strategies to compare)", 
        type=["xlsm", "xlsx"],
        accept_multiple_files=True
    )
else:
    uploaded_file = st.file_uploader(
        "Upload DEAP NSGA Readable Excel file for optimization", 
        type=["xlsm", "xlsx"]
    )
    uploaded_files = [uploaded_file] if uploaded_file else []

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
            units = merged_df['Unit'].tolist()
            
            # Get year information - handle both numeric and string years
            if 'Year' in merged_df.columns:
                years = []
                for y in merged_df['Year']:
                    if pd.isna(y):
                        years.append('N/A')
                    else:
                        years.append(str(int(y)) if isinstance(y, (int, float)) else str(y))
            else:
                years = ['N/A'] * len(materials)
            
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
                'units': units,
                'years': years,
                'impact_matrix': impact_matrix,
                'impact_columns': impact_columns,
                'total_cost': total_cost,
                'total_gwp': gwp
            })
            
        except Exception as e:
            st.error(f"Error reading {uploaded_file.name}: {e}")
            continue
    
    return farms_data if len(farms_data) > 0 else None

# DEAP Evaluation Functions
def evaluate_cost_gwp_constrained(ind, costs, impact_matrix, impact_cols, base_amounts, 
                                  baseline_trees, scale_materials_mask, efficiency_materials_mask, 
                                  max_scale_deviation, max_efficiency_deviation):
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
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
    except ValueError:
        pass
    
    return total_cost + penalty, gwp + penalty

def evaluate_cost_only_constrained(ind, costs, base_amounts, baseline_trees, 
                                   scale_materials_mask, efficiency_materials_mask,
                                   max_scale_deviation, max_efficiency_deviation):
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
                         baseline_trees, scale_mask, efficiency_mask, max_scale_dev, max_eff_dev):
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
        ind = [random.uniform(1 - max_scale_dev, 1 + max_scale_dev)]
        for _ in range(num_efficiency_materials):
            ind.append(random.uniform(1 - max_eff_dev, 1 + max_eff_dev))
        return ind
    
    toolbox.register("individual", tools.initIterate, creator.Individual, create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_cost_gwp_constrained, 
                    costs=costs, impact_matrix=matrix, impact_cols=impact_cols, 
                    base_amounts=base_amounts, baseline_trees=baseline_trees,
                    scale_materials_mask=scale_mask, efficiency_materials_mask=efficiency_mask,
                    max_scale_deviation=max_scale_dev, max_efficiency_deviation=max_eff_dev)
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
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=popsize)
    
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    for gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb and i < len(offspring):
                child1, child2 = toolbox.mate(offspring[i-1], offspring[i])
                offspring[i-1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        
        pop = toolbox.select(pop + offspring, popsize)
    
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single_constrained(obj_func, popsize, ngen, cxpb, mutpb, base_amounts, baseline_trees, 
                          scale_mask, efficiency_mask, max_scale_dev, max_eff_dev, *args):
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
    toolbox.register("evaluate", obj_func, *args, base_amounts=base_amounts, baseline_trees=baseline_trees,
                    scale_materials_mask=scale_mask, efficiency_materials_mask=efficiency_mask,
                    max_scale_deviation=max_scale_dev, max_efficiency_deviation=max_eff_dev)
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
    
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    hof = tools.HallOfFame(1)
    
    for gen in range(ngen):
        offspring = toolbox.select(pop, popsize)
        offspring = [creator.Individual(list(ind)) for ind in offspring]
        
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb and i < len(offspring):
                child1, child2 = toolbox.mate(offspring[i-1], offspring[i])
                offspring[i-1] = creator.Individual(child1)
                offspring[i] = creator.Individual(child2)
        
        for i in range(len(offspring)):
            if random.random() < mutpb:
                mutated, = toolbox.mutate(offspring[i])
                offspring[i] = creator.Individual(mutated)
        
        for ind in offspring:
            ind.fitness.values = toolbox.evaluate(ind)
        
        pop[:] = offspring
        hof.update(pop)
    
    return hof[0]

# Load data
farms_data = load_multiple_farms(uploaded_files)

# ========== MULTI-FARM COMPARISON MODE ==========
if mode == "📊 Multi-Farm Comparison & Hybrid Strategy":
    st.header("📊 Multi-Farm Comparison & Hybrid Strategy")
    
    if farms_data is not None and len(farms_data) > 0:
        st.success(f"✅ Loaded {len(farms_data)} farm management strategies!")
        
        # Display comparison table with baseline metrics
        st.markdown("## Farm Strategy Comparison - Baseline Metrics")
        
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
        
        pareto_farms = []
        for i, farm_i in enumerate(farms_data):
            is_dominated = False
            for j, farm_j in enumerate(farms_data):
                if i != j:
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
        
        # Comparative Material Breakdown
        if len(farms_data) > 1:
            st.markdown("---")
            st.markdown("## 💰 Comparative Cost Breakdown (Top 10 Materials)")
            
            fig_comp_cost, axes = plt.subplots(1, len(farms_data), figsize=(7*len(farms_data), 6))
            if len(farms_data) == 1:
                axes = [axes]
            
            for idx, farm in enumerate(farms_data):
                material_costs = farm['base_amounts'] * farm['costs']
                cost_breakdown = pd.DataFrame({
                    'Material': farm['materials'],
                    'Cost': material_costs
                }).sort_values('Cost', ascending=False).head(10)
                
                axes[idx].barh(cost_breakdown['Material'], cost_breakdown['Cost'])
                axes[idx].set_xlabel('Cost ($)')
                axes[idx].set_title(f'{farm["name"]}\nBaseline: ${farm["total_cost"]:.2f}')
                axes[idx].invert_yaxis()
            
            plt.tight_layout()
            st.pyplot(fig_comp_cost)
            
            # Comparative GWP Breakdown
            if "kg CO2-Eq/Unit" in farms_data[0]['impact_columns']:
                st.markdown("## 🌍 Comparative GWP Breakdown (Top 10 Materials)")
                
                fig_comp_gwp, axes = plt.subplots(1, len(farms_data), figsize=(7*len(farms_data), 6))
                if len(farms_data) == 1:
                    axes = [axes]
                
                for idx, farm in enumerate(farms_data):
                    gwp_idx = farm['impact_columns'].index("kg CO2-Eq/Unit")
                    material_gwp = farm['base_amounts'] * farm['impact_matrix'][:, gwp_idx]
                    gwp_breakdown = pd.DataFrame({
                        'Material': farm['materials'],
                        'GWP': material_gwp
                    }).sort_values('GWP', ascending=False).head(10)
                    
                    axes[idx].barh(gwp_breakdown['Material'], gwp_breakdown['GWP'], 
                                 color='#2ecc71', edgecolor='none')
                    axes[idx].set_xlabel('GWP (kg CO2-Eq)')
                    axes[idx].set_title(f'{farm["name"]}\nBaseline: {farm["total_gwp"]:.2f} kg CO2-Eq')
                    axes[idx].invert_yaxis()
                
                plt.tight_layout()
                st.pyplot(fig_comp_gwp)
        
        # Multi-Farm Hybrid Strategy
        if len(farms_data) > 1:
            st.markdown("---")
            st.markdown("## 🔀 Create Hybrid Strategy")
            st.markdown("Combine the best materials from different farms to find an optimal hybrid approach.")
            
            strategy_type = st.radio(
                "How should we create the hybrid?",
                ["Take lowest cost material for each", 
                 "Take lowest GWP material for each",
                 "Optimize cost-GWP tradeoff for each material"]
            )
            
            if st.button("🚀 Generate Hybrid Strategy"):
                with st.spinner("Creating hybrid strategy..."):
                    unique_materials = set()
                    for farm in farms_data:
                        unique_materials.update(farm['materials'])
                    
                    hybrid_materials = []
                    hybrid_amounts = []
                    hybrid_costs = []
                    hybrid_units = []
                    hybrid_years = []
                    hybrid_gwp_values = []
                    hybrid_sources = []
                    
                    for material in unique_materials:
                        candidates = []
                        for farm in farms_data:
                            if material in farm['materials']:
                                idx = farm['materials'].index(material)
                                amount = farm['base_amounts'][idx]
                                cost = farm['costs'][idx]
                                unit = farm['units'][idx]
                                year = farm['years'][idx]
                                
                                if "kg CO2-Eq/Unit" in farm['impact_columns']:
                                    gwp_idx = farm['impact_columns'].index("kg CO2-Eq/Unit")
                                    gwp_per_unit = farm['impact_matrix'][idx, gwp_idx]
                                else:
                                    gwp_per_unit = 0
                                
                                candidates.append({
                                    'farm': farm['name'],
                                    'amount': amount,
                                    'cost': cost,
                                    'unit': unit,
                                    'year': year,
                                    'gwp_per_unit': gwp_per_unit,
                                    'total_cost': amount * cost,
                                    'total_gwp': amount * gwp_per_unit
                                })
                        
                        if candidates:
                            if strategy_type == "Take lowest cost material for each":
                                best = min(candidates, key=lambda x: x['total_cost'])
                            elif strategy_type == "Take lowest GWP material for each":
                                best = min(candidates, key=lambda x: x['total_gwp'])
                            else:
                                max_cost = max(c['total_cost'] for c in candidates)
                                max_gwp = max(c['total_gwp'] for c in candidates) if max(c['total_gwp'] for c in candidates) > 0 else 1
                                best = min(candidates, key=lambda x: (x['total_cost']/max_cost + x['total_gwp']/max_gwp))
                            
                            hybrid_materials.append(material)
                            hybrid_amounts.append(best['amount'])
                            hybrid_costs.append(best['cost'])
                            hybrid_units.append(best['unit'])
                            hybrid_years.append(best['year'])
                            hybrid_gwp_values.append(best['total_gwp'])
                            hybrid_sources.append(best['farm'])
                    
                    hybrid_total_cost = sum(hybrid_amounts[i] * hybrid_costs[i] for i in range(len(hybrid_materials)))
                    hybrid_total_gwp = sum(hybrid_gwp_values)
                    
                    st.success(f"🎉 Hybrid Strategy Created!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Hybrid Total Cost", f"${hybrid_total_cost:.2f}")
                        
                        for farm in farms_data:
                            savings = farm['total_cost'] - hybrid_total_cost
                            st.markdown(f"💰 vs {farm['name']}: ${savings:+.2f} ({savings/farm['total_cost']*100:+.1f}%)")
                    
                    with col2:
                        st.metric("Hybrid Total GWP", f"{hybrid_total_gwp:.2f} kg CO2-Eq")
                        
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
                    
                    # Hybrid Implementation Plan
                    st.markdown("---")
                    st.markdown("### 📋 Hybrid Strategy Implementation Plan")
                    
                    hybrid_df = pd.DataFrame({
                        'Material': hybrid_materials,
                        'Amount': hybrid_amounts,
                        'Unit': hybrid_units,
                        'Year': hybrid_years,
                        'Unit Cost ($)': hybrid_costs,
                        'Total Cost ($)': [hybrid_amounts[i] * hybrid_costs[i] for i in range(len(hybrid_materials))],
                        'Total GWP (kg CO2-Eq)': hybrid_gwp_values,
                        'Source Strategy': hybrid_sources
                    })
                    
                    # Proper sorting
                    def get_sort_key(row):
                        year_str = str(row['Year'])
                        if year_str == 'N/A' or year_str == 'nan':
                            return (999999, -row['Total Cost ($)'])
                        try:
                            year_int = int(float(year_str))
                            return (year_int, -row['Total Cost ($)'])
                        except:
                            return (999999, -row['Total Cost ($)'])
                    
                    hybrid_df['_sort_key'] = hybrid_df.apply(get_sort_key, axis=1)
                    hybrid_df = hybrid_df.sort_values('_sort_key').drop('_sort_key', axis=1)
                    
                    st.dataframe(hybrid_df, use_container_width=True)
                    
                    # Summary by source
                    st.markdown("### 📊 Materials by Source Strategy")
                    source_summary = hybrid_df.groupby('Source Strategy').agg({
                        'Material': 'count',
                        'Total Cost ($)': 'sum',
                        'Total GWP (kg CO2-Eq)': 'sum'
                    }).rename(columns={'Material': 'Number of Materials'})
                    
                    st.dataframe(source_summary, use_container_width=True)
                    
                    # Visual breakdown
                    fig_sources, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                    
                    cost_by_source = hybrid_df.groupby('Source Strategy')['Total Cost ($)'].sum()
                    ax1.pie(cost_by_source.values, labels=cost_by_source.index, autopct='%1.1f%%', startangle=90)
                    ax1.set_title('Cost Distribution by Source Strategy')
                    
                    gwp_by_source = hybrid_df.groupby('Source Strategy')['Total GWP (kg CO2-Eq)'].sum()
                    ax2.pie(gwp_by_source.values, labels=gwp_by_source.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
                    ax2.set_title('GWP Distribution by Source Strategy')
                    
                    plt.tight_layout()
                    st.pyplot(fig_sources)
    else:
        st.info("📂 Upload multiple Excel files to compare farm management strategies and create hybrid solutions.")

# ========== SINGLE FARM OPTIMIZATION MODE ==========
elif mode == "🔬 Single Farm Optimization":
    st.header("🔬 Single Farm Optimization")
    
    if farms_data is not None and len(farms_data) > 0:
        farm = farms_data[0]
        merged_df = farm['merged_df']
        materials = farm['materials']
        base_amounts = farm['base_amounts']
        costs = farm['costs']
        impact_matrix = farm['impact_matrix']
        traci_impact_cols = farm['impact_columns']
        
        st.success(f"✅ File loaded: **{farm['name']}**")
        st.dataframe(merged_df)
        
        # Material Classification
        st.sidebar.markdown("### 🔧 Material Classification")
        st.sidebar.info("Classify materials as either scaling with production or efficiency-based")
        
        # Auto-detect scale materials
        default_scale = []
        scale_keywords = ['transplant', 'seedling', 'plant', 'tree']
        for i, mat in enumerate(materials):
            if any(keyword in mat.lower() for keyword in scale_keywords):
                default_scale.append(i)
        
        with st.sidebar.expander("Customize Material Types", expanded=True):
            st.markdown("**Scale Materials** change proportionally with production (transplants, etc.)")
            st.markdown("**Efficiency Materials** can be optimized independently (fertilizers, pesticides, fuel)")
            
            scale_material_indices = st.multiselect(
                "Select Scale Materials",
                options=list(range(len(materials))),
                default=default_scale,
                format_func=lambda x: f"{materials[x]}"
            )
            
            efficiency_material_indices = st.multiselect(
                "Select Efficiency Materials",
                options=[i for i in range(len(materials)) if i not in scale_material_indices],
                default=[i for i in range(len(materials)) if i not in default_scale],
                format_func=lambda x: f"{materials[x]}"
            )
        
        # Create masks
        scale_materials_mask = np.zeros(len(materials), dtype=bool)
        scale_materials_mask[scale_material_indices] = True
        
        efficiency_materials_mask = np.zeros(len(materials), dtype=bool)
        efficiency_materials_mask[efficiency_material_indices] = True
        
        # Display material status
        st.sidebar.markdown("**Material Status:**")
        st.sidebar.markdown(f"📏 **Scale Materials:** {sum(scale_materials_mask)}")
        st.sidebar.markdown(f"⚡ **Efficiency Materials:** {sum(efficiency_materials_mask)}")
        
        # Production Configuration
        st.sidebar.markdown("### 🌳 Production Settings")
        baseline_trees = st.sidebar.number_input("Baseline Trees", min_value=1, value=1900, step=1)
        
        st.sidebar.markdown("### 📊 Optimization Bounds")
        max_scale_deviation_pct = st.sidebar.slider("Production Scale ±%", 5, 50, 20)
        max_scale_deviation = max_scale_deviation_pct / 100.0
        
        max_efficiency_deviation_pct = st.sidebar.slider("Material Efficiency ±%", 5, 30, 20)
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
        baseline_cost_per_tree = baseline_cost / baseline_trees
        
        baseline_gwp = 0.0
        if "kg CO2-Eq/Unit" in traci_impact_cols:
            gwp_idx = traci_impact_cols.index("kg CO2-Eq/Unit")
            baseline_gwp = np.dot(base_amounts, impact_matrix[:, gwp_idx])
            baseline_gwp_per_tree = baseline_gwp / baseline_trees
            
            st.info(f"📌 **Baseline ({baseline_trees} trees):** ${baseline_cost:.2f} total (${baseline_cost_per_tree:.2f}/tree) | {baseline_gwp:.2f} kg CO2-Eq total ({baseline_gwp_per_tree:.2f} kg/tree)")
        
        if st.button("🚀 Run Optimization"):
            st.subheader(f"Running: {scenario}")
            
            if scenario == "Optimize Cost vs GWP (Tradeoff)":
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
                            "Total Cost": total_cost,
                            "Total GWP": total_gwp,
                            "Trees": actual_trees,
                            "Cost/Tree": cost_per_tree,
                            "GWP/Tree": gwp_per_tree,
                            "Individual": ind
                        })
                    
                    df_pareto = pd.DataFrame(results)
                    st.dataframe(df_pareto[["Total Cost", "Total GWP", "Trees", "Cost/Tree", "GWP/Tree"]])
                    
                    # Plot Pareto front
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                    
                    ax1.scatter(df_pareto["Total Cost"], df_pareto["Total GWP"], alpha=0.6, s=50)
                    ax1.scatter([baseline_cost], [baseline_gwp], color='red', s=150, marker='*', label='Baseline', zorder=5)
                    ax1.set_xlabel("Total Cost ($)")
                    ax1.set_ylabel("Total GWP (kg CO2-Eq)")
                    ax1.set_title("Pareto Front: Total Values")
                    ax1.legend()
                    ax1.grid(True, alpha=0.3)
                    
                    ax2.scatter(df_pareto["Cost/Tree"], df_pareto["GWP/Tree"], alpha=0.6, s=50)
                    ax2.scatter([baseline_cost_per_tree], [baseline_gwp_per_tree], color='red', s=150, marker='*', label='Baseline', zorder=5)
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
                        "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                    })
                    
                    st.dataframe(df_materials)
                    
                    # Summary
                    total_cost_opt = np.dot(final_amounts, costs)
                    total_gwp_opt = np.dot(final_amounts, impact_matrix[:, gwp_idx])
                    actual_trees = baseline_trees * production_scale
                    cost_savings = baseline_cost - total_cost_opt
                    gwp_reduction = baseline_gwp - total_gwp_opt
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Cost", f"${total_cost_opt:.2f}", f"-${cost_savings:.2f}")
                    with col2:
                        st.metric("Total GWP", f"{total_gwp_opt:.2f} kg", f"-{gwp_reduction:.2f} kg")
                    with col3:
                        st.metric("Trees Produced", f"{actual_trees:.0f}", f"{actual_trees - baseline_trees:+.0f}")
                    
                    st.success(f"**Per-Tree Metrics:** ${total_cost_opt/actual_trees:.2f}/tree | {total_gwp_opt/actual_trees:.2f} kg CO2-Eq/tree")
            
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
                    
                    total_cost = np.dot(final_amounts, costs)
                    actual_trees = baseline_trees * production_scale
                    cost_per_tree = total_cost / actual_trees
                    cost_savings = baseline_cost - total_cost
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Cost", f"${total_cost:.2f}", f"-${cost_savings:.2f}")
                    with col2:
                        st.metric("Cost per Tree", f"${cost_per_tree:.2f}", f"-${(baseline_cost_per_tree - cost_per_tree):.2f}")
                    with col3:
                        st.metric("Trees Produced", f"{actual_trees:.0f}", f"{actual_trees - baseline_trees:+.0f}")
                    
                    df_materials = pd.DataFrame({
                        "Material": materials,
                        "Type": material_type,
                        "Base Amount": base_amounts,
                        "Optimized Amount": final_amounts,
                        "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                    })
                    
                    st.dataframe(df_materials)
            
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
                    
                    impact_idx = traci_impact_cols.index(selected_impact)
                    total_impact = np.dot(final_amounts, impact_matrix[:, impact_idx])
                    actual_trees = baseline_trees * production_scale
                    impact_per_tree = total_impact / actual_trees
                    
                    baseline_impact = np.dot(base_amounts, impact_matrix[:, impact_idx])
                    baseline_impact_per_tree = baseline_impact / baseline_trees
                    impact_reduction = baseline_impact - total_impact
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric(f"Total {selected_impact}", f"{total_impact:.4f}", f"-{impact_reduction:.4f}")
                    with col2:
                        st.metric(f"{selected_impact} per Tree", f"{impact_per_tree:.4f}",
                                 f"-{(baseline_impact_per_tree - impact_per_tree):.4f}")
                    with col3:
                        st.metric("Trees Produced", f"{actual_trees:.0f}", f"{actual_trees - baseline_trees:+.0f}")
                    
                    df_materials = pd.DataFrame({
                        "Material": materials,
                        "Type": material_type,
                        "Base Amount": base_amounts,
                        "Optimized Amount": final_amounts,
                        "Change (%)": ((final_amounts - base_amounts) / base_amounts * 100)
                    })
                    
                    st.dataframe(df_materials)
        
        # Download functionality
        st.markdown("---")
        if st.button("📥 Download Merged Data as Excel"):
            output = BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
            output.seek(0)
            st.download_button(
                label="Download Excel File",
                data=output,
                file_name=f"optimized_{farm['name']}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    else:
        st.info("📂 Upload a single Excel file to begin single-farm optimization.\n\n**This mode allows you to:**\n- Optimize cost vs environmental impact tradeoffs\n- Minimize specific TRACI impact categories\n- Find cost-optimal production strategies\n- Adjust production scale and material efficiency")
