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
            units = merged_df['Unit'].tolist()
            
            # Get year information if available
            years = merged_df['Year'].tolist() if 'Year' in merged_df.columns else ['N/A'] * len(materials)
            
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

# DEAP Evaluation Functions (keeping existing functions)
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
    actual_trees = baseline_trees * production_scale
    
    gwp = 0.0
    try:
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        gwp = np.dot(final_amounts, impact_matrix[:, gwp_idx])
    except ValueError:
        pass
    
    return total_cost + penalty, gwp + penalty

# Load data
farms_data = load_multiple_farms(uploaded_files)

if farms_data is not None and len(farms_data) > 0:
    st.success(f"✅ Loaded {len(farms_data)} farm management strategies!")
    
    # Display comparison table with baseline metrics
    st.markdown("## 📊 Farm Strategy Comparison - Baseline Metrics")
    
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
    else:
        # Single farm breakdown
        farm = farms_data[0]
        
        st.markdown("### 💰 Cost Breakdown (Top 10 Materials)")
        material_costs = farm['base_amounts'] * farm['costs']
        cost_breakdown = pd.DataFrame({
            'Material': farm['materials'],
            'Cost': material_costs
        }).sort_values('Cost', ascending=False).head(10)
        
        fig_cost, ax_cost = plt.subplots(figsize=(10, 6))
        ax_cost.barh(cost_breakdown['Material'], cost_breakdown['Cost'])
        ax_cost.set_xlabel('Cost ($)')
        ax_cost.set_title(f'Top 10 Cost Contributors - {farm["name"]}')
        ax_cost.invert_yaxis()
        plt.tight_layout()
        st.pyplot(fig_cost)
        
        if "kg CO2-Eq/Unit" in farm['impact_columns']:
            st.markdown("### 🌍 GWP Breakdown (Top 10 Materials)")
            gwp_idx = farm['impact_columns'].index("kg CO2-Eq/Unit")
            material_gwp = farm['base_amounts'] * farm['impact_matrix'][:, gwp_idx]
            gwp_breakdown = pd.DataFrame({
                'Material': farm['materials'],
                'GWP': material_gwp
            }).sort_values('GWP', ascending=False).head(10)
            
            fig_gwp, ax_gwp = plt.subplots(figsize=(10, 6))
            ax_gwp.barh(gwp_breakdown['Material'], gwp_breakdown['GWP'], 
                       color='#2ecc71', edgecolor='none')
            ax_gwp.set_xlabel('GWP (kg CO2-Eq)')
            ax_gwp.set_title(f'Top 10 GWP Contributors - {farm["name"]}')
            ax_gwp.invert_yaxis()
            plt.tight_layout()
            st.pyplot(fig_gwp)
    
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
                
                # Hybrid Implementation Plan with Year and Unit
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
                
                # Sort by year then cost
                hybrid_df = hybrid_df.sort_values(['Year', 'Total Cost ($)'], ascending=[True, False])
                
                st.dataframe(hybrid_df, use_container_width=True)
                
                # Show summary by source
                st.markdown("### 📊 Materials by Source Strategy")
                source_summary = hybrid_df.groupby('Source Strategy').agg({
                    'Material': 'count',
                    'Total Cost ($)': 'sum',
                    'Total GWP (kg CO2-Eq)': 'sum'
                }).rename(columns={'Material': 'Number of Materials'})
                
                st.dataframe(source_summary, use_container_width=True)
                
                # Create visual breakdown by source
                fig_sources, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                
                cost_by_source = hybrid_df.groupby('Source Strategy')['Total Cost ($)'].sum()
                ax1.pie(cost_by_source.values, labels=cost_by_source.index, autopct='%1.1f%%', startangle=90)
                ax1.set_title('Cost Distribution by Source Strategy')
                
                gwp_by_source = hybrid_df.groupby('Source Strategy')['Total GWP (kg CO2-Eq)'].sum()
                ax2.pie(gwp_by_source.values, labels=gwp_by_source.index, autopct='%1.1f%%', startangle=90, colors=['lightgreen', 'lightcoral'])
                ax2.set_title('GWP Distribution by Source Strategy')
                
                plt.tight_layout()
                st.pyplot(fig_sources)
    
    st.markdown("---")
    st.info("✅ Hybrid strategy analysis complete! Upload files to continue with individual farm optimization if desired.")
    
else:
    st.info("📂 Upload one or more Excel files to begin analysis.\n\n**Tips:**\n- Upload multiple files to compare different farm management strategies\n- Each file should contain material inputs, costs, and environmental impacts\n- The app will identify which strategy is most efficient")
