# This Streamlit application performs multi-objective optimization of material amounts
# using the DEAP (Distributed Evolutionary Algorithms in Python) library.
# It allows users to upload material data, costs, and environmental impacts from an Excel file,
# define optimization scenarios (Cost vs GWP, Cost only, Single Impact, Combined Impact, Total Quantity),
# set bounds for material amounts, configure genetic algorithm parameters, and run
# the optimization to find optimal material compositions. Results are displayed in tables
# and plots, with an option to download data and view optimization history.

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
# import subprocess # Already imported
# import sys # Already imported
from io import BytesIO

# Ensure Streamlit is installed (check already in first cell)
# Ensure DEAP is installed (check already in first cell)

random.seed(42)

# --- File Uploader and Data Loading Section ---
# File uploader
st.title("Multi-Objective Optimization with DEAP")
st.write("Upload your Excel file containing material inputs, costs, and environmental impacts to perform multi-objective optimization.")
uploaded_file = st.file_uploader("Upload the 'DEAP NSGA Readable file.xlsm' Excel file", type=["xlsm"], help="Please upload an Excel file with at least three sheets: Inputs (Material, Unit, Amount), Costs (Material, Unit, Unit Cost ($)), and Impacts (Material, Unit, TRACI 2.1 impacts).")

@st.cache_data
def load_data(uploaded_file):
    """
    Loads and processes data from an uploaded Excel file.

    Args:
        uploaded_file (UploadedFile): The Excel file uploaded by the user.

    Returns:
        tuple: A tuple containing merged_df, materials, base_amounts, costs,
               impact_matrix, and traci_impact_cols if successful, otherwise None values.
    """
    if uploaded_file is not None:
        with st.spinner('Loading and processing data...'):
            try:
                # Read all sheets from the Excel file
                df = pd.read_excel(uploaded_file, sheet_name=None)

                # Check for expected sheets
                expected_sheets = list(df.keys())
                if len(expected_sheets) < 3:
                     raise ValueError("The Excel file should contain at least three sheets.")

                # Access dataframes by sheet order
                inputs_df = df[expected_sheets[0]]
                cost_df = df[expected_sheets[1]]
                impact_df = df[expected_sheets[2]]

                # Check for required columns in inputs_df
                if 'Material' not in inputs_df.columns or 'Unit' not in inputs_df.columns or 'Amount' not in inputs_df.columns:
                     raise ValueError("The first sheet must contain 'Material', 'Unit', and 'Amount' columns.")

                # Check for required columns in cost_df
                if 'Material' not in cost_df.columns or 'Unit' not in cost_df.columns or 'Unit Cost ($)' not in cost_df.columns:
                     raise ValueError("The second sheet must contain 'Material', 'Unit', and 'Unit Cost ($)' columns.")

                # Check for required columns in impact_df
                if 'Material' not in impact_df.columns or 'Unit' not in impact_df.columns:
                    raise ValueError("The third sheet must contain 'Material' and 'Unit' columns.")

                # Merge the dataframes
                merged_df = inputs_df.merge(cost_df, on=["Material", "Unit"], how="left")
                merged_df = merged_df.merge(impact_df, on=["Material", "Unit"], how="left")
                merged_df = merged_df.fillna(0) # Fill missing values with 0

                # Extract relevant data for optimization
                materials = merged_df['Material'].tolist()
                base_amounts = merged_df['Amount'].values
                costs = merged_df['Unit Cost ($)'].values
                # Identify impact columns (excluding identifiers and Year if present)
                impact_columns = [col for col in impact_df.columns if col in merged_df.columns and col not in ['Material', 'Unit', 'Year']]
                impact_matrix = merged_df[impact_columns].values
                traci_impact_cols = impact_columns # Store TRACI impact column names

                return merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols

            except FileNotFoundError:
                st.error("Error: File not found. Please upload the correct file.")
                return None, None, None, None, None, None
            except ValueError as ve:
                st.error(f"Data format error: {ve}")
                return None, None, None, None, None, None
            except KeyError as ke:
                st.error(f"Error accessing data: Missing sheet or column - {ke}")
                return None, None, None, None, None, None
            except Exception as e:
                st.error(f"An unexpected error occurred during file processing: {e}")
                return None, None, None, None, None, None
    else:
        return None, None, None, None, None, None

# --- DEAP Evaluation Functions Section ---
# These functions define the objectives to be minimized by the genetic algorithm.

def evaluate_cost_gwp(ind, costs, impact_matrix, impact_cols):
    """
    Evaluates the total cost and Global Warming Potential (GWP) for a given individual.

    Args:
        ind (list): A list representing the amounts of each material.
        costs (np.ndarray): Array of unit costs for each material.
        impact_matrix (np.ndarray): Matrix of impact values (rows=materials, columns=impact categories).
        impact_cols (list): List of impact category column names.

    Returns:
        tuple: A tuple containing (total_cost, total_gwp). Returns (total_cost, 0.0) if GWP column is not found.
    """
    # Ensure individual amounts are non-negative and convert to numpy array
    x = np.maximum(0.0, np.array(ind))
    # Calculate total cost
    cost = np.dot(x, costs)
    gwp = 0.0
    try:
        # Find the index of the GWP column
        gwp_idx = impact_cols.index("kg CO2-Eq/Unit")
        # Calculate total GWP
        gwp = np.dot(x, impact_matrix[:, gwp_idx])
    except ValueError:
        # Handle case where GWP column is not found
        st.warning("GWP column 'kg CO2-Eq/Unit' not found in impact data. GWP calculation skipped.")
    # Return fitness values as a tuple (required by DEAP)
    return cost, gwp

def evaluate_cost_only(ind, costs):
    """
    Evaluates the total cost for a given individual.

    Args:
        ind (list): A list representing the amounts of each material.
        costs (np.ndarray): Array of unit costs for each material.

    Returns:
        tuple: A tuple containing (total_cost,).
    """
    # Ensure individual amounts are non-negative and convert to numpy array
    x = np.maximum(0.0, np.array(ind))
    # Calculate total cost
    cost = np.dot(x, costs)
    # Return fitness value as a tuple (required by DEAP)
    return (cost,)

def evaluate_combined(ind, costs, impact_matrix):
    """
    Evaluates the sum of total cost and total combined impact for a given individual.

    Args:
        ind (list): A list representing the amounts of each material.
        costs (np.ndarray): Array of unit costs for each material.
        impact_matrix (np.ndarray): Matrix of impact values (rows=materials, columns=impact categories).

    Returns:
        tuple: A tuple containing (total_cost + total_combined_impact,).
    """
    # Ensure individual amounts are non-negative and convert to numpy array
    x = np.maximum(0.0, np.array(ind))
    # Calculate total cost
    cost = np.dot(x, costs)
    # Calculate total combined impact (sum of impacts across all categories for each material)
    combined_impact = np.sum(np.dot(x, impact_matrix))
    # Return fitness value as a tuple (required by DEAP)
    return (cost + combined_impact,)

def evaluate_single_impact(ind, matrix, colname, cols):
    """
    Evaluates the total impact for a single specified impact category for a given individual.

    Args:
        ind (list): A list representing the amounts of each material.
        matrix (np.ndarray): Matrix of impact values (rows=materials, columns=impact categories).
        colname (str): The name of the impact category column to evaluate.
        cols (list): List of impact category column names.

    Returns:
        tuple: A tuple containing (total_single_impact,). Returns (0.0,) if the specified column is not found.
    """
    # Ensure individual amounts are non-negative and convert to numpy array
    x = np.maximum(0.0, np.array(ind))
    try:
        # Find the index of the specified impact column
        idx = cols.index(colname)
        # Calculate total impact for the specified category
        single_impact = np.dot(x, matrix[:, idx])
        # Return fitness value as a tuple (required by DEAP)
        return (single_impact,)
    except ValueError:
        # Handle case where the specified impact column is not found
        st.warning(f"Impact column '{colname}' not found. Single impact calculation skipped.")
        return (0.0,)

def evaluate_total_quantity(ind):
    """
    Evaluates the total quantity of materials for a given individual.

    Args:
        ind (list): A list representing the amounts of each material.

    Returns:
        tuple: A tuple containing (total_quantity,).
    """
    # Ensure individual amounts are non-negative and convert to numpy array
    x = np.maximum(0.0, np.array(ind))
    # Calculate total quantity
    total_quantity = np.sum(x)
    # Return fitness value as a tuple (required by DEAP)
    return (total_quantity,)


# --- Optimization Logic Section ---
# Functions for running the genetic algorithms using DEAP.

def run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, matrix, impact_cols):
    """
    Runs the NSGA-II multi-objective optimization algorithm.

    Args:
        popsize (int): Population size.
        ngen (int): Number of generations.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        lows (np.ndarray): Array of lower bounds for material amounts.
        highs (np.ndarray): Array of upper bounds for material amounts.
        costs (np.ndarray): Array of unit costs for each material.
        matrix (np.ndarray): Matrix of impact values.
        impact_cols (list): List of impact category column names.

    Returns:
        list: The first front of non-dominated individuals (Pareto front).
    """
    # Attempt to delete DEAP creator classes to prevent AttributeErrors on re-runs
    try:
        del creator.FitnessMin
        del creator.Individual
    except AttributeError:
        pass # Classes don't exist yet, which is fine

    # Define the fitness function (minimizing two objectives: cost and GWP)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0))
    # Define the individual as a list with the defined fitness attribute
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Register a function to create random floats within the defined bounds for each material
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    # Register a function to create an individual using the attr_float function
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    # Register a function to create a population (list) of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register the evaluation function for cost and GWP
    toolbox.register("evaluate", evaluate_cost_gwp, costs=costs, impact_matrix=matrix, impact_cols=impact_cols)
    # Register the crossover operator (Blend Crossover)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        """
        Mutates an individual using Gaussian mutation and applies boundary constraints.

        Args:
            ind (creator.Individual): The individual to mutate.
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.
            indpb (float): Independent probability of each attribute being mutated.

        Returns:
            tuple: A tuple containing the mutated individual.
        """
        # Perform Gaussian mutation
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        # Apply boundary constraints element-wise
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    # Register the bounded mutation operator
    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    # Register the selection operator (NSGA-II selection)
    toolbox.register("select", tools.selNSGA2)

    # Create the initial population
    pop = toolbox.population(n=popsize)
    # Evaluate the fitness of the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Run the NSGA-II algorithm
    # mu+lambda algorithm: mu individuals are selected from the population and reproduce
    # with lambda offspring. The next generation is selected from the union of mu and lambda individuals.
    algorithms.eaMuPlusLambda(pop, toolbox, mu=popsize, lambda_=popsize, cxpb=cxpb, mutpb=mutpb, ngen=ngen, verbose=False)
    # Return the first front of non-dominated individuals (Pareto front)
    return tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

def run_single(obj_func, popsize, ngen, cxpb, mutpb, lows, highs, *args):
    """
    Runs a single-objective genetic algorithm (Simple Evolutionary Algorithm).

    Args:
        obj_func (function): The objective function to minimize.
        popsize (int): Population size.
        ngen (int): Number of generations.
        cxpb (float): Crossover probability.
        mutpb (float): Mutation probability.
        lows (np.ndarray): Array of lower bounds for material amounts.
        highs (np.ndarray): Array of upper bounds for material amounts.
        *args: Additional arguments to pass to the objective function.

    Returns:
        creator.Individual: The best individual found by the algorithm.
    """
    # Attempt to delete DEAP creator classes to prevent AttributeErrors on re-runs
    try:
        del creator.FitnessMin
        del creator.Individual
    except AttributeError:
        pass # Classes don't exist yet, which is fine

    # Define the fitness function (minimizing a single objective)
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    # Define the individual as a list with the defined fitness attribute
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    # Register a function to create random floats within the defined bounds for each material
    toolbox.register("attr_float", lambda: [random.uniform(l, h) for l, h in zip(lows, highs)])
    # Register a function to create an individual using the attr_float function
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.attr_float)
    # Register a function to create a population (list) of individuals
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register the specified single objective evaluation function
    toolbox.register("evaluate", obj_func, *args)
    # Register the crossover operator (Blend Crossover)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)

    def bounded_mutate(ind, mu, sigma, indpb):
        """
        Mutates an individual using Gaussian mutation and applies boundary constraints.

        Args:
            ind (creator.Individual): The individual to mutate.
            mu (float): Mean of the Gaussian distribution.
            sigma (float): Standard deviation of the Gaussian distribution.
            indpb (float): Independent probability of each attribute being mutated.

        Returns:
            tuple: A tuple containing the mutated individual.
        """
        # Perform Gaussian mutation
        mutated, = tools.mutGaussian(ind, mu, sigma, indpb)
        # Apply boundary constraints element-wise
        for i in range(len(mutated)):
            mutated[i] = min(max(mutated[i], lows[i]), highs[i])
        return mutated,

    # Register the bounded mutation operator
    toolbox.register("mutate", bounded_mutate, mu=0, sigma=0.1, indpb=0.2)
    # Register the selection operator (Tournament selection)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Create the initial population
    pop = toolbox.population(n=popsize)
    # Evaluate the fitness of the initial population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Create a Hall of Fame to store the best individual
    hof = tools.HallofFame(1)
    # Run the Simple Evolutionary Algorithm
    algorithms.eaSimple(pop, toolbox, cxpb=cxpb, mutpb=mutpb, ngen=ngen, halloffame=hof, verbose=False)
    # Return the best individual from the Hall of Fame
    return hof[0]

# Load data
merged_df, materials, base_amounts, costs, impact_matrix, traci_impact_cols = load_data(uploaded_file)

# --- Main Content Area Section ---
# Display data preview and optimization results here.

if merged_df is not None:
    st.success("File uploaded and data processed successfully!")

    # Input Data Preview
    st.subheader("Input Data Preview")
    st.write("This table shows the merged data loaded from your Excel file.")
    st.dataframe(merged_df)

    # --- Sidebar Controls Section ---
    # User inputs for optimization parameters and scenarios are placed here.
    st.sidebar.title("Optimization Settings")

    st.sidebar.subheader("Optimization Scenario")
    scenario_options = [
        "Optimize Cost vs GWP (Tradeoff)",
        "Optimize Cost + Combined Impact",
        "Optimize Single Impact",
        "Optimize Cost Only",
        "Minimize Total Material Quantity" # Added new scenario
    ]
    scenario = st.sidebar.selectbox(
        "Select the optimization objective(s):",
        scenario_options,
        help="Choose the optimization scenario. 'Cost vs GWP' finds a set of solutions trading off cost and Global Warming Potential. Others optimize for a single objective."
    )

    st.sidebar.subheader("Bounds Settings")
    global_dev = st.sidebar.slider(
        "Global ±% Deviation",
        0, 100, 20,
        help="Set a global percentage deviation from the base amount for all materials. The optimized amount will be within this range."
    )
    use_custom_bounds = st.sidebar.checkbox("Set per-material bounds", help="Check this box to set individual deviation bounds for each material.")

    lows = np.copy(base_amounts)
    highs = np.copy(base_amounts)
    if use_custom_bounds:
        st.sidebar.write("Set individual ±% deviation for each material:")
        for i, mat in enumerate(materials):
            dev = st.sidebar.number_input(
                f"{mat}",
                min_value=0,
                max_value=100,
                value=global_dev,
                key=f"dev_{mat}",
                help=f"Percentage deviation from the base amount for {mat}."
            )
            # Ensure lows are not negative
            lows[i] = max(0.0, base_amounts[i] * (1 - dev/100.0))
            highs[i] = base_amounts[i] * (1 + dev/100.0)
    else:
        # Ensure lows are not negative
        lows = np.maximum(0.0, base_amounts * (1 - global_dev / 100.0))
        highs = base_amounts * (1 + global_dev / 100.0)

    st.sidebar.subheader("Genetic Algorithm Parameters")
    popsize = st.sidebar.slider(
        "Population Size",
        10, 200, 50,
        help="The number of individuals (potential solutions) in each generation."
    )
    ngen = st.sidebar.slider(
        "Generations",
        10, 200, 40,
        help="The number of generations the algorithm will run for."
    )
    cxpb = st.sidebar.slider(
        "Crossover Probability",
        0.0, 1.0, 0.6,
        help="The probability that two individuals will crossover to create offspring."
    )
    mutpb = st.sidebar.slider(
        "Mutation Probability",
        0.0, 1.0, 0.3,
        help="The probability that an individual will be mutated."
    )

    selected_impact = None
    run_button_disabled = False
    if scenario == "Optimize Single Impact":
        if not traci_impact_cols:
            st.warning("No TRACI impact columns found in the data. 'Optimize Single Impact' scenario is not available.")
            run_button_disabled = True
            selected_impact = None # Ensure selected_impact is None if no columns
        else:
            selected_impact = st.selectbox(
                "Select TRACI Impact to Optimize:",
                traci_impact_cols,
                help="Choose the specific TRACI 2.1 impact category to minimize."
            )
            if selected_impact is None: # Also disable if selectbox is empty (shouldn't happen if traci_impact_cols is not empty)
                 run_button_disabled = True


    st.sidebar.subheader("Run Optimization")

    # Input validation checks
    inputs_valid = True
    if popsize <= 0:
        st.error("Population Size must be a positive integer.")
        inputs_valid = False
    if ngen <= 0:
        st.error("Generations must be a positive integer.")
        inputs_valid = False
    if not (0.0 <= cxpb <= 1.0):
        st.error("Crossover Probability must be between 0.0 and 1.0.")
        inputs_valid = False
    if not (0.0 <= mutpb <= 1.0):
        st.error("Mutation Probability must be between 0.0 and 1.0.")
        inputs_valid = False
    # Check if highs are greater than or equal to lows for all materials
    if not np.all(highs >= lows):
         st.error("Upper bounds must be greater than or equal to lower bounds for all materials.")
         inputs_valid = False
    # Additional check for Single Impact scenario if no columns
    if scenario == "Optimize Single Impact" and (not traci_impact_cols or selected_impact is None):
         inputs_valid = False # This scenario is already handled by disabling the button, but keep the logic here for consistency.

    if st.sidebar.button("Run Optimization", disabled=run_button_disabled or not inputs_valid):
        if not inputs_valid:
            st.warning("Please fix the input errors before running the optimization.")
        elif scenario == "Optimize Single Impact" and (not traci_impact_cols or selected_impact is None):
             st.warning("Cannot run 'Optimize Single Impact' without valid TRACI impact columns.")
        else:
            st.subheader(f"Running: {scenario}")

            with st.spinner("Optimizing..."):
                if scenario == "Optimize Cost vs GWP (Tradeoff)":
                    pareto = run_nsga2(popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix, traci_impact_cols)
                    results_df = pd.DataFrame([[ind.fitness.values[0], ind.fitness.values[1]] for ind in pareto],
                                      columns=["Total Cost", "Total GWP"])
                    st.subheader("Optimization Results: Pareto Front (Cost vs GWP)")
                    st.write("The table and plot below show the set of non-dominated solutions, representing the best tradeoffs between Cost and Global Warming Potential.")
                    st.dataframe(results_df)

                    # Pareto Front Plot
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results_df["Total Cost"], results_df["Total GWP"])
                    ax.set_xlabel("Total Cost ($)")
                    ax.set_ylabel("Total GWP (kg CO2-Eq)")
                    ax.set_title("Pareto Front: Cost vs GWP")
                    st.pyplot(fig)
                    plt.close(fig) # Close the figure to free memory

                    # Download Pareto front data
                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Pareto Front Data as CSV",
                        data=csv_output,
                        file_name="pareto_front_cost_gwp.csv",
                        mime="text/csv",
                        key="download_pareto_csv"
                    )


                    st.session_state.history.append({"scenario": scenario, "results": results_df, "plot": fig})

                elif scenario in ["Optimize Cost + Combined Impact", "Optimize Cost Only", "Minimize Total Material Quantity"] or (scenario == "Optimize Single Impact" and selected_impact):
                    if scenario == "Optimize Cost + Combined Impact":
                        best = run_single(evaluate_combined, popsize, ngen, cxpb, mutpb, lows, highs, costs, impact_matrix)
                        objective_value = best.fitness.values[0]
                        objective_name = "Minimized Objective Value (Cost + Combined Impact)"
                        results_title = "Optimization Results: Minimum Cost + Combined Impact"
                        results_description = "The table below shows the optimized material amounts that minimize the sum of total cost and all TRACI 2.1 impacts."
                        file_prefix = "optimized_cost_combined"

                    elif scenario == "Optimize Cost Only":
                        best = run_single(evaluate_cost_only, popsize, ngen, cxpb, mutpb, lows, highs, costs)
                        objective_value = best.fitness.values[0]
                        objective_name = "Minimized Total Cost"
                        results_title = "Optimization Results: Minimum Cost"
                        results_description = "The table below shows the optimized material amounts that minimize the total cost."
                        file_prefix = "optimized_cost_only"

                    elif scenario == "Optimize Single Impact" and selected_impact:
                        best = run_single(evaluate_single_impact, popsize, ngen, cxpb, mutpb, lows, highs, impact_matrix, selected_impact, traci_impact_cols)
                        objective_value = best.fitness.values[0]
                        objective_name = f"Minimized Total {selected_impact}"
                        results_title = f"Optimization Results: Minimum {selected_impact}"
                        results_description = f"The table below shows the optimized material amounts that minimize the total {selected_impact} impact."
                        file_prefix = f"optimized_single_impact_{selected_impact.replace(' ', '_').replace('/', '_')}"

                    elif scenario == "Minimize Total Material Quantity": # Added new scenario logic
                         best = run_single(evaluate_total_quantity, popsize, ngen, cxpb, mutpb, lows, highs) # evaluate_total_quantity doesn't need costs or impact_matrix
                         objective_value = best.fitness.values[0]
                         objective_name = "Minimized Total Material Quantity"
                         results_title = "Optimization Results: Minimum Total Material Quantity"
                         results_description = "The table below shows the optimized material amounts that minimize the total quantity of all materials."
                         file_prefix = "optimized_total_quantity"


                    results_df = pd.DataFrame({
                        "Material": materials,
                        "Base Amount": base_amounts,
                        "Optimized Amount": best
                    })

                    st.subheader(results_title)
                    st.write(results_description)
                    st.metric(objective_name, f"{objective_value:.2f}")
                    st.dataframe(results_df)

                    # Bar chart for single objective results
                    fig, ax = plt.subplots(figsize=(12, 6))
                    bar_width = 0.35
                    x = np.arange(len(materials))
                    ax.bar(x - bar_width/2, results_df["Base Amount"], bar_width, label='Base Amount', color='skyblue')
                    ax.bar(x + bar_width/2, results_df["Optimized Amount"], bar_width, label='Optimized Amount', color='lightcoral')
                    ax.set_ylabel("Amount")
                    ax.set_title(f"Base vs. Optimized Material Amounts ({scenario})")
                    ax.set_xticks(x)
                    ax.set_xticklabels(materials, rotation=45, ha="right")
                    ax.legend()
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig) # Close the figure to free memory

                    # Download single objective results data
                    csv_output = results_df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="Download Optimization Results as CSV",
                        data=csv_output,
                        file_name=f"{file_prefix}_results.csv",
                        mime="text/csv",
                        key=f"download_{file_prefix}_csv"
                    )

                    st.session_state.history.append({"scenario": scenario, "results": results_df, "plot": fig})


    st.sidebar.markdown("---") # Separator
    st.sidebar.subheader("Download Data")

    if st.sidebar.button("Download Merged Data as Excel"):
        output = BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            merged_df.to_excel(writer, index=False, sheet_name='Merged Data')
        output.seek(0)
        st.download_button(
            label="Download Excel File",
            data=output,
            file_name="merged_data.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            help="Download the combined input, cost, and impact data as an Excel file."
        )

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    # --- Optimization History Section ---
    # Display previous optimization runs and results here.
    st.subheader("Optimization History")
    st.write("View the results of previous optimization runs.")
    with st.expander("\U0001F4C8 View Optimization History"):
        if st.session_state['history']:
            for i, record in enumerate(st.session_state['history'], 1):
                st.write(f"**Run {i}: {record['scenario']}**")
                st.dataframe(record['results'])
                # Re-plot the saved figure from history if available
                if "plot" in record and record["plot"] is not None:
                     # Create a new figure for display from saved data
                     if record['scenario'] == "Optimize Cost vs GWP (Tradeoff)":
                         fig, ax = plt.subplots(figsize=(10, 6))
                         ax.scatter(record['results']["Total Cost"], record['results']["Total GWP"])
                         ax.set_xlabel("Total Cost ($)")
                         ax.set_ylabel("Total GWP (kg CO2-Eq)")
                         ax.set_title("Pareto Front: Cost vs GWP")
                         st.pyplot(fig)
                         plt.close(fig)
                     else: # Single objective plots
                          fig, ax = plt.subplots(figsize=(12, 6))
                          bar_width = 0.35
                          x = np.arange(len(record['results']["Material"]))
                          ax.bar(x - bar_width/2, record['results']["Base Amount"], bar_width, label='Base Amount', color='skyblue')
                          ax.bar(x + bar_width/2, record['results']["Optimized Amount"], bar_width, label='Optimized Amount', color='lightcoral')
                          ax.set_ylabel("Amount")
                          ax.set_title(f"Base vs. Optimized Material Amounts ({record['scenario']})")
                          ax.set_xticks(x)
                          ax.set_xticklabels(record['results']["Material"], rotation=45, ha="right")
                          ax.legend()
                          plt.tight_layout()
                          st.pyplot(fig)
                          plt.close(fig)

        else:
            st.info("No optimization runs recorded yet.")
else:
    st.info("Upload a valid Excel file to begin.")
