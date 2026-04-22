#!/usr/bin/env python3
"""
Diverse High-Fidelity Dataset Generation for Training a ML Surrogate Model

This script generates a training dataset for a machine learning surrogate model of a NuScale-like small modular reactor 
(SMR), using k-effective as the target. It employs Latin Hypercube Sampling (LHS) to provide diverse combinations of input 
parameters: fuel temperature, cladding temperature, moderator temperature, boron concentration and Bank D control rod 
insertion step.

Methodology:
    -LHS creates diverse random combinations of inputs for each sample in 'num_samples'
    -Iterative loop for each sample:
        -calculates water density using IAPWS97 thermodynamic library
        -writes current samples inputs to 'inputs.txt' to be passed to the code injection in 'build-core-fresh-v2.py'
        -runs 'build-core-fresh-v2' which outputs the OpenMC reactor geometry .xml files to the 'core-fresh' folder 
        -runs the OpenMC simulation
        -pulls k-effective and standard deviation from OpenMC statepoint output file (.h5)
        -appends results to the 'results' matrix
    -Finally, reformats the 'results' matrix using pandas and outputs to a .csv file
    -Note: The script contains multiple features such as try/except/finally blocks to ensure any run time errors do not 
    effect the simulation of the other samples. An interrupt mid run still saves the computed results to a .csv file

Additional Files:
    -ENDF/B-VII.1 nuclear cross sections
    -build-core-fresh-v2.py (the MIT ExaSMR benchmark core builder script with input injection block)
    -/smr (part of the MIT ExaSMR benchmark)

Author: Joseph Dehghani - 201725400
Course: Individual Project, Civil Engineering, The University of Liverpool
Date: 31/03/2026

References:
    -NuScale Power LLC (2016). NuScale standard plant design certification application, part 2 - tier 2, chapter 4: reactor 
    (Revision 0), Available at: https://www.nrc.gov/docs/ML2509/ML25099A249.pdf (Accessed: 31 March 2026)
    -Romano, P.K. et al. (2024) OpenMC (Version 0.14.0) [Computer program]. Available at: https://openmc.org/ (Accessed: 
    1 April 2026).
    -Virtanen, P. et al. (2020) 'SciPy 1.0: fundamental algorithms for scientific computing in Python', Nature Methods, 
    17(3). Available at: https://doi.org/10.1038/s41592-019-0686-2.
"""

# Required Imports
import os
import sys 
import subprocess #needed for openmc, to run core-fresh.py script
import openmc #the physics engine
from pathlib import Path #quick file handling (from MIT scripts)
import glob #allows searching for files with * (to make batch no. easily editable)
import pandas as pd #used to save results in a "dataframe"
from scipy.stats import qmc #scipy's Quasi-Monte-Carlo (for the latin hypercube)
from iapws import IAPWS97 #thermodynamic library for water density calculation
from tqdm import tqdm #creates progress bar to visulaise compute progress
import time #for timer
import numpy as np #for rounding cr insertion to specified step heights

# ---- Assigning File Paths ---- #
# All files relative to the location of the python scripts (/OpenMC)
# ------------------------------ #

# Required file paths 
BASE_FOLDER = Path(__file__).parent.resolve() # The base OpenMC folder
SCRIPT = BASE_FOLDER / 'build-core-fresh-v2.py' # Core builder script path
CORE_FRESH = BASE_FOLDER / 'core-fresh' # Core geometry output folder path of build-core-fresh.py
INPUT_FILE = BASE_FOLDER / "inputs.txt" # Messanger file path in base directory 
OUTPUT_CSV = BASE_FOLDER / 'dataset.csv' # Raw dataset output path

# Nuclear cross-section library path (ENDF/B-VII.1)
cross_section_path = os.path.abspath("endfb71/endfb-vii.1-hdf5/cross_sections.xml") # Point to cross_sections.xml 
os.environ['OPENMC_CROSS_SECTIONS'] = cross_section_path # Link OpenMC to ENDF/BVII.1 nuclear cross sections


# ---- Latin Hypercube Sampling ---- #
# Generates combinations of inputs using LHS for diverse coverage of the parameter space. LHS reduces the number of 
# samples needed compared to traditional random sampling (Alizadeh et al, 2020).
# ---------------------------------- #

print("Creating Random Inputs")
num_samples = 1000 # Define the number of samples

# initiate LHS matrix
sampler = qmc.LatinHypercube(d=5) # generates 5D (for each input parameter) matrix for LHS
sample_matrix = sampler.random(n=num_samples) # generates LHS 5D matrix for all samples

# Set the physical reactor bounds (defined in section 3.2.2 of the final report) 
# [Fuel_temp, Clad_temp, Mod_temp, Boron_ppm, CR_step]
lower_bounds = [500.0, 500.0, 500.0, 0.01, 0]
upper_bounds = [1200.0, 700.0, 600.0, 1235.0,248]

# Scale the LHS output to the upper and lower boundaries
matrix = qmc.scale(sample_matrix, lower_bounds, upper_bounds) 

# Unpack the scaled matrix into the individual variables' array
f_temps = matrix[:, 0]  # Fuel Temperature array
c_temps = matrix[:, 1]  # Clad Temperature array
m_temps = matrix[:, 2]  # Mod Temperature array
b_ppms  = matrix[:, 3]  # Boron Concentration array
cr_steps_raw = matrix[:, 4] # Rod Insertion Step array
cr_steps = np.clip(np.round(cr_steps_raw).astype(int), 0, 248) # Ensure rod step is within 0-248 at specific steps

# ---- Main Simulation Loop ----#
# For each LHS sample: 1) Calculate moderator density using IAPWS library, 2) Inject parameters via inputs.txt, 3) run 
# 'build-core-fresh-v2' to build core geometry, 4) run OpenMC simulation, 5) extract keff and std, 6) cleanup files. 
# Try/except/finally structure ensures partial results are saved even if execution is interrupted (Ctrl-C or syntax error).
# ----------------------------- #

print(f"Starting simulations for sample size: {num_samples}")
results = [] # Initialise the results matrix
total_duration = 0 
 
try:
    for i in tqdm(range(num_samples), desc="Simulating"): # individually increment through each sample
        
        # Start lap timer 
        start_time = time.time()

# 1) Calculate Moderator Density

        #Based on temperature using the IAPWS97 thermodynamic library
        # P=12.76 MPa (nominal primary system pressure from __init__.py)
        water = IAPWS97(T=m_temps[i], P=12.76) 
        m_dens = round(water.rho * 0.001, 5) # kg/m^3 -> g/cm^3

# 2) Write the current sample's variables to the messenger file (inputs.txt)
    
        with open(INPUT_FILE, "w") as f: # open the inputs.txt as write
            f.write(f"{f_temps[i]},{c_temps[i]},{m_temps[i]},{m_dens},{b_ppms[i]},{cr_steps[i]}") # write current sample's parameters to inpts.txt

# 3) Run core builder 

        # Runs build-core-fresh-v2.py which reads inputs.txt and injects parameters into materials and geometry
        # Outputs reactor geometry to /core-fresh as .xml files
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=BASE_FOLDER)
        if result.returncode != 0: # In the case of an error, output the error and continue
            print(f"Run {i}: Core builder failed. Error:\n{result.stderr}")
            continue

# 4) Run physics engine

        # Executes OpenMC with materials.xml, geometry.xml, settings.xml produced by 'build-core-fresh-v2.py'
        result = subprocess.run(["openmc"], capture_output=True, text=True, cwd=str(CORE_FRESH))
        if result.returncode != 0: # In the case of an error, output the error and continue
            print(f"Run {i}: OpenMC failed. Error:\n{result.stderr}")
            continue

# 5) Extract k-effective from statepoint file

        # Find the most recent statepoint OpenMC output file (using glob paths as the name of statepoint depends on batch number)
        statepoint_files = glob.glob(str(CORE_FRESH / 'statepoint.*.h5'))
        if not statepoint_files: # In the case of a file not found error, output and continue
            print(f"Run {i}: No statepoint file found")
            continue

        # Navigate to the final batch simulation
        sp_path = sorted(statepoint_files)[-1] # code will fail with >999 batches due to alphabetical sorting 
        
        # Extract k-effective + standard deviation from the .h5 statepoint file 
        try:
            sp = openmc.StatePoint(sp_path) 
            keff = sp.keff.nominal_value # Extract k-effective
            std =  sp.keff.std_dev # Extract standard deviation
            sp.close() # Close the statepoint file
        except Exception as e: # In the case of an error, output the error and continue
            print(f"\nRun {i}: Error reading statepoint file: {e}")
            keff, std = None, None

        # Stop lap timer
        run_duration = time.time() - start_time # Calculate current run duration
        total_duration = total_duration + run_duration # append the total run duration with the current run

        # Append the current sample's simulation results to the results matrix
        results.append({
            'Fuel_Temperature_K': f_temps[i], #fue; temperature 
            'Cladding_Temperature_K': c_temps[i], #cladding temperature
            'Moderator_Temperature_K': m_temps[i], #moderator temperature
            'Moderator_Density_gcc': m_dens, # Moderator/water desnsity 
            'Boron_ppm': b_ppms[i], #Boron ppm
            'CR_Step':   cr_steps[i],
            'K_eff': keff, #k effective
            'standard_deviation': std, #standard deviation
            'Compute_Time_sec': round(run_duration, 2) #compute time (2DP)
        })

# 6) Delete large output files to conserve disk space

        # Removes statepoint.*.h5, summary.h5, tallies.out
        # (Results already extracted and stored in 'results' matrix)
        for pattern in ['statepoint.*.h5', 'summary.h5', 'tallies.out']: # Iterates through each OpenMC output 
            for leftovers in glob.glob(str(CORE_FRESH / pattern)):
                os.remove(leftovers) # Deletes all large OpenMC outputs
        if keff is None: # If k-effective cannot be extracted, continue
            continue

        # Output results of completed run
        print(f"Run {i+1}/{num_samples}: keff={keff:.5f} ± {std:.5f} | CR={cr_steps[i]}/248 | Run Time={run_duration}")

# Exception stops the for loop mid-run when interrupted (press Ctrl-C while running) to ensure results are not lost on a failed run
except KeyboardInterrupt:
    print("\nInterrupted, Saving Results") 

# ---- Exporting Results ---- #
# Exports results to CSV even if execution was interrupted. Provides preliminary summary statistics.
# --------------------------- #

finally: # Finally block ensures results are always saved even if the main loop is interrupted
    df = pd.DataFrame(results) #use pandas to create a 'spreadsheet' with results matrix
    df.to_csv(OUTPUT_CSV, index=False) #saves 'spreadsheet' to dataset.csv

    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total Completed Runs: {len(results)} / {num_samples}") # Output total completed runs
    print(f"Total Compute Time: {total_duration} seconds") # Output the total run duration

