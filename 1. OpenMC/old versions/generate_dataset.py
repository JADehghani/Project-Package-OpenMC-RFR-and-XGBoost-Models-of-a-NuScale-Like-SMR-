#!/usr/bin/env python3
import os #operating system
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

#assign important file paths to shortcuts
BASE_FOLDER = Path(__file__).parent.resolve()
SCRIPT = BASE_FOLDER / 'build-core-fresh.py' #core builder script
CORE_FRESH = BASE_FOLDER / 'core-fresh' #output of build-core-fresh.py

# Link to endfb71 nuclear cross sections 
cross_section_path = os.path.abspath("endfb71/endfb-vii.1-hdf5/cross_sections.xml")
os.environ['OPENMC_CROSS_SECTIONS'] = cross_section_path

#start stopwatch
script_start_time = time.time()
total_duration = 0

# GENERATE 4D(for each input parameter) MATRIX
print("Creating Random Inputs")
num_samples = 200
sampler = qmc.LatinHypercube(d=4)
sample_matrix = sampler.random(n=num_samples)

# Set the physical reactor bounds
lower_bounds = [600.0, 580.0, 550.0, 0.01]
upper_bounds = [1200.0, 700.0, 615.0, 1200.0]
matrix = qmc.scale(sample_matrix, lower_bounds, upper_bounds)

# Unpack the matrix into the variables
f_temps = matrix[:, 0]  # Fuel Temp
c_temps = matrix[:, 1]  # Clad Temp
m_temps = matrix[:, 2]  # Mod Temp
b_ppms  = matrix[:, 3]  # Boron ppm

results = [] #initialise results matrix

print(f"Starting dataset generation for sample size: {num_samples}")

#for each sample 
for i in tqdm(range(num_samples), desc="Simulating"):
    
    #Start Lap Timer 
    start_time = time.time()

    #calculate water (moderator) density based on temperature
    water = IAPWS97(T=m_temps[i], P=15.51) # P=15.51 nominal primary system pressure
    m_dens = round(water.rho * 0.001, 3)

    # write the current run's variables to the messenger file (inputs.txt)
    with open("inputs.txt", "w") as f:
        f.write(f"{f_temps[i]},{c_temps[i]},{m_temps[i]},{m_dens},{b_ppms[i]}")

    # run core builder (build-core-fresh.py)
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=BASE_FOLDER)
    if result.returncode != 0:
        continue

    #run physics engine (openmc)
    result = subprocess.run(["openmc"], capture_output=True, text=True, cwd=str(CORE_FRESH))
    if result.returncode != 0:
        continue

    # Read statepoint file using glob path
    statepoint_files = glob.glob(str(CORE_FRESH / 'statepoint.*.h5'))
    sp_path = sorted(statepoint_files)[-1]
      
    # extract k-effective + std from the statepoint file (uing openMC documentation)
    try:
        sp = openmc.StatePoint(sorted(sp_files)[-1]) 
        keff = sp.keff.nominal_value
        std =  sp.keff.std_dev
        sp.close()
    except Exception as e:
        keff, std = None, None

    # Delete statepoint file using glob paths (to free up hard drive space)
    for pattern in ['statepoint.*.h5', 'summary.h5', 'tallies.out']:
        for f in glob.glob(str(CORE_FRESH / pattern)):
            os.remove(f)
    if keff is None:
        continue

    # stops stopwatch
    run_duration = time.time() - start_time
    total_duration = total_duration + run_duration

    # save the data to results matrix
    results.append({
        'Fuel_Temperature_K': f_temps[i], #fue; temperature 
        'Cladding_Temperature_K': c_temps[i], #cladding temperature
        'Moderator_Temperature_K': m_temps[i], #moderator temperature
        'Moderator_Density_gcc': m_dens, # Moderator/water desnsity 
        'Boron_ppm': b_ppms[i], #Boron ppm
        'K_eff': keff, #k effective
        'standard_deviation': std, #standard deviation
        'Compute_Time_sec': round(run_duration, 2) #compute time (2DP)
    })
    
# SAVE FINAL RESULTS TO .CSV FILE
df = pd.DataFrame(results) #use pandas to create a 'spreadsheet' with results
df.to_csv("dataset.csv", index=False) #saves 'spreadsheet' to dataset.csv

#print values regarding compute time
print(f"Total Completed Runs: {len(results)} / {num_samples}")
print(f"Total Time: {total_duration} seconds")

