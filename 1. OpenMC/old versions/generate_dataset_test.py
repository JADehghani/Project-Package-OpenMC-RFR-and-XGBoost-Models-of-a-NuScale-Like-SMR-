#!/usr/bin/env python3
import glob
import os
from pathlib import Path
import subprocess
import pandas as pd
import openmc
from scipy.stats import qmc
from tqdm import tqdm
from iapws import IAPWS97 #imports thermodynamic library for water density calculation
import sys

BASE_DIR = Path(__file__).parent.resolve()
SCRIPT = BASE_DIR / 'build-core-fresh-test.py'
CORE_DIR = BASE_DIR / 'core-fresh'

# Link to endfb71 nuclear cross sections 
cross_section_path = os.path.abspath("endfb71/endfb-vii.1-hdf5/cross_sections.xml")
os.environ['OPENMC_CROSS_SECTIONS'] = cross_section_path

#starts stopwatch
import time
script_start_time = time.time()
total_duration = 0

print("Initializing Latin Hypercube Sampling...")

# 1. GENERATE THE 4D MATHEMATICAL MATRIX
num_samples = 200
sampler = qmc.LatinHypercube(d=4)#4, for each input
sample_matrix = sampler.random(n=num_samples)

# Set the physical reactor bounds
lower_bounds = [600.0, 580.0, 550.0, 0.01]
upper_bounds = [1200.0, 700.0, 615.0, 2000.0]
scaled_matrix = qmc.scale(sample_matrix, lower_bounds, upper_bounds)

# Unpack the matrix into the variables
f_temps = scaled_matrix[:, 0]  # Fuel Temp
c_temps = scaled_matrix[:, 1]  # Clad Temp
m_temps = scaled_matrix[:, 2]  # Mod Temp
b_ppms  = scaled_matrix[:, 3]  # Boron ppm

results = [] #initialise results matrix

print(f"Starting dataset generation for {num_samples} sample size...")

# 2. THE FACTORY AUTOMATION LOOP
for i in tqdm(range(num_samples), desc="Simulating"):
    
    #Start Lap Timer 
    run_start_time = time.time()

    #calculate water density based on temperature
    water = IAPWS97(T=m_temps[i], P=15.51) # nominal primary system pressure
    m_dens = round(water.rho * 0.001, 3)

    # Writes the current run's variables to the messenger file
    with open("ml_inputs.txt", "w") as f:
        f.write(f"{f_temps[i]},{c_temps[i]},{m_temps[i]},{m_dens},{b_ppms[i]}")
    print(f"\nRun {i}: ml_inputs.txt written")

    # B. Command the computer to build the core (this triggers the edit to materials.xml through buil-core-fresh.py)
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=BASE_DIR)
    print(f"Run {i}: build script returncode={result.returncode}")
    print(f"Run {i}: build stdout={result.stdout[-500:]}")  # last 500 chars
    print(f"Run {i}: build stderr={result.stderr[-500:]}")
    if result.returncode != 0:
        continue
    
    print(f"Run {i}: CORE_DIR exists = {CORE_DIR.exists()}")
    print(f"Run {i}: CORE_DIR contents = {list(CORE_DIR.iterdir()) if CORE_DIR.exists() else 'N/A'}")

    result = subprocess.run(["openmc"], capture_output=True, text=True, cwd=str(CORE_DIR))
    print(f"Run {i}: openmc returncode={result.returncode}")
    print(f"Run {i}: openmc stdout={result.stdout[-500:]}")
    print(f"Run {i}: openmc stderr={result.stderr[-500:]}")
    if result.returncode != 0:
        continue

# Read statepoint from absolute path
    
    sp_files = glob.glob(str(CORE_DIR / 'statepoint.*.h5'))
    if not sp_files:
        print(f"\nRun {i}: No statepoint file found")
        continue
    sp_path = sorted(sp_files)[-1]
    
# D. Extract just the k-effective from the massive output file
    
    try:
        sp = openmc.StatePoint(sorted(sp_files)[-1])
        keff = sp.keff.nominal_value
        std =  sp.keff.std_dev
        sp.close()
    except Exception as e:
        print(f"\nError reading results: {e}")
        keff, std = None, None
        
# Cleanup using absolute paths
    for pattern in ['statepoint.*.h5', 'summary.h5', 'tallies.out']:
        for f in glob.glob(str(CORE_DIR / pattern)):
            os.remove(f)
    if keff is None:
        continue

    # stops stopwatch
    run_duration = time.time() - run_start_time
    total_duration = total_duration + run_duration

    # E. Save the data to our bucket
    results.append({
        'Fuel_Temp_K': f_temps[i],
        'Clad_Temp_K': c_temps[i],
        'Mod_Temp_K': m_temps[i],
        'Mod_Density_gcc': m_dens,
        'Boron_ppm': b_ppms[i],
        'K_eff': keff,
        'std': std,
        'Compute_Time_sec': round(run_duration, 2)
    })
    
    # F. Delete the heavy binary file to save hard drive space, then step back out

# 3. SAVE THE FINAL CSV
df = pd.DataFrame(results)
df.to_csv("rfr_dataset.csv", index=False)

#print important values regarding compute time
print(f"Total Runs Completed : {len(results)}")
print(f"Total Wall-Clock Time: {round(total_duration / 60, 2)} minutes")
if not df.empty:
    print(f"Average Time per Run : {round(df['Compute_Time_sec'].mean(), 2)} seconds")
else:
    print("No successful runs — check error messages above.")
