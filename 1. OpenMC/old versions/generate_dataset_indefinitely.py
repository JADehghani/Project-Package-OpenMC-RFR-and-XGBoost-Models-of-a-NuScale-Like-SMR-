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
import numpy as np #for rounding cr insertion to specified step heights

#assign important file paths to shortcuts
BASE_FOLDER = Path(__file__).parent.resolve()
SCRIPT = BASE_FOLDER / 'build-core-fresh-v2.py' #core builder script
CORE_FRESH = BASE_FOLDER / 'core-fresh' #output of build-core-fresh.py

# Link to endfb71 nuclear cross sections 
cross_section_path = os.path.abspath("endfb71/endfb-vii.1-hdf5/cross_sections.xml")
os.environ['OPENMC_CROSS_SECTIONS'] = cross_section_path

#start stopwatch
script_start_time = time.time()
total_duration = 0

# GENERATE 4D(for each input parameter) MATRIX
sampler = qmc.Halton(d=5, scramble=True)
run_count = 0

print("Running indefinitely — Ctrl+C to stop and save...")

# Set the physical reactor bounds

# Unpack the matrix into the variables


results = [] #initialise results matrix

print(f"Starting dataset generation for sample size:")

#for each sample 
try:
    while True:
        run_count += 1
            
        # Generate one fresh sample
        sample = qmc.scale(sampler.random(n=1), 
                          [600.0, 580.0, 550.0, 0.01, 0],
                          [1200.0, 700.0, 600.0, 1200.0, 248])[0]
            
        f_temp  = sample[0]
        c_temp  = sample[1]
        m_temp  = sample[2]
        b_ppm   = sample[3]
        cr_step = int(np.clip(np.round(sample[4]), 0, 248))
        #Start Lap Timer 
        start_time = time.time()

        #calculate water (moderator) density based on temperature
        water = IAPWS97(T=m_temp, P=12.76) # P= nominal primary system pressure from __init__.py
        m_dens = round(water.rho * 0.001, 3)

        # write the current run's variables to the messenger file (inputs.txt)
        input_path = BASE_FOLDER / "inputs.txt"
        with open(input_path, "w") as f:
            f.write(f"{f_temp},{c_temp},{m_temp},{m_dens},{b_ppm},{cr_step}")
            
        # run core builder (build-core-fresh.py)
        result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, cwd=BASE_FOLDER)
        if result.returncode != 0:
            print(f"Run {run_count}: Core builder failed! Error:\n{result.stderr}")
            continue

        #run physics engine (openmc)
        result = subprocess.run(["openmc"], capture_output=True, text=True, cwd=str(CORE_FRESH))
        if result.returncode != 0:
            print(f"Run {run_count}: OpenMC failed! Error:\n{result.stderr}")
            continue

        # Read statepoint file using glob path
        statepoint_files = glob.glob(str(CORE_FRESH / 'statepoint.*.h5'))
        if not statepoint_files:
            print(f"Run {run_count}: No statepoint file generated. Skipping...")
            continue
        sp_path = sorted(statepoint_files)[-1]
          
        # extract k-effective + std from the statepoint file (uing openMC documentation)
        try:
            sp = openmc.StatePoint(sp_path) 
            keff = sp.keff.nominal_value
            std =  sp.keff.std_dev
            sp.close()
        except Exception as e:
            print(f"\nRun {run_count}: Error reading statepoint: {e}")
            keff, std = None, None

        # Delete statepoint file using glob paths (to free up hard drive space)
        for pattern in ['statepoint.*.h5', 'summary.h5', 'tallies.out']:
            for leftover in glob.glob(str(CORE_FRESH / pattern)):
                os.remove(leftover)
        if keff is None:
            continue

        # stops stopwatch
        run_duration = time.time() - start_time
        total_duration = total_duration + run_duration

        # save the data to results matrix
        results.append({
            'Fuel_Temperature_K': f_temp, #fue; temperature 
            'Cladding_Temperature_K': c_temp, #cladding temperature
            'Moderator_Temperature_K': m_temp, #moderator temperature
            'Moderator_Density_gcc': m_dens, # Moderator/water desnsity 
            'Boron_ppm': b_ppm, #Boron ppm
            'CR_Insertion': cr_step,
            'K_eff': keff, #k effective
            'standard_deviation': std, #standard deviation
            'Compute_Time_sec': round(run_duration, 2) #compute time (2DP)
        })
        print(f"Run {run_count}: keff={keff:.5f} ± {std:.5f} | CR={cr_step} | B={b_ppm:.0f}ppm")
except KeyboardInterrupt:
    print("\nStopped — saving...")

finally:
    df = pd.DataFrame(results)
    df.to_csv('dataset.csv', index=False)
    print(f"Saved {len(results)} runs to dataset.csv")
# SAVE FINAL RESULTS TO .CSV FILE

#print total duration
print(f"Total Time: {total_duration} seconds")


