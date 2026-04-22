#!/usr/bin/env python3
"""
Surrogate Model Test and Performance Benchmark for Nuclear Reactor K-Effective Prediction

This script compares the predictive performance of a Random Forest Regressor (RFR) and an 
Extreme Gradient Boosting (XGBoost) surrogate models against a 200-sample unseen Monte Carlo test dataset (OpenMC). 
It calculates error in pcm, records computational demand, and generates a parity plot for a NuScale-like small 
modular reactor (SMR) (the MIT ExaSMR Benchmark model).

Methodology:
    - Loads previously optimized RFR and XGBoost models using joblib
    - Loads a 200-sample test dataset with five reactor input features and OpenMC k_eff ground truth
    - Isolates the relevant features 
    - Executes the ML models to predict k_eff
    - Calculates prediction error in pcm units (standard for reactor physics)
    - Appends predictions, errors, and compute times to a CSV dataset for later comparison
    - Generates parity plots to visualise the accuracy of each model. 

Additional Files:
    - 200_test_dataset.csv (unseenground truth dataset from generate_dataset.py)
    - RFR/rfr_k_eff_surrogate.pkl (trained RFR surrogate model)
    - XGB/xgb_k_eff_surrogate.pkl (trained XGBoost surrogate model)
    - Comparison_dataset.csv (full dataset output with all predictions and errors)
    - Parity_plot.png (visual comparison plot output, using matplotlib)

Author: Joseph Dehghani - 201725400
Course: Individual Project, Civil Engineering, The University of Liverpool
Date: 31/03/2026

References:
    - Bisong, E. (2019). Introduction to Scikit-learn. Building machine learning and deep learning models on google 
    cloud platform: a comprehensive guide for beginners. available at: 
    https://link.springer.com/chapter/10.1007/978-1-4842-4470-8_18 (Accessed: 31 March 2026).
    - Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. 
    available at: https://www.nature.com/articles/s41592-019-0686-2.pdf (Accessed: 31 March 2026).
    - Alizadeh, R., et al. (2020). Managing computational complexity using surrogate models: a critical 
    review. available at: https://link.springer.com/article/10.1007/s00163-020-00336-7 (Accessed: 31 March 2026).
    - Rhinehart, R. R. (2016). Nonlinear regression modeling for engineering applications: modeling, model validation, 
    and enabling design of experiments. available at: https://onlinelibrary.wiley.com/doi/book/10.1002/9781118597972 
    (Accessed: 31 March 2026).
"""

# Required Imports
import pandas as pd # handles csv dataset
import joblib # For loading the saved ML models
import time # For timing 
import matplotlib.pyplot as plt # for generating the parity plot
import numpy as np # handles the involved matrices
import os # to create an output folder for the visual graphs

# ---- Loading the Models and Data ---- #
# Loads the trained surrogate models and the unseen 200-sample test dataset. Isolates the 
# five input features the models were trained.
# ---------------------------------------- #

# Load the Saved Surrogate Models
print("Loading Surrogate Models")
rfr_model = joblib.load('RFR/rfr_k_eff_surrogate.pkl')
xgb_model = joblib.load('XGB/xgb_k_eff_surrogate.pkl')
print("Models loaded successfully")

# Load the 200-sample test dataset
new_samples_df = pd.read_csv('200_test_dataset.csv') 

# Isolate 5 Features and ground truth k_eff from OpenMC
feature_columns = [
    'Fuel_Temperature_K', 'Cladding_Temperature_K', 
    'Moderator_Temperature_K', 'Boron_ppm', 'CR_Step'
]
X_test_features = new_samples_df[feature_columns] # feature matrix
openmc_k_eff = new_samples_df['K_eff'] # Ground truth target matrix

# ---- Model Predictions and Compute Time ---- #
# Executes predictions for both models on the feature matrix. Uses precise performance 
# counters to capture the exact time required for the surrogate model predictions.
# ---------------------------------------- #

# Predict and record compute times: Random Forest Regressor
print("Running Random Forest predictions")
start_rfr = time.perf_counter() # start RFR stopwatch
rfr_preds = rfr_model.predict(X_test_features) # predict target K_effs in RFR predictions matrix
rfr_time = time.perf_counter() - start_rfr # stop RFR stopwatch 

# 5. Predict and record compute times: XGBoost
print("Running XGBoost predictions")
start_xgb = time.perf_counter() # start XGBoost stopwatch
xgb_preds = xgb_model.predict(X_test_features) # predict target K_effs in XGBoost predictions matrix
xgb_time = time.perf_counter() - start_xgb # stop XGBoost stopwatch

# Calculate per-sample times for comparison dataset
rfr_per_sample = rfr_time / len(new_samples_df)
xgb_per_sample = xgb_time / len(new_samples_df)
print(f"\nRFR Total Time: {rfr_time:.6f} s, Per Sample: {rfr_per_sample:.8f} s")
print(f"XGB Total Time: {xgb_time:.6f} s, Per Sample: {xgb_per_sample:.8f} s")

# ---- Error Calculation and Saving---- #
# Calculates the error between ML predictions and OpenMC ground truth. Converts 
# the fraction into nuclear-specific pcm units and exports all data as a comparison CSV.
# ---------------------------------------- #

# Append Results matrix and Calculate Errors in pcm (100,000 multiplier)
# results matrix is appended to the end of the raw OpenMC dataset for easy comparison
new_samples_df['RFR_Predicted_K_eff'] = rfr_preds
new_samples_df['RFR_Error_pcm'] = abs(openmc_k_eff - rfr_preds) * 100000
new_samples_df['RFR_Compute_Time_sec'] = rfr_per_sample 

new_samples_df['XGB_Predicted_K_eff'] = xgb_preds
new_samples_df['XGB_Error_pcm'] = abs(openmc_k_eff - xgb_preds) * 100000
new_samples_df['XGB_Compute_Time_sec'] = xgb_per_sample 

# Save to the final results CSV
output_filename = 'Comparison_dataset.csv' # output file path
new_samples_df.to_csv(output_filename, index=False) # write results matrix to csv file
print(f"\npredictions and errors for both models saved to '{output_filename}'")

# ------ Matplotlib/ Seaborn Visual Plots ------ #
# initially coded using 'Claude.ai'. and altered to match the purpose and consistent style of my report

# ---- Parity Plot Visualization ---- #
# Generates a publication-ready side-by-side scatter plot with matplotlib comparing the OpenMC ground truth 
# against the surrogate predictions (Rhinehart, 2016). 
# ---------------------------------------- #

print("Generating Parity Plot...")

# globally set all font properties to match othe figures
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold'
})

# Calculate axes limits so both plots have the exact same scale
min_val = min(openmc_k_eff.min(), rfr_preds.min(), xgb_preds.min())
max_val = max(openmc_k_eff.max(), rfr_preds.max(), xgb_preds.max())
buffer = (max_val - min_val) * 0.05 
axis_min = min_val - buffer
axis_max = max_val + buffer

# Set up a side-by-side figure layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')
fig.suptitle('OpenMC Ground Truth vs. Surrogate Model Predictions', 
             fontsize=16, fontname='sans-serif', y=1.05)

# --- Subplot 1: Random Forest ---
ax1.scatter(openmc_k_eff, rfr_preds, color='black', marker='x', alpha=0.85, linewidth=1.0, s=20, label='RFR Prediction')
ax1.plot([axis_min, axis_max], [axis_min, axis_max], color='red', linestyle='--', linewidth=1.5, label='Perfect Correlation')
ax1.set_title('A: Random Forest Regressor', fontsize=13, fontname='sans-serif', pad=10)
ax1.set_xlabel('OpenMC Ground Truth keff', fontsize=11, fontname='sans-serif')
ax1.set_ylabel('Surrogate Predicted keff', fontsize=11, fontname='sans-serif')

# --- Subplot 2: XGBoost ---
ax2.scatter(openmc_k_eff, xgb_preds, color='black', marker='x', alpha=0.85, linewidth=1.0, s=20, label='XGBoost Prediction')
ax2.plot([axis_min, axis_max], [axis_min, axis_max], color='red', linestyle='--', linewidth=1.5, label='Perfect Correlation')
ax2.set_title('B: Gradient Boosting (XGBoost)', fontsize=13, fontname='sans-serif', pad=10)
ax2.set_xlabel('OpenMC Ground Truth keff', fontsize=11, fontname='sans-serif')

# Apply consistent formatting to both plots
for ax in [ax1, ax2]:
    ax.set_xlim(axis_min, axis_max)
    ax.set_ylim(axis_min, axis_max)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5, color='#A6A6A6')
    # Force the legend font to be bold as well
    prop = {'weight': 'bold', 'size': 10}
    ax.legend(loc='upper left', frameon=True, prop=prop)

# creates output folder if it doesnt exist
os.makedirs('Test Dataset Plots', exist_ok=True)

plt.tight_layout()
plt.savefig('Test Dataset Plots/Parity_plot.png', dpi=300, bbox_inches='tight')
print("Saved Parity_plot.png")
plt.show()

# ---- Error Distribution Visualization ---- #
# Generates a side-by-side violin plot comparing the absolute prediction error distributions 
# for RFR and XGBoost models (Bisong, 2019) directly revealing precision, central tendency, and outliers 
# ---------------------------------------- #

print("\nGenerating Error Distribution Plot...")

rfr_errors = new_samples_df['RFR_Error_pcm'].values
xgb_errors = new_samples_df['XGB_Error_pcm'].values

# Calculate medians for the legends
rfr_median = np.median(rfr_errors)
xgb_median = np.median(xgb_errors)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), facecolor='white')

# --- Subplot 1: Random Forest Error Distribution ---
violin1 = ax1.violinplot([rfr_errors], positions=[1], widths=0.7, 
                         showmeans=False, showmedians=True, showextrema=True)

for pc in violin1['bodies']:
    pc.set_facecolor('#595959') # Steel Gray
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)
    pc.set_alpha(0.7)
violin1['cmedians'].set_color('#D92A20') # Safety Red
violin1['cmedians'].set_linewidth(2.5)
violin1['cmaxes'].set_color('black')
violin1['cmins'].set_color('black')
violin1['cbars'].set_color('black')

ax1.set_title('A: Random Forest Regressor', fontsize=13, fontname='sans-serif', pad=10)
ax1.set_ylabel('Absolute Error (pcm)', fontsize=11, fontname='sans-serif')
ax1.set_xticks([1])
ax1.set_xticklabels(['RFR'], fontweight='bold')
ax1.grid(True, axis='y', linestyle='--', alpha=0.5, color='#A6A6A6')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Add dynamically calculated median to the legend
ax1.plot([], [], color='#D92A20', linewidth=2.5, label=f'Median Error = {rfr_median:.1f} pcm')
ax1.legend(loc='upper right', frameon=True, prop={'weight': 'bold', 'size': 10})

# --- Subplot 2: XGBoost Error Distribution ---
violin2 = ax2.violinplot([xgb_errors], positions=[1], widths=0.7,
                         showmeans=False, showmedians=True, showextrema=True)

for pc in violin2['bodies']:
    pc.set_facecolor('#005A9C') 
    pc.set_edgecolor('black')
    pc.set_linewidth(1.5)
    pc.set_alpha(0.7)
violin2['cmedians'].set_color('#D92A20') # Safety Red
violin2['cmedians'].set_linewidth(2.5)
violin2['cmaxes'].set_color('black')
violin2['cmins'].set_color('black')
violin2['cbars'].set_color('black')

ax2.set_title('B: Gradient Boosting (XGBoost)', fontsize=13, fontname='sans-serif', pad=10)
ax2.set_xticks([1])
ax2.set_xticklabels(['XGBoost'], fontweight='bold')
ax2.grid(True, axis='y', linestyle='--', alpha=0.5, color='#A6A6A6')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Add dynamically calculated median to the legend
ax2.plot([], [], color='#D92A20', linewidth=2.5, label=f'Median Error = {xgb_median:.1f} pcm')
ax2.legend(loc='upper right', frameon=True, prop={'weight': 'bold', 'size': 10})

# Synchronize y-axis limits
max_error = max(rfr_errors.max(), xgb_errors.max())
y_buffer = max_error * 0.1
for ax in [ax1, ax2]:
    ax.set_ylim(0, max_error + y_buffer)

plt.tight_layout()
plt.savefig('Test Dataset Plots/Error_distribution.png', dpi=300, bbox_inches='tight')
print("Saved Error_distribution.png")
plt.show()


# ---- Spatial Error / Dominant Input Breakdown ---- #
# Generates plots of the most dominant features against keff for all 3 models used in the comparison dataset
# directly revealing if the tree-based models are learning the non-linear underlying physics within the
# baseline dataset (Bisong, 2019)
# -------------------------------------------------- #

print("\nGenerating Model Breakdown Over Dominant Inputs Plot...")

# Apply global font standardization for consistency across Chapter 4
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold'
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), facecolor='white')

# Updated styling: Changed Ground Truth to crisp crosses ('x') and removed edgecolors='none'
gt_style = {'color': 'black', 'marker': 'x', 's': 35, 'alpha': 0.9, 'linewidth': 1.5, 'label': 'OpenMC (Ground Truth)', 'zorder': 3}
rfr_style = {'color': '#595959', 'marker': '^', 's': 20, 'alpha': 0.6, 'label': 'RFR Prediction', 'edgecolors': 'none', 'zorder': 2}
xgb_style = {'color': '#005A9C', 'marker': 's', 's': 20, 'alpha': 0.6, 'label': 'XGBoost Prediction', 'edgecolors': 'none', 'zorder': 2}

# --- Subplot 1: Control Rod Step (Plot A) ---
ax1.scatter(new_samples_df['CR_Step'], openmc_k_eff, **gt_style)
ax1.scatter(new_samples_df['CR_Step'], rfr_preds, **rfr_style)
ax1.scatter(new_samples_df['CR_Step'], xgb_preds, **xgb_style)

ax1.set_title('A: Control Rod Position Impact', fontsize=13, fontname='sans-serif', pad=10)
ax1.set_xlabel('Control Rod Step', fontsize=11, fontname='sans-serif', fontweight='bold')
ax1.set_ylabel('Effective Multiplication Factor ($k_{eff}$)', fontsize=11, fontname='sans-serif', fontweight='bold')

# --- Subplot 2: Boron Concentration (Plot B) ---
ax2.scatter(new_samples_df['Boron_ppm'], openmc_k_eff, **gt_style)
ax2.scatter(new_samples_df['Boron_ppm'], rfr_preds, **rfr_style)
ax2.scatter(new_samples_df['Boron_ppm'], xgb_preds, **xgb_style)

ax2.set_title('B: Boron Concentration Impact', fontsize=13, fontname='sans-serif', pad=10)
ax2.set_xlabel('Boron Concentration (ppm)', fontsize=11, fontname='sans-serif', fontweight='bold')

# Apply consistent professional formatting to both plots
for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.5, color='#A6A6A6', zorder=1) # Pushed grid to the very back
    ax.legend(loc='best', frameon=True, prop={'weight': 'bold', 'size': 9})

plt.tight_layout()
plt.savefig('Test Dataset Plots/Dominant_input_breakdown.png', dpi=300, bbox_inches='tight')
print("Saved Dominant_input_breakdown.png")
plt.show()