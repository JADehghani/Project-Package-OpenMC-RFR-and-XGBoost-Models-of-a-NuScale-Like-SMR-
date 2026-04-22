#!/usr/bin/env python3
"""
Extreme Gradient Booster Training and Hyperparameter Optimization for Nuclear Reactor Surrogate Modeling

This script develops and optimizes an Extreme Gradient Boosting (XGBoost) surrogate model for predicting the neutron 
multiplication factor (k-effective) of a NuScale-like small modular reactor. The model is trained on a Latin Hypercube 
Sampled dataset based on a verified OpenMC model and employs GridSearchCV with 5-fold cross-validation to identify the
optimal hyperparameters.

Methodology:
    - Loads cleaned Monte Carlo dataset containing five input features and k-effective target 
    - Defines nuclear specific scoring metric (RMSE in pcm, standard for reactor physics)
    - Implements GridSearchCV across a comprehensive hyperparameter space including:
        * Number of estimators (1000, 1500, 2000)
        * Learning rate (0.01, 0.03, 0.05)
        * Maximum tree depth (3, 4, 5)
        * Subsample fraction (0.5, 0.65, 0.8)
    - Performs 5-fold cross-validation to ensure robust generalization performance
    - Extracts and validates the best-performing model hyperparamters
    - Saves the optimized model for deployment using joblib
    - Visualizes feature importance using Gini impurity for interpretability

Additional Files:
    - dataset_cleaned.csv (cleaned training dataset from generate_dataset.py)
    - xgb_k_eff_surrogate.pkl (trained model joblib output)
    - Chart_Feature_Importance_XGB.png (feature importance visualization output, using matplotlib)

Author: Joseph Dehghani - 201725400
Course: Individual Project, Civil Engineering, The University of Liverpool
Date: 31/03/2026

References:
    - Bisong, E. (2019). Introduction to Scikit-learn. Building machine learning and deep learning models on google 
    cloud platform: a comprehensive guide for beginners. available at: 
    https://link.springer.com/chapter/10.1007/978-1-4842-4470-8_18 (Accessed: 31 March 2026).
    - Virtanen, P., et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. 
    available at: https://www.nature.com/articles/s41592-019-0686-2.pdf (Accessed: 31 March 2026).
    - Chen, T., et al. (2019). Package ‘xgboost’.available at: 
    http://r.meteo.uni.wroc.pl/web/packages/xgboost/xgboost.pdf (Accessed: 31 March 2026).
    - Alizadeh, R., et al. (2020). Managing computational complexity using surrogate models: a critical 
    review. available at: https://link.springer.com/article/10.1007/s00163-020-00336-7 (Accessed: 31 March 2026).
"""

# Required Imports
import numpy as np # handles the involved matrices
import pandas as pd # handles csv dataset
import joblib # For Saving the XGB model
import time # For precise timing (as a measure of computational demand)
from xgboost import XGBRegressor # import the XGBoost algorithm
from sklearn.model_selection import KFold, GridSearchCV # import cross validation tools
from sklearn.metrics import make_scorer, mean_squared_error # for scoring metrics (Uncertainty)
import matplotlib.pyplot as plt # for feature importance plot

# ---- Nuclear Specific Performance Metric ---- #
# Function calculates error in pcm (parts per hundred thousand, good for nuclear reactor applications). 
# Arguments:
#     y_true: True k-effective values from Monte Carlo simulations
#     y_pred: Predicted k_eff values from surrogate model
#    Returns:
#        RMSE error expressed in pcm units (1 pcm = 0.00001 Δk_eff)
# -------------------------------------------- #

def custom_domain_metric(y_true, y_pred):
    
    mse = mean_squared_error(y_true, y_pred) # set arguments
    rmse = np.sqrt(mse)
    pcm_error = rmse * 100000  # Convert fraction k_eff error to pcm
    return pcm_error

# make pcm understandable for GridSearchCV, lower pcm is a better model 
scorer = make_scorer(custom_domain_metric, greater_is_better=False)

# ---- Data Loading and Preprocessing ---- #
# Loads the cleaned Monte Carlo dataset using pandas and separates features from the target. The dataset 
# contains input features: fuel temperature, cladding temperature, moderator temperature, boron concentration, 
# and control rod step. With target k_eff.
# ---------------------------------------- #

df = pd.read_csv('dataset_cleaned.csv') # load cleaned dataset
X = df.drop(columns=['K_eff'])  # Feature matrix: all columns except target
y = df['K_eff']  # Target matrix: k_eff
feature_names = X.columns.tolist()  # Keep feature names for importance analysis

# ---- Model Initialization and Cross-Validation Strategy ---- #
# Configures Extreme Gradient Boosting with a set random state for reproducibility (Chen, 2019), and sets up 
# 5-fold cross-validation to prevent bias in the distribution of data (Bisong, 2019).
# ------------------------------------------------------------ #

# Initialize base XGB model (random state = 42)
xgb = XGBRegressor(random_state=42, objective='reg:squarederror', n_jobs=1)
# 5-fold cross-validation, training on 4 chunks and testing on 1 (random state = 42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ---- Hyperparameter Definition ---- #
# Defines a comprehensive grid of hyperparameters to check. Tuning controls the tree complexity and
# prevents overfitting on small datasets (Alizadeh, et al. 2020)
# ------------------------------------------------ #

param_grid = {
    'n_estimators': [1000, 1500, 2000],  # Number of trees / sequential correction steps
    'learning_rate': [0.01, 0.03, 0.05], # Fine-tuning the step size for corrections
    'max_depth': [3, 4, 5],              # Maximum tree depth / complexity
    'subsample': [0.5, 0.65, 0.8]        # Fraction of data used per tree (Heavy regularization)
}

# ---- Grid Search Execution ---- #
# Performs hyperparameter optimization using 5-fold cross-validation. Each configuration is tested across 
# all 5 folds, and the combination with the lowest cross-validated pcm error is selected as the best model.
# ------------------------------- #

# set up GridSearch cross validation model
grid_search = GridSearchCV(
    estimator=xgb, # asign base model as estimator
    param_grid=param_grid, # sets the hyperparameter options from param_grid
    cv=kf, # sets cross validator as K-fold 
    scoring=scorer,  # Use nuclear specific pcm scoring metric
    n_jobs=-1  #  Uses all available processors for faster computation
)

# Start training stopwatch
print("Starting GridSearch...")
start_time = time.perf_counter()  

 # Execute hyperparameter optimization / gridsearch with feature and target matrices
grid_search.fit(X, y) 

# Stop training stopwatch
end_time = time.perf_counter()  

# Calculate and display total training time
training_time = end_time - start_time
print(f"GridSearch Completed in {training_time} seconds.\n")

# ---- Saving the Model ---- #
# Retrieves the best hyperparameter configuration and reports its cross-validated performance. CV provides 
# an estimate of generalization error on unseen reactor states. (Bisong, 2019)
# ----------------------------------------- #

best_xgb = grid_search.best_estimator_  # Extract optimized model
print(f"Best Hyperparameters: {grid_search.best_params_}") # display best hyperparameters

# Extract cross-validation score (GridSearchCV stores errors as negative) (Bisong, 2019)
best_cv_pcm_error = abs(grid_search.best_score_)
print(f"Cross-Validation Score (RMSE): {best_cv_pcm_error:.1f} pcm") # display best RMSE (pcm)

# ---- Model Serialization ---- #
# Saves the trained surrogate model using joblib in a pickle file for future deployment in tests.
# ----------------------------- #

joblib.dump(best_xgb, 'xgb_k_eff_surrogate.pkl')
print("\nModel saved as pickle file: 'xgb_k_eff_surrogate.pkl'")

# ---- Feature Importance Visualization ---- #
# Generates a bar chart displaying the relative importance of each reactor parameter in predicting k_eff. 
# Gini impurity measures importance across all trees, providing insight into which features influence 
# reactor criticality the most. Created and optimised using 'Claude.ai'.
# ------------------------------------------ #

importances = best_xgb.feature_importances_  # Extract Gini importance scores
indices = np.argsort(importances)  # Sort features by importance (ascending)

# Configure global matplotlib font settings
plt.rcParams.update({
    'font.weight': 'bold',
    'axes.labelweight': 'bold',
    'axes.titleweight': 'bold',
    'figure.titleweight': 'bold'
})

plt.figure(figsize=(9, 6), facecolor='white')

# Create horizontal bar chart
plt.barh(range(len(indices)), importances[indices], color='#005A9C', align='center', edgecolor='black', linewidth=0.5)

# Map feature names to clean presentation labels
clean_labels = {
    'Fuel_Temperature_K': 'Fuel Temp',
    'Cladding_Temperature_K': 'Clad Temp',
    'Moderator_Temperature_K': 'Mod Temp',
    'Boron_ppm': 'Boron Conc',
    'CR_Step': 'CR Step'
}

# Apply clean labels to y-axis
display_names = [clean_labels.get(feature_names[i], feature_names[i]) for i in indices]
plt.yticks(range(len(indices)), display_names, fontsize=11, fontname='sans-serif')
plt.xlabel('Relative Importance (Gini)', fontsize=12, fontname='sans-serif')
plt.title('B: XGBoost Feature Importance', fontsize=14, pad=15, fontname='sans-serif')

# Add subtle grid lines for improved readability
plt.grid(axis='x', linestyle='--', alpha=0.6, color='#A6A6A6')

# Remove top and right spines for clean style
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Feature_Importance_XGB.png', dpi=300, bbox_inches='tight')
print("Saved Feature Importance Chart")
plt.show()