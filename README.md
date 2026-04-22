# Project-Package-OpenMC-RFR-and-XGBoost-Models-of-a-NuScale-Like-SMR

## Overview
This repository contains the 3 stage computational pipeline developed to train, validate, and execute tree-based ML surrogate models (Random Forest and XGBoost) for predicting the effective multiplication factor (k_eff) of a Small Modular Reactor (SMR). The project successfully captures the complex underlying reactor physics while achieving microsecond prediction times, offering a resolution to the computational bottlenecks of traditional Monte Carlo simulations.
The baseline OpenMC Model (from https://github.com/mit-crpg/ecp-benchmarks) and the ML surrogate models come with limitations discussed in Section 3.5 of the final project report (saved in the root of this repository).

## Package Layout
The project is divided into 3 stages, each contained within its respective folder:

### 1. OpenMC (Data Generation)
Generates the high-fidelity baseline training data using stochastic Monte Carlo simulations.
* `generate_dataset.py`: The overarching script that executes the Latin Hypercube Sampling (LHS) and runs OpenMC simulations for each sample.
* `build-core-fresh-v2.py`: The MIT ExaSMR benchmark core builder script defining the .xml model for the OpenMC simulation, with an additional code injection to change 5 specific input parameters.
* `/smr`: Folder containing scripts necessary for 'build-core-fresh-v2.py'. Provided in the MIT ExaSMR Package. 
* `/results`: Folder containing the output '.csv' datasets. 

### 2. Data Cleaning
Processes the raw OpenMC outputs into a format for ML model training.
* `Raw_data_cleaner.py`: Jupyter Notebook that cleans the raw dataset and produces visual plots for the baseline dataset.
* `dataset_cleaned.csv`: The finalised dataset used for ML training.
* `/Cleaned Dataset Plots`: Folder containing the plots used for assessing and validating the baseline dataset.

### 3. Surrogate Models
Trains, optimises, and tests the tree-based machine learning models.
* `/XGB` & `/RFR`: Folders containing the respective training scripts and the saved ML models (`.pkl` files) ready for deployment during testing.
* `/Test dataset plots`: Folder containing the parity plots, violin error distributions, and Gini importance rankings for assessing the performance of the surrogate models.
* `Surrogate_model_test.py`: The main testing script that evaluates all 3 SMR models against an unseen 200-sample dataset (`200_test_dataset.csv`).

## User Guides and Terminal Outputs
Each stage in the project pipeline has a dedicated `User Guide.txt` with specific execution instructions and requirements. To replicate the study, navigate through the folders sequentially (1 -> 2 -> 3), following the User Guides. The included `Outputs.txt` files provide the terminal outputs for all '.py' scripts.

## Requirements
* Python 3.14.4
* OpenMC 0.15.3 and ENDF/B-VII.1 Nuclear Cross-Section Library
* Jupyter 7.5.5 in VSCode with a stable python kernel
* XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
