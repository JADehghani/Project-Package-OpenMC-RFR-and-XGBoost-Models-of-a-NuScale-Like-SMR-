# Project-Package-OpenMC-RFR-and-XGBoost-Models-of-a-NuScale-Like-SMR-
# SMR Surrogate Modelling: OpenMC to Machine Learning Pipeline

## Overview
This repository contains the complete computational pipeline developed to train, validate, and execute machine learning surrogate models (Random Forest and XGBoost) for predicting the effective multiplication factor (k_eff) of a Small Modular Reactor (SMR). The project successfully maps complex, non-linear reactor physics to achieve microsecond inference times, resolving the computational bottlenecks of traditional Monte Carlo simulations.

## Pipeline Architecture
The project is divided into three sequential stages, each contained within its respective directory:

### 1. OpenMC (Data Generation)
Generates the high-fidelity baseline training data using stochastic Monte Carlo simulations.
* `generate_dataset.py`: The overarching script that executes the Latin Hypercube Sampling (LHS) across the operational parameter space.
* `build-core-fresh-v2.py`: The core builder script defining the physical geometry, materials, and settings for the OpenMC simulation.
* `/smr` & `/results`: Directories containing the geometry outputs and raw simulation state points.

### 2. Data Cleaning
Processes the raw OpenMC outputs into a machine-learning-ready format.
* `Raw_data_cleaner.py`: Extracts the input parameters and resulting k_eff (with statistical uncertainties) from the OpenMC state points, removing failed simulation runs.
* `dataset_cleaned.csv`: The finalized, structured dataset used for ML training.
* `/cleaned dataset Plots`: Visualizations of the data distribution and feature sparsity.

### 3. Surrogate Models
Trains, optimizes, and tests the tree-based machine learning models.
* `Surrogate_model_test.py`: The main testing script that evaluates the trained models against an unseen 200-sample dataset (`200_test_dataset.csv`).
* `/XGB` & `/RFR`: Directories containing the respective training scripts and the serialized, optimized model weights (`.pkl` files) ready for instant deployment.
* `/test dataset plots`: Outputs including parity plots, violin error distributions, and Gini importance rankings.

## Getting Started
Each primary directory contains a dedicated `User Guide.txt` with specific execution instructions and environment requirements. To replicate the study, navigate through the folders sequentially (1 -> 2 -> 3), executing the overarching scripts as defined in the guides. Included `Outputs.txt` files provide the expected terminal outputs for verification.

## Requirements
* Python 3.8+
* OpenMC (Nuclear Data Libraries: ENDF/B-VII.1 or later)
* XGBoost, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
