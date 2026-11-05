# Vehicle Aerodynamic Drag Surrogate — Gaussian Process Regression

> Predicting drag coefficient with uncertainty-aware Gaussian Process regression

## Overview

This project demonstrates surrogate modelling for aerodynamic design exploration.

Instead of repeatedly evaluating a high-cost CFD model, a Gaussian Process Regressor is trained on a synthetic aerodynamic design dataset and used to predict drag coefficient across the design space.

## Model

The implementation uses scikit-learn's `GaussianProcessRegressor` with a Matérn 5/2 covariance kernel.

Key ideas:
- nonlinear surrogate modelling;
- uncertainty-aware prediction;
- train/test evaluation;
- design-space visualization;
- residual and prediction diagnostics.

## Data Scope

The aerodynamic dataset is **synthetic / physics-inspired**. It is intended to demonstrate surrogate-modelling methodology and does not replace validated CFD or wind-tunnel data.

The reported error metrics therefore describe performance on the generated dataset, not generalization to real vehicle aerodynamics.

## Results

Saved metrics include:
- R²;
- RMSE in drag coefficient;
- prediction diagnostics.

Generated figures are stored under `plots/`, with supporting notes under `docs/`.

## Run

```bash
git clone https://github.com/bharath153/Aero_Drag_GPR.git
cd Aero_Drag_GPR
pip install -r requirements.txt
python src/train_gpr.py
```

## Repository Structure

```text
Aero_Drag_GPR/
├── src/
│   └── train_gpr.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Engineering Interpretation

Gaussian Processes are useful in engineering optimization because they provide both a prediction and an uncertainty estimate. This repository is a compact precursor to the Bayesian-optimization workflow in [Aero_BayesOpt](https://github.com/bharath153/Aero_BayesOpt).
