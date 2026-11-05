# Technical Notes — Aerodynamic Drag Gaussian-Process Surrogate

## Purpose

This repository demonstrates uncertainty-aware surrogate modelling for an aerodynamic design objective.

## Data

The design-response dataset is synthetic / physics-inspired. It is intended to mimic a smooth nonlinear aerodynamic response surface for methodology demonstration.

## Model

The implementation uses scikit-learn's `GaussianProcessRegressor` with a Matérn-family covariance kernel.

The key advantage of the Gaussian Process is that it returns both:
- a mean prediction;
- a predictive uncertainty estimate.

## Evaluation

The repository reports held-out regression metrics and visualization of prediction quality across the generated design space.

## Limitations

The surrogate is not trained on CFD or wind-tunnel data. Reported error therefore quantifies interpolation on the generated objective landscape, not predictive accuracy for a physical vehicle.

A higher-fidelity extension would use CFD-derived design samples, geometric constraints and cross-validation across design regions.
