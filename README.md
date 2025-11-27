# FEniCS Data Generation Module

This directory contains the full numerical pipeline used to generate the training dataset for the benzene concentration forecasting PINN. All simulations were produced with FEniCS by solving the two-dimensional transient advection–diffusion–reaction (ADR) equation under realistic atmospheric conditions, variable source strengths, and scenario-dependent wind fields.

## Overview

A total of **400 independently parameterized scenarios** were simulated. Each scenario varied:

* Horizontal wind vector fields
* Diffusion coefficients derived from Pasquill–Gifford stability
* First-order chemical decay
* Gaussian source intensity and spread

The solver is executed on a high-resolution spatial mesh and fine temporal grid. Across all scenarios, the workflow produced **~42 million spatiotemporal concentration samples**, forming the ground-truth dataset for supervising the PINN and defining the solution space the deployed model must generalize across.

## Governing Equation

The code solves the ADR equation in conservative form:

$$
\frac{\partial \phi}{\partial t} + \mathbf{u}\cdot\nabla \phi - D \nabla^2 \phi + k_1 \phi = S(x,y)
$$

Where:
* $\phi(x,y,t)$: Concentration
* $\mathbf{u}=(u,v)$: Scenario-specific wind field
* $D$: Diffusion coefficient (stability-dependent)
* $k_1 = 1\times10^{-5} \text{ s}^{-1}$: First-order decay
* $S(x,y)$: Gaussian emission source

The source term is implemented as:

$$
S(x,y)=Q_0\exp\left[-\frac{(x-x_s)^2+(y-y_s)^2}{2\sigma^2}\right], \quad Q_0=\frac{Q_{\text{total}}}{2\pi\sigma^2}
$$

## Boundary Conditions

The weak form employs **do-nothing** transport boundaries:
* **Outflow** ($u \cdot n > 0$): mass exits the domain
* **Inflow** ($u \cdot n < 0$): no pollutant enters the domain

This is encoded through the boundary integral:

$$
\int_{\partial\Omega} (u_n^+) \phi v \, ds, \quad u_n^+=\max(\mathbf{u}\cdot\mathbf{n},0)
$$

## Summary

This module integrates the full ADR system with dynamic diffusion, Gaussian emissions, chemical decay, and wind-aligned outflow boundaries, stabilized via SUPG. The resulting dataset constitutes the physics-consistent reference used to train the high-speed surrogate PINN and underpins all downstream forecasting tasks.

A rendered example of a scenario output is shown below:
<img width="1354" height="1179" alt="image" src="https://github.com/user-attachments/assets/23238be9-970a-4fca-ae56-8057a8ce4517" />
