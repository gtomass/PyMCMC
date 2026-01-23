# PyMCMC-Phys
**A robust MCMC library for complex physical model fitting.**

PyMCMC-Phys is a Bayesian inference tool designed for scientific research. It facilitates the transition from simple curve fitting to rigorous statistical analysis by handling data correlations, multimodal distributions, and providing automatic convergence diagnostics.

---

## Core Features

* **Advanced Sampling Algorithms**: Includes DRAM (Delayed Rejection Adaptive Metropolis), AM (Adaptive Metropolis), and standard Metropolis-Hastings.
* **Parallel Tempering (MCMCMC)**: Global exploration for multimodal posterior distributions to prevent the sampler from becoming trapped in local minima.
* **Full Data Covariance Support**: Supports data covariance matrices to correctly account for correlated measurement errors.
* **Scientific Diagnostics**: Automatic calculation of Burn-in (Geweke), R-hat statistic (Gelman-Rubin), and Effective Sample Size (ESS).
* **Information Criteria**: Supports model comparison using AIC, BIC, and WAIC (Watanabe-Akaike Information Criterion).
* **Professional Visualization**: Integrated tools for corner plots, chain traces, and posterior predictive uncertainty bands.

---

## Installation

```bash
git clone https://github.com/gtomass/PyMCMC.git
cd PyMCMC
pip install -r requirements.txt
```

## Quick Start

Fit a simple linear model with uncertainties:

```python
import numpy as np
from PyMCMC import FunctionFitter, Prior, MCMCAnalyzer

# 1. Define the model
def linear_model(x, p):
    return p[0] * x + p[1]

# 2. Configure the fitter
fitter = FunctionFitter(
    model_func=linear_model,
    x_data=x_obs,
    y_data=y_obs,
    y_err=y_sigma,
    priors=[
        Prior('uniform', (0, 10)), 
        Prior('gaussian', (5, 2))
    ]
)

# 3. Sampling (DRAM used by default)
chain = fitter.fit(
    initial_params=[1.0, 5.0], 
    n_iterations=20000, 
    show_progress=True
)

# 4. Analysis
analyzer = MCMCAnalyzer(chain, fitter=fitter)
analyzer.print_summary()
analyzer.plot_corner()
```

## Advanced Usage

**Multimodal Exploration (Parallel Tempering)**

Essential for models with multiple mathematical solutions separated by low-probability regions.

```python
from PyMCMC import ParallelTempering

# Initialize with 8 chains and a max temperature of 50.0
pt = ParallelTempering(fitter, n_temps=8, t_max=50.0)
cold_chain = pt.run(
    initial_params=[1.0, 5.0], 
    total_steps=50000, 
    show_progress=True
)
```

**Correlated Errors (Covariance Matrix)**

Account for the noise structure of your measurement instruments using a full covariance matrix.

```python
fitter = FunctionFitter(
    model_func=my_model,
    x_data=x,
    y_data=y,
    data_cov=my_covariance_matrix # NxN matrix
)
```

## Project Structure

* ```chain.py```: High-performance memory management for sample storage and basic statistics.

* ```sampler.py``` : Core sampling engine implementing MH, AM, and DRAM algorithms.

* ```fitter.py``` : User interface for defining priors, likelihoods, and managing the fit process.

* ```analyzer.py``` : Post-processing tools for statistical reporting and plotting.

* ```parallel_tempering.py``` : Manager for multi-temperature chain execution.

* ```parallel_mcmc.py``` : Facilitates running independent chains across multiple CPU cores.

## Convergence Diagnostics

The library provides several tools to ensure the validity of the posterior distribution:

* **R-hat (Gelman-Rubin)** : Compares multiple chains to verify they have converged to the same stationary distribution.

* **Geweke Z-score** : Automatically determines the end of the "Burn-in" phase by comparing segment means.

* **HDI (Highest Density Interval)** : Calculates the narrowest Bayesian credible intervals for parameters.

## Unit Testing

To ensure code stability and mathematical correctness, run the test suite:

```bash
pytest tests/
```
