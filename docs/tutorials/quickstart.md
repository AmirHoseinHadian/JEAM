# Quick Start: Circular Diffusion Model

This tutorial demonstrates how to simulate data from a **Circular Diffusion Model (CDM)** and recover model parameters using likelihood-based estimation with JEAM.

## 1. Import required packages


```python
import numpy as np              # Numerical computations
import pandas as pd             # Data handling

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Optimization
from scipy.optimize import differential_evolution

# JEAM model
from jeam.Models.Circular import CircularDiffusionModel as CDM
```

## 2. Create a circular diffusion model

To instantiate a model, create an object of the `CircularDiffusionModel` class and specify the threshold dynamics.

The `threshold_dynamic` argument determines whether the decision boundary radius changes over time.

Available options:

- `'fixed'` (default)
- `'linear'`
- `'exponential'`
- `'hyperbolic'`
- `'custom'`


```python
model = CDM(threshold_dynamic='fixed')
```

The model instance provides two key methods:

-  `simulate(...)`, which simulates data (response times and response angles) from the model for a given parameters.
- `joint_lpdf(...)`, which calculates the joint log-likelihood of the model for the given data and set of parameters.

## 3. Simulate continuous-response decision data

```python
# Ground-truth parameters
threshold = 2.0                    # Decision threshold
drift_vector = np.array([2.0, 0.0]) # Drift in x and y directions
ndt = 0.25                         # Non-decision time

# Simulate data
sim_data = model.simulate(
    drift_vector,
    ndt,
    threshold,
    n_sample=1000
)

sim_data.head()
```

|    |    rt |   response |
|---:|------:|-----------:|
|  0 | 0.623 |  -0.288496 |
|  1 | 1.178 |  -0.213882 |
|  2 | 2.552 |   0.710925 |
|  3 | 0.894 |  -0.255075 |
|  4 | 1.242 |  -0.152316 |


The simulated dataset contains:

- `rt`: response times
- `response`: continuous angular responses on the circle in radian

## 4. Visualizing simulated data


```python
plt.figure(figsize=(9, 4), layout='constrained')

plt.subplot(121)
sns.histplot(sim_data['rt'], stat='density')
plt.xlabel("Response time")

plt.subplot(122)
sns.histplot(sim_data['response'], stat='density')
plt.xlabel("Response angle (radians)");
```


    
![png](../imgs/_outputs/output_10_1_quick_start.png)
    


## 5. Define the likelihood function

We estimate parameters by **maximizing the joint likelihood** of response times and continuous response angles.


```python
def negative_log_likelihood(prms, rt, theta, model):
    threshold =  prms[0] 
    ndt = prms[1] 
    drift_vec = np.array([prms[2], prms[3]])
    
    logpdf = model.joint_lpdf(rt, theta, drift_vec, ndt, threshold)
    
    return -np.sum(logpdf)
```

## 6. Estimate parameters using differential evolution

After defining the `negative_log_likelihood` function, users can pass this function to any optimization routine. For instance, users can use `scipy.optimize.differential_evolution`. This optimizer requires the parameters' range. In other words, `differential_evolution` searches for the global minimum of the function within a **bounded space** defined by the user. Therefore, if users wish to employ `differential_evolution` as the optimizer, providing the parameter bounds is essential. 

!!! warning "Parameter order must match bounds" 

    The order of parameters in bounds must exactly match the order in which parameters are unpacked inside the `negative_log_likelihood` function. **A mismatch will lead to incorrect estimation.**
      


```python
# Parameter bounds
bounds = [
    (0.05, 5.0),   # threshold
    (0.0, 2.0),    # non-decision time
    (-5.0, 5.0),   # drift_x
    (-5.0, 5.0),   # drift_y
]

param_names = ["threshold", "ndt", "drift_x", "drift_y"]


result = differential_evolution(
    negative_log_likelihood,
    bounds=bounds,
    args=(
        sim_data["rt"].values, 
        sim_data["response"].values, 
        model
    ),
)
```

The optimization result contains:

- `result.x`: the optimized parameter values  
- `result.fun` - the optimized negative log-likelihood value, 

`result.fun` can be used for calculating the goodness-of-fit metrics like AIC or BIC.

## 7. Inspect recovered parameters


```python
for name, value in zip(param_names, result.x):
    print(f"{name}: {value:.3f}")
```

    threshold: 1.939
    ndt: 0.276
    drift_x: 1.899
    drift_y: 0.028


The recovered parameters should be close to the values used for simulation, demonstrating correct likelihood evaluation and parameter recovery.

## Notes

- Collapsing thresholds (`'linear'`, `'exponential'`, `'hyperbolic'`) follow the same workflow.
- Other response geometries (e.g., projected spherical or hyperspherical models) use the same estimation structure.
