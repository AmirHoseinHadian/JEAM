<div style="text-align: center">
    <img src="imgs/_logo/JEAM_logo.svg" alt="JEAM logo" width="500">
</div>

## **Overview**
JEAM is a Python package for evidence accumulation modeling of continuous judgment tasks.

The package provides fast and numerically stable likelihood evaluation for multi-dimensional diffusion models based on the integral equation method proposed by [Hadian Rasanan et al. (2025)](https://doi.org/10.3758/s13428-025-02810-3). 

JEAM is designed for researchers in **cognitive science**, **mathematical psychology**, and **neuroscience** who study decision-making with continuous-response spaces. 

!!! Note "Required data: response time and response location"
    JEAM implements diffusion-based models that jointly explain **response times** and
    **continuous response locations**. To use these models appropriately, your dataset must include:

    - a response time (RT) for each decision, and  
    - a continuous response value (e.g., angle, position, 2D coordinate) for each decision.

    These models are designed for tasks in which response times reflect the dynamics of
    evidence accumulation. If response times are not recorded, are heavily constrained by the task
    design (e.g., delayed response prompts, fixed response windows), or are not theoretically
    meaningful measures of decision latency, diffusion-based modeling may not be appropriate.

    In such cases, models that focus solely on response error distributions (e.g., descriptive or
    static mixture models) may be more suitable.

## **Features**
**1. Support for diverse continuous response scales:**

JEAM supports a wide range of continuous-response scales commonly used in experimental research, including:

- Circular scales (e.g., panels a, b, and c),
- Bounded one-dimensional scales (e.g., panels d and e),
- Two-dimensional scales (e.g., panel f).

Examples of continuous-response decision tasks are illustrated below:

![Examples of continuous-report decision tasks.](imgs/Continuous_tasks.png "Examples of continuous-report decision tasks.")

**2. Time-dependent decision thresholds:**

JEAM supports diffusion models with time-dependent decision thresholds and enables likelihood estimation for models with arbitrary threshold dynamics. Specifically, the package includes:

- Fixed threshold,
- Linear collapsing threshold,
- Exponential collapsing threshold,
- Hyperbolic collapsing threshold,
- User-defined threshold functions.

This flexibility allows researchers to model urgency signals and dynamic decision boundaries in a principled way.

---
## **Diffusion models of continuous-response taks**

JEAM includes four classes of multi-dimensional diffusion models, tailored to different response geometries:

- **Circular Diffusion Models** for experiments with circular response scales (e.g., panels a, b, and c in figure above),
- **(Hyper-)Spherical Diffusion Models** for experiments with circular response scales (e.g., panels a, b, and c in figure above),
- **Projected Spherical Diffusion Models** for experiments with one-dimensional bounded scales (e.g., panels d and e in figure above),
- **Projected Hyper-spherical Diffusion Models** for experiments with two-dimensional bounded scales (e.g., panel f in figure above).

![Diffusion models of continuous response tasks.](imgs/Diffusion_models.png "Diffusion models of continuous response tasks.")

---
## **Credits**

This package was developed by me, [Amir Hosein Hadian Rasanan](https://scholar.google.com/citations?hl=en&user=qbOoaykAAAAJ),
with support from [Dr. Nathan J Evans](https://scholar.google.com/citations?user=2hG7r90AAAAJ&hl=en) and [Prof. Dr. Jörg Rieskamp](https://scholar.google.com/citations?user=6Y5X1xUAAAAJ&hl=en). 

When using this package or part of the code for your own research, I ask you to cite us:

> Hadian Rasanan, A. H., Evans, N. J., and Rieskamp, J. (in prepration). JEAM: A Python Package for Evidence Accumulation Modeling of Continuous Judgments

*Also don't forget to cite the original paper for each model.* 

- **Circular Diffusion Model**: Smith, P. L. (2016). Diffusion theory of decision making in continuous report. *Psychological Review*, 123(4), 425–451. [https://doi.org/10.1037/rev0000023](https://doi.org/10.1037/rev0000023)
- **Hyper-spherical Diffusion Model**: Smith, P. L., & Corbett, E. A. (2019). Speeded multielement decision-making as diffusion in a hypersphere: Theory and application to double-target detection. *Psychonomic Bulletin & Review*, 26(1), 127-162. [https://doi.org/10.3758/s13423-018-1491-0](https://doi.org/10.3758/s13423-018-1491-0)
- **Projected Spherical Diffusion Model**: Hadian Rasanan, A. H., Olschewski, S., & Rieskamp, J. (2026) The Projected Spherical Diffusion Model: A Theory of Evidence Accumulation for Continuous Estimation Tasks. [https://doi.org/10.31234/osf.io/mhj6v_v1](https://doi.org/10.31234/osf.io/mhj6v_v1)
- **Multi-dimensional Diffusion Models with Collapsing Decision Threshold**: Hadian Rasanan, A. H., Evans, N. J., Amani Rad, J., & Rieskamp, J. (2025). Parameter estimation of hyper-spherical diffusion models with a time-dependent threshold: An integral equation method. *Behavior Research Methods*, 57(10), 283. [https://doi.org/10.3758/s13428-025-02810-3](https://doi.org/10.3758/s13428-025-02810-3)

---
## **Support**

If you have questions, encounter issues, or would like to contribute, please contact: [amir.h.hadian@gmail.com](mailto:amir.h.hadian@gmail.com)