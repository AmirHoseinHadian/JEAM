# Data Requirements and Model Selection

This guide explains:

1. What type of data is required to use models implemented in JEAM.
2. Which model is appropriate for different response geometries.

---

## 1. What Data Do You Need?

JEAM implements multi-dimensional diffusion models that jointly explain **response time (RT)** and **continuous response location** and assume that response times reflect **evidence accumulation process**.
Therefore, to use these models appropriately, your dataset must contain:

- **Response time (RT)** for each decision  
- **Continuous response value** for each decision (e.g., angle, slider position, 2D coordinate)

Each row of your dataset should correspond to a single trial.

!!! warning "JEAM may *not* be appropriate if:"

    - The response scale was not continuous
    - Response times were not recorded.
    - Response times are artificially constrained (e.g., fixed response windows).
    - The task design makes response times unrelated to decision dynamics.
    - You are only interested in modeling response error distributions without response times.

---

## 2. Model Selection by Response Scale

JEAM provides different diffusion models depending on the geometry of the response scale.

### A. Circular Response Scales

**Examples:**

- Color judgment tasks (using a color wheel)
- Orientation judgment tasks

**Suitable Model:** `CircularDiffusionModel`

Use when responses lie on a circle (0–2π or 0–360°).

---

### B. Circular Response Scales with Multimodal Response Error

**Examples:**

- Random dot motion task
- Color judgment tasks (using a color wheel)

**Suitable Models:** `SphericalDiffusionModel` and `HyperSphericalDiffusionModel`

Use when responses lie on a circle (0–2π or 0–360°) and the **response error is multimodal**.

---

### C. One-Dimensional Bounded Scales

**Examples:**

- Numerosity estimation tasks
- Pricing tasks

**Suitable Model:** `ProjectedSphericalDiffusionModel`

Use when responses are bounded but not circular (i.e., endpoints are distinct). For example, sliders with lower and upper bounds or arc-shaped are suitable.

---

### D. Two-Dimensional Bounded Planes

**Examples:**

- Centriod estimation tasks
- Spatial working memory tasks

**Suitable Model:** `ProjectedHyperSphericalDiffusionModel`

Use when responses are continuous in 2D and bounded within a region.

---

## 3. Quick Model Selection Table

| Response Geometry          | Example Task                          | Suitable Model in JEAM                                      |
|----------------------------|---------------------------------------|-------------------------------------------------------------|
| Circular (1D wrapped)      | Color or orientation judgement        | `CircularDiffusionModel`                                    |
| Circular (1D wrapped)      | Random dot motion                     | `SphericalDiffusionModel` or `HyperSphericalDiffusionModel` |
| 1D bounded (not wrapped)   | Estimation, pricing                   | `ProjectedSphericalDiffusionModel`                          |
| 2D bounded plane           | Centriod estimation                   | `ProjectedHyperSphericalDiffusionModel`                     |

---

## 4. Summary

JEAM is appropriate when:

- The response scale is continuous.
- You have trial-level RTs.
- You have trial-level continuous responses.
- RTs meaningfully reflect decision latency.

Choosing the correct model depends entirely on the **structure of the response scale**.
