<div align="center">
  <img src="docs/imgs/_logo/JEAM_logo.svg" alt="JEAM logo" width="400">
</div>

JEAM is a Python package for evidence accumulation modeling of continuous judgment tasks. 

The package provides fast and numerically stable likelihood evaluation for multi-dimensional diffusion decision models using the integral equation method proposed by [Hadian Rasanan et al.,(2025)](https://doi.org/10.3758/s13428-025-02810-3). JEAM supports a wide range of continuous response scale, that can be employed in experimental studies including:
- Bounded one-dimensional scales (e.g., arcs or sliders),
- Circular scales (e.g., color wheels),
- Two-dimensional scales (e.g., 2D planes).

JEAM is designed for researchers in cognitive science, mathematical psychology, and neuroscience who work with diffusion models of continuous responses.

---

## Install
### Install via `pip`
The package can be installed via pip:
```bash
pip istall jeam
```
### Install from source
Alternatively, the latest version of the package can be installed directly from Gihub repository:
```bash
pip install git+https://github.com/AmirHoseinHadian/JEAM.git
```

---

## Dependencies
JEAM requires the following Python packages:

- `numpy`
- `scipy`
- `pandas`
- `numba`
- `mpmath`

All dependencies are installed automatically when using `pip`.

---

### Conda environment (suggested)
If you have Andaconda or miniconda installed and you would like to create a separate environment:

```bash
conda create --name jeam_env python=3
conda activate jeam_env
pip install jeam
```

## Documentation

The latest documentation can be found here: **[amirhoseinhadian.github.io/JEAM/](https://amirhoseinhadian.github.io/JEAM/)**

## Credits

This package was developed by me, [Amir Hosein Hadian Rasanan](https://scholar.google.com/citations?hl=en&user=qbOoaykAAAAJ),
with support from [Dr. Nathan J Evans](https://scholar.google.com/citations?user=2hG7r90AAAAJ&hl=en) and [Prof. Dr. Jörg Rieskamp](https://scholar.google.com/citations?user=6Y5X1xUAAAAJ&hl=en). 

When using this package or part of the code for your own research, I ask you to cite us:

> Hadian Rasanan, A. H., Evans, N. J., and Rieskamp, J. (2026). JEAM: A Framework and Tutorial for Applying Evidence Accumulation Models to Continuous Judgments

---

## Selected References

For background on diffusion models for continuous response tasks and the estimation methods implemented in JEAM, see:

- Hadian Rasanan, A. H., Evans, N. J., Amani Rad, J., & Rieskamp, J. (2025). Parameter estimation of hyper-spherical diffusion models with a time-dependent threshold: An integral equation method. *Behavior Research Methods*, 57(10), 283. https://doi.org/10.3758/s13428-025-02810-3

- Hadian Rasanan, A. H., Olschewski, S., & Rieskamp, J. (2026). The Projected Spherical Diffusion Model: An Evidence Accumulation Theory for Estimation. https://doi.org/10.31234/osf.io/mhj6v_v1

- Smith, P. L. (2016). Diffusion theory of decision making in continuous report. *Psychological Review*, 123 (4), 425–451, https://doi.org/10.1037/rev0000023

- Smith, P.L., & Corbett, E.A. (2019). Speeded multielement decision-making as diffusion in a hypersphere: Theory and application to double-target detection. *Psychonomic Bulletin & Review*, 26, https://doi.org/10.3758/s13423-018-1491-0
