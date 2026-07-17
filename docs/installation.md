# Installation

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