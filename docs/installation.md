# Installation

### Install via `pip`
The package can be installed via pip:
```bash
pip istall jeam
```
### Install from source
Alternatively, clone or download the source code and install locally:
```bash
python -m setup.py
```

---

## Dependencies
JEAM requires the following Python packages:

- `numpy`
- `scipy`
- `pandas`
- `numba`

All dependencies are installed automatically when using `pip`.

---

### Conda environment (suggested)
If you have Andaconda or miniconda installed and you would like to create a separate environment:

```bash
conda create --n jeam python=3 numpy scipy pandas numba
conda activate jeam
pip install jeam
```