import numpy as np
from mpmath import besseljzero
from scipy.special import jv
from scipy.special import jn_zeros

# Precompute Bessel function zeros and values for first-passage time distribution
zeros_0 = jn_zeros(0, 100)
JVZ1 = jv(1, zeros_0)

zeros_5 = np.asarray([float(besseljzero(0.5, i+1)) for i in range(100)])
JVZ5 = jv(1.5, zeros_5)

zeros_1 = jn_zeros(1, 100)
JVZ2 = jv(2, zeros_1)