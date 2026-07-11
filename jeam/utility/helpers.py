import numpy as np
from numba import jit, njit

# ----------------------------
# Lanczos log-gamma (NumPy-only)
# ----------------------------
@jit(nopython=False)
def _loggamma_lanczos(z):
    """
    log(Gamma(z)) for real z using Lanczos approximation + reflection.
    Valid for real z not equal to non-positive integers.
    """
    # Coefficients for g=7, n=9 (common choice)
    p0 = 0.99999999999980993
    p1 = 676.5203681218851
    p2 = -1259.1392167224028
    p3 = 771.32342877765313
    p4 = -176.61502916214059
    p5 = 12.507343278686905
    p6 = -0.13857109526572012
    p7 = 9.9843695780195716e-6
    p8 = 1.5056327351493116e-7

    pi = np.pi

    if z < 0.5:
        # Reflection: Gamma(z)Gamma(1-z) = pi/sin(pi z)
        # logGamma(z) = log(pi) - log(sin(pi z)) - logGamma(1-z)
        return np.log(pi) - np.log(np.sin(pi * z)) - _loggamma_lanczos(1.0 - z)

    z1 = z - 1.0
    x = p0
    x += p1 / (z1 + 1.0)
    x += p2 / (z1 + 2.0)
    x += p3 / (z1 + 3.0)
    x += p4 / (z1 + 4.0)
    x += p5 / (z1 + 5.0)
    x += p6 / (z1 + 6.0)
    x += p7 / (z1 + 7.0)
    x += p8 / (z1 + 8.0)

    g = 7.0
    t = z1 + g + 0.5
    return 0.5 * np.log(2.0 * np.pi) + (z1 + 0.5) * np.log(t) - t + np.log(x)


@jit(nopython=False)
def _gamma(z):
    return np.exp(_loggamma_lanczos(z))


# ----------------------------
# I_v(x) helpers
# ----------------------------
@jit(nopython=False)
def _iv_series(v, x, tol=1e-14, max_terms=20000):
    """
    Power series:
      I_v(x) = sum_{k>=0} (1/(k! Gamma(k+v+1))) (x/2)^{2k+v}

    Computed via recurrence to avoid factorials / repeated gammas.
    Assumes x >= 0 and v not too negative (Gamma(v+1) finite).
    """
    if x == 0.0:
        if v == 0.0:
            return 1.0
        # For v>0, I_v(0)=0. For negative non-integers it diverges;
        # you should not call this in that regime.
        return 0.0

    halfx = 0.5 * x

    # term0 = (x/2)^v / Gamma(v+1)
    term = np.exp(v * np.log(halfx) - _loggamma_lanczos(v + 1.0))
    s = term

    x2_over4 = (x * x) * 0.25
    k = 0
    while k < max_terms:
        k1 = k + 1
        # term_{k+1} = term_k * (x^2/4) / ((k+1)(k+v+1))
        term = term * x2_over4 / (k1 * (k1 + v))
        s_new = s + term
        # relative convergence check
        if np.abs(term) <= tol * np.abs(s_new):
            return s_new
        s = s_new
        k = k1

    return s


@jit(nopython=False)
def _ive_asymptotic(v, x, n_terms=12):
    """
    Asymptotic for exponentially scaled ive(v,x) = exp(-x) I_v(x), x>0:

    exp(-x) I_v(x) ~ 1/sqrt(2*pi*x) * [1 - (mu-1)/(8x) + (mu-1)(mu-9)/(2!(8x)^2) - ...]
    where mu = 4 v^2.

    Uses a stable recurrence for the bracket series.
    """
    mu = 4.0 * v * v
    inv8x = 1.0 / (8.0 * x)

    # bracket series
    s = 1.0
    term = 1.0
    for k in range(1, n_terms + 1):
        m = 2.0 * k - 1.0
        term = term * (mu - m * m) * inv8x / k
        term = -term  # alternating signs
        s += term

    return s / np.sqrt(2.0 * np.pi * x)


# ----------------------------
# Public API: iv and ive
# ----------------------------
@jit(nopython=False)
def ive_numba(v, x):
    """
    Exponentially scaled modified Bessel I:
      ive(v,x) = exp(-abs(x)) * I_v(x)
    for real v and real x.

    This implementation assumes x >= 0 for best behavior.
    """
    ax = np.abs(x)
    if ax == 0.0:
        # ive(v,0) = I_v(0)
        if v == 0.0:
            return 1.0
        return 0.0

    # Threshold where asymptotic is usually accurate and faster than the series
    if ax < 18.0:
        # compute I_v(ax) then scale
        iv = _iv_series(v, ax)
        return iv * np.exp(-ax)
    else:
        return _ive_asymptotic(v, ax)


@jit(nopython=False)
def iv_numba(v, x):
    """
    Modified Bessel function of the first kind I_v(x) for real v and real x.
    Uses series for small x and asymptotic for large x.

    WARNING: I_v(x) grows like exp(x)/sqrt(x), so iv_numba will overflow to inf
    for sufficiently large x (around x > ~709 in float64).
    """
    ax = np.abs(x)
    if ax == 0.0:
        if v == 0.0:
            return 1.0
        return 0.0

    if ax < 18.0:
        return _iv_series(v, ax)
    else:
        # Use ive asymptotic then multiply by exp(ax) if safe
        scaled = _ive_asymptotic(v, ax)
        # exp(709.78...) is near float64 max
        if ax > 709.0:
            return np.inf
        return scaled * np.exp(ax)
    
@njit(cache=True, fastmath=True)
def trapz_1d(y, x):
    """
    Trapezoidal integral of y over x. Both 1D, same length.
    """
    s = 0.0
    for i in range(x.shape[0] - 1):
        dx = x[i + 1] - x[i]
        s += 0.5 * (y[i] + y[i + 1]) * dx
    return s