import numpy as np

from numba import jit
from scipy.special import iv, ive

from CRDDM.utility.helpers import iv_numba, ive_numba
from CRDDM.utility.Constants import zeros_0, zeros_1, zeros_5
from CRDDM.utility.Constants import JVZ1, JVZ2, JVZ5

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def cdm_short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**2) / (np.sqrt(x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*zeros_0[0]**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def cdm_long_t_fpt_z(t, threshold, sigma=1):
    fpt_z = np.zeros(t.shape)
    for i in range(t.shape[0]):
        series = np.sum((zeros_0/JVZ1) * np.exp(-(zeros_0**2 * sigma**2)/(2*threshold**2)*t[i]))
        fpt_z[i] = sigma**2/threshold**2 * series
    return fpt_z

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def hsdm_short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**3) / ((x + t) * np.sqrt(x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*zeros_1[0]**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def hsdm_long_t_fpt_z(t, threshold, sigma=1):
    fpt_z = np.zeros(t.shape)
    for i in range(t.shape[0]):
        series = np.sum((zeros_1**2/JVZ2) * np.exp(-(zeros_1**2 * sigma**2)/(2*threshold**2)*t[i]))
        fpt_z[i] = sigma**2/threshold**2 * series
    return fpt_z

# The firs-passage time distribution of zero-drift process for small RTs
@jit(nopython=True)
def sdm_short_t_fpt_z(t, x):
    term1 = ((1 - x)*(1 + t)**2.5) / ((x + t) * t**1.5)
    term2 = np.exp(-.5*(1-x)**2/t -  0.5*np.pi**2*t)
    return term1*term2

# The firs-passage time distribution of zero-drift process
@jit(nopython=True)
def sdm_long_t_fpt_z(t, threshold, sigma=1):
    fpt_z = np.zeros(t.shape)
    for i in range(t.shape[0]):
        series = np.sum((zeros_5**1.5/JVZ5) * np.exp(-(zeros_5**2 * sigma**2)/(2*threshold**2)*t[i]))
        fpt_z[i] = sigma**2/threshold**2 * series
    return fpt_z


@jit(nopython=False)
def k_linear(threshold, decay, t, q, sigma=2):
    da = -2*decay * (threshold - decay*t)
    return 0.5 * (q - 0.5*sigma - da)

@jit(nopython=False)
def psi_linear(threshold, decay, t, z, tau, q, sigma=2):
    kk = k_linear(threshold, decay, t, q, sigma)
    
    a = (threshold - decay*t)**2
    da = -2*decay * (threshold - decay*t)
    
    if 2*np.sqrt(a*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (a + z)/(sigma*(t-tau)))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = iv_numba(q/sigma-1, 2*np.sqrt(a*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * iv_numba(q/sigma, 2*np.sqrt(a*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = ive_numba(q/sigma-1, (a + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * ive_numba(q/sigma, (a + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

@jit(nopython=False)
def ie_fpt_linear(threshold, decay, q, z, sigma=2, dt=0.1, T_max=2):
    g = np.zeros((int(T_max/dt)+2,))
    T = np.zeros((int(T_max/dt)+2,))
    if threshold - decay*dt > 0:
        g[1] = -2*psi_linear(threshold, decay, dt, z, 0, q, sigma)
    T[1] = dt
    
    for n in range(2, int(T_max/dt)+2):
        if threshold - decay*(n*dt) <= 0:
            T[n] = n*dt
            continue

        s = -2 * psi_linear(threshold, decay, n*dt, z, 0, q, sigma)

        for j in range(1, n):
            if threshold - decay*(j*dt) <= 0:
                continue
            
            s += 2 * dt * g[j] * psi_linear(threshold, decay, n*dt, (threshold - decay*(j*dt))**2, j*dt, q, sigma)

        g[n] = s
        T[n] = n*dt

    return g, T

@jit(nopython=False)
def k_exponential(threshold, decay, t, q, sigma=2):
    da = -2*decay*threshold*np.exp(-decay*t) * (threshold * np.exp(-decay*t))
    return 0.5 * (q - 0.5*sigma - da)

@jit(nopython=False)
def psi_exponential(threshold, decay, t, z, tau, q, sigma=2):
    kk = k_exponential(threshold, decay, t, q, sigma)

    a = (threshold * np.exp(-decay*t))**2
    da = -2*decay*threshold*np.exp(-decay*t) * (threshold * np.exp(-decay*t))

    if 2*np.sqrt(a*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (a + z)/(sigma*(t-tau)))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = iv_numba(q/sigma-1, 2*np.sqrt(a*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * iv_numba(q/sigma, 2*np.sqrt(a*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = ive_numba(q/sigma-1, (a + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * ive_numba(q/sigma, (a + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

@jit(nopython=False)
def ie_fpt_exponential(threshold, decay, q, z, sigma=2, dt=0.1, T_max=2):
    g = np.zeros((int(T_max/dt)+2,))
    T = np.zeros((int(T_max/dt)+2,))

    g[1] = -2*psi_exponential(threshold, decay, dt, z, 0, q, sigma)
    T[1] = dt
    
    for n in range(2, int(T_max/dt)+2):        
        s = -2 * psi_exponential(threshold, decay, n*dt, z, 0, q, sigma)

        for j in range(1, n):
            s += 2 * dt * g[j] * psi_exponential(threshold, decay, n*dt, (threshold * np.exp(-decay*(j*dt)))**2, j*dt, q, sigma)

        g[n] = s
        T[n] = n*dt

    return g, T

@jit(nopython=False)
def k_hyperbolic(threshold, decay, t, q, sigma=2):
    da = -2*decay*threshold/(1 + decay*t)**2 * (threshold / (1 + decay*t))
    return 0.5 * (q - 0.5*sigma - da)

@jit(nopython=False)
def psi_hyperbolic(threshold, decay, t, z, tau, q, sigma=2):
    kk = k_hyperbolic(threshold, decay, t, q, sigma)

    a = (threshold / (1 + decay*t))**2
    da = -2*decay*threshold/(1 + decay*t)**2 * (threshold / (1 + decay*t))

    if 2*np.sqrt(a*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (a + z)/(sigma*(t-tau)))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = iv_numba(q/sigma-1, 2*np.sqrt(a*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * iv_numba(q/sigma, 2*np.sqrt(a*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (a/z)**(0.5*(q-sigma)/sigma)
        term3 = da - (a/(t-tau)) + kk
        term4 = ive_numba(q/sigma-1, (a + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(a*z)/(t-tau)) * ive_numba(q/sigma, (a + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

@jit(nopython=False)
def ie_fpt_hyperbolic(threshold, decay, q, z, sigma=2, dt=0.1, T_max=2):
    g = np.zeros((int(T_max/dt)+2,))
    T = np.zeros((int(T_max/dt)+2,))

    g[1] = -2*psi_hyperbolic(threshold, decay, dt, z, 0, q, sigma)
    T[1] = dt
    
    for n in range(2, int(T_max/dt)+2):
        s = -2 * psi_hyperbolic(threshold, decay, n*dt, z, 0, q, sigma)
        for j in range(1, n):           
            s += 2 * dt * g[j] * psi_hyperbolic(threshold, decay, n*dt, (threshold / (1 + decay*(j*dt)))**2, j*dt, q, sigma)

        g[n] = s
        T[n] = n*dt

    return g, T

def k_custom(dt_threshold_function, t, q, sigma=2):
    return 0.5 * (q - 0.5*sigma - dt_threshold_function(t))

def psi_custom(threshold_function, dt_threshold_function, t, z, tau, q, sigma=2):
    kk = k_custom(dt_threshold_function, t, q, sigma)
    
    if 2*np.sqrt(threshold_function(t)*z)/(sigma*(t-tau))<=700:
        term1 = 1./(sigma*(t - tau)) * np.exp(- (threshold_function(t) + z)/(sigma*(t-tau)))
        term2 = (threshold_function(t)/z)**(0.5*(q-sigma)/sigma)
        term3 = dt_threshold_function(t) - (threshold_function(t)/(t-tau)) + kk
        term4 = iv(q/sigma-1, 2*np.sqrt(threshold_function(t)*z)/(sigma*(t-tau)))
        term5 = (np.sqrt(threshold_function(t)*z)/(t-tau)) * iv(q/sigma, 2*np.sqrt(threshold_function(t)*z)/(sigma*(t-tau)))
    else:
        term1 = 1./(sigma*(t - tau))
        term2 = (threshold_function(t)/z)**(0.5*(q-sigma)/sigma)
        term3 = dt_threshold_function(t) - (threshold_function(t)/(t-tau)) + kk
        term4 = ive(q/sigma-1, (threshold_function(t) + z)/(sigma*(t-tau)))
        term5 = (np.sqrt(threshold_function(t)*z)/(t-tau)) * ive(q/sigma, (threshold_function(t) + z)/(sigma*(t-tau)))
    
    return term1 * term2 * (term3 * term4 + term5)

def ie_fpt_custom(threshold_function, dt_threshold_function, q, z, sigma=2, dt=0.1, T_max=2):
    g = np.zeros((int(T_max/dt)+2,))
    T = np.zeros((int(T_max/dt)+2,))
    
    if threshold_function(dt) > 0:
        g[1] = -2*psi_custom(threshold_function, dt_threshold_function, dt, z, 0, q, sigma)
    T[1] = dt

    for n in range(2, int(T_max/dt)+2):
        if threshold_function(n*dt) <= 0:
            T[n] = n*dt
            continue

        s = -2 * psi_custom(threshold_function, dt_threshold_function, n*dt, z, 0, q, sigma)

        for j in range(1, n):
            if threshold_function(j*dt) <= 0:
                continue

            s += 2 * dt * g[j] * psi_custom(threshold_function, dt_threshold_function, n*dt, threshold_function(j*dt), j*dt, q, sigma)

        g[n] = s
        T[n] = n*dt
        
    g = np.asarray(g)
    T = np.asarray(T)
    
    return g, T