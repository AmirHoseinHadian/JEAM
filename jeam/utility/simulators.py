import numpy as np
from numba import jit

@jit(nopython=True)
def simulate_CDM_trial(threshold, drift_vec, ndt, threshold_dynamic='fixed', decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the circular diffusion model (CDM).

    Parameters
    ----------
    threshold : float
        A positive floating number representing the decision threshold.
    drift_vec : array_like, shape (2,)
        A two-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    threshold_dynamic : {'fixed', 'linear', 'exponential', 'hyperbolic'}, optional
        Type of threshold collapse. Default is 'fixed'.
    decay : float, optional
        Decay rate of the collapsing boundary. Default is 0.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : float
        Response angle between [-pi, pi].
    '''
    x = np.zeros((2,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(2)
    else:
        mu_t = drift_vec

    if threshold_dynamic == 'fixed':
        while np.linalg.norm(x) < threshold:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
            rt += dt
    elif threshold_dynamic == 'linear':
        while np.linalg.norm(x) < threshold - decay*rt:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
            rt += dt
    elif threshold_dynamic == 'exponential':
        while np.linalg.norm(x) < threshold * np.exp(-decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
            rt += dt
    elif threshold_dynamic == 'hyperbolic':
        while np.linalg.norm(x) < threshold / (1 + decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
            rt += dt
    theta = np.arctan2(x[1], x[0]) 
    return ndt_t+rt, theta


def simulate_custom_threshold_CDM_trial(threshold_function, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the circular diffusion model (CDM) with a user defined threshold function.

    Parameters
    ----------
    threshold_function : callable
        A function that takes time t and returns the threshold at time t.
    drift_vec : array_like, shape (2,)
        A two-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    drift_vec : array_like, shape (2,)
        A two-dimensional array representing the drift vector.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : float
        Response angle between [-pi, pi].
    '''
    x = np.zeros((2,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand() - s_t
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(2)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold_function(rt):
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(2)
        rt += dt
    
    theta = np.arctan2(x[1], x[0]) 
    
    return ndt_t+rt, theta


@jit(nopython=True)
def simulate_SDM_trial(threshold, drift_vec, ndt, threshold_dynamic='fixed', decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the spherical diffusion model (SDM).

    Parameters
    ----------
    threshold : float
        A positive floating number representing the decision threshold.
    drift_vec : array_like, shape (3,)
        A three-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    threshold_dynamic : {'fixed', 'linear', 'exponential', 'hyperbolic'}, optional
        Type of threshold collapse. Default is 'fixed'.
    decay : float, optional
        Decay rate of the collapsing boundary. Default is 0.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2); theta[0] between [0, pi] and theta[1] between [-pi, pi]
    '''
    x = np.zeros((3,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(3)
    else:
        mu_t = drift_vec

    if threshold_dynamic == 'fixed':
        while np.linalg.norm(x) < threshold:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'linear':
        while np.linalg.norm(x) < threshold - decay*rt:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'exponential':
        while np.linalg.norm(x) < threshold * np.exp(-decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'hyperbolic':
        while np.linalg.norm(x) < threshold / (1 + decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt

    theta1 = np.arctan2(np.sqrt(x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(x[2], x[1])

    return ndt_t+rt, (theta1, theta2)


def simulate_custom_threshold_SDM_trial(threshold_function, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the spherical diffusion model (SDM) with a user defined threshold function.

    Parameters
    ----------
    threshold_function : callable
        A function that takes time t and returns the threshold at time t.
    drift_vec : array_like, shape (3,)
        A three-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001.

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2); theta[0] between [0, pi] and theta[1] between [-pi, pi]
    '''
 
    x = np.zeros((3,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(3)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold_function(rt):
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
        rt += dt

    theta1 = np.arctan2(np.sqrt(x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(x[2], x[1])

    return ndt_t+rt, (theta1, theta2)


@jit(nopython=True)
def simulate_HSDM_trial(threshold, drift_vec, ndt, threshold_dynamic='fixed', decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the hyper spherical diffusion model (HSDM).

    Parameters
    ----------
    threshold : float
        A positive floating number representing the decision threshold.
    drift_vec : array_like, shape (4,)
        A four-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    threshold_dynamic : {'fixed', 'linear', 'exponential', 'hyperbolic'}, optional
        Type of threshold collapse. Default is 'fixed'.
    decay : float, optional
        Decay rate of the collapsing boundary. Default is 0.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2, theta3); theta[0] and theta[1] between [0, pi], and theta[2] between [-pi, pi]
    '''
    x = np.zeros((4,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(4)
    else:
        mu_t = drift_vec

    if threshold_dynamic == 'fixed':
        while np.linalg.norm(x) < threshold:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'linear':
        while np.linalg.norm(x) < threshold - decay*rt:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'exponential':
        while np.linalg.norm(x) < threshold * np.exp(-decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'hyperbolic':
        while np.linalg.norm(x) < threshold / (1 + decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt

    theta1 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2), x[1])
    theta3 = np.arctan2(x[3], x[2])

    return ndt_t+rt, (theta1, theta2, theta3)

def simulate_custom_threshold_HSDM_trial(threshold_function, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the hyper spherical diffusion model (HSDM) with a user defined threshold function.

    Parameters
    ----------
    threshold_function : callable
        A function that takes time and returns the threshold at that time.
    drift_vec : array_like, shape (4,)
        A four-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2, theta3); theta[0] and theta[1] between [0, pi], and theta[2] between [-pi, pi]
    '''
    x = np.zeros((4,))
    
    rt = 0

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if s_v>0:
        mu_t = drift_vec + s_v*np.random.randn(4)
    else:
        mu_t = drift_vec

    while np.linalg.norm(x) < threshold_function(rt):
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
        rt += dt

    theta1 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2 + x[1]**2), x[0])
    theta2 = np.arctan2(np.sqrt(x[3]**2 + x[2]**2), x[1])
    theta3 = np.arctan2(x[3], x[2])

    return ndt_t+rt, (theta1, theta2, theta3)

@jit(nopython=True)
def simulate_PSDM_trial(threshold, drift_vec, ndt, threshold_dynamic='fixed', decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the projected spherical diffusion model (PSDM).

    Parameters
    ----------
    threshold : float
        A positive floating number representing the decision threshold.
    drift_vec : array_like, shape (2,)
        A two-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    threshold_dynamic : {'fixed', 'linear', 'exponential', 'hyperbolic'}, optional
        Type of threshold collapse. Default is 'fixed'.
    decay : float, optional
        Decay rate of the collapsing boundary. Default is 0.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : float
        Response angle between [0, pi]
    '''
    x = np.zeros((3,))
    muz = drift_vec[0]
    eta = drift_vec[1]
    
    norm_mu = np.sqrt(eta**2 + muz**2)
    theta_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta_mu) * np.sin(rphi)

    mu = np.array([mux, muy, muz])

    if s_v>0:
        mu_t = mu + s_v*np.random.randn(3)
    else:
        mu_t = mu

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if threshold_dynamic == 'fixed':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'linear':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold - decay*rt:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'exponential':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold * np.exp(-decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    elif threshold_dynamic == 'hyperbolic':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold / (1 + decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
            rt += dt
    
    theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])    
    
    return ndt_t+rt, theta

def simulate_custom_threshold_PSDM_trial(threshold_function, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the projected spherical diffusion model (PSDM) with a user defined threshold function.

    Parameters
    ----------
    threshold_function : callable
        A function that takes time and returns threshold.
    drift_vec : array_like, shape (2,)
        A two-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : float
        Response angle between [0, pi]
    '''
    x = np.zeros((3,))
    muz = drift_vec[0]
    eta = drift_vec[1]
    
    norm_mu = np.sqrt(eta**2 + muz**2)
    theta_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta_mu) * np.sin(rphi)

    mu = np.array([mux, muy, muz])

    if s_v>0:
        mu_t = mu + s_v*np.random.randn(3)
    else:
        mu_t = mu

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2) < threshold_function(rt):
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(3)
        
        rt += dt

    theta = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])    
    
    return ndt_t+rt, theta


@jit(nopython=True)
def simulate_PHSDM_trial(threshold, drift_vec, ndt, threshold_dynamic='fixed', decay=0, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the projected hyper spherical diffusion model (PHSDM).

    Parameters
    ----------
    threshold : float
        A positive floating number representing the decision threshold.
    drift_vec : array_like, shape (3,)
        A three-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    threshold_dynamic : {'fixed', 'linear', 'exponential', 'hyperbolic'}, optional
        Type of threshold collapse. Default is 'fixed'.
    decay : float, optional
        Decay rate of the collapsing boundary. Default is 0.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2); theta[0] and theta[1] between [0, pi]
    '''

    x = np.zeros((4,))
    muw = drift_vec[0]
    muz = drift_vec[1]
    eta = drift_vec[2]
    
    norm_mu = np.sqrt(eta**2 + muz**2 + muw**2)
    theta1_mu = np.arctan2(eta, muw)
    theta2_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta1_mu) * np.sin(theta2_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta1_mu) * np.sin(theta2_mu) * np.sin(rphi)

    mu = np.array([mux, muy, muz, muw])

    if s_v>0:
        mu_t = mu + s_v*np.random.randn(4)
    else:
        mu_t = mu

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt

    if threshold_dynamic == 'fixed':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) < threshold:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'linear':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) < threshold - decay*rt:
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'exponential':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) < threshold * np.exp(-decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    elif threshold_dynamic == 'hyperbolic':
        while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) < threshold / (1 + decay*rt):
            x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
            rt += dt
    
    theta1 = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[3])
    theta2 = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])   
    
    return ndt_t+rt, (theta1, theta2)
    

def simulate_custom_threshold_PHSDM_trial(threshold_function, drift_vec, ndt, s_v=0, s_t=0, sigma=1, dt=0.001):
    '''
    Simulate a single trial of the projected hyper spherical diffusion model (PHSDM) with a user defined threshold function.

    Parameters
    ----------
    threshold_function : callable
        A function that takes a time value and returns the threshold at that time.
    drift_vec : array_like, shape (3,)
        A three-dimensional array representing the drift vector.
    ndt : float
        A positive floating number representing the non-decision time.
    s_v : float, optional
        Standard deviation of drift rate variability. Default is 0.
    s_t : float, optional
        Range of non-decision time variability. Default is 0.
    sigma : float, optional
        Diffusion coefficient (standard deviation of the diffusion process). Default is 1.
    dt : float, optional
        Time step for the simulation. Default is 0.001. 

    Returns
    -------
    rt : float
        Response time in seconds.
    theta : tuple
        A tuple of response angles (theta1, theta2); theta[0] and theta[1] between [0, pi]
    '''
    x = np.zeros((4,))
    muw = drift_vec[0]
    muz = drift_vec[1]
    eta = drift_vec[2]
    
    norm_mu = np.sqrt(eta**2 + muz**2 + muw**2)
    theta1_mu = np.arctan2(eta, muw)
    theta2_mu = np.arctan2(eta, muz)
    
    rt = 0
    rphi = np.pi/4 # it is not important (just a dummpy value)
    mux = norm_mu * np.sin(theta1_mu) * np.sin(theta2_mu) * np.cos(rphi)
    muy = norm_mu * np.sin(theta1_mu) * np.sin(theta2_mu) * np.sin(rphi)

    mu = np.array([mux, muy, muz, muw])

    if s_v>0:
        mu_t = mu + s_v*np.random.randn(4)
    else:
        mu_t = mu

    if s_t>0:
        ndt_t = ndt + s_t*np.random.rand()
    else:
        ndt_t = ndt


    while np.sqrt(x[0]**2 + x[1]**2 + x[2]**2 + x[3]**2) < threshold_function(rt):
        x += mu_t*dt + sigma*np.sqrt(dt)*np.random.randn(4)
        rt += dt
    
    
    theta1 = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[3])
    theta2 = np.arctan2(np.sqrt(x[0]**2 + x[1]**2), x[2])   
    
    return ndt_t+rt, (theta1, theta2)