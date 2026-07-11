import numpy as np
import pandas as pd
from scipy.special import iv

from CRDDM.utility.helpers import trapz_1d
from CRDDM.utility.simulators import simulate_CDM_trial, simulate_custom_threshold_CDM_trial
from CRDDM.utility.fpts import cdm_short_t_fpt_z, cdm_long_t_fpt_z, ie_fpt_linear, ie_fpt_exponential, ie_fpt_hyperbolic, ie_fpt_custom

class CircularDiffusionModel:
    '''
    Circular Diffusion Model
    '''

    def __init__(self, threshold_dynamic='fixed'):
        '''
        Parameters
        ----------
        threshold_dynamic : str, optional
            The type of threshold collapse ('fixed', 'linear', 'exponential', 'hyperbolic', or 'custom'), default is 'fixed'
        '''
        self.name = 'Circular Diffusion Model'

        if threshold_dynamic in ['fixed', 'linear', 'exponential', 'hyperbolic', 'custom']:
            self.threshold_dynamic = threshold_dynamic
        else:
            raise ValueError("\'threshold_dynamic\' must be one of \'fixed\', \'linear\', \'exponential\', \'hyperbolic\', or \'custom\'. However, got \'{}\'".format(threshold_dynamic))
        


    def simulate(self, drift_vec, ndt, threshold=1, decay=0, threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.001, n_sample=1):
        '''
        Simulate data from the Circular Diffusion Model with collapsing boundaries

        Parameters
        ----------
        drift_vec : array-like, shape (2,)
            The drift vector [drift_x, drift_y]
        ndt : float
            The non-decision time
        threshold : float
            The initial decision threshold (default is 1)
        decay : float
            The decay rate of the threshold (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)
        dt : float, optional
            The time step for simulation (default is 0.001)
        n_sample : int, optional
            The number of samples to simulate (default is 1)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing simulated response times and choice angles
        '''
        RT = np.empty((n_sample,))
        Choice = np.empty((n_sample,))

        if drift_vec.ndim == 1:
            drift_vec = drift_vec * np.ones((n_sample, 2))
        elif drift_vec.shape[0] != n_sample:
            raise ValueError("Number of rows in drift_vec must be equal to n_sample")
        
        if isinstance(ndt, (float, np.floating)) or isinstance(ndt, (int, np.integer)):
            ndt = np.full((n_sample,), ndt)
        elif len(ndt) != n_sample:
            raise ValueError("Length of ndt must be equal to n_sample")
        
        if isinstance(threshold, (float, np.floating)) or isinstance(threshold, (int, np.integer)):
            threshold = np.full((n_sample,), threshold)
        elif len(threshold) != n_sample:
            raise ValueError("Length of threshold must be equal to n_sample")
        
        if isinstance(decay, (float, np.floating)) or isinstance(decay, (int, np.integer)):
            decay = np.full((n_sample,), decay)
        elif len(decay) != n_sample:
            raise ValueError("Length of decay must be equal to n_sample")
        
        if threshold_function is None and self.threshold_dynamic == 'custom':
            raise ValueError("threshold_function must be provided when threshold_dynamic is 'custom'")
        
        if threshold_function is not None and self.threshold_dynamic != 'custom':
            raise ValueError("threshold_function should be None when threshold_dynamic is not 'custom'")
        
        if s_v < 0:
            raise ValueError("s_v must be non-negative")
        if s_t < 0:
            raise ValueError("s_t must be non-negative")
        
        if self.threshold_dynamic != 'custom':
            for n in range(n_sample):
                RT[n], Choice[n] = simulate_CDM_trial(threshold[n], drift_vec[n, :].astype(np.float64), ndt[n], 
                                                      threshold_dynamic=self.threshold_dynamic,
                                                      decay=decay[n], s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        else:
            for n in range(n_sample):
                RT[n], Choice[n] = simulate_custom_threshold_CDM_trial(threshold_function,
                                                                       drift_vec[n, :].astype(np.float64), ndt[n], 
                                                                       s_v=s_v, s_t=s_t, sigma=sigma, dt=dt)
        
        return pd.DataFrame(np.c_[RT, Choice], columns=['rt', 'response'])


    def joint_lpdf(self, rt, theta, drift_vec, ndt, threshold, decay=0, threshold_function=None, dt_threshold_function=None, s_v=0, s_t=0, sigma=1, dt=0.01):
        '''
        Compute the joint log-probability density function of response time and choice angle

        Parameters
        ----------
        rt : array-like
            Response times
        theta : array-like
            Choice angles in radians
        drift_vec : array-like, shape (2,) or (n_samples, 2)
            The drift vector [drift_x, drift_y]
        ndt : float
            The non-decision time
        threshold : float
            The initial decision threshold
        decay : float
            The decay rate of the threshold (default is 0)
        threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the threshold at time t
        dt_threshold_function : callable, if threshold_dynamic is 'custom'
            A function that takes time t and returns the derivative of the threshold at time t
        s_v : float, optional
            The standard deviation of drift variability (default is 0)
        s_t : float, optional
            The standard deviation of non-decision time variability (default is 0)
        sigma : float, optional
            The diffusion coefficient (default is 1)
        dt : float, optional
            The time step for numerical estimation of first-passage time densities (default is 0.01)

        Returns
        -------
        array-like
            The joint log-probability density evaluated at (rt, theta) with same shape as rt and theta
        '''

        if drift_vec.ndim == 1:
            drift_vec = drift_vec * np.ones((rt.shape[0], 2))

        if drift_vec.shape[1] != 2 or drift_vec.ndim != 2:
            raise ValueError("drift_vec must have shape (2,) or (n_samples, 2)")

        tt = np.maximum(rt - ndt, 0)

        # first-passage time density of zero drift process
        if self.threshold_dynamic == 'fixed':
            a = threshold
            s0 = 0.002
            s1 = 0.02
            if s_t == 0:
                s = tt/threshold**2
                w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
                fpt_lt = cdm_long_t_fpt_z(tt, threshold, sigma=sigma)
                fpt_st = sigma**2/threshold**2 * cdm_short_t_fpt_z(sigma**2 * tt/threshold**2, sigma**2 * 0.1**8/threshold**2)   
            else:
                T = np.arange(0, tt.max()+0.05, 0.05)
                s = T/threshold**2
                w = np.minimum(np.maximum((s - s0) / (s1 - s0), 0), 1)
                fpt_lt = cdm_long_t_fpt_z(T, threshold, sigma=sigma)
                fpt_st = sigma**2/threshold**2 * cdm_short_t_fpt_z(sigma**2 * T/threshold**2, sigma**2 * 0.1**8/threshold**2)   
            fpt_z =  (1 - w) * fpt_st + w * fpt_lt
        elif self.threshold_dynamic == 'linear':
            a = threshold - decay*tt
            T_max = min(rt.max(), threshold/decay)
            g_z, T = ie_fpt_linear(threshold, decay, 2*sigma**2, 0.000001, sigma=2*sigma**2, dt=dt, T_max=T_max)
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'exponential':
            a = threshold * np.exp(-decay*tt)
            g_z, T = ie_fpt_exponential(threshold, decay, 2*sigma**2, 0.000001, sigma=2*sigma**2, dt=dt, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'hyperbolic':
            a = threshold / (1 + decay*tt)
            g_z, T = ie_fpt_hyperbolic(threshold, decay, 2*sigma**2, 0.000001, sigma=2*sigma**2, dt=dt, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)
        elif self.threshold_dynamic == 'custom':
            threshold_function2 = lambda t: threshold_function(t)**2
            dt_threshold_function2 = lambda t: 2 * dt_threshold_function(t) * threshold_function(t)
            a = threshold_function(tt)
            g_z, T = ie_fpt_custom(threshold_function2, dt_threshold_function2, 2*sigma**2, 0.000001, sigma=2*sigma**2, dt=dt, T_max=rt.max())
            fpt_z = np.interp(tt, T, g_z)

        fpt_z = np.maximum(fpt_z, 0.1**14)

        # Girsanov:
        if s_v == 0:
            # No drift variability
            mu_dot_x0 = drift_vec[:, 0] * np.cos(theta)
            mu_dot_x1 = drift_vec[:, 1] * np.sin(theta)

            if s_t == 0:
                # No non-decision time variability
                term1 = a * (mu_dot_x0 + mu_dot_x1) / sigma**2
                term2 = 0.5 * (drift_vec[:, 0]**2 + drift_vec[:, 1]**2) * tt / sigma**2
                log_density = term1 - term2 + np.log(fpt_z) - np.log(2*np.pi)
            else:
                # With non-decision time variability
                log_density = np.log(0.1**14) * np.ones(rt.shape[0])
                eps = np.linspace(0, s_t, max(2, int(s_t//0.02)))
                norm2_drift = drift_vec[:, 0]**2 + drift_vec[:, 1]**2
                mu_dot_x = (mu_dot_x0 + mu_dot_x1) / sigma**2
                for i in range(rt.shape[0]):
                    if tt[i] - s_t > 0:
                        if self.threshold_dynamic == 'fixed':
                            integrand = np.exp(- 0.5 * norm2_drift[i] * (tt[i] - eps)/sigma**2) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                            density =  np.exp(threshold * mu_dot_x[i]) * trapz_1d(integrand, eps) * 0.5/np.pi
                        elif self.threshold_dynamic == 'linear':
                            integrand = np.exp((threshold - decay * (tt[i] - eps)) * mu_dot_x[i] - 0.5 * norm2_drift[i] * (tt[i] - eps)/sigma**2) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                            density = trapz_1d(integrand, eps) * 0.5/np.pi
                        elif self.threshold_dynamic == 'exponential':
                            integrand = np.exp(threshold*np.exp(-decay * (tt[i] - eps)) * mu_dot_x[i] - 0.5 * norm2_drift[i] * (tt[i] - eps)/sigma**2) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                            density = trapz_1d(integrand, eps) * 0.5/np.pi
                        elif self.threshold_dynamic == 'hyperbolic':
                            integrand = np.exp(threshold/(1  + decay * (tt[i] - eps)) * mu_dot_x[i] - 0.5 * norm2_drift[i] * (tt[i] - eps)/sigma**2) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                            density = trapz_1d(integrand, eps) * 0.5/np.pi
                        elif self.threshold_dynamic == 'custom':
                            integrand = np.exp(threshold_function(tt[i] - eps) * mu_dot_x[i] - 0.5 * norm2_drift[i] * (tt[i] - eps)/sigma**2) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                            density = trapz_1d(integrand, eps) * 0.5/np.pi
                        
                        if density > 0.1**14:
                            log_density[i] = np.log(density)

        else:
            # With drift variability
            if s_t == 0:
                # No non-decision time variability     
                s_v2 = s_v**2
                x0 =  a * np.cos(theta)
                x1 =  a * np.sin(theta)
                fixed = 1/(np.sqrt(s_v2/sigma**2 * tt + 1))
                exponent0 = -0.5*drift_vec[:, 0]**2/s_v2 + 0.5*(x0 * s_v2/sigma**2 + drift_vec[:, 0])**2 / (s_v2 * (s_v2/sigma**2 * tt + 1))
                exponent1 = -0.5*drift_vec[:, 1]**2/s_v2 + 0.5*(x1 * s_v2/sigma**2 + drift_vec[:, 1])**2 / (s_v2 * (s_v2/sigma**2 * tt + 1))

                log_density = 2*np.log(fixed) + exponent0 + exponent1 + np.log(fpt_z) - np.log(2*np.pi)
            else:
                # With non-decision time variability
                log_density = np.log(0.1**14) * np.ones(rt.shape[0])
                eps = np.linspace(0, s_t, max(2, int(s_t//0.02)))
                s_v2 = s_v**2
                for i in range(rt.shape[0]):
                    if tt[i] - s_t > 0:
                        if self.threshold_dynamic == 'fixed':
                            x0 =  threshold * np.cos(theta[i])
                            x1 =  threshold * np.sin(theta[i])
                        elif self.threshold_dynamic == 'linear':
                            x0 =  (threshold - decay * (tt[i]-eps)) * np.cos(theta[i])
                            x1 =  (threshold - decay * (tt[i]-eps)) * np.sin(theta[i])
                        elif self.threshold_dynamic == 'exponential':
                            x0 =  (threshold * np.exp(-decay * (tt[i]-eps))) * np.cos(theta[i])
                            x1 =  (threshold * np.exp(-decay * (tt[i]-eps))) * np.sin(theta[i])
                        elif self.threshold_dynamic == 'hyperbolic':
                            x0 =  (threshold / (1 + decay * (tt[i]-eps))) * np.cos(theta[i])
                            x1 =  (threshold / (1 + decay * (tt[i]-eps))) * np.sin(theta[i])
                        elif self.threshold_dynamic == 'custom':
                            x0 =  threshold_function(tt[i]-eps) * np.cos(theta[i])
                            x1 =  threshold_function(tt[i]-eps) * np.sin(theta[i])
                        fixed = 1/(np.sqrt(s_v2/sigma**2 * (tt[i]-eps) + 1))
                        exponent0 = -0.5*drift_vec[i, 0]**2/s_v2 + 0.5*(x0 * s_v2/sigma**2 + drift_vec[i, 0])**2 / (s_v2 * (s_v2/sigma**2 * (tt[i]-eps) + 1))
                        exponent1 = -0.5*drift_vec[i, 1]**2/s_v2 + 0.5*(x1 * s_v2/sigma**2 + drift_vec[i, 1])**2 / (s_v2 * (s_v2/sigma**2 * (tt[i]-eps) + 1))

                        integrand = fixed**2 * np.exp(exponent0 + exponent1) * np.interp(tt[i]-eps, T, fpt_z)/s_t
                        density = trapz_1d(integrand, eps) * 0.5/np.pi
                        if density > 0.1**14:
                            log_density[i] = np.log(density)

        log_density[rt - ndt - s_t <= 0] = np.log(0.1**14)
        log_density = np.maximum(log_density, np.log(0.1**14))
            
        return log_density

    