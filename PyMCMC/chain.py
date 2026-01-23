from __future__ import annotations
import os
import numpy as np
from typing import Optional, Tuple, Dict, Any

class Chain:
    """
    Data structure to store and analyze MCMC sampling results.
    
    Handles memory management (auto-resizing), statistical calculations 
    (mean, covariance, HDI), and convergence diagnostics (Geweke, ESS).
    """

    def __init__(self, ndim: int, max_steps: int = 10000):
        """
        Initialize the MCMC chain object.

        Args:
            ndim: Number of dimensions (parameters) in the model.
            max_steps: Initial maximum number of steps allocated. Defaults to 10000.
        """
        self.ndim = ndim
        self.max_steps = max_steps
        
        # Pre-allocation of arrays for performance
        self.samples = np.zeros((max_steps, ndim))
        self.ln_likelihoods = np.zeros(max_steps)
        self.weights = np.zeros(max_steps)
        
        self.n_entries = 0
        self.current_capacity = max_steps
        
        # Tracking the best state (MAP)
        self.best_ln_likelihood = -np.inf
        self.best_params: Optional[np.ndarray] = None

    def add_state(self, state: np.ndarray, ln_likelihood: float, weight: float = 1.0) -> None:
        """
        Add a new sample state to the chain.

        Args:
            state: Array of parameter values.
            ln_likelihood: The log-likelihood of this state.
            weight: The weight of the state (useful for thinned chains). Defaults to 1.0.
        """
        if self.n_entries >= self.current_capacity:
            self._resize_storage()

        self.samples[self.n_entries] = state
        self.ln_likelihoods[self.n_entries] = float(np.asarray(ln_likelihood).item())
        self.weights[self.n_entries] = weight

        # Update Maximum A Posteriori (MAP)
        if float(ln_likelihood) > self.best_ln_likelihood:
            self.best_ln_likelihood = float(ln_likelihood)
            self.best_params = state.copy()

        self.n_entries += 1

    @property
    def current_samples(self) -> np.ndarray:
        """Return the valid samples (excluding empty pre-allocated slots)."""
        return self.samples[:self.n_entries]

    @property
    def current_ln_likelihoods(self) -> np.ndarray:
        """Return the valid log-likelihoods."""
        return self.ln_likelihoods[:self.n_entries]

    def get_mean(self) -> np.ndarray:
        """Calculate the weighted mean of the parameters."""
        if self.n_entries == 0:
            return np.zeros(self.ndim)
        
        w = self.weights[:self.n_entries]
        total_w = np.sum(w)
        
        if total_w <= 0:
            return self.samples[0]
            
        return np.average(self.current_samples, axis=0, weights=w)

    def get_covariance(self) -> np.ndarray:
        """Calculate the weighted covariance matrix."""
        if self.n_entries < 2:
            return np.eye(self.ndim)
            
        w = self.weights[:self.n_entries]
        if np.sum(w) <= 0:
            return np.eye(self.ndim)
        
        cov = np.atleast_2d(np.cov(self.current_samples, rowvar=False, aweights=w))
        return cov

    def get_correlation(self) -> np.ndarray:
        """Calculate the correlation matrix."""
        cov = self.get_covariance()
        std = np.sqrt(np.diag(cov))
        # Avoid division by zero for constant parameters
        std[std == 0] = 1.0
        return cov / np.outer(std, std)

    def get_map_estimate(self) -> Optional[np.ndarray]:
        """Return the best parameter set found (MAP)."""
        return self.best_params

    def _resize_storage(self) -> None:
        """Double the capacity of the storage arrays when full."""
        new_capacity = self.current_capacity * 2
        self.samples = np.resize(self.samples, (new_capacity, self.ndim))
        self.ln_likelihoods = np.resize(self.ln_likelihoods, new_capacity)
        self.weights = np.resize(self.weights, new_capacity)
        self.current_capacity = new_capacity

    def estimate_autocorr(self, max_lag: int = 100) -> np.ndarray:
        """
        Estimate the autocorrelation function for each parameter.
        """
        autocorr = np.zeros((max_lag, self.ndim))
        samples = self.current_samples
        
        for i in range(self.ndim):
            data = samples[:, i]
            mean = np.mean(data)
            var = np.var(data)
            
            if var == 0:
                autocorr[:, i] = 1.0
                continue
                
            centered_data = data - mean
            for lag in range(max_lag):
                if lag == 0:
                    autocorr[lag, i] = 1.0
                else:
                    # Optimized autocorrelation using NumPy indexing
                    c = np.mean(centered_data[lag:] * centered_data[:-lag])
                    autocorr[lag, i] = c / var
        return autocorr

    def get_ess(self) -> np.ndarray:
        """Estimate the Effective Sample Size (ESS) per parameter."""
        max_lag = min(self.n_entries // 10, 500)
        rho = self.estimate_autocorr(max_lag=max_lag)
        # Sum of correlations (Integrated Autocorrelation Time)
        tau = 1 + 2 * np.sum(rho[1:], axis=0)
        return self.n_entries / tau

    def compute_hdi(self, cred_mass: float = 0.95) -> np.ndarray:
        """Calculate the Highest Density Interval (HDI)."""
        hdi = np.zeros((self.ndim, 2))
        for i in range(self.ndim):
            values = np.sort(self.current_samples[:, i])
            n_samples = len(values)
            interval_idx_inc = int(np.floor(cred_mass * n_samples))
            n_intervals = n_samples - interval_idx_inc
            
            # Find the narrowest interval containing the required mass
            interval_width = values[interval_idx_inc:] - values[:n_intervals]
            min_idx = np.argmin(interval_width)
            hdi[i] = [values[min_idx], values[min_idx + interval_idx_inc]]
        return hdi

    def calculate_geweke_z(self, first_frac: float = 0.1, last_frac: float = 0.5) -> np.ndarray:
        """Geweke Z-score diagnostic for stationarity."""
        n_first = int(first_frac * self.n_entries)
        n_last = int(last_frac * self.n_entries)
        
        if n_first == 0 or n_last == 0:
            return np.zeros(self.ndim)

        seg_first = self.current_samples[:n_first]
        seg_last = self.current_samples[-n_last:]
        
        z_scores = np.zeros(self.ndim)
        for i in range(self.ndim):
            m1, m2 = np.mean(seg_first[:, i]), np.mean(seg_last[:, i])
            v1 = np.var(seg_first[:, i]) / n_first if n_first > 1 else 1.0
            v2 = np.var(seg_last[:, i]) / n_last if n_last > 1 else 1.0
            
            denom = np.sqrt(v1 + v2)
            z_scores[i] = (m1 - m2) / denom if denom > 0 else 0.0
            
        return z_scores

    def save(self, filename: str) -> None:
        """Save the chain to a compressed .npz file."""
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        np.savez_compressed(
            filename, 
            samples=self.current_samples,
            ln_likelihoods=self.current_ln_likelihoods,
            weights=self.weights[:self.n_entries],
            best_params=self.best_params,
            best_ln_likelihood=self.best_ln_likelihood,
            ndim=self.ndim,
            max_steps=self.max_steps
        )

    @classmethod
    def load(cls, filename: str) -> Chain:
        """Load and reconstruct a Chain object from a .npz file."""
        data = np.load(filename, allow_pickle=True)
        
        obj = cls(ndim=int(data['ndim']), max_steps=int(data['max_steps']))
        n_val = len(data['samples'])
        
        if n_val > obj.current_capacity:
            obj.samples = np.resize(obj.samples, (n_val, obj.ndim))
            obj.ln_likelihoods = np.resize(obj.ln_likelihoods, n_val)
            obj.weights = np.resize(obj.weights, n_val)
            obj.current_capacity = n_val

        obj.samples[:n_val] = data['samples']
        obj.ln_likelihoods[:n_val] = data['ln_likelihoods']
        obj.weights[:n_val] = data['weights']
        obj.n_entries = n_val
        obj.best_params = data['best_params']
        obj.best_ln_likelihood = float(data['best_ln_likelihood'])
        
        return obj