from __future__ import annotations
import os
import numpy as np
from scipy import linalg, special
from typing import Optional, List, Callable, Tuple, Union, Any

from .sampler import Sampler
from .chain import Chain
from .parallel_mcmc import run_parallel_mcmc

class Prior:
    """
    Defines prior probability distributions for model parameters.
    """

    def __init__(self, distribution_type: str = 'uniform', params: Tuple[float, float] = (0.0, 1.0)):
        """
        Initialize the Prior.

        Args:
            distribution_type: Type of distribution ('uniform', 'gaussian', 'log_normal', 'beta').
            params: Parameters for the distribution.
        """
        self.type = distribution_type.lower()
        self.params = params

    def compute_log_pdf(self, x: float) -> float:
        """
        Calculate the log-probability density function for the prior at x.
        """
        if self.type == 'uniform':
            low, high = self.params
            return 0.0 if low <= x <= high else -np.inf
        
        if self.type == 'gaussian':
            mu, sigma = self.params
            return -0.5 * ((x - mu) / sigma)**2
        
        if self.type == 'log_normal':
            if x <= 0:
                return -np.inf
            mu, sigma = self.params
            return -0.5 * ((np.log(x) - mu) / sigma)**2 - np.log(x * sigma * np.sqrt(2 * np.pi))
        
        if self.type == 'beta':
            a, b = self.params
            if not (0 <= x <= 1):
                return -np.inf
            # Using scipy.special for the beta function for precision
            return (a - 1) * np.log(x) + (b - 1) * np.log(1 - x) - special.betaln(a, b)
            
        return 0.0


class FunctionFitter:
    """
    Orchestrates the fitting process by linking models, data, and sampling methods.
    """

    def __init__(
        self,
        model_func: Optional[Callable] = None,
        x_data: Optional[np.ndarray] = None,
        y_data: Optional[np.ndarray] = None,
        y_err: Optional[np.ndarray] = None,
        data_cov: Optional[np.ndarray] = None,
        priors: Optional[List[Prior]] = None,
        initial_proposal_cov: Optional[np.ndarray] = None,
        custom_log_lik: Optional[Callable] = None,
        adapt_every: int = 100
    ):
        self.model_func = model_func
        self.x_data = x_data
        self.y_data = y_data
        self.priors = priors
        self.custom_log_lik = custom_log_lik
        self.adapt_every = adapt_every
        self.initial_proposal_cov = initial_proposal_cov

        # Data Error Handling
        self.data_cov = data_cov
        if self.data_cov is not None:
            # Pre-compute Cholesky for speed: lnL = -0.5 * (r.T @ Cov^-1 @ r)
            self.L_data = linalg.cholesky(self.data_cov, lower=True)
            # Log-determinant of Cov is 2 * sum(log(diag(L)))
            self.log_det_cov = 2.0 * np.sum(np.log(np.diag(self.L_data)))
        else:
            self.y_err = y_err if y_err is not None else np.ones_like(y_data, dtype=float)
            self.log_det_cov = 2.0 * np.sum(np.log(self.y_err))

    def compute_log_prior(self, params: np.ndarray) -> float:
        """Calculate the cumulative log-prior for all parameters."""
        if self.priors is None:
            return 0.0
        
        total_lp = 0.0
        for p, prior_dist in zip(params, self.priors):
            lp = prior_dist.compute_log_pdf(p)
            if lp == -np.inf:
                return -np.inf
            total_lp += lp
        return total_lp

    def compute_log_likelihood(self, params: np.ndarray) -> float:
        """Calculate the log-likelihood (diagonal or full covariance)."""
        if self.custom_log_lik:
            return self.custom_log_lik(params)

        if self.model_func is None or self.y_data is None:
            raise ValueError("Model function and y_data are required for likelihood calculation.")

        model_prediction = self.model_func(self.x_data, params)
        residuals = self.y_data - model_prediction

        if self.data_cov is not None:
            # Solve L*alpha = residuals -> chi2 = alpha.T @ alpha
            alpha = linalg.solve_triangular(self.L_data, residuals, lower=True)
            chi2 = np.sum(alpha**2)
        else:
            chi2 = np.sum((residuals / self.y_err)**2)

        n_points = len(self.y_data)
        return -0.5 * (chi2 + self.log_det_cov + n_points * np.log(2 * np.pi))

    def _initialize_proposal_cov(self, value: np.ndarray) -> np.ndarray:
        """Generate an initial proposal covariance matrix if none is provided."""
        if self.initial_proposal_cov is not None:
            return np.atleast_2d(self.initial_proposal_cov)
        
        # Default: 5% of initial parameter values as standard deviation
        sigmas = np.abs(value) * 0.05
        # Guard against zeros, NaNs or Infs
        sigmas[sigmas == 0] = 1.0
        sigmas[~np.isfinite(sigmas)] = 1.0
        
        return np.diag(sigmas**2)

    def fit(
        self,
        initial_params: np.ndarray,
        n_iterations: int = 10000,
        method: str = 'DRAM',
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 1000,
        resume: bool = False,
        thin: int = 1,
        show_progress: bool = False
    ) -> Chain:
        """
        Execute the MCMC fit.
        """
        start_step = 1
        current_p = np.array(initial_params)
        chain_obj = None

        if resume and checkpoint_path and os.path.exists(checkpoint_path):
            chain_obj = Chain.load(checkpoint_path)
            # Adjust max_steps if necessary
            chain_obj.max_steps = n_iterations
            start_step = chain_obj.n_entries
            current_p = chain_obj.get_map_estimate()
            
        proposal_cov = self._initialize_proposal_cov(current_p)

        sampler = Sampler(
            log_lik_func=self.compute_log_likelihood,
            initial_state=current_p,
            proposal_covariance=proposal_cov,
            log_prior_func=self.compute_log_prior,
            max_steps=n_iterations,
            method=method,
            adapt_every=self.adapt_every,
            thin=thin
        )

        if chain_obj:
            sampler.chain = chain_obj
            sampler.n_steps = start_step

        return sampler.run(
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            show_progress=show_progress
        )

    def fit_parallel(
        self,
        initial_params_list: List[np.ndarray],
        n_iterations: int = 10000,
        method: str = 'DRAM',
        num_workers: Optional[int] = None,
        record_full_trace: bool = True
    ) -> List[Chain]:
        """Run multiple chains in parallel."""
        return run_parallel_mcmc(
            fitter=self,
            initial_params_list=initial_params_list,
            max_steps=n_iterations,
            method=method,
            num_workers=num_workers,
            record_full_trace=record_full_trace
        )