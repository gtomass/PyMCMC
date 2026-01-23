from __future__ import annotations
import numpy as np
import multiprocessing as mp
from typing import List, Tuple, Optional, Any

from .sampler import Sampler
from .chain import Chain
from .fitter import FunctionFitter

def _worker_step_wrapper(args: Tuple[Sampler, int]) -> Sampler:
    """
    Worker function for multiprocessing. 
    Must be at the top level of the module to be picklable.
    """
    sampler, n_steps = args
    for _ in range(n_steps):
        sampler.step()
    return sampler


class ParallelTempering:
    """
    Parallel Tempering (MCMCMC) engine.
    
    Runs multiple chains at different temperatures to explore multimodal 
    posterior distributions and swaps states to improve mixing.
    """

    def __init__(self, fitter: FunctionFitter, n_temps: int = 4, t_max: float = 10.0):
        """
        Initialize the Parallel Tempering engine.

        Args:
            fitter: The FunctionFitter instance containing the model and likelihood.
            n_temps: Number of chains at different temperatures.
            t_max: Maximum temperature (the highest chain).
        """
        self.fitter = fitter
        self.n_temps = n_temps
        # Geometric spacing for inverse temperatures (betas)
        self.betas = 1.0 / np.logspace(0, np.log10(t_max), n_temps)

    def run(
        self, 
        initial_params: np.ndarray, 
        total_steps: int = 20000, 
        swap_every: int = 100, 
        show_progress: bool = False
    ) -> Chain:
        """
        Run the Parallel Tempering sampling.

        Args:
            initial_params: Starting parameter vector.
            total_steps: Total number of iterations per chain.
            swap_every: Number of steps between swap attempts.
            show_progress: If True, prints progress to console.

        Returns:
            The cold chain (beta=1.0) containing the final samples.
        """
        # 1. Initialize Samplers for each temperature
        initial_p = np.array(initial_params)
        # Using the fitter's internal method to get a starting covariance
        start_cov = self.fitter._initialize_proposal_cov(initial_p)

        samplers = [
            Sampler(
                log_lik_func=self.fitter.compute_log_likelihood,
                initial_state=initial_p,
                proposal_covariance=start_cov,
                log_prior_func=self.fitter.compute_log_prior,
                max_steps=total_steps,
                beta=self.betas[i]
            ) for i in range(self.n_temps)
        ]

        n_blocks = total_steps // swap_every
        n_swaps_accepted = 0

        # 2. Multiprocessing execution
        # We use a single Pool for the entire run to avoid process creation overhead
        with mp.Pool(processes=self.n_temps) as pool:
            for block_idx in range(n_blocks):
                # Run chains in parallel for 'swap_every' steps
                tasks = [(s, swap_every) for s in samplers]
                samplers = pool.map(_worker_step_wrapper, tasks)

                # 3. Perform swaps between adjacent chains (sequential)
                # This logic ensures information flows from hot chains to the cold one
                for i in range(self.n_temps - 1):
                    j = i + 1
                    s_i, s_j = samplers[i], samplers[j]
                    
                    # Extract raw (un-tempered) log-likelihoods
                    log_lik_i = s_i.current_log_lik / s_i.beta
                    log_lik_j = s_j.current_log_lik / s_j.beta
                    
                    # Exchange probability: (beta_i - beta_j) * (log_L_j - log_L_i)
                    swap_log_prob = (s_i.beta - s_j.beta) * (log_lik_j - log_lik_i)
                    
                    if np.log(np.random.rand()) < swap_log_prob:
                        # Swap states
                        temp_state = s_i.current_state.copy()
                        s_i.current_state = s_j.current_state.copy()
                        s_j.current_state = temp_state
                        
                        # Re-calculate tempered likelihoods for the new states
                        s_i.current_log_lik = s_i.beta * log_lik_j
                        s_j.current_log_lik = s_j.beta * log_lik_i
                        
                        # Re-calculate priors
                        s_i.current_log_prior = s_i.log_prior_func(s_i.current_state)
                        s_j.current_log_prior = s_j.log_prior_func(s_j.current_state)
                        
                        n_swaps_accepted += 1

                if show_progress and (block_idx % 5 == 0):
                    progress = 100 * (block_idx + 1) / n_blocks
                    print(f"Parallel Tempering: {progress:.1f}% | Swaps: {n_swaps_accepted}", end='\r')

        # Final Statistics
        total_attempts = n_blocks * (self.n_temps - 1)
        swap_rate = (n_swaps_accepted / total_attempts) * 100 if total_attempts > 0 else 0
        
        if show_progress:
            print(f"\nFinal Swaps accepted: {n_swaps_accepted}, Swap Acceptance Rate: {swap_rate:.2f}%")
        
        # Return only the cold chain (beta = 1.0)
        return samplers[0].chain