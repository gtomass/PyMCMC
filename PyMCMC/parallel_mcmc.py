from __future__ import annotations
import multiprocessing as mp
import numpy as np
from typing import List, Optional, Callable, Any, Tuple

from .sampler import Sampler
from .chain import Chain

def _worker_task_wrapper(
    log_lik_func: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    prop_cov: np.ndarray,
    log_prior_func: Callable[[np.ndarray], float],
    max_steps: int,
    method: str,
    adapt_every: int,
    record_full_trace: bool
) -> Chain:
    """
    Isolated worker function to execute a single MCMC chain in a separate process.
    
    This function is defined at the module level to ensure it can be 
    pickled for multiprocessing.
    """
    sampler = Sampler(
        log_lik_func=log_lik_func,
        initial_state=initial_state,
        proposal_covariance=prop_cov,
        log_prior_func=log_prior_func,
        max_steps=max_steps,
        method=method,
        adapt_every=adapt_every
    )
    return sampler.run(record_full_trace=record_full_trace)


def run_parallel_mcmc(
    fitter: Any,  # Avoid circular import with FunctionFitter
    initial_params_list: List[np.ndarray],
    max_steps: int = 10000,
    method: str = 'DRAM',
    num_workers: Optional[int] = None,
    record_full_trace: bool = True
) -> List[Chain]:
    """
    Run multiple MCMC chains in parallel using a multiprocessing pool.

    Args:
        fitter: A configured instance of the FunctionFitter class.
        initial_params_list: List of starting parameter vectors (one per chain).
        max_steps: Number of iterations to perform per chain.
        method: Sampling method ('MH', 'AM', or 'DRAM').
        num_workers: Number of CPU cores to utilize. Defaults to system count.
        record_full_trace: If True, records all steps (essential for R-hat).

    Returns:
        A list containing the resulting Chain objects.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    # Initialize the proposal covariance matrix using the fitter's logic
    # We use the first set of parameters to define the scale
    first_params = np.array(initial_params_list[0])
    initial_cov = fitter._initialize_proposal_cov(first_params)
    
    # Prepare the arguments for each worker process
    tasks = []
    for start_p in initial_params_list:
        tasks.append((
            fitter.compute_log_likelihood,
            np.array(start_p),
            initial_cov,
            fitter.compute_log_prior,
            max_steps,
            method,
            fitter.adapt_every,
            record_full_trace
        ))

    # Determine the number of processes (don't create more than there are chains)
    n_chains = len(initial_params_list)
    pool_size = min(num_workers, n_chains)

    # Execute the tasks in parallel
    with mp.Pool(processes=pool_size) as pool:
        # starmap allows passing multiple arguments to the worker function
        chains = pool.starmap(_worker_task_wrapper, tasks)
    
    return chains