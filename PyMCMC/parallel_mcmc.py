# PyMCMC/parallel_mcmc.py
from __future__ import annotations
import multiprocessing as mp
import numpy as np
from tqdm import tqdm
from typing import List, Optional, Callable, Any, Tuple
from threadpoolctl import threadpool_limits

from .sampler import Sampler
from .chain import Chain

DEBUG = False
if DEBUG:
    import cProfile, pstats, os

def _worker_task_wrapper(
    log_lik_func: Callable[[np.ndarray], float],
    initial_state: np.ndarray,
    prop_cov: np.ndarray,
    log_prior_func: Callable[[np.ndarray], float],
    max_steps: int,
    method: str,
    adapt_every: int,
    record_full_trace: bool,
    progress_queue: Optional[Any] = None,
    threads_per_worker: int = 1 # Reçoit la queue
) -> Chain:
    """
    Isolated worker function to execute a single MCMC chain.
    """
    with threadpool_limits(limits=threads_per_worker, user_api='blas'):
        if DEBUG:
            print("Worker PID:", os.getpid())
            print("Initialize profiler")
            profiler = cProfile.Profile()
            profiler.enable()
            print("Profiler started")


        
        sampler = Sampler(
            log_lik_func=log_lik_func,
            initial_state=initial_state,
            proposal_covariance=prop_cov,
            log_prior_func=log_prior_func,
            max_steps=max_steps,
            method=method,
            adapt_every=adapt_every
        )
        # On passe la queue à la méthode run
        results = sampler.run(record_full_trace=record_full_trace, progress_queue=progress_queue)

        if DEBUG:
            print("Disabling profiler")
            profiler.disable()
            # Sauvegarde le profil du processus (un par PID)
            profiler.dump_stats(f"/Users/gtomassini/Documents/Git_Repo/DustyPY/profile_worker_{os.getpid()}.prof")
            print("Profiler data saved")

        return results

def run_parallel_mcmc(
    fitter: Any,
    initial_params_list: List[np.ndarray],
    max_steps: int = 10000,
    method: str = 'DRAM',
    num_workers: Optional[int] = None,
    record_full_trace: bool = True,
    show_progress: bool = False,
    threads_per_worker: int = 1
) -> List[Chain]:
    """
    Run multiple MCMC chains in parallel with real-time iteration progress.
    """
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    first_params = np.array(initial_params_list[0])
    initial_cov = fitter._initialize_proposal_cov(first_params)
    
    n_chains = len(initial_params_list)
    pool_size = min(num_workers, n_chains)

    # Utilisation d'un Manager pour la Queue partagée entre processus
    with mp.Manager() as manager:
        progress_queue = manager.Queue() if show_progress else None
        
        # On utilise apply_async pour ne pas bloquer le thread principal
        # et pouvoir lire la queue en temps réel.
        with mp.Pool(processes=pool_size) as pool:
            async_results = []
            for start_p in initial_params_list:
                args = (
                    fitter.compute_log_likelihood,
                    np.array(start_p),
                    initial_cov,
                    fitter.compute_log_prior,
                    max_steps,
                    method,
                    fitter.adapt_every,
                    record_full_trace,
                    progress_queue,
                    threads_per_worker
                )
                async_results.append(pool.apply_async(_worker_task_wrapper, args))

            # Affichage de la barre de progression globale
            if show_progress:
                total_iterations = n_chains * max_steps
                with tqdm(total=total_iterations, desc="Parallel MCMC Sampling") as pbar:
                    completed_iterations = 0
                    while completed_iterations < total_iterations:
                        # On récupère les increments de progression (bloquant court)
                        try:
                            # On vide la queue par paquets pour plus de fluidité
                            count = 0
                            while not progress_queue.empty():
                                count += progress_queue.get_nowait()
                            
                            if count > 0:
                                pbar.update(count)
                                completed_iterations += count
                        except:
                            pass
                        
                        # Vérifier si tous les processus ont fini ou crashé 
                        # pour éviter une boucle infinie
                        if all(r.ready() for r in async_results) and progress_queue.empty():
                            break
            
            # Récupération finale des chaînes
            chains = [r.get() for r in async_results]
    
    return chains