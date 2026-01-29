from __future__ import annotations
import numpy as np
from typing import Optional, Tuple, Callable, Dict, Type, Any
from .chain import Chain

# Optional progress bar support
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


class MCMCMethod:
    """Base class for MCMC sampling strategies (MH, AM, DRAM)."""

    def __init__(self, sampler: Sampler):
        self.sampler = sampler

    def _calculate_log_alpha(self, log_pi_old: float, log_pi_new: float) -> float:
        """Calculate ln(alpha) with protection against -inf."""
        if np.isinf(log_pi_new) and log_pi_new < 0:
            return -np.inf # Rejet immédiat si le nouveau est invalide
        if np.isinf(log_pi_old) and log_pi_old < 0:
            return 0.0 # Acceptation si l'ancien était invalide et le nouveau est "mieux" (ou moins pire)
            
        return min(0.0, log_pi_new - log_pi_old)

    def _log_one_minus_exp(self, val: float) -> float:
        """Numerically stable ln(1 - exp(val)) for val <= 0."""
        if val >= 0:
            return -1e20  # Numerical approximation of log(0)
        if val < -700:
            return 0.0  # exp(val) is effectively zero
        return np.log1p(-np.exp(val))

    def step(self) -> Tuple[np.ndarray, float, bool]:
        """Perform one iteration. Must be implemented by subclasses."""
        raise NotImplementedError


class MetropolisHastings(MCMCMethod):
    """Standard Metropolis-Hastings implementation."""

    def step(self) -> Tuple[np.ndarray, float, bool]:
        proposal = self.sampler.propose(self.sampler.proposal_cov)
        log_prior_prop = self.sampler.log_prior_func(proposal)

        if log_prior_prop == -np.inf:
            return self.sampler.current_state, self.sampler.current_log_lik, False

        log_lik_prop = self.sampler.evaluate_log_lik(proposal)
        
        # Log-acceptance ratio
        current_prob = self.sampler.current_log_lik + self.sampler.current_log_prior
        proposed_prob = log_lik_prop + log_prior_prop
        
        if np.log(np.random.rand()) < (proposed_prob - current_prob):
            return proposal, log_lik_prop, True
        
        return self.sampler.current_state, self.sampler.current_log_lik, False


class AM(MetropolisHastings):
    """Adaptive Metropolis. Logic is identical to MH; adaptation is handled in Sampler."""
    pass


class DRAM(MCMCMethod):
    """Delayed Rejection Adaptive Metropolis (DRAM) with two-stage rejection."""

    def __init__(self, sampler: Sampler, gamma: float = 0.2):
        super().__init__(sampler)
        self.gamma = gamma

    def step(self) -> Tuple[np.ndarray, float, bool]:
        x = self.sampler.current_state
        log_pi_x = self.sampler.current_log_lik + self.sampler.current_log_prior

        # --- STAGE 1 ---
        prop1 = self.sampler.propose(self.sampler.proposal_cov)
        lp1 = self.sampler.log_prior_func(prop1)
        ll1 = self.sampler.evaluate_log_lik(prop1) if lp1 != -np.inf else -np.inf
        log_pi_y1 = ll1 + lp1

        log_alpha1 = self._calculate_log_alpha(log_pi_x, log_pi_y1)
        acceptance_u = np.log(np.random.rand())

        if acceptance_u < log_alpha1:
            return prop1, ll1, True

        # --- STAGE 2 (Delayed Rejection) ---
        # Propose with scaled down covariance
        prop2 = self.sampler.propose(self.sampler.proposal_cov * self.gamma)
        lp2 = self.sampler.log_prior_func(prop2)
        ll2 = self.sampler.evaluate_log_lik(prop2) if lp2 != -np.inf else -np.inf
        log_pi_y2 = ll2 + lp2

        # Symmetry term for reversibility
        log_alpha1_y2_y1 = self._calculate_log_alpha(log_pi_y2, log_pi_y1)

        # DRAM Alpha2 formula
        term1 = log_pi_y2 - log_pi_x
        term2 = self._log_one_minus_exp(log_alpha1_y2_y1)
        term3 = self._log_one_minus_exp(log_alpha1)

        log_alpha2 = min(0.0, term1 + term2 - term3)

        if acceptance_u < log_alpha2:
            return prop2, ll2, True

        return x, self.sampler.current_log_lik, False


class Sampler:
    """
    MCMC Engine responsible for state transitions, adaptation, and data recording.
    """

    def __init__(
        self,
        log_lik_func: Callable[[np.ndarray], float],
        initial_state: np.ndarray,
        proposal_covariance: np.ndarray,
        log_prior_func: Optional[Callable[[np.ndarray], float]] = None,
        max_steps: int = 10000,
        method: str = "DRAM",
        adapt_every: int = 100,
        thin: int = 1,
        beta: float = 1.0,
    ):
        # Functions and hyperparameters
        self.raw_log_lik = log_lik_func
        self.log_prior_func = log_prior_func if log_prior_func else lambda p: 0.0
        self.beta = beta
        
        # State variables
        self.current_state = np.array(initial_state)
        self.ndim = len(initial_state)
        self.proposal_cov = np.array(proposal_covariance)
        self.max_steps = max_steps
        self.adapt_every = adapt_every
        self.thin = thin
        
        # Counters
        self.n_steps = 1
        self.n_accepted = 0
        
        # Initial evaluations
        self.current_log_prior = self.log_prior_func(self.current_state)
        self.current_log_lik = self.evaluate_log_lik(self.current_state)
        
        # Chain storage
        self.chain = Chain(self.ndim, max_steps)
        self.chain.add_state(self.current_state, self.current_log_lik)

        # Strategy pattern for MCMC methods
        method_map: Dict[str, Type[MCMCMethod]] = {
            "MH": MetropolisHastings,
            "AM": AM,
            "DRAM": DRAM
        }
        if method not in method_map:
            raise ValueError(f"Method {method} not supported. Choose from MH, AM, DRAM.")
        
        self.method_engine = method_map[method](self)

    def evaluate_log_lik(self, params: np.ndarray) -> float:
        """Calculate tempered log-likelihood."""
        result = self.beta * self.raw_log_lik(params)

        return float(np.asarray(result).item())

    def step(self, record_full_trace: bool = True) -> None:
        """Execute a single MCMC step and update internal state."""
        new_state, new_ll, accepted = self.method_engine.step()

        if accepted:
            self.current_state = new_state
            self.current_log_lik = new_ll
            self.current_log_prior = self.log_prior_func(new_state)
            self.n_accepted += 1

        # Chain recording logic
        if self.n_steps % self.thin == 0:
            if accepted or record_full_trace:
                self.chain.add_state(self.current_state, self.current_log_lik)
            else:
                # Increment weight of the existing last state (compressed storage)
                if self.chain.n_entries > 0:
                    self.chain.weights[self.chain.n_entries - 1] += 1

        # Covariance Adaptation
        if self.n_steps % self.adapt_every == 0 and self.chain.n_entries > self.ndim * 2:
            self._adapt_covariance()

        self.n_steps += 1

    def run(
        self,
        record_full_trace: bool = True,
        checkpoint_path: Optional[str] = None,
        checkpoint_every: int = 1000,
        show_progress: bool = False,
        progress_queue: Optional[Any] = None # Ajout de la queue pour le parallélisme
    ) -> Chain:
        """Run the sampler for the allocated steps."""
        remaining_steps = max(0, self.max_steps - (self.n_steps - 1))

        # Progress bar setup (mode série)
        use_tqdm = show_progress and tqdm is not None and progress_queue is None
        pbar = None
        if use_tqdm:
            pbar = tqdm(total=self.max_steps, initial=self.n_steps-1, desc="MCMC Sampling")
        
        for _ in range(int(remaining_steps)):
            self.step(record_full_trace=record_full_trace)
            
            if pbar:
                pbar.update(1)
            
            # Envoi du signal de progression au processus principal
            if progress_queue:
                progress_queue.put(1)
            
            # Periodic Checkpointing
            if checkpoint_path and (self.n_steps % checkpoint_every == 0):
                self.chain.save(checkpoint_path)
                
        if pbar:
            pbar.close()

        if checkpoint_path:
            self.chain.save(checkpoint_path)
            
        return self.chain

    def _adapt_covariance(self) -> None:
        """Update proposal covariance using the Gelman/Rubin scaling factor."""
        # Optimal scaling for Gaussian targets: (2.38^2 / ndim)
        scaling_factor = (2.38**2) / self.ndim
        current_cov = self.chain.get_covariance()
        
        # Add small regularization term to ensure positive-definiteness
        self.proposal_cov = scaling_factor * (current_cov + np.eye(self.ndim) * 1e-6)

    def propose(self, covariance: np.ndarray) -> np.ndarray:
        """Generate a new state proposal using Cholesky decomposition for speed."""
        try:
            # Using Cholesky is significantly faster than multivariate_normal
            cholesky_l = np.linalg.cholesky(covariance)
            return self.current_state + cholesky_l @ np.random.randn(self.ndim)
        except np.linalg.LinAlgError:
            # Fallback for non-positive definite matrices during early adaptation
            return np.random.multivariate_normal(self.current_state, covariance)
        
    def get_acceptance_rate(self) -> float:
        """Calculate the current acceptance rate."""
        return self.n_accepted / self.n_steps if self.n_steps > 0 else 0.0