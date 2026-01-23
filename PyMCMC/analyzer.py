from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Union, Optional, Dict, Any
from scipy.special import logsumexp

from .chain import Chain
from .fitter import FunctionFitter

class MCMCAnalyzer:
    """
    Post-processing tool for MCMC results. 
    
    Provides convergence diagnostics (R-hat, Geweke), statistical summaries, 
    and publication-quality visualizations (Corner plots, Posterior Predictive).
    """

    def __init__(
        self, 
        chains: Union[Chain, List[Chain]], 
        fitter: Optional[FunctionFitter] = None, 
        burn_in_fraction: Union[float, str] = 'auto', 
        check_convergence: bool = True
    ):
        """
        Initialize the analyzer and prepare merged data.

        Args:
            chains: Single Chain object or a list of Chain objects.
            fitter: The FunctionFitter instance used for the sampling.
            burn_in_fraction: Fraction to discard (e.g., 0.2) or 'auto' for Geweke-based detection.
            check_convergence: If True, calculates the Gelman-Rubin R-hat.
        """
        self.fitter = fitter
        
        # Ensure chains are in a list
        self.original_chains = [chains] if isinstance(chains, Chain) else chains
        self.ndim = self.original_chains[0].ndim

        # 1. Burn-in handling
        if burn_in_fraction == 'auto':
            self.burn_in_fraction = self._find_optimal_burn_in()
            print(f"Automatic Burn-in detected: {self.burn_in_fraction*100:.1f}%")
        else:
            self.burn_in_fraction = float(burn_in_fraction)

        # 2. Convergence Check
        if len(self.original_chains) > 1 and check_convergence:
            r_hat = self.calculate_gelman_rubin(self.original_chains)
            print(f"Convergence Diagnostic (R-hat): {r_hat}")
            if np.any(r_hat > 1.1):
                print("WARNING: Chains may not have converged (R-hat > 1.1).")
        
        # 3. Merging and Data Prep
        self.merged_chain = self._merge_chains(self.original_chains, self.burn_in_fraction)
        self.samples = self.merged_chain.current_samples
        self.ln_likelihoods = self.merged_chain.current_ln_likelihoods
        # Chi2 defined as -2 * lnL (including constants if present)
        self.chi2_trace = -2 * self.ln_likelihoods

    def _merge_chains(self, chains: List[Chain], burn_in: float) -> Chain:
        """Discard burn-in and combine multiple chains into one."""
        all_samples, all_ln_liks = [], []
        
        for c in chains:
            start_idx = int(c.n_entries * burn_in)
            all_samples.append(c.samples[start_idx:c.n_entries])
            all_ln_liks.append(c.ln_likelihoods[start_idx:c.n_entries])
            
        total_steps = sum(len(s) for s in all_samples)
        merged = Chain(self.ndim, total_steps)
        merged.samples = np.vstack(all_samples)
        merged.ln_likelihoods = np.concatenate(all_ln_liks)
        merged.weights = np.ones(total_steps)
        merged.n_entries = total_steps
        
        # Update MAP for the merged set
        best_idx = np.argmax(merged.ln_likelihoods)
        merged.best_params = merged.samples[best_idx].copy()
        merged.best_ln_likelihood = merged.ln_likelihoods[best_idx]
        
        return merged

    @staticmethod
    def calculate_gelman_rubin(chains: List[Chain]) -> np.ndarray:
        """Calculate the R-hat convergence statistic with safety for zero variance."""
        m = len(chains)
        n = chains[0].n_entries
        
        means = np.array([c.get_mean() for c in chains])
        vars_diag = np.array([np.diag(c.get_covariance()) for c in chains])
        
        grand_mean = np.mean(means, axis=0)
        b = (n / (m - 1)) * np.sum((means - grand_mean)**2, axis=0)
        w = np.mean(vars_diag, axis=0)
        
        var_plus = ((n - 1) / n) * w + (b / n)
        
        # Handling zero within-chain variance
        # ----------------------------------------
        if np.any(w == 0):
            r_hat = np.ones_like(w)
            # If W=0 but means differ (B>0), R-hat is infinite (total non-convergence)
            # If W=0 and B=0, chains are identical and stationary (R-hat = 1.0)
            mask_zero_w = (w == 0)
            r_hat[mask_zero_w] = np.where(b[mask_zero_w] > 0, np.inf, 1.0)
            
            # Classic calculation for parameters with non-zero variance
            mask_positive_w = (w > 0)
            if np.any(mask_positive_w):
                r_hat[mask_positive_w] = np.sqrt(var_plus[mask_positive_w] / w[mask_positive_w])
            return r_hat
        # ----------------------------------------

        return np.sqrt(var_plus / w)

    def compute_information_criteria(self) -> Dict[str, float]:
        """Calculate AIC, BIC, and WAIC (if model data is available)."""

        y_data = self.fitter.y_data if (self.fitter and hasattr(self.fitter, 'y_data')) else None
        n_data = len(y_data) if y_data is not None else None
        k = self.ndim
        max_ln_lik = self.merged_chain.best_ln_likelihood
        
        aic = 2 * k - 2 * max_ln_lik
        results = {"AIC": aic, "BIC": np.nan, "WAIC": np.nan}

        if n_data:
            results["BIC"] = k * np.log(n_data) - 2 * max_ln_lik
            
            # WAIC Calculation (Pointwise approximation)
            n_sub = min(1000, len(self.samples))
            idx = np.random.choice(len(self.samples), n_sub, replace=False)
            sub_samples = self.samples[idx]
            
            log_lik_vec = np.array([self.fitter.compute_log_likelihood(p) for p in sub_samples])
            lppd = logsumexp(log_lik_vec) - np.log(n_sub)
            p_waic = np.var(log_lik_vec)
            results["WAIC"] = -2 * (lppd - p_waic)

        return results

    def print_summary(self, param_names: Optional[List[str]] = None, cred_mass: float = 0.95) -> None:
        """Print a full statistical report of the posterior."""
        means = self.merged_chain.get_mean()
        stds = np.sqrt(np.diag(self.merged_chain.get_covariance()))
        map_p = self.merged_chain.get_map_estimate()
        hdi = self.merged_chain.compute_hdi(cred_mass=cred_mass)
        names = param_names if param_names else [f"P{i}" for i in range(self.ndim)]
        
        print("\n" + "="*85)
        print(f"{'Parameter':<15} | {'Mean ± std':<20} | {'HDI '+str(int(cred_mass*100))+'%':<20} | {'MAP':<10}")
        print("-" * 85)
        for i, (m, s, h, mp) in enumerate(zip(means, stds, hdi, map_p)):
            str_meansd = f"{m:.4f} ± {s:.4f}"
            str_hdi = f"[{h[0]:.3f}, {h[1]:.3f}]"
            print(f"{names[i]:<15} | {str_meansd:<20} | {str_hdi:<20} | {mp:8.4f}")
        print("="*85)

        chi2_min = self.get_minimal_chi2()
        n_data = len(self.fitter.y_data) if self.fitter.y_data is not None else None

        if n_data:
            dof = n_data - self.ndim
            print(f"\n Minimum reduced Chi2: {chi2_min/dof:.3f}")
            ic = self.compute_information_criteria()
            print(f" AIC: {ic['AIC']:.3f} | BIC: {ic['BIC']:.3f} | WAIC: {ic['WAIC']:.3f}")
        else:
            print(f"\n Minimum Chi2 (-2*lnL): {chi2_min:.3f}")
            print(f" AIC: {2 * k + chi2_min:.3f}")

    def get_minimal_chi2(self) -> float:
        """Recalculate pure Chi2 at MAP or return -2*max_lnL."""
        if self.fitter.model_func is None or self.fitter.y_data is None:
            return np.min(self.chi2_trace)
            
        best_p = self.merged_chain.get_map_estimate()
        pred = self.fitter.model_func(self.fitter.x_data, best_p)
        res = self.fitter.y_data - pred
        
        if self.fitter.data_cov is not None:
            from scipy import linalg
            alpha = linalg.solve_triangular(self.fitter.L_data, res, lower=True)
            return np.sum(alpha**2)
        return np.sum((res / self.fitter.y_err)**2)

    def plot_traces(self, true_params: Optional[np.ndarray] = None) -> None:
        """Visualize chain mixing for each parameter."""
        fig, axes = plt.subplots(self.ndim, 1, figsize=(10, max(1.8 * self.ndim, 6)), sharex=True)
        if self.ndim == 1: axes = [axes]
        
        for i, ax in enumerate(axes):
            for idx, c in enumerate(self.original_chains):
                ax.plot(c.samples[:c.n_entries, i], alpha=0.5, label=f"Chain {idx}")
            ax.set_ylabel(f"P{i}")
            if true_params is not None:
                ax.axhline(true_params[i], color='red', linestyle='--')
        
        axes[0].legend(loc='upper right', ncol=len(self.original_chains), fontsize='small')
        plt.tight_layout()
        plt.show()

    def plot_posterior_predictive(self, n_samples: int = 100, intervals: List[int] = [68, 95]) -> None:
        """Plot model uncertainty bands against the data."""
        if self.fitter.x_data is None or self.fitter.model_func is None:
            print("Cannot plot predictive: Missing model or x_data.")
            return

        x_grid = np.linspace(self.fitter.x_data.min(), self.fitter.x_data.max(), 250)
        idx = np.random.choice(len(self.samples), n_samples, replace=False)
        sub_samples = self.samples[idx]
        
        preds = np.array([self.fitter.model_func(x_grid, p) for p in sub_samples])
        
        plt.figure(figsize=(10, 6))
        for i, perc in enumerate(sorted(intervals, reverse=True)):
            low = np.percentile(preds, (100 - perc) / 2, axis=0)
            high = np.percentile(preds, 100 - (100 - perc) / 2, axis=0)
            plt.fill_between(x_grid, low, high, color='blue', alpha=0.1 * (i+1), label=f"{perc}% CI")

        plt.errorbar(self.fitter.x_data, self.fitter.y_data, yerr=self.fitter.y_err, fmt='.k', alpha=0.3, label="Data")
        plt.plot(x_grid, self.fitter.model_func(x_grid, self.merged_chain.get_map_estimate()), 'r-', label="MAP")
        plt.legend()
        plt.title("Posterior Predictive Distribution")
        plt.show()

    def plot_corner(self, param_names: Optional[List[str]] = None, bins: int = 30) -> None:
        """Create a triangle plot of parameter correlations."""
        names = param_names if param_names else [f"P{i}" for i in range(self.ndim)]
        map_p = self.merged_chain.get_map_estimate()
        
        fig, axes = plt.subplots(self.ndim, self.ndim, figsize=(max(1.8 * self.ndim, 10), max(1.8 * self.ndim, 6)))
        if self.ndim == 1: axes = np.array([[axes]])

        for i in range(self.ndim):
            for j in range(self.ndim):
                ax = axes[i, j]
                if j > i:
                    ax.axis('off')
                elif i == j:
                    ax.hist(self.samples[:, i], bins=bins, color='gray', alpha=0.7)
                    ax.axvline(map_p[i], color='red', linestyle='--')
                    ax.set_title(names[i], fontsize=10)
                else:
                    ax.hist2d(self.samples[:, j], self.samples[:, i], bins=bins, cmap='Blues')
                    ax.axvline(map_p[j], color='red', alpha=0.3)
                    ax.axhline(map_p[i], color='red', alpha=0.3)
                
                if i < self.ndim - 1: ax.set_xticklabels([])
                if j > 0: ax.set_yticklabels([])
                if i == self.ndim - 1: ax.set_xlabel(names[j])
                if j == 0: ax.set_ylabel(names[i])
                ax.set_aspect('auto')

        plt.tight_layout()
        plt.show()

    def plot_autocorrelation(self, max_lag: int = 100) -> None:
        """
        Display the autocorrelation function for each parameter to check for sample independence.

        Args:
            max_lag (int): Maximum lag to calculate and display.
        """
        try:
            rho = self.merged_chain.estimate_autocorr(max_lag=max_lag)
        except AttributeError:
            print("Error: Chain class must implement estimate_autocorr.")
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        for i in range(self.ndim):
            ax.plot(rho[:, i], label=f"P{i}")

        ax.axhline(0, color='k', linestyle='--', alpha=0.5)
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        ax.set_title("Autocorrelation Functions")
        ax.legend()
        plt.show()

    def plot_correlation_matrix(self, param_names: Optional[List[str]] = None, annot: bool = True) -> None:
        """
        Display the parameter correlation matrix using Matplotlib.

        Uses a RdBu_r diverging colormap centered at zero.

        Args:
            param_names: Parameter labels.
            annot: Whether to annotate each cell with the correlation coefficient.
        """
        corr = self.merged_chain.get_correlation()
        names = param_names if param_names else [f"P{i}" for i in range(self.ndim)]

        # Mask for the upper triangle
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        corr_masked = np.ma.masked_where(mask, corr)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Correlation Coefficient", rotation=270, labelpad=15)

        if annot:
            for i in range(self.ndim):
                for j in range(self.ndim):
                    # Only plot lower triangle and diagonal
                    if i >= j:  
                        val = corr[i, j]
                        color = "white" if abs(val) > 0.5 else "black"
                        ax.text(j, i, f"{val:.2f}",
                                ha="center", va="center", color=color, fontsize=10)

        ax.set_xticks(range(self.ndim))
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_yticks(range(self.ndim))
        ax.set_yticklabels(names)

        # Aesthetics
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_xticks(np.arange(self.ndim+1)-.5, minor=True)
        ax.set_yticks(np.arange(self.ndim+1)-.5, minor=True)
        ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
        ax.tick_params(which="minor", bottom=False, left=False)

        ax.set_title("Parameter Correlation Matrix", fontsize=14, pad=20)
        plt.tight_layout()
        plt.show()

    def _find_optimal_burn_in(self, max_burn: float = 0.5) -> float:
        """Use Geweke test to find where stationarity begins."""
        for burn in np.arange(0, max_burn, 0.05):
            all_stationary = True
            for c in self.original_chains:
                start = int(c.n_entries * burn)
                temp_c = Chain(self.ndim, c.n_entries - start)
                temp_c.samples = c.samples[start:c.n_entries]
                temp_c.n_entries = c.n_entries - start
                
                if np.any(np.abs(temp_c.calculate_geweke_z()) > 2.0):
                    all_stationary = False
                    break
            if all_stationary:
                return burn
        return max_burn