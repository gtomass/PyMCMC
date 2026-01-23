import numpy as np
import pytest
from PyMCMC import Sampler, ParallelTempering, FunctionFitter

def test_covariance_adaptation():
    s = Sampler(lambda p: -0.5*p**2, np.array([0.0]), np.eye(1), adapt_every=10)
    for i in range(15): s.chain.add_state(np.array([float(i)]), 0.0)
    old_cov = s.proposal_cov.copy()
    s._adapt_covariance()
    assert not np.array_equal(old_cov, s.proposal_cov)
    assert s.proposal_cov[0,0] > 1.0

def test_pt_swap_probability():
    fitter = FunctionFitter(custom_log_lik=lambda p: -0.5 * p**2)
    pt = ParallelTempering(fitter, n_temps=2, t_max=10.0)
    diff_beta = pt.betas[0] - pt.betas[1]
    prob = np.exp(diff_beta * (-2.0 - 0.0))
    assert prob == pytest.approx(0.16529, abs=1e-4)