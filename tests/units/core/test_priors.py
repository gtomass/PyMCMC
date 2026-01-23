import numpy as np
import pytest
from PyMCMC import Prior

def test_uniform_prior():
    p = Prior('uniform', (0, 10))
    assert p.compute_log_pdf(5) == 0.0
    assert p.compute_log_pdf(-1) == -np.inf

def test_gaussian_prior():
    p = Prior('gaussian', (0, 1))
    assert p.compute_log_pdf(0) == 0.0
    assert p.compute_log_pdf(1) == -0.5

def test_log_normal_prior():
    p_log = Prior('log_normal', (0, 1))
    assert p_log.compute_log_pdf(1.0) == pytest.approx(-0.9189, abs=1e-4)
    assert p_log.compute_log_pdf(-1.0) == -np.inf

def test_beta_prior():
    p_beta = Prior('beta', (2, 2))
    assert p_beta.compute_log_pdf(0.5) == pytest.approx(np.log(1.5)) 
    assert p_beta.compute_log_pdf(1.5) == -np.inf