import numpy as np
import pytest
import os
from PyMCMC import Chain

def test_chain_save_load(tmp_path):
    filename = os.path.join(tmp_path, "test_chain.npz")
    c = Chain(ndim=2, max_steps=100)
    c.add_state(np.array([1.0, 2.0]), ln_likelihood=-0.5)
    c.add_state(np.array([1.1, 2.1]), ln_likelihood=-0.2)
    c.save(filename)
    
    c_loaded = Chain.load(filename)
    assert c_loaded.ndim == 2
    assert c_loaded.n_entries == 2
    assert np.allclose(c_loaded.samples[1], [1.1, 2.1])
    assert c_loaded.best_ln_likelihood == -0.2

def test_hdi_calculation():
    c = Chain(ndim=1, max_steps=100)
    data = []
    for i in range(1, 11): data.extend([i] * i)
    for val in data: c.add_state(np.array([float(val)]), ln_likelihood=0.0)
    
    hdi = c.compute_hdi(cred_mass=0.8)
    assert hdi[0, 0] <= hdi[0, 1]
    assert hdi[0, 1] == 10.0
    assert hdi[0, 0] > 1.0

def test_ess_calculation_deterministic():
    c = Chain(ndim=1, max_steps=100)
    for _ in range(100): c.add_state(np.array([5.0]), ln_likelihood=0.0)
    ess = c.get_ess()
    assert ess[0] < 10.0

def test_chain_weighted_stats():
    c = Chain(ndim=1, max_steps=10)
    c.add_state(np.array([1.0]), ln_likelihood=0.0, weight=1.0)
    c.add_state(np.array([2.0]), ln_likelihood=0.0, weight=3.0)
    assert c.get_mean()[0] == 1.75
    
def test_chain_resizing():
    c = Chain(ndim=1, max_steps=2)
    c.add_state(np.array([1.0]), 0.0)
    c.add_state(np.array([2.0]), 0.0)
    c.add_state(np.array([3.0]), 0.0)
    assert c.current_capacity == 4
    assert c.n_entries == 3