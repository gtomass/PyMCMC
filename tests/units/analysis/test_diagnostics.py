import numpy as np
from PyMCMC import Chain, MCMCAnalyzer

def test_gelman_rubin_logic():
    c1, c2 = Chain(ndim=1, max_steps=100), Chain(ndim=1, max_steps=100)
    for i in range(100):
        c1.add_state(np.array([0.0]), ln_likelihood=0.0)
        c2.add_state(np.array([10.0]), ln_likelihood=0.0)
    r_hat = MCMCAnalyzer.calculate_gelman_rubin([c1, c2])
    assert r_hat[0] > 1.1

def test_automatic_burn_in_logic():
    c = Chain(1, 1000)
    for i in range(200): c.add_state(np.array([100 * (1 - i/200)]), ln_likelihood=0.0)
    for i in range(800): c.add_state(np.array([np.random.normal(0, 0.1)]), ln_likelihood=0.0)
    ana = MCMCAnalyzer(c, fitter=None, burn_in_fraction='auto')
    assert ana.burn_in_fraction >= 0.2