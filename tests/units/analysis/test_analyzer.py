import numpy as np
from PyMCMC import Chain, MCMCAnalyzer, FunctionFitter

def test_analyzer_merging():
    c1, c2 = Chain(ndim=1, max_steps=100), Chain(ndim=1, max_steps=100)
    for i in range(100):
        c1.add_state(np.array([1.0]), ln_likelihood=-1.0)
        c2.add_state(np.array([2.0]), ln_likelihood=-0.5)

    ana = MCMCAnalyzer([c1, c2], fitter=None, burn_in_fraction=0.2)
    assert ana.ndim == 1
    assert len(ana.samples) == 160
    assert ana.merged_chain.best_ln_likelihood == -0.5

def test_information_criteria():
    c = Chain(ndim=1, max_steps=100)
    for i in range(100): c.add_state(np.array([1.0]), ln_likelihood=-1.5)
    fitter = FunctionFitter(model_func=lambda x, p: p, x_data=np.arange(10), y_data=np.ones(10))
    ana = MCMCAnalyzer(c, fitter=fitter, burn_in_fraction=0.1)
    
    ic = ana.compute_information_criteria()
    assert "WAIC" in ic
    assert ic["AIC"] > 0
    assert not np.isnan(ic["BIC"])