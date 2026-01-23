import numpy as np
import os
import pytest
from PyMCMC import FunctionFitter

def test_fit_resume_workflow(tmp_path):
    ckpt = os.path.join(tmp_path, "ckpt.npz")
    fitter = FunctionFitter(lambda x, p: p[0]*x, np.linspace(0,10,5), 2.0*np.linspace(0,10,5))
    c1 = fitter.fit([1.0], n_iterations=500, checkpoint_path=ckpt)
    c2 = fitter.fit([1.0], n_iterations=1000, checkpoint_path=ckpt, resume=True)
    assert c2.n_entries == 1001
    assert np.array_equal(c1.samples[:501], c2.samples[:501])

def test_thinning_workflow():
    fitter = FunctionFitter(lambda x, p: p[0]*x, np.array([1]), np.array([2]))
    chain = fitter.fit([1.0], n_iterations=100, thin=10)
    assert chain.n_entries == 11

def test_likelihood_covariance_workflow():
    fitter = FunctionFitter(model_func=lambda x, p: p, x_data=np.array([1.0]), 
                            y_data=np.array([1.0]), data_cov=np.array([[0.25]]))
    lnL = fitter.compute_log_likelihood(np.array([1.0]))
    expected = -0.5 * (np.log(0.25) + np.log(2 * np.pi))
    assert lnL == pytest.approx(expected)

def test_ui_progress_bar():
    fitter = FunctionFitter(lambda x, p: p[0]*x, np.linspace(0,1,10), 2.0*np.linspace(0,1,10))
    chain = fitter.fit([1.0], n_iterations=100, show_progress=True)
    assert chain.n_entries == 101