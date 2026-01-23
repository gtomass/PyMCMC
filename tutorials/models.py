import numpy as np

def linear_model(x, p):
    """A simple linear model for testing."""
    return p[0] * x + p[1]

def multimodal_log_likelihood(params):
    """A complex likelihood with two peaks."""
    p = params[0]
    peak1 = np.exp(-0.5 * (p - 2.0)**2 / 0.1**2)
    peak2 = np.exp(-0.5 * (p - 8.0)**2 / 0.1**2)
    return np.log(peak1 + peak2 + 1e-100)