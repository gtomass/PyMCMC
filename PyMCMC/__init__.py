# PyMCMC/__init__.py

from .chain import Chain
from .fitter import FunctionFitter, Prior
from .analyzer import MCMCAnalyzer
from .sampler import Sampler
from .parallel_tempering import ParallelTempering
from .parallel_mcmc import run_parallel_mcmc

# Optionnel : Définir la version de votre bibliothèque
__version__ = "1.0.0"

# Définir ce qui est exporté lors d'un "from PyMCMC import *"
__all__ = [
    "Chain",
    "FunctionFitter",
    "Prior",
    "MCMCAnalyzer",
    "Sampler",
    "ParallelTempering",
    "run_parallel_mcmc",
]