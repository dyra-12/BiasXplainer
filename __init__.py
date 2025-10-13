__version__ = "1.0.0"
__author__ = "Dyuti Dasmahapatra"
__description__ = "A framework for auditing and mitigating gendered stereotypes in career recommendation systems"

from .main import BiasGuardPro
from .core.bias_detector import BiasDetector
from .core.explainer import SHAPExplainer
from .core.counterfactuals import CounterfactualGenerator

__all__ = [
    "BiasGuardPro",
    "BiasDetector", 
    "SHAPExplainer",
    "CounterfactualGenerator"
]