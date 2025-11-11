"""BiasXplainer package public API.

Provides a small set of convenience exports for the main analyzer and core
components.
"""

__version__ = "1.0.0"
__author__ = "Dyuti Dasmahapatra"
__description__ = "A framework for auditing and mitigating gendered stereotypes in career recommendation systems"

from .core.bias_detector import BiasDetector
from .core.counterfactuals import CounterfactualGenerator
from .core.explainer import SHAPExplainer
from .main import BiasGuardPro

__all__ = ["BiasGuardPro", "BiasDetector", "SHAPExplainer", "CounterfactualGenerator"]
