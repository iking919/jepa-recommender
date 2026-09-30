"""
Model architectures for the JEPA Recommender project.
"""

from .bert4rec import BERT4Rec
from .jepa import JEPA
from .transformer import TransformerEncoder

__all__ = [
    "BERT4Rec",
    "JEPA",
    "TransformerEncoder",
]