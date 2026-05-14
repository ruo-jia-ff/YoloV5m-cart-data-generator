"""POPS Engine — modular cart tracking + classification + scoring pipeline."""
from .scoring import compute_pops, classify_event
from .config import SAMPLE_VIDEOS

__all__ = [
    "compute_pops", "classify_event",
    "SAMPLE_VIDEOS",
]
