"""Minimal public implementation of the finalized AMYCLIP model."""

from .model import AMYCLIP, load_checkpoint

__all__ = ["AMYCLIP", "load_checkpoint"]
