"""Pluggable protein/drug encoders kept out of models.py.

models.py imports from here and does nothing else encoder-specific, so new
encoders land in this package rather than growing the model file - which keeps
branches like 3d-saprot mergeable against each other.
"""
