"""
Main parameters for simulation experiment environment and agent body.
"""
from pyevolver.evolution import MIN_SEARCH_VALUE, MAX_SEARCH_VALUE
import numpy as np
from numpy.random import RandomState

EVOLVE_GENE_RANGE = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)

ENV_LENGTH: float = 300             # lenght of a circle
AGENT_WIDTH: float = 4              # width of players (and their ghosts)
OBJ_WIDTH: float  = 4               # width of objects
SHADOW_DELTA: float = ENV_LENGTH/4  # distance between agnet and its shadow

AGENTS_INITIAL_POS = RandomState(0).uniform(low=0, high=ENV_LENGTH, size=(100,2))
