"""
Main parameters for simulation experiment environment and agent body.
"""
from pyevolver.evolution import MIN_SEARCH_VALUE, MAX_SEARCH_VALUE
import numpy as np

EVOLVE_GENE_RANGE = (MIN_SEARCH_VALUE, MAX_SEARCH_VALUE)

ENV_LENGTH: float = 600             # lenght of a circle
AGENT_WIDTH: float = 4              # width of players (and their ghosts)
OBJ_WIDTH: float  = 4               # width of objects
SHADOW_DELTA: float = 150           # distance between agnet and its shadow

AGENTS_INITIAL_POS = ENV_LENGTH * np.array([
    [0,   1/2],
    [1/2, 0],
    [1/4, 3/4],
    [3/4, 1/4],

])