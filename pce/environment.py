import numpy as np
from typing import List
from dataclasses import dataclass
from pce.agent import Agent


@dataclass
class Environment:
    # agents
    agents: List[Agent]
    init_state: float
    
    # env setting
    env_length: float
    agent_width:float
    no_shadow: bool
    shadow_delta: float
    agents_pos: np.ndarray             # agents starting position - must be provided    
    objs_pos: np.ndarray    
    
    # initialized    
    agents_signal: np.ndarray = None    # agents sensors
    objs_pos: np.ndarray = None         # objects positions
    shadows_pos: np.ndarray = None      # shadow positions

    def __post_init__(self):
        self.num_agents = len(self.agents)
        self.num_objects = len(self.objs_pos)
        self.env_length_half = self.env_length / 2
        self.agents_signal = np.zeros(self.num_agents)  

        for a in self.agents:
            a.init_params(self.init_state)

        self.compute_agents_signals() 

    def wrap_around(self, data):
        return data % self.env_length

    def wrap_around_diff(self, a, b):
        abs_diff = abs(a - b)
        return min(self.env_length - abs_diff, abs_diff)

    def wrap_around_diff_array(self, a):
        abs_diff = np.abs(np.diff(a, axis=1))
        a = np.column_stack([self.env_length - abs_diff, abs_diff])
        return np.min(a, axis=1)
        
    def compute_agents_signals(self):
        if not self.no_shadow:
            self.shadows_pos = self.wrap_around(self.agents_pos - self.shadow_delta)

        for a in range(self.num_agents):                
            self.agents_signal[a] = (
                # agent overlaps with any other agent
                any( 
                    self.wrap_around_diff(self.agents_pos[a], self.agents_pos[j]) <= self.agent_width
                    for j in range(self.num_agents) if j != a
                )
                or
                # agent a overlaps with any other shadow                    
                (
                    not self.no_shadow # only if shadows are active
                    and
                    any(                     
                        self.wrap_around_diff(self.agents_pos[a], self.shadows_pos[j]) <= self.agent_width
                        for j in range(self.num_agents) if j != a
                    )
                )
                or
                any( # the agent a overlaps with one of the objects
                    self.wrap_around_diff(self.agents_pos[a], self.objs_pos[o]) <= self.agent_width
                    for o in range(self.num_objects)
                )
            )

    def make_one_step(self):
        self.compute_agents_signals()

        agents_pos_copy = np.copy(self.agents_pos)

        for i, a in enumerate(self.agents):
            a.compute_brain_input(self.agents_signal[i]) # updates brain_inputs
            a.brain.euler_step() # updeates brain_states and brain_outputs
            a.compute_motor_outputs()    
            self.agents_pos[i] += a.get_velocity()
        
        self.agents_pos = self.wrap_around(self.agents_pos)        

        return agents_pos_copy



