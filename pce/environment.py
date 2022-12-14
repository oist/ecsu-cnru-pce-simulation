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
    objects_facing_agents: bool
    agents_pos: np.ndarray             # agents starting position - must be provided    
    agents_reverse_motors: List        # list of boolean, one per agent, where True means the agent is facing outside, False inside
    objs_pos: np.ndarray
    
    # initialized    
    agents_signal: np.ndarray = None    # agents sensors
    objs_pos: np.ndarray = None         # objects positions
    shadows_pos: np.ndarray = None      # shadow positions
    
    # previous values
    agents_prev_pos: np.ndarray = None              # agents pos before step
    agents_prev_neural_states: np.ndarray = None    # agents neural states before step
    agents_prev_neural_outputs: np.ndarray = None   # agents neural outputs before step

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

    def wrap_around_diff_rel(self, a, b):
        diff = a - b
        abs_diff = abs(a -b)
        if (diff >=  -300 and diff <= -150) or (diff >= 0 and diff <= 150): # agent b is Left side of agent a
            return min(self.env_length - abs_diff, abs_diff)
        else: # agent b is Right side of agent a
            return -min(self.env_length -abs_diff, abs_diff)

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
                # the agent overlaps with the facing object
                (
                    self.objects_facing_agents
                    and
                    self.wrap_around_diff(self.agents_pos[a], self.objs_pos[a]) <= self.agent_width
                )
                or
                # the agent overlaps with one of the objects on line
                ( 
                    not self.objects_facing_agents
                    and
                    any( 
                        self.wrap_around_diff(self.agents_pos[a], self.objs_pos[o]) <= self.agent_width
                        for o in range(self.num_objects)
                    )
                )
            )

    def copy_var_previous_step(self):
        self.agents_prev_pos = np.copy(self.agents_pos)
        self.agents_prev_neural_states = np.array([a.brain.states for a in self.agents])
        self.agents_prev_neural_outputs = np.array([a.brain.output for a in self.agents])

    def make_one_step(self, step, ghost_index=None, ghost_pos_trial=None, last_step=False):
        self.copy_var_previous_step()
        self.compute_agents_signals()
        for i, a in enumerate(self.agents):
            if ghost_index==i:
                if not last_step:
                    # otherwise step+1 goes out of bounds                    
                    self.agents_pos[ghost_index] = ghost_pos_trial[step+1]
                continue
            a.compute_brain_input(self.agents_signal[i]) # updates brain_inputs
            a.brain.euler_step() # updates brain_states and brain_outputs            
            a.compute_motor_outputs()
            self.agents_pos[i] += a.get_velocity(reverse=self.agents_reverse_motors[i])
                
        self.agents_pos = self.wrap_around(self.agents_pos)        



