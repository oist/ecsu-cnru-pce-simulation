import numpy as np
from numpy.random import RandomState
from typing import List
from dataclasses import dataclass, field
from pce.agent import Agent
from pce.utils import overlaps


@dataclass
class Environment:
    agents: List[Agent]
    
    # env setting
    env_length: float
    agent_width:float
    obj_width: float
    shadow_delta: float
    agents_pos: np.ndarray             # agents starting position - must be provided    
    objs_pos: np.ndarray
    
    # initialized    
    agents_signal: np.ndarray = None    # agents sensors
    objs_pos: np.ndarray = None         # objects positions
    shadows_pos: np.ndarray = None      # shadow positions

    def __post_init__(self):
        self.objs_left_pos = self.wrap_around(self.objs_pos-self.obj_width/2)
        self.objs_right_pos = self.wrap_around(self.objs_pos+self.obj_width/2)
        self.compute_shadow_pos()
        self.agents_signal = np.zeros(2)  

        init_ctrnn_state = 0. # todo: consider to change this based on random_state
        
        for a in self.agents:
            a.init_params(init_ctrnn_state) 

    def wrap_around(self, data):
        return data % self.env_length
        
    def compute_shadow_pos(self):
        self.shadows_pos = self.wrap_around(self.agents_pos-self.shadow_delta)

    def compute_agents_signals(self):
        agents_left_pos = self.wrap_around(self.agents_pos-self.agent_width/2)
        agents_right_pos = self.wrap_around(self.agents_pos+self.agent_width/2)
        if overlaps(*agents_left_pos, *agents_right_pos):
            # the two agents overalp
            self.agents_signal = np.ones(2)
        else:            
            for a in range(2):                
                self.agents_signal[a] = (
                    # agent a overlaps with the other shadow
                    overlaps(agents_left_pos[a], self.shadows_pos[1-a], agents_right_pos[a], self.shadows_pos[1-a])
                    or
                    any( # the agent a overlaps with one of the two object                                        
                        overlaps(agents_left_pos[a], self.objs_left_pos[o], agents_right_pos[a], self.objs_right_pos[o])
                        for o in range(2)
                    )
                )

    def make_one_step(self):
        agents_pos = np.copy(self.agents_pos)        
        shadows_pos = np.copy(self.shadows_pos)
        objs_pos = np.copy(self.objs_pos)
        agents_signal = np.copy(self.agents_signal)
        agents_vel = np.zeros(2)

        self.compute_agents_signals()

        for i, a in enumerate(self.agents):
            a.compute_brain_input(self.agents_signal[i])
            a.brain.euler_step() # compute brain_output
            a.compute_motor_outputs()    
            agents_vel[i] = a.get_velocity()
        
        self.agents_pos = self.wrap_around(self.agents_pos + agents_vel)

        self.compute_shadow_pos()
        
        return agents_pos, agents_vel, shadows_pos, objs_pos, agents_signal



