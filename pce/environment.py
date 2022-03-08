import numpy as np
from numpy.random import RandomState
from typing import List
from dataclasses import dataclass, field
from pce.agent import Agent
from pce.utils import overlaps
from pce import params

def wrap_around(data):
    return data % params.ENV_LENGTH

@dataclass
class Environment:
    agents: List[Agent]
    num_trials: int
    trial_idx: int
    random_state: RandomState = None    

    # initialized
    agents_pos: np.ndarray = None       # agents starting position
    agents_signal: np.ndarray = None    # agents sensors
    objs_pos: np.ndarray = None         # objects positions
    shadows_pos: np.ndarray = None      # shadow positions

    def __post_init__(self):
        # dispacement = params.ENV_LENGTH/self.num_trials
        self.agents_pos = params.AGENTS_INITIAL_POS[self.trial_idx] 

        if self.random_state is None:
            self.objs_pos = np.array([params.ENV_LENGTH / 4, 3 * params.ENV_LENGTH / 4]) # todo: change based on random_state and num_trial
        else:
            self.objs_pos = self.random_state.uniform(low=0, high=params.ENV_LENGTH, size=2)
        
        self.objs_left_pos = wrap_around(self.objs_pos-params.OBJ_WIDTH/2)
        self.objs_right_pos = wrap_around(self.objs_pos+params.OBJ_WIDTH/2)
        self.compute_shadow_pos()
        self.agents_signal = np.zeros(2)  

        init_ctrnn_state = 0. # todo: consider to change this based on random_state
        
        for a in self.agents:
            a.init_params(init_ctrnn_state) 
        
    def compute_shadow_pos(self):
        self.shadows_pos = wrap_around(self.agents_pos-params.SHADOW_DELTA)

    def compute_agents_signals(self):
        agents_left_pos = wrap_around(self.agents_pos-params.AGENT_WIDTH/2)
        agents_right_pos = wrap_around(self.agents_pos+params.AGENT_WIDTH/2)
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
        
        self.agents_pos = wrap_around(self.agents_pos + agents_vel)

        self.compute_shadow_pos()
        
        return agents_pos, agents_vel, shadows_pos, objs_pos, agents_signal



