"""
Main code for experiment simulation.
"""

from measures.utils.jidt import initJVM
initJVM()


from dataclasses import dataclass, asdict
import json
import numpy as np
from numpy.random import RandomState
from joblib import Parallel, delayed
from pyevolver.json_numpy import NumpyListJsonEncoder
from pce.agent import Agent
from pce.environment import Environment
from pce import gen_structure
from pce import utils
from itertools import product

# from measures.entropy_shannon_binned import get_shannon_entropy_dd_simplified
from measures.parametric.entropy_shannon_binned import get_shannon_entropy_dd_simplified
from measures.jidt.mi_kraskov import compute_mi_kraskov # JVM must be started already
from measures.jidt.transfer_entropy_continuous import compute_transfer_entropy_kraskov_reciprocal


@dataclass
class Simulation:

    # pairing dynamics
    # num_pairing: int = 1 # 0=split, 1=1-1 pairs, n>1: 1 agent paired with n agents
    num_pop: int = None
    pop_size: int = None
    self_pairing: bool = False

    # agents settings
    num_agents: int = 2 # number of agents    
    num_neurons: int = 2 # number of brain neurons
    brain_step_size: float = 0.1
    init_state: float = 0.

    # sim settings
    num_steps: int = 500
    num_trials: int = 10    
    alternate_sides: bool = False # whether to place the two agents on opposite side of the 1-d space (and alternate their motors so that direction is not fixed based on neuron activity)
    objects_facing_agents: bool = True # whether object are facing the respective agents (otherwise they are placed on the line)    
    performance_function: str = 'OVERLAPPING_STEPS' # 'OVERLAPPING_STEPS', 'DISTANCE', 'SHANNON_ENTROPY', 'MI', 'TE'
    transient_period: bool = False # whether to evaluate only on the second half of the simulation (only applicable for OVERLAPPING_STEPS)
    aggregation_function: str = 'MIN' # 'MEAN', 'MIN'
    normalize_perf: bool = True # only for OVERLAPPING_STEPS TODO: to be removed and used by default
    num_cores: int = 1    

    # env sttings    
    env_length: float = 300             # lenght of a circle
    num_objects: int = 2                # number of objects
    agent_width: float = 4              # width of players (and their ghosts)
    no_shadow: bool = False             # if to avoid using the shadows
    shadow_delta: float = None          # distance between agent and its shadow    

    # random seed is used for initializing simulation settings 
    # (e.g., initial pos of agents)
    sim_seed: int = 0 

    def __post_init__(self):

        self.__check_params__()   

        self.genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(self.num_neurons)

        self.genotype_size = gen_structure.get_genotype_size(self.genotype_structure)
        
        self.agents = [
            Agent(
                genotype_structure=self.genotype_structure,
                brain_step_size=self.brain_step_size,
            )
            for _ in range(self.num_agents)
        ]      

        self.prepare_simulation()

    def __check_params__(self):
        if self.shadow_delta is None and not self.no_shadow:
            self.shadow_delta = self.env_length/4
        if self.transient_period:
            assert self.performance_function == 'OVERLAPPING_STEPS' ,\
            'Transient period is applicable only to OVERLAPPING_STEPS'
        assert self.aggregation_function in ['MIN', 'MEAN']
        assert self.performance_function in ['OVERLAPPING_STEPS', 'DISTANCE', 'SHANNON_ENTROPY', 'MI', 'TE']
        if self.performance_function in ['OVERLAPPING_STEPS', 'MI']:
            assert self.num_agents == 2
        if self.objects_facing_agents:
            assert self.num_objects == self.num_agents

    def save_to_file(self, file_path):
        with open(file_path, 'w') as f_out:
            obj_dict = asdict(self)
            json.dump(obj_dict, f_out, indent=3, cls=NumpyListJsonEncoder)

    @staticmethod
    def load_from_file(file_path, verbose=True, **kwargs):
        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        if kwargs:
            for k,v in kwargs.items():
                if v is None or k not in obj_dict:
                    continue                
                old_v = obj_dict[k]
                if v == old_v:
                    continue
                if verbose:
                    print(f'Overriding {k} from {old_v} to {v}')
                obj_dict[k] = v
        
        sim = Simulation(**obj_dict)

        return sim            

    def set_agents_genotype_phenotype(self):
        
        self.genotypes = np.array([
            self.genotype_populations[i,self.genotype_index]
            for i in range(self.num_agents)
        ])

        self.phenotypes = [{} for _ in range(self.num_agents)]

        for i, a in enumerate(self.agents):
            a.genotype_to_phenotype(
                self.genotypes[i],
                phenotype_dict=self.phenotypes[i]
            )

        if self.data_record is not None:
            self.data_record['genotypes'] = self.genotypes
            self.data_record['phenotypes'] = self.phenotypes            
            self.data_record['signatures'] = [
                utils.get_numpy_signature(gt) 
                for gt in self.genotypes
            ]

    def init_data_record(self):
        if self.data_record is None:
            return

        self.data_record['shadows_pos'] = np.zeros((self.num_trials, self.num_steps, self.num_agents))        
        self.data_record['objs_pos'] = np.zeros((self.num_trials, self.num_steps, self.num_objects))        
        self.data_record['agents_pos'] = np.zeros((self.num_trials, self.num_steps, self.num_agents))        
        self.data_record['agents_delta'] = np.zeros((self.num_trials, self.num_steps))        
        self.data_record['agents_vel'] = np.zeros((self.num_trials, self.num_steps, self.num_agents))        
        self.data_record['signal'] =     np.zeros((self.num_trials, self.num_steps, self.num_agents))
        self.data_record['sensor'] =     np.zeros((self.num_trials, self.num_steps, self.num_agents))
        self.data_record['motors'] = np.zeros((self.num_trials, self.num_steps, self.num_agents, 2))
        self.data_record['brain_inputs'] =      np.zeros((self.num_trials, self.num_steps, self.num_agents, self.num_neurons))
        self.data_record['brain_derivatives'] = np.zeros((self.num_trials, self.num_steps, self.num_agents, self.num_neurons))
        self.data_record['brain_states'] =      np.zeros((self.num_trials, self.num_steps, self.num_agents, self.num_neurons))
        self.data_record['brain_outputs'] =     np.zeros((self.num_trials, self.num_steps, self.num_agents, self.num_neurons))

        if self.ghost_index is not None:
            copied_keys = [
                'agents_vel', 'signal', 'sensor', 'motors',
                'brain_inputs', 'brain_states', 'brain_derivatives', 'brain_outputs', 
            ]
            for k in copied_keys:
                # original_data_record may have more steps if overriden in current sim
                self.data_record[k][:,:,self.ghost_index] = \
                self.original_data_record[k][:,:self.num_steps,self.ghost_index] 

    def save_data_record_step(self, t, s):

        agents_pos = self.environment.agents_prev_pos
        self.store_step_data_for_performance(s, agents_pos)

        if self.data_record is None:
            return

        self.data_record['agents_pos'][t][s] = agents_pos        
        self.data_record['shadows_pos'][t][s] = self.environment.shadows_pos # shadows_pos
        self.data_record['objs_pos'][t][s] = self.environment.objs_pos # objs_pos        

        # delta between agents        
        if self.num_agents==2:
            agents_delta = self.environment.wrap_around_diff(*agents_pos)
            self.data_record['agents_delta'][t][s] = agents_delta        

        for i, a in enumerate(self.agents):
            if i == self.ghost_index:
                continue # data already written in init_data_record
            self.data_record['agents_vel'][t][s][i] = a.get_velocity(reverse=self.agents_reverse_motors[i])
            self.data_record['signal'][t][s][i] = self.environment.agents_signal[i]
            self.data_record['sensor'][t][s][i] = a.sensor
            self.data_record['motors'][t][s][i] = a.motors
            self.data_record['brain_inputs'][t][s][i] = a.brain.input
            self.data_record['brain_derivatives'][t][s][i] = a.brain.dy_dt
            self.data_record['brain_states'][t][s][i] = self.environment.agents_prev_neural_states[i] # a.brain.states
            self.data_record['brain_outputs'][t][s][i] = self.environment.agents_prev_neural_outputs[i] # a.brain.output
        
    def prepare_simulation(self):
        rs = RandomState(self.sim_seed)
        self.agents_initial_pos_trials = \
            rs.uniform(low=0, high=self.env_length, size=(self.num_trials,self.num_agents))
        
        self.objects_initial_pos_trials = \
            rs.uniform(low=0, high=self.env_length, size=(self.num_trials,self.num_objects))
        
        # todo: consider making shadow delta random

    def prepare_trial(self, t, ghost_index=None, ghost_pos_trial=None):                    
        
        # init environemnts       
        agents_pos = self.agents_initial_pos_trials[t]
        objs_pos = self.objects_initial_pos_trials[t]        
        # objs_pos = np.array([self.env_length / 4, 3 * self.env_length / 4])

        if ghost_index is not None:
            agents_pos[ghost_index] = ghost_pos_trial[0]
            objs_pos[ghost_index] = self.original_data_record['objs_pos'][t,0,self.ghost_index]        
        
        # reverse motors
        # when True, the respective agent faces OUT
        if self.alternate_sides:
            combinations = list(product([True, False], repeat=2)) # (t,t), (t,f) (f,t) (f,f)
            self.agents_reverse_motors = combinations[t%4] # Setting to True to agent on the outer side (first in even trials)
        else:            
            self.agents_reverse_motors = [True, False]

        self.environment = Environment(
            agents = self.agents,
            init_state = self.init_state,
            env_length = self.env_length,
            agent_width = self.agent_width,
            no_shadow = self.no_shadow,
            shadow_delta = self.shadow_delta,
            objects_facing_agents = self.objects_facing_agents,
            agents_pos = agents_pos,
            agents_reverse_motors = self.agents_reverse_motors,
            objs_pos = objs_pos            
        )
        
        # to collect the data to compute performance
        if self.performance_function in ['OVERLAPPING_STEPS', 'DISTANCE']:
            # agents positions
            self.data_for_performance = np.zeros((self.num_steps, self.num_agents)) 
        else: #self.performance_function in ['SHANNON_ENTROPY', 'MI', 'TE']:
            self.data_for_performance = np.zeros((self.num_agents, self.num_steps, self.num_neurons)) 

    def store_step_data_for_performance(self, s, agents_pos):
        if self.performance_function in ['OVERLAPPING_STEPS', 'DISTANCE']:
            self.data_for_performance[s] = agents_pos
        else: #self.performance_function in ['SHANNON_ENTROPY', 'MI', 'TE']:
            for i,a in enumerate(self.agents):
                self.data_for_performance[i,s] = a.brain.output

    def compute_trial_performance(self):
        # sum of all abs difference of the two agents' agents_pos
        if self.performance_function in ['OVERLAPPING_STEPS', 'DISTANCE']:
            if self.transient_period:
                # only evaluate on second half of simulation
                num_step_half = int(self.num_steps/2)
                self.data_for_performance = self.data_for_performance[num_step_half:] 
            delta_agents = self.environment.wrap_around_diff_array(self.data_for_performance)                        
            if self.performance_function == 'OVERLAPPING_STEPS':
                perf = np.sum(delta_agents < self.agent_width)
                if self.normalize_perf:
                    perf /= len(self.data_for_performance)
            else:
                # distance
                env_length_half = self.env_length/2
                perf = 1 - np.mean(delta_agents/env_length_half)            
            return perf
        if self.performance_function == 'SHANNON_ENTROPY':
            return np.mean(
                [
                    get_shannon_entropy_dd_simplified(self.data_for_performance[i])
                    for i in range(self.num_agents) if i!=self.ghost_index
                ]
            )
        if self.performance_function == 'MI':
            return compute_mi_kraskov(
                self.data_for_performance[0],
                self.data_for_performance[1]
            )
        else: # if self.performance_function == 'TE':            
            assert self.num_neurons == 1
            self.data_for_performance = self.data_for_performance.squeeze()
            return compute_transfer_entropy_kraskov_reciprocal(
                self.data_for_performance[0],
                self.data_for_performance[1]
            )
            # compute_transfer_entropy_discrete(
            #     self.data_for_performance[0], self.data_for_performance[1], 
            #     delay=1, reciprocal=True, bins=100, min_v=0., max_v=1.
            # )

                
    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_index, 
        data_record=None, ghost_index=None, original_data_record=None):
        '''
        Main function to run simulation
        '''
        
        assert genotype_index < self.pop_size, \
            f'genotype_index ({genotype_index}) must be < pop_size ({self.pop_size})'

        assert (ghost_index == None) == (original_data_record==None), \
            f'ghost_index and original_data_record should be both None or different from None'

        if ghost_index is not None:
            assert self.performance_function in ['OVERLAPPING_STEPS', 'SHANNON_ENTROPY'], \
            f'invalid performance measure with ghost'
            
        self.genotype_index = genotype_index
        self.data_record = data_record   
        self.ghost_index = ghost_index     
        self.original_data_record = original_data_record

        # SIMULATIONS START
        
        self.set_agents_genotype_phenotype()

        trials_performances = []

        # INITIALIZE DATA RECORD
        self.init_data_record()        

        # TRIALS START
        for t in range(self.num_trials):

            ghost_pos_trial = \
                None if self.ghost_index is None \
                else self.original_data_record['agents_pos'][t,:,self.ghost_index]

            # setup trial (agents agents_pos and angles)
            self.prepare_trial(t, self.ghost_index, ghost_pos_trial)                  

            for s in range(self.num_steps):                                 
                last_step = s == self.num_steps - 1
                self.environment.make_one_step(s, self.ghost_index, ghost_pos_trial, last_step)
                self.save_data_record_step(t, s)
                
                
                
            trials_performances.append(self.compute_trial_performance())

        # TRIALS END

        # mean performances between all trials        
        if self.aggregation_function == 'MEAN':
            performance = np.mean(trials_performances)
        elif self.aggregation_function == 'MIN':
            performance = np.min(trials_performances)

        if self.data_record:
            self.data_record.update({
                'genotype_index': self.genotype_index,
                'genotype_distance': utils.genotype_group_distance(self.genotypes),
                'trials_performances': trials_performances,
                'performance': performance,
            })

        return performance
    
    def set_genotype_populations(self, genotype_populations):

        assert genotype_populations.ndim == 3
        
        self.genotype_populations = genotype_populations

        self.num_pop, self.pop_size, gen_size = genotype_populations.shape
        
        self.split_population = not self.self_pairing and self.num_pop == 1 and self.num_agents>1

        if self.self_pairing:            
            assert self.num_pop == 1, \
                f"In self-pairing only 1 population is required"

        if self.self_pairing:
            self.genotype_populations = np.repeat(genotype_populations, self.num_agents, axis=0)
            self.num_pop = self.num_agents
        elif self.split_population:
            assert self.pop_size % self.num_agents == 0, \
                f"pop_size ({self.pop_size}) must be a multiple of num_agents {self.num_agents}"
            self.genotype_populations = np.array(
                np.split(self.genotype_populations[0], self.num_agents)
            )
            self.num_pop, self.pop_size, _ = genotype_populations.shape

        assert self.num_pop == self.num_agents, \
            f'num_pop ({self.num_pop}) must be equal to num_agents ({self.num_agents})'

        assert gen_size == self.genotype_size, \
            f'invalid gen_size ({gen_size}) must be {self.genotype_size}'


    ##################
    # EVAL FUNCTION
    ##################
    def evaluate(self, genotype_populations, random_seed):

        self.set_genotype_populations(genotype_populations)

        if self.num_cores == 1:
            # single core                
            perf_list = [
                self.run_simulation(i)
                for i in range(self.pop_size)
            ]
        else:
            # run parallel job            
            perf_list = Parallel(n_jobs=self.num_cores)(
                delayed(self.run_simulation)(i) \
                for i in range(self.pop_size)
            )

        if self.split_population:
            # population was split in num_agents parts
            # we need to repeat the performance of each group of agents 
            # (those sharing the same index)
            performances = np.tile([perf_list], self.num_agents) # shape: (num_agents,)            
        elif not self.self_pairing:
            # we have num_agents populations
            # so we need to repeat the performances num_agents times
            performances = np.repeat([perf_list], self.num_agents, axis=0)
        else: # self_pairing
            performances = np.expand_dims(perf_list, axis=0)
        
        assert performances.shape == genotype_populations.shape[:2]

        return performances


# --- END OF SIMULATION CLASS

def export_data_trial_to_tsv(tsv_file, data_record, trial_idx):
    import csv
    if not tsv_file.endswith('.tsv'):
        tsv_file += '.tsv'

    with open(tsv_file, 'w') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        num_steps = len(data_record['agents_pos'][trial_idx])
        num_agents = len(data_record['agents_pos'][trial_idx,0])        
        num_objects = len(data_record['objs_pos'][trial_idx,0])        
        headers = (
            [f'agents_pos_{n}' for n in range(1,num_agents+1)] +
            [f'shadows_pos_{n}' for n in range(1,num_agents+1)] +
            [f'objs_pos_{n}' for n in range(1,num_objects+1)] +
            [f'signal_{n}' for n in range(1,num_agents+1)]            
        )
        writer.writerow(headers)    
        
        for s in range(num_steps):
            row = []
            for h in headers:
                data_key, n = h.rsplit('_', 1)
                data = data_record[data_key][trial_idx,s,int(n)-1]
                row.append(data)
            writer.writerow(row)

# TEST

def test_simulation(num_agents=2, num_neurons=2, num_steps=500, seed=None, **kwargs):
    
    from pyevolver.evolution import Evolution    

    if seed is None:
        seed = utils.random_int()

    print("Seed: ", seed)

    sim = Simulation(
        num_agents,
        num_neurons,
        num_steps=num_steps,
        sim_seed=seed,
        **kwargs
    )

    rs = RandomState(seed)

    genotype_populations = np.array([
        [Evolution.get_random_genotype(rs, sim.genotype_size)]
        for _ in range(num_agents)
    ])

    data_record = {}

    sim.set_genotype_populations(genotype_populations)

    performance = sim.run_simulation(
        genotype_index=0, 
        data_record=data_record
    )
    print("Performance: ", performance)
    print("Trial Performances: ", data_record['trials_performances'])

    # print('Phenotype 1:')
    # print(data_record['phenotypes'][0])
    # print('Phenotype 2:')
    # print(data_record['phenotypes'][1])

    return sim, data_record


if __name__ == "__main__":
    test_simulation(
        num_agents=3,
        num_neurons=2,
        seed=None
    )
