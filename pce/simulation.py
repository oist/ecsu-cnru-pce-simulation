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

# from measures.entropy_shannon_binned import get_shannon_entropy_dd_simplified
from measures.entropy_shannon_binned import get_shannon_entropy_dd_simplified
from measures.jidt.mi_kraskov import compute_mi_kraskov # JVM must be started already
from measures.jidt.transfer_entropy import compute_transfer_entropy_kraskov_reciprocal, compute_transfer_entropy_discrete # JVM must be started already


@dataclass
class Simulation:

    # agents settings
    num_agents: int = 2 # number of agents    
    num_neurons: int = 2 # number of brain neurons
    brain_step_size: float = 0.1
    init_state: float = 0.

    # sim settings
    num_steps: int = 2000
    num_trials: int = 10    
    performance_function: str = 'OVERLAPPING_STEPS' # 'OVERLAPPING_STEPS', 'SHANNON_ENTROPY', 'MI', 'TE'
    aggregation_function: str = 'MIN' # 'MEAN', 'MIN'
    num_cores: int = 1    

    # env sttings    
    env_length: float = 300             # lenght of a circle
    num_objects: int = 2                # number of objects
    agent_width: float = 4              # width of players (and their ghosts)
    no_shadow: bool = False             # if to avoid using the shadows
    shadow_delta: float = env_length/4  # distance between agent and its shadow

    # random seed is used for initializing simulation settings 
    # (e.g., initial pos of agents)
    random_seed: int = 0 

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
        assert self.aggregation_function in ['MIN', 'MEAN']
        assert self.performance_function in ['OVERLAPPING_STEPS', 'SHANNON_ENTROPY', 'MI', 'TE']
        if self.performance_function in ['OVERLAPPING_STEPS', 'MI']:
            assert self.num_agents == 2

    def save_to_file(self, file_path):
        with open(file_path, 'w') as f_out:
            obj_dict = asdict(self)
            json.dump(obj_dict, f_out, indent=3, cls=NumpyListJsonEncoder)

    @staticmethod
    def load_from_file(file_path, **kwargs):
        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        if kwargs:
            for k,v in kwargs.items():
                if v is None or k not in obj_dict:
                    continue
                old_v = obj_dict[k]
                if v == old_v:
                    continue
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

    def save_data_record_step(self, t, s, agents_pos):

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
            self.data_record['agents_vel'][t][s][i] = a.get_velocity()
            self.data_record['signal'][t][s][i] = self.environment.agents_signal[i]
            self.data_record['sensor'][t][s][i] = a.sensor
            self.data_record['motors'][t][s][i] = a.motors
            self.data_record['brain_inputs'][t][s][i] = a.brain.input
            self.data_record['brain_derivatives'][t][s][i] = a.brain.dy_dt
            self.data_record['brain_states'][t][s][i] = a.brain.states
            self.data_record['brain_outputs'][t][s][i] = a.brain.output
        
    def prepare_simulation(self):
        rs = RandomState(self.random_seed)
        self.agents_initial_pos_trials = \
            rs.uniform(low=0, high=self.env_length, size=(self.num_trials,self.num_agents))
        
        self.objects_initial_pos_trials = \
            rs.uniform(low=0, high=self.env_length, size=(self.num_trials,self.num_objects))

        # todo: consider making shadow delta random

    def prepare_trial(self, t):                    
        # init environemnts       
        agents_pos = self.agents_initial_pos_trials[t]
        
        objs_pos = self.objects_initial_pos_trials[t]
        # objs_pos = np.array([self.env_length / 4, 3 * self.env_length / 4])

        self.environment = Environment(
            agents = self.agents,
            init_state = self.init_state,
            env_length = self.env_length,
            agent_width = self.agent_width,
            no_shadow = self.no_shadow,
            shadow_delta = self.shadow_delta,
            agents_pos = agents_pos,
            objs_pos = objs_pos,            
        )
        
        # to collect the data to compute performance
        if self.performance_function == 'OVERLAPPING_STEPS':
            # agents positions
            self.data_for_performance = np.zeros((self.num_steps, self.num_agents)) 
        else: #self.performance_function in ['SHANNON_ENTROPY', 'MI', 'TE']:
            self.data_for_performance = np.zeros((self.num_agents, self.num_steps, self.num_neurons)) 

    def store_step_data_for_performance(self, s, agents_pos):
        if self.performance_function == 'OVERLAPPING_STEPS':
            self.data_for_performance[s] = agents_pos
        else: #self.performance_function in ['SHANNON_ENTROPY', 'MI', 'TE']:
            for i,a in enumerate(self.agents):
                self.data_for_performance[i,s] = a.brain.output

    def compute_trial_performance(self):
        # sum of all abs difference of the two agents' agents_pos
        if self.performance_function == 'OVERLAPPING_STEPS':
            delta_agents = self.environment.wrap_around_diff_array(self.data_for_performance)
            return np.sum(delta_agents < self.agent_width)        
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
    def run_simulation(self, genotype_populations, genotype_index, 
        data_record=None, ghost_index=None, original_data_record=None):
        '''
        Main function to run simulation
        '''
        
        num_pop, pop_size, gen_size = genotype_populations.shape
        
        assert num_pop == self.num_agents, \
            f'num_pop ({num_pop}) must be equal to num_agents ({self.num_agents})'

        assert genotype_index < pop_size, \
            f'genotype_index ({genotype_index}) must be < pop_size ({pop_size})'

        assert gen_size == self.genotype_size, \
            f'invalid gen_size ({gen_size}) must be {self.genotype_size}'

        assert (ghost_index == None) == (original_data_record==None), \
            f'ghost_index and original_data_record should be both None or different from None'

        if ghost_index is not None:
            assert self.performance_function in ['OVERLAPPING_STEPS', 'SHANNON_ENTROPY'], \
            f'invalid performance measure with ghost'
            
        self.genotype_populations = genotype_populations
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

            # setup trial (agents agents_pos and angles)
            self.prepare_trial(t)  

            for s in range(self.num_steps): 

                ghost_pos = \
                    None if self.ghost_index is None \
                    else self.original_data_record['agents_pos'][t][s][self.ghost_index]
                
                # retured pos and angles are before moving the agents
                agents_pos = self.environment.make_one_step(self.ghost_index, ghost_pos)
                
                self.save_data_record_step(t, s, agents_pos)
                
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
    

    ##################
    # EVAL FUNCTION
    ##################
    def evaluate(self, genotype_populations, random_seed):

        assert genotype_populations.ndim == 3

        num_pop, pop_size, gen_size = genotype_populations.shape

        expected_perf_shape = (num_pop, pop_size)

        split_population = num_pop == 1

        if split_population:
            assert pop_size % self.num_agents == 0, \
                f"pop_size ({pop_size}) must be a multiple of num_agents {self.num_agents}"
            genotype_populations = np.array(
                np.split(genotype_populations[0], self.num_agents)
            )
            num_pop, pop_size, gen_size = genotype_populations.shape

        if self.num_cores == 1:
            # single core                
            perf = [
                self.run_simulation(genotype_populations, i)
                for i in range(pop_size)
            ]
        else:
            # run parallel job            
            perf = Parallel(n_jobs=self.num_cores)(
                delayed(self.run_simulation)(genotype_populations, i) \
                for i in range(pop_size)
            )

        if split_population:
            # population was split in num_agents parts
            # we need to repeat the performance of each group of agents 
            # (those sharing the same index)
            performances = np.tile([perf], self.num_agents) # shape: (num_agents,)            
        else:
            # we have num_agents populations
            # so we need to repeat the performances num_agents times
            performances = np.repeat([perf], self.num_agents, axis=0)
        
        assert performances.shape == expected_perf_shape

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
        random_seed=seed,
        **kwargs
    )

    rs = RandomState(seed)

    genotype_populations = np.array([
        [Evolution.get_random_genotype(rs, sim.genotype_size)]
        for _ in range(num_agents)
    ])

    data_record = {}

    performance = sim.run_simulation(
        genotype_populations=genotype_populations, 
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
