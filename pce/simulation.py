"""
Main code for experiment simulation.
"""

from dataclasses import dataclass, asdict
import json
import numpy as np
from numpy.random import RandomState
from joblib import Parallel, delayed
from ranges import Range, RangeSet
from pyevolver.json_numpy import NumpyListJsonEncoder
from pce.agent import Agent
from pce.environment import Environment
from pce import gen_structure
from pce import utils
from scipy.spatial import distance
from pce import params

@dataclass
class Simulation:

    num_neurons: int = 2 # number of brain neurons
    brain_step_size: float = 0.1

    num_trials: int = 4
    num_steps: int = 500
    num_cores: int = 1    

    def __post_init__(self):

        self.__check_params__()   

        self.genotype_structure = gen_structure.DEFAULT_GEN_STRUCTURE(self.num_neurons)

        self.genotype_size = gen_structure.get_genotype_size(self.genotype_structure)
        
        self.agents = [
            Agent(
                genotype_structure=self.genotype_structure,
                brain_step_size=self.brain_step_size,
            )
            for _ in range(2)
        ]      

        # init centroid (agents center of mass) for computing performance
        self.centroid_pos = np.zeros((self.num_trials, self.num_steps, 2))
        self.centroid_segments = np.zeros((self.num_trials, self.num_steps-1, 2))
                
        # if agents are in formation (close together to centroid)
        self.formation = np.zeros((self.num_trials, self.num_steps), dtype=bool)



    def __check_params__(self):
        pass


    def save_to_file(self, file_path):
        with open(file_path, 'w') as f_out:
            obj_dict = asdict(self)
            json.dump(obj_dict, f_out, indent=3, cls=NumpyListJsonEncoder)

    @staticmethod
    def load_from_file(file_path, **kwargs):
        with open(file_path) as f_in:
            obj_dict = json.load(f_in)

        if kwargs:
            obj_dict.update(kwargs)
        
        sim = Simulation(**obj_dict)

        return sim            

    def set_agents_genotype_phenotype(self):
        
        self.genotypes = np.array([
            self.genotype_populations[i,self.genotype_index]
            for i in range(2)
        ])

        self.phenotypes = [{} for _ in range(2)]

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
        self.data_record['agents_pos'] = np.zeros((self.num_trials, self.num_steps, 2))        
        self.data_record['agents_vel'] = np.zeros((self.num_trials, self.num_steps, 2))        
        self.data_record['shadows_pos'] = np.zeros((self.num_trials, self.num_steps, 2))        
        self.data_record['objs_pos'] = np.zeros((self.num_trials, self.num_steps, 2))        
        self.data_record['signal'] =    np.zeros((self.num_trials, self.num_steps, 2))
        self.data_record['sensor'] =    np.zeros((self.num_trials, self.num_steps, 2))

        self.data_record['brain_inputs'] = np.zeros((self.num_trials, self.num_steps, 2, self.num_neurons))
        self.data_record['brain_states'] = np.zeros((self.num_trials, self.num_steps, 2, self.num_neurons))
        self.data_record['brain_derivatives'] = np.zeros((self.num_trials, self.num_steps, 2, self.num_neurons))
        self.data_record['brain_outputs'] = np.zeros((self.num_trials, self.num_steps, 2, self.num_neurons))
        
        self.data_record['motors'] = np.zeros((self.num_trials, self.num_steps, 2, 2))

    def save_data_record_step(self, t, s, agents_pos, agents_vel, shadows_pos, objs_pos, agents_signal):
        if self.data_record is None:
            return

        self.data_record['agents_pos'][t][s] = agents_pos
        self.data_record['agents_vel'][t][s] = agents_vel
        self.data_record['shadows_pos'][t][s] = shadows_pos
        self.data_record['objs_pos'][t][s] = objs_pos
        self.data_record['signal'][t][s] = agents_signal
        
        for i, a in enumerate(self.agents):
            self.data_record['sensor'][t][s][i] = a.sensor
            self.data_record['brain_inputs'][t][s][i] = a.brain.input
            self.data_record['brain_states'][t][s][i] = a.brain.states
            self.data_record['brain_derivatives'][t][s][i] = a.brain.dy_dt
            self.data_record['brain_outputs'][t][s][i] = a.brain.output
            self.data_record['motors'][t][s][i] = a.motors
        

    def prepare_trial(self, t, random_state):                    
        # init environemnts       
        self.environment = Environment(self.agents, self.num_trials, t, random_state)
        self.trial_agents_pos = np.zeros((self.num_steps, 2))


    def compute_trial_performance(self):
        # sum of all abs difference of the two agents' agents_pos
        return np.sum(np.abs(np.diff(self.trial_agents_pos))<params.AGENT_WIDTH)


    #################
    # MAIN FUNCTION
    #################
    def run_simulation(self, genotype_populations, genotype_index, 
                       random_state=None, data_record=None):
        '''
        Main function to run simulation
        '''
        
        num_pop, pop_size, gen_size = genotype_populations.shape
        
        assert num_pop == 2, \
            f'num_pop ({num_pop}) must be equal to num_agents ({2})'

        assert genotype_index < pop_size, \
            f'genotype_index ({genotype_index}) must be < pop_size ({pop_size})'

        assert gen_size == self.genotype_size, \
            f'invalid gen_size ({gen_size}) must be {self.genotype_size}'
            
        self.genotype_populations = genotype_populations
        self.genotype_index = genotype_index
        self.data_record = data_record        

        # SIMULATIONS START
        
        self.set_agents_genotype_phenotype()

        trials_performances = []

        # INITIALIZE DATA RECORD
        self.init_data_record()

        # TRIALS START
        for t in range(self.num_trials):

            # setup trial (agents agents_pos and angles)
            self.prepare_trial(t, random_state)  

            for s in range(self.num_steps): 

                # retured pos and angles are before moving the agents
                agents_pos, agents_vel, shadows_pos, objs_pos, agents_signal = self.environment.make_one_step()                                
                self.trial_agents_pos[s] = agents_pos

                self.save_data_record_step(t, s, agents_pos, agents_vel, shadows_pos, objs_pos, agents_signal)
                
            trials_performances.append(self.compute_trial_performance())

        # TRIALS END

        # mean performances between all trials
        performance = np.mean(trials_performances)

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
            assert pop_size % 2 == 0, \
                f"pop_size ({pop_size}) must be a multiple of num_agents {2}"
            genotype_populations = np.array(
                np.split(genotype_populations[0], 2)
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
            performances = np.tile([perf], 2) # shape: (num_agents,)            
        else:
            # we have num_agents populations
            # so we need to repeat the performances num_agetns times
            performances = np.repeat([perf], 2, axis=0)
        
        assert performances.shape == expected_perf_shape

        return performances


# --- END OF SIMULATION CLASS


# TEST

def test_simulation(num_neurons=2, num_steps=500, seed=None):
    
    from pyevolver.evolution import Evolution    

    if seed is None:
        seed = utils.random_int()

    print("Seed: ", seed)

    sim = Simulation(
        num_neurons,
        num_steps=num_steps
    )

    rs = RandomState(seed)

    genotype_populations = np.array([
        [Evolution.get_random_genotype(rs, sim.genotype_size)]
        for _ in range(2)
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
        num_neurons=2,
        seed=None
    )
