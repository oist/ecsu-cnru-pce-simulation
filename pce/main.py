"""
Runs evolutionary code for the main simulation.
Run from command line as
python -m pce.main args
See help for required arguments:
python -m pce.main --help
"""

import os
import argparse
from pytictoc import TicToc
import numpy as np
from pyevolver.evolution import Evolution
from pce import utils
from pce.simulation import Simulation
from measures.utils.jidt import initJVM, shutdownJVM


def main(raw_args=None):
    parser = argparse.ArgumentParser(
        description='Run the Formation Simulation'
    )

    # evolution arguments
    parser.add_argument('--evo_seed', type=int, default=0, help='Random seed used in pyevolver')
    parser.add_argument('--dir', type=str, default=None, help='Output directory')
    parser.add_argument('--gen_zfill', action='store_true', default=False,
                        help='whether to fill genotipes with zeros otherwize random (default)')    
    parser.add_argument('--num_pop', type=int, default=1, help='Number of populations')
    parser.add_argument('--pop_size', type=int, default=96, help='Population size')
    parser.add_argument('--noshuffle', action='store_true', default=False, help='Weather to shuffle agents before eval function')
    parser.add_argument('--max_gen', type=int, default=10, help='Number of generations')

    # simulation arguments        
    parser.add_argument('--num_agents', type=int, default=2, help='Number of agents in the simulation')
    parser.add_argument('--num_neurons', type=int, default=2, help='Number of brain neurons in each agent')
    parser.add_argument('--num_objects', type=int, default=2, help='Number of static objects')
    parser.add_argument('--self_pairing', action='store_true', default=False, help='Weather to pair each agent with itself')
    parser.add_argument('--noshadow', action='store_true', default=False, help='Whether to avoid placing shadows')
    parser.add_argument('--num_trials', type=int, default=10, help='Number of simulation trials')            
    parser.add_argument('--num_steps', type=int, default=500, help='Number of simulation steps')                
    parser.add_argument('--alternate_sides', action='store_true', default=False, 
        help='whether to place the two agents on opposite side of the 1-d space \
            (and alternate their motors so that direction is not fixed based on neuron activity)')    
    parser.add_argument('--transient_period', action='store_true', default=False, help='Whether to evaluate only for second half of the simulation')
    parser.add_argument('--perf_func', type=str, default='OVERLAPPING_STEPS', 
        choices=['OVERLAPPING_STEPS', 'DISTANCE', 'SHANNON_ENTROPY', 'MI', 'TE'], help='Type of performance function')
    parser.add_argument('--agg_func', type=str, default='MIN', 
        choices=['MEAN', 'MIN'], help='Type of aggregation function over trial performances')
    parser.add_argument('--cores', type=int, default=1, help='Number of cores')

    # Gather the provided arguements as an array.
    args = parser.parse_args(raw_args)

    t = TicToc()
    t.tic()

    if args.dir is not None:
        # create default path if it specified dir already exists
        if os.path.isdir(args.dir):
            subdir = 'pce_'
            if args.alternate_sides: 
                subdir += 'alt_'
            subdir += f'{args.perf_func.split("_")[0].lower()}_'
            subdir += f'{args.agg_func.lower()}_'
            subdir += f'{args.num_pop}p_{args.num_agents}a_{args.num_neurons}n_{args.num_objects}o'
            if args.noshadow:
                subdir += '_noshadow'
            if args.gen_zfill:
                subdir += '_zfill'
            if args.noshuffle:
                subdir += f'_noshuffle'
            if args.self_pairing:
                subdir += '_self'
            if args.transient_period:
                subdir += '_tp'
            evo_seed_dir = 'seed_{}'.format(str(args.evo_seed).zfill(3))
            outdir = os.path.join(args.dir, subdir, evo_seed_dir)
        else:
            # use the specified dir if it doesn't exist 
            outdir = args.dir
        utils.make_dir_if_not_exists_or_replace(outdir)
    else:
        outdir = None

    checkpoint_interval = int(np.ceil(args.max_gen / 10))

    sim = Simulation(
        num_pop = args.num_pop,
        pop_size = args.pop_size,
        self_pairing = args.self_pairing,
        num_agents = args.num_agents,
        num_neurons = args.num_neurons,
        num_objects = args.num_objects,
        no_shadow = args.noshadow,
        num_trials = args.num_trials,    
        num_steps = args.num_steps,    
        alternate_sides = args.alternate_sides,    
        transient_period = args.transient_period,
        performance_function = args.perf_func,
        aggregation_function = args.agg_func,
        num_cores=args.cores
    )

    genotype_size = sim.genotype_size

    if outdir is not None:
        sim_config_json = os.path.join(outdir, 'simulation.json')
        sim.save_to_file(sim_config_json)

    population = None  # by default randomly initialized in evolution

    if args.gen_zfill:
        # all genotypes initialized with zeros
        population = np.zeros(
            (args.num_pop, args.pop_size, genotype_size)
        )

    evo = Evolution(
        random_seed=args.evo_seed,
        population=population,
        num_populations=args.num_pop,
        shuffle_agents=not args.noshuffle,
        population_size=args.pop_size,
        genotype_size=genotype_size,
        evaluation_function=sim.evaluate,
        performance_objective='MAX',
        fitness_normalization_mode='FPS',  # 'NONE', 'FPS', 'RANK', 'SIGMA' -> NO NORMALIZATION
        selection_mode='RWS',  # 'UNIFORM', 'RWS', 'SUS'
        reproduce_from_elite=False,
        reproduction_mode='GENETIC_ALGORITHM',  # 'HILL_CLIMBING',  'GENETIC_ALGORITHM'
        mutation_variance=0.2,  # mutation noice with variance 0.1
        elitist_fraction=0.1,  # elite fraction of the top 4% solutions
        mating_fraction=0.9,  # the remaining mating fraction (consider leaving something for random fill)
        crossover_probability=0.3,
        crossover_mode='UNIFORM',
        crossover_points=None,  # genotype_structure['crossover_points'],
        folder_path=outdir,
        max_generation=args.max_gen,
        termination_function=None,
        checkpoint_interval=checkpoint_interval
    )
    print('Output path: ', outdir)
    print('n_elite, n_mating, n_filling: ', evo.n_elite, evo.n_mating, evo.n_fillup)
    evo.run()

    print('Ellapsed time: {}'.format(t.tocvalue()))

    if args.perf_func == 'MI':
        shutdownJVM()

    return sim, evo


if __name__ == "__main__":
    main()
