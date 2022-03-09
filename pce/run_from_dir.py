"""
Rerun a simulation of a given seed and optionally visualize
animation and data plots of behavior and neural activity.
Run as
python -m dol.run_from_dir --help
"""

import os
from pathlib import Path
import json
from numpy.random import RandomState
from pce.simulation import Simulation
from pyevolver.evolution import Evolution
from pce import utils
import numpy as np
from pce.utils import get_numpy_signature
from pce.analyze_results import get_non_flat_neuron_data

def run_simulation_from_dir(dir, gen=None, genotype_idx=0, random_seed=None,
                            write_data=False, quiet=True, **kwargs):
    """
    Utitity function to get data from a simulation
    """
    verbose = not quiet
    evo_files = sorted([f for f in os.listdir(dir) if f.startswith('evo_')])
    assert len(evo_files) > 0, "Can't find evo files in dir {}".format(dir)
    file_num_zfill = len(evo_files[0].split('_')[1].split('.')[0])
    
    if gen is None:
        evo_json_filepath = os.path.join(dir, evo_files[-1])
        gen = int(evo_files[-1].split('_')[1].split('.')[0])
    else:
        generation_str = str(gen).zfill(file_num_zfill)
        evo_json_filepath = os.path.join(dir, 'evo_{}.json'.format(generation_str))    

    random_state = RandomState(random_seed) if random_seed is not None else None

    sim_json_filepath = os.path.join(dir, 'simulation.json')    
    # json_data = json.load(open(sim_json_filepath))
    
    # sim_class = SimulationSynergy if "synergy_padding" in json_data else Simulation
    sim = Simulation.load_from_file(sim_json_filepath)
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)

    # overloaded params
    perf_func = kwargs.get('perf_func', None)
    if perf_func is not None:
        sim.performance_function = perf_func

    data_record = {}

    expect_same_results = True

    original_genotype_populations = evo.population_unsorted

    # get the indexes of the populations as they were before being sorted by performance
    # we only need to do this for the first population (index 0)
    original_genotype_idx = evo.population_sorted_indexes[0][genotype_idx]

    original_agent_genotype = original_genotype_populations[0][original_genotype_idx]        

    num_pop, pop_size, gen_size = original_genotype_populations.shape

    if num_pop == 1:
        # split_population
        original_genotype_populations = np.array(
            np.split(original_genotype_populations[0], 2)
        )
        num_pop, pop_size, gen_size = original_genotype_populations.shape
        original_genotype_idx = original_genotype_idx % pop_size # where in the pop

        
    performance  = sim.run_simulation(
        genotype_populations=original_genotype_populations,
        genotype_index=original_genotype_idx,
        random_state=random_state,
        data_record=data_record
    )

    trials_performances = data_record['trials_performances']

    if verbose:        
        original_agent_signature = get_numpy_signature(original_agent_genotype)        
        print('original agent:', original_agent_signature)
        perf_orig = evo.performances[0][genotype_idx]
        print("Performance original: {}".format(perf_orig))
        print("Performance recomputed: {}".format(performance))
        if expect_same_results:
            diff_perfomance = abs(perf_orig - performance)
            if diff_perfomance > 1e-5:
                print(f'Warning: diff_perfomance: {diff_perfomance}')
        print('Trials Performances:', trials_performances)

    if write_data:
        outdir = os.path.join(dir, 'data')
        utils.make_dir_if_not_exists_or_replace(outdir)
        for k, v in data_record.items():
            outfile = os.path.join(outdir, '{}.json'.format(k))
            utils.save_json_numpy_data(v, outfile)

    if verbose:
        print('Agent(s) signature(s):', data_record['signatures'])
        non_flat_neurons = get_non_flat_neuron_data(data_record, 'brain_outputs')
        print(f'Non flat neurons: {non_flat_neurons}')

    return performance, trials_performances, evo, sim, data_record


if __name__ == "__main__":
    import argparse
    from pce import plot
    from pce.visual import Visualization

    parser = argparse.ArgumentParser(
        description='Rerun simulation'
    )

    # args for run_simulation_from_dir
    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--quiet', action='store_true', default=False, help='Print extra information (e.g., originale performance)')
    parser.add_argument('--gen', type=int, help='Number of generation to load')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index of agent in population to load')
    parser.add_argument('--random_seed', type=int, help='Random seed for randomized trials')
    parser.add_argument('--write_data', action='store_true', default=False, help='Whether to output data (same directory as input)')

    # overloading sim default params
    parser.add_argument('--perf_func', type=str, default='OVERLAPPING_STEPS', 
        choices=['OVERLAPPING_STEPS', 'SHANNON_ENTROPY'], help='Type of performance function')

    # additional args
    parser.add_argument('--viz', action='store_true', help='Visualize trial')
    parser.add_argument('--mp4', action='store_true', help='Save visualization to video')
    parser.add_argument('--fps', type=int, default=20, help='Frame per seconds')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    parser.add_argument('--trial', type=int, help='Whether to visualize/plot a specif trial (1-based)')

    args = parser.parse_args()

    perf, trials_perfs, evo, sim, data_record = \
        run_simulation_from_dir(**vars(args))


    if args.plot:
        trial_idx = args.trial - 1 if args.trial is not None else 'all'
        if trial_idx == 'all':
            print(f"Plotting all trials")
        else:
            print(f"Plotting trial: {trial_idx+1}/{sim.num_trials}")
        plot.plot_results(evo, sim, trial_idx, data_record)
    if args.viz or args.mp4:
        trial_idx = args.trial - 1 if args.trial is not None else np.argmax(trials_perfs)
        video_path = \
            os.path.join(
                'video',
                '_'.join([
                    os.path.basename(Path(args.dir).parent),
                    os.path.basename(args.dir),
                    f't{trial_idx+1}.mp4'
                ])
            ) \
            if args.mp4 else None        
        viz = Visualization(
            sim=sim,
            video_path=video_path,
            fps=args.fps,
        )        
        print(f"Visualizing trial {trial_idx+1}/{sim.num_trials}")
        viz.start(data_record, trial_idx)
        if video_path:
            print('Output video:', video_path)

