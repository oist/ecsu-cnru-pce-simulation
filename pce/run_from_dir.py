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
from pce.simulation import Simulation, export_data_trial_to_tsv
from pyevolver.evolution import Evolution
from pce import utils
import numpy as np
from pce.utils import get_numpy_signature
from pce import file_utils
from pce.analyze_results import get_non_flat_neuron_data
from pce.network import plot_network

def run_simulation_from_dir(dir, gen=None, genotype_idx=0, write_data=False, quiet=True, get_data=True, **kwargs):
    """
    Utitity function to get data from a simulation
    """
    
    # func_arguments = locals()   
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

    sim_json_filepath = os.path.join(dir, 'simulation.json')    
    # json_data = json.load(open(sim_json_filepath))
    
    # loading evo
    evo = Evolution.load_from_file(evo_json_filepath, folder_path=dir)
    
    # loading sim with overriding params
    sim = Simulation.load_from_file(sim_json_filepath, verbose=verbose, **kwargs)

    # just in case there were overridden params
    sim.prepare_simulation()

    if get_data:
        data_record = {}
    else:
        data_record = None

    original_genotype_populations = evo.population_unsorted

    # get the indexes of the populations as they were before being sorted by performance
    # we only need to do this for the first population (index 0)
    original_genotype_idx = evo.population_sorted_indexes[0][genotype_idx]

    original_agent_genotype = original_genotype_populations[0][original_genotype_idx]        

    sim.num_pop, sim.pop_size, _ = original_genotype_populations.shape

    sim.set_genotype_populations(original_genotype_populations)

    if sim.split_population:
        original_genotype_idx = original_genotype_idx % sim.pop_size # where in the pop

    ghost_index = kwargs.get('ghost_index', None)
    if ghost_index is not None:
        # get original results without overloading (e.g, ghost, random seed, ...)            
        sim_original = Simulation.load_from_file(sim_json_filepath)
        sim_original.set_genotype_populations(original_genotype_populations)
        original_data_record = {}
        original_performance  = sim_original.run_simulation(
            genotype_index=original_genotype_idx,
            data_record=original_data_record
        )
        print("Original performance (without ghost and overriding params): {}".format(original_performance))        
    else:                        
        original_data_record = None
        
    performance  = sim.run_simulation(
        genotype_index=original_genotype_idx,
        ghost_index=ghost_index,
        data_record=data_record,
        original_data_record=original_data_record
    )

    if verbose:        
        trials_performances = data_record['trials_performances']
        original_agent_signature = get_numpy_signature(original_agent_genotype)        
        print('Agent signature:', original_agent_signature)
        perf_orig = evo.performances[0][genotype_idx]
        print("Performance (in json): {}".format(perf_orig))
        print("Performance recomputed: {}".format(performance))
        if len(kwargs)==0: # expect_same_results
            diff_perfomance = abs(perf_orig - performance)
            if diff_perfomance > 1e-5:
                print(f'\t⚠️ Warning: diff_perfomance: {diff_perfomance}')
        print('Trials Performances:', trials_performances)
        print('Agent(s) signature(s):', data_record['signatures'])
        non_flat_neurons = get_non_flat_neuron_data(data_record, 'brain_outputs')
        print(f'Non flat neurons: {non_flat_neurons}')

    if write_data:
        outdir = os.path.join(dir, 'data')
        utils.make_dir_if_not_exists_or_replace(outdir)
        for k, v in data_record.items():
            outfile = os.path.join(outdir, '{}.json'.format(k))
            utils.save_json_numpy_data(v, outfile)

    return evo, sim, performance, data_record


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
    parser.add_argument('--gen', type=int, help='Generation number to load. Defaults to the last one.')
    parser.add_argument('--genotype_idx', type=int, default=0, help='Index (0-based) of agent in population to load. Defaults to 0 (best agent).')
    parser.add_argument('--ghost_index', type=int, help='Ghost index')
    parser.add_argument('--sim_seed', type=int, help='Overriding sim seed')
    parser.add_argument('--num_steps', type=int, help='Overriding sim num steps')
    parser.add_argument('--num_trials', type=int, help='Overriding sim num trials')
    parser.add_argument('--num_objects', type=int, help='Overriding sim num objects')
    parser.add_argument('--transient_period', action='store_true', default=None, help='Overriding sim num objects')
    parser.add_argument('--alternate_sides',action='store_true', default=None, help='whether to place the two agents on opposite side of the 1-d space (and alternate their motors so that direction is not fixed based on neuron activity)')
    # parser.add_argument('--objects_facing_agents',action='store_true', default=None, help='whether to place the two agents on opposite side of the 1-d space (and alternate their motors so that direction is not fixed based on neuron activity)')
    parser.add_argument('--init_state', type=float, help='Overriding initial state of agents')    
    parser.add_argument('--write_data', action='store_true', default=False, help='Whether to output data (same directory as input)')

    # overloading sim default params
    parser.add_argument('--performance_function', type=str, help='Type of performance function')
    parser.add_argument('--aggregation_function', type=str, help='Type of aggregation function')
    parser.add_argument('--shadow_delta', type=float, help='Shadow distance')

    # additional args
    parser.add_argument('--viz', action='store_true', help='Visualize trial')
    parser.add_argument('--mp4', action='store_true', help='Save visualization to video')
    parser.add_argument('--fps', type=int, default=20, help='Frame per seconds')
    parser.add_argument('--plot', action='store_true', help='Whether to plot the data')
    parser.add_argument('--network', action='store_true', help='Whether to plot the diagram with networks and weights')
    parser.add_argument('--trial', help='Whether to visualize/plot a specif trial (1-based). Defaults to the trial with worst performance.')
    parser.add_argument('--tsv', help='TSV file where to export the trial results')

    args = parser.parse_args()

    evo, sim, performance, data_record = \
        run_simulation_from_dir(**vars(args))

    trials_perfs = data_record['trials_performances']    

    best_trial_idx = np.argmax(trials_perfs)
    worst_trial_idx = np.argmin(trials_perfs)
    
    trial_idx =  (
        worst_trial_idx if args.trial in [None, 'worst']
        else best_trial_idx if args.trial == 'best'
        else int(args.trial)-1 if utils.is_int(args.trial)
        else args.trial if args.trial in ['all'] 
        else None
    )

    assert trial_idx is not None, "Wrong value for param 'trial'"

    if type(trial_idx) is not str:
        print(f'Performance of selected trial ({trial_idx+1}/{sim.num_trials}): {trials_perfs[trial_idx]}')

    if args.tsv:
        export_data_trial_to_tsv(args.tsv, data_record, trial_idx)
    if args.plot:
        if trial_idx == 'all':
           print(f"Plotting all trials")
        else:
            print(f"Plotting trial: {trial_idx+1}/{sim.num_trials}")
        plot.plot_results(evo, sim, trial_idx, data_record)
    if args.viz or args.mp4:
        video_path = \
            os.path.join(
                file_utils.SAVE_FOLDER,
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
    if args.network:
        phenotypes = data_record['phenotypes']
        plot_network(sim.num_neurons, phenotypes)
