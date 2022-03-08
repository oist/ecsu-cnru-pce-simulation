"""
Given a directory with N simulation seeds,
returns a bar plot with best performance of last generation
of all seeds.
Run from command line as
python -m dol.analyze_results ./data/exp_dir
where 'exp_dir' contains all the simulation seeds
"""
import os
import json
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd

def plot_best_exp_performance(best_exp_performance, seeds):
    # best_exp_performance is a an array of best performances (one per seed)
    fig, ax = plt.subplots(figsize=(12, 6))
    seeds_str = [str(s) for s in seeds]
    ax.bar(seeds_str, best_exp_performance)
    plt.xlabel('Seeds')
    plt.ylabel('Performance')    
    plt.show()

def bar_plot_seeds_data_list(seed_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title)
    seeds = seed_data.keys()
    ind = np.arange(len(seeds))        
    num_bars = len(list(seed_data.values())[0])
    width = 0.7 / num_bars
    for p in range(num_bars):
        p_series = [b[p] for b in seed_data.values()]
        x_pos = ind + p * width + width/2
        ax.bar(x_pos, p_series, width, label=f'A{p+1}')
    ax.set_xticks(ind + 0.7 / 2)
    ax.set_xticklabels(seeds)
    plt.xlabel('Seeds')
    plt.ylabel('Non Flat Elements')
    plt.legend(bbox_to_anchor=(-0.15, 1.10), loc='upper left')
    plt.show()    

def bar_plot_seeds_data_value(seed_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle(title)
    seeds = seed_data.keys()
    ind = np.arange(len(seeds))        
    width = 0.7
    p_series = list(seed_data.values())
    x_pos = ind + width/2
    ax.bar(x_pos, p_series, width)
    ax.set_xticks(ind + 0.7 / 2)
    ax.set_xticklabels(seeds)
    plt.xlabel('Seeds')
    plt.show()    

def scatter_plot_tse_performance(seed_tse, performances):
    fig, ax = plt.subplots(figsize=(12, 6))
    seeds = seed_tse.keys()
    tse_values = list(seed_tse.values())
    ax.scatter(tse_values, performances)
    for i, seed_txt in enumerate(seeds):
        ax.annotate(seed_txt, (tse_values[i], performances[i]))
    plt.xlabel('TSE')
    plt.ylabel('Performance')    
    plt.show()


def get_non_flat_neuron_data(data_record, key, variance_threshold = 1e-6):
    brain_data = data_record[key] # shape: (num_trials, sim_steps(500), num_agents, num_dim (num_neurons))
    # move axis, source, destination
    brain_data = np.moveaxis(brain_data, (2,3), (0,1)) # (num_agents, num_dim (num_neurons), num_trials, sim_steps(500))
    brain_data = brain_data[:,:,:,100:] # cut the firs 100 point in each trial (brain outputs needs few steps to converge)        
    var = np.var(brain_data, axis=3) 
    max_var = np.max(var, axis=2) # for each agent, each neuron what is the max variance across trials  
    non_flat_neurons = np.sum(max_var > variance_threshold, axis=1)    
    return non_flat_neurons

def flat_elements_stats(values):
    # values is a list of pairs [x,y] (e.g., [1,2])
    # indicating how many non-flat elements in the corresponding seed
    tuple_list = [tuple(sorted(x)) for x in values]
    sorted_set = sorted(
        set(x for x in tuple_list),
        key = lambda x: np.sum(x)
    )
    if len(sorted_set)==1:
        l = list(sorted_set[0])
        avg = str(l)
        min_max = str(l)        
    else:
        avg = np.mean(tuple_list, axis=0).round(1).tolist()
        min_max = f'{list(sorted_set[0])} ... {list(sorted_set[-1])}'        
    return avg, min_max

def get_last_performance_seeds(base_dir, print_stats=True, 
    print_values=False, plot=False, export_to_csv=False,
    best_sim_stats=False, first_20_seeds=False):

    from pce.run_from_dir import run_simulation_from_dir
    from pce.simulation import Simulation
    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])
    best_exp_performance = []  # the list of best performances of last generation for all seeds
    all_gen_best_performances = [] # all best performances for all seeds for all generations
    seeds = []
    seed_exp_dir = {}
    for n, seed_str in enumerate(seed_dirs,1):
        if first_20_seeds and n>20:
            break
        exp_dir = os.path.join(base_dir, seed_str)
        evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
        if len(evo_files) == 0:
            # no evo files
            continue
        last_evo_file = evo_files[-1]
        evo_file = os.path.join(exp_dir, last_evo_file)
        if not os.path.isfile(evo_file):
            continue
        with open(evo_file) as f_in:
            sim_json_filepath = os.path.join(exp_dir, 'simulation.json')    
            # sim = sim_class.load_from_file(sim_json_filepath)
            exp_evo_data = json.load(f_in)
            s = exp_evo_data['random_seed']
            seeds.append(s)
            seed_exp_dir[s] = exp_dir
            gen_best_perf = np.array(exp_evo_data['best_performances']) # one per population            

            # make sure it's monotonic increasing(otherwise there is a bug)
            # assert all(gen_best_perf[i] <= gen_best_perf[i+1] for i in range(len(gen_best_perf)-1))

            perf_index = lambda a: '|'.join(['{:.5f}'.format(x) for x in a])

            # take only per of first population since 
            # all populations have same best performance
            last_best_performance = gen_best_perf[-1][0] 
            if print_values:
                print('{} {}'.format(seed_str, perf_index(last_best_performance)))            
            best_exp_performance.append(last_best_performance)
            all_gen_best_performances.append(gen_best_perf)

    seeds_perf = {s:p for s,p in zip(seeds,best_exp_performance)}

    if best_sim_stats:
        best_stats_non_flat_neur_outputs = {}
        best_stats_non_flat_neur_states = {}
        best_stats_non_flat_motors = {}
        best_stats_genetic_distance = {}
        best_stats_tse = {}

        for s in seeds:
            s_exp_dir = seed_exp_dir[s]
            performance, trials_performances, evo, sim, data_record = run_simulation_from_dir(s_exp_dir, quiet=True)
            best_stats_non_flat_neur_outputs[s] = get_non_flat_neuron_data(data_record, 'brain_outputs')
            best_stats_non_flat_neur_states[s] = get_non_flat_neuron_data(data_record, 'brain_states')
            best_stats_non_flat_motors[s] = get_non_flat_neuron_data(data_record, 'motors')
            best_stats_genetic_distance[s] = data_record['genotype_distance']

    if print_stats:
        # print('Selected evo: {}'.format(last_evo_file))
        # print('Num seeds:', len(best_exp_performance))
        print('Stats:', stats.describe(best_exp_performance))
        print('\tNum Seeds:', len(seeds_perf))
        print('\tSeed/perf:', seeds_perf)
        # print(f'Non converged ({len(non_converged_seeds)}):', non_converged_seeds)

        if best_sim_stats:
            print(f'Average genetic distance: {np.mean(list(best_stats_genetic_distance.values()))}')            
            flat_neurons_avg, flat_neurons_min_max = flat_elements_stats(best_stats_non_flat_neur_outputs.values())
            print(f'Non flat neurons outputs for each agent (min-max): {flat_neurons_min_max}')
            print(f'Non flat neurons outputs for each agent (avg): {flat_neurons_avg}')
            # for s in best_stats_seeds:
            #     print(f'\tSeed {str(s).zfill(3)}: {best_stats_non_flat_neur_outputs[s]}')            
            # flat_states_avg, flat_states_min_max = flat_elements_stats(best_stats_non_flat_neur_states.values())
            # print(f'Non flat neurons states for each agent (min-max): {flat_states_min_max}')
            # print(f'Non flat neurons states for each agent (avg): {flat_states_avg}')
            # for s in best_stats_seeds:
            #     print(f'\tSeed {str(s).zfill(3)}: {best_stats_non_flat_neur_states[s]}')

    if export_to_csv:
        # save file to csv
        f_name = os.path.join(base_dir,'gen_seeds_error.csv')
        print('saving csv:', f_name)
        all_gen_best_performances = np.transpose(np.array(all_gen_best_performances))
        num_agents, num_gen, num_seeds = all_gen_best_performances.shape
        if num_agents==1:
            all_gen_best_performances = all_gen_best_performances[0,:,:]
            seeds_str = [f'seed_{str(s)}' for s in seeds]            
        else:
            assert num_agents==2
            # num_agents, num_gen, num_seeds -> num_gen, num_seeds, num_agents
            all_gen_best_performances = np.moveaxis(all_gen_best_performances, 0, 2) 
            all_gen_best_performances = np.reshape(all_gen_best_performances,  (num_gen, 2*num_seeds))
            seeds_str = [ [f'seed_{str(s)}A', f'seed_{str(s)}B'] for s in seeds]
            seeds_str = [i for g in seeds_str for i in g] # flattening
        df = pd.DataFrame(all_gen_best_performances, columns=seeds_str)  
        df.to_csv(f_name, index=False)

    if plot:
        plot_best_exp_performance(best_exp_performance, seeds)
        if best_sim_stats:
            # bar_plot_seeds_data_value(best_stats_genetic_distance, 'Genetic distance')
            bar_plot_seeds_data_list(best_stats_non_flat_neur_outputs, 'Non flat neurons outputs')
            # bar_plot_seeds_data_list(best_stats_non_flat_neur_states, 'Non flat neurons states')
            bar_plot_seeds_data_list(best_stats_non_flat_motors, 'Non flat motors')
            scatter_plot_tse_performance(best_stats_tse, best_exp_performance)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze results'
    )

    parser.add_argument('--dir', type=str, help='Directory path')
    parser.add_argument('--print_values', action='store_true', default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--best_sim_stats', action='store_true', default=False, help='Detailed stats of best simulation in each seed')
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--csv', action='store_true', default=False, help='Whether to export results to csv in same dir')
    parser.add_argument('--first20', action='store_true', default=False, help='Whether to run analysis only on first 20 seeds')

    args = parser.parse_args()

    get_last_performance_seeds(
        base_dir=args.dir, 
        print_stats=True, 
        print_values=args.print_values, 
        plot=args.plot, 
        export_to_csv=args.csv,
        best_sim_stats=args.best_sim_stats,
        first_20_seeds=args.first20
    )
