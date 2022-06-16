
"""
Given a directory with N simulation seeds,
returns a bar plot with best performance of last generation
of all seeds.
Run from command line as
python -m dol.analyze_results ./data/exp_dir
where 'exp_dir' contains all the simulation seeds
"""
import os
from tqdm import tqdm
from pce.run_from_dir import run_simulation_from_dir
import json
from joblib import Parallel, delayed
from pce.simulation import Simulation
import matplotlib.pyplot as plt
import numpy as np

def box_plot_seeds_data_value(seed_data, title):
    fig, ax = plt.subplots(figsize=(12, 6))
    # fig.suptitle(title)
    # seeds = seed_data.keys()
    # ind = np.arange(len(seeds))        
    # width = 0.7
    # p_series = list(seed_data.values())
    # x_pos = ind + width/2
    # ax.bar(x_pos, p_series, width)
    # ax.set_xticks(ind + 0.7 / 2)
    # ax.set_xticklabels(seeds)
    # plt.xlabel('Seeds')
    # plt.show()    
    print(seed_data)
    print(seed_data.values())
    print(seed_data.keys())
    print(seed_data[1])
    print(seed_data[1][1])
    #for i in seed_data.keys():

    #print(seed_data.values().items())
    #ax.boxplot(seed_data[1:10][1], showmeans=True)
    ax.boxplot([seed_data[i][1] for i in seed_data.keys()], showmeans=True)
    # ax.violinplot(trials_performances)
    plt.xlabel('Seeds')
    plt.ylabel('Performance')    
    plt.show()

def test_robustness_single(base_dir, seed_str, **kwargs):
    exp_dir = os.path.join(base_dir, seed_str)
    evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
    if len(evo_files)==0:
        return None
    _, _, data_record = run_simulation_from_dir(
            exp_dir, quiet=True,
            **kwargs
        )
    #print(data_record['trials_performances'])
    #print(data_record['performance'])
    all_perf = data_record['trials_performances']
    perf = float(data_record['performance'])
    return perf, all_perf

def test_robustness_seeds(base_dir, **kwargs):

    num_cores = kwargs.get('num_cores', 1)

    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])

    seed_num = [int(s.split('_')[1]) for s in seed_dirs]

    if len(seed_dirs) == 0:
        subdirs = [
            os.path.join(base_dir, d) 
            for d in os.listdir(base_dir) 
            if os.path.isdir(os.path.join(base_dir, d))
        ]
        for d in subdirs:
            print('Runinng on subdir: ', d)
            test_robustness_seeds(d, **kwargs)
        return

    sim_json_filepath = os.path.join(base_dir, seed_dirs[0], 'simulation.json')    
    sim = Simulation.load_from_file(sim_json_filepath, **kwargs)
    random_seed = sim.random_seed

    if num_cores == 1:
        # single core                
        perf = [
            test_robustness_single(base_dir, seed_str, **kwargs)
            for seed_str in tqdm(seed_dirs)
        ]
    else:
        # run parallel job            
        perf = Parallel(n_jobs=num_cores)(
            delayed(test_robustness_single)(base_dir, seed_str, **kwargs) \
            for seed_str in tqdm(seed_dirs)
        )

    seed_pef = {
        s:p for s,p in zip(seed_num, perf)
        if p is not None
    }

    if kwargs.get('plot', False):
        box_plot_seeds_data_value(seed_pef, "")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze robustnes'
    )

    parser.add_argument('--dir', type=str, required=True, help='Directory path')
    parser.add_argument('--num_cores', type=int, default=5, help='Number of cores')
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to plot results')

    #parser.add_argument('--random_seed', type=int, default=123, help='Overriding sim random seed')    
    parser.add_argument('--random_seed', type=int, default=0, help='Overriding sim random seed')    
    parser.add_argument('--performance_function', type=str, help='Type of performance function')
    parser.add_argument('--aggregation_function', type=str, help='Type of aggregation function')

    args = parser.parse_args()

    base_dir=args.dir

    args_dict = vars(args)
    del args_dict['dir'] # delete dir to avoid conflic with run_from_dir

    test_robustness_seeds(
        base_dir, 
        **args_dict
    )
