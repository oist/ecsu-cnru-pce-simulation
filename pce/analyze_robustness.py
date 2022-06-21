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

def test_robustness_single(base_dir, seed_str, **kwargs):
    exp_dir = os.path.join(base_dir, seed_str)
    evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
    if len(evo_files)==0:
        return None
    _, _, data_record = run_simulation_from_dir(
            exp_dir, quiet=True,
            **kwargs
        )
    perf = float(data_record['performance'])
    return perf

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
    sim_seed = sim.sim_seed

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
        bar_plot_seeds_data_value(seed_pef, "")

    seed_pef = {k:v for k,v in sorted(seed_pef.items(), key=lambda x: -x[1])}
    print(json.dumps(seed_pef, indent=3))

    out_file = os.path.join(base_dir, 'robustness.json')
    if os.path.exists(out_file):
        robusteness = json.load(open(out_file))
    else:
        robusteness = {}
    robusteness[str(sim_seed)] = seed_pef
    json.dump(robusteness, open(out_file, 'w'), indent=3)


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze robustnes'
    )

    parser.add_argument('--dir', type=str, required=True, help='Directory path')
    parser.add_argument('--gen', type=int, help='Generation number')
    parser.add_argument('--num_cores', type=int, default=5, help='Number of cores')
    parser.add_argument('--plot', action='store_true', default=False, help='Whether to plot results')

    parser.add_argument('--sim_seed', type=int, default=123, help='Overriding sim seed')    
    parser.add_argument('--num_trials', type=int, help='Overriding num trials')    
    parser.add_argument('--shadow_delta', type=float, help='Overriding shadow distance')    
    parser.add_argument('--performance_function', type=str, help='Type of performance function')
    parser.add_argument('--aggregation_function', type=str, default='MIN', help='Type of aggregation function')

    args = parser.parse_args()

    base_dir=args.dir

    args_dict = vars(args)
    del args_dict['dir'] # delete dir to avoid conflic with run_from_dir

    test_robustness_seeds(
        base_dir, 
        **args_dict
    )
