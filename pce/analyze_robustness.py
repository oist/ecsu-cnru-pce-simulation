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

def test_robustness_single(base_dir, seed_str, random_seed, num_trials, aggregation_function):
    exp_dir = os.path.join(base_dir, seed_str)
    evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
    if len(evo_files)==0:
        return -1
    _, _, data_record = run_simulation_from_dir(
            exp_dir, quiet=True,
            random_seed=random_seed, 
            num_trials=num_trials, 
            aggregation_function=aggregation_function
        )
    perf = float(data_record['performance'])
    return perf

def test_robustness_seeds(
    base_dir, random_seed=123, num_trials=10, aggregation_function='MIN',
    num_cores=1):

    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])

    if num_cores == 1:
        # single core                
        perf = [
            test_robustness_single(base_dir, seed_str, random_seed, num_trials, aggregation_function)
            for seed_str in tqdm(seed_dirs)
        ]
    else:
        # run parallel job            
        perf = Parallel(n_jobs=num_cores)(
            delayed(test_robustness_single)(base_dir, seed_str, random_seed, num_trials, aggregation_function) \
            for seed_str in tqdm(seed_dirs)
        )

    seed_pef = {k:v for k,v in sorted(zip(seed_dirs, perf), key=lambda x: -x[1])}
    print(json.dumps(seed_pef, indent=3))

    out_file = os.path.join(base_dir, 'robustness.json')
    if os.path.exists(out_file):
        robusteness = json.load(open(out_file))
    else:
        robusteness = {}
    robusteness[random_seed] = seed_pef
    json.dump(robusteness, open(out_file, 'w'), indent=3)

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Analyze robustnes'
    )

    parser.add_argument('--dir', type=str, required=True, help='Directory path')
    parser.add_argument('--random_seed', type=int, default=123, help='Overriding sim random seed')
    parser.add_argument('--num_cores', type=int, default=1, help='Number of cores')

    args = parser.parse_args()

    test_robustness_seeds(
        base_dir=args.dir, 
        random_seed=args.random_seed,
        num_cores=args.num_cores
    )
