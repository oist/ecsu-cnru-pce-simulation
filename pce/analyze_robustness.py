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

def test_robustness_seeds(base_dir, random_seed=123, num_trials=10, aggregation_function='MIN'):

    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])
    seed_pef = {}
    for seed_str in tqdm(seed_dirs):
        exp_dir = os.path.join(base_dir, seed_str)
        evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
        if len(evo_files)==0:
            print('No evo file, skipping...')
            continue
        _, _, data_record = run_simulation_from_dir(
            exp_dir, quiet=True,
            random_seed=random_seed, 
            num_trials=num_trials, 
            aggregation_function=aggregation_function
        )
        seed_pef[seed_str] = float(data_record['performance'])
    seed_pef = {k:v for k,v in sorted(seed_pef.items(), key=lambda x: -x[1])}
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

    args = parser.parse_args()

    test_robustness_seeds(
        base_dir=args.dir, 
    )
