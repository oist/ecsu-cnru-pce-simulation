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
from pce.run_from_dir import run_simulation_from_dir
from pce.simulation import Simulation


def test_robustness_seeds(base_dir, random_seed=123, num_trials=10, aggregation_function='MIN'):

    seed_dirs = sorted([d for d in os.listdir(base_dir) if d.startswith('seed_')])
    for n, seed_str in enumerate(seed_dirs,1):
        exp_dir = os.path.join(base_dir, seed_str)
        seed_num = int(seed_str.split('_')[1])
        print('Seed: ', seed_num)
        evo_files = sorted([f for f in os.listdir(exp_dir) if 'evo_' in f])
        if len(evo_files)==0:
            print('No evo file, skipping...')
            continue
        evo, sim, data_record = run_simulation_from_dir(
            exp_dir, quiet=True,
            random_seed=random_seed, 
            num_trials=num_trials, 
            aggregation_function=aggregation_function
        )        
        for key in ['performance']:
            print(key, data_record[key])
        print()


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description='Test robustnes'
    )

    parser.add_argument('--dir', type=str, help='Directory path')

    args = parser.parse_args()

    test_robustness_seeds(
        base_dir=args.dir, 
    )
