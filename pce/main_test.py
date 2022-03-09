from pce.main import main

def test(p, n, noshuffle):
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        # '--gen_zfill',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_neurons', str(n), 
        '--perf_func', 'OVERLAPPING_STEPS', # OVERLAPPING_STEPS, SHANNON_ENTROPY
        '--agg_func', 'MEAN', # MEAN, MIN
        '--max_gen', '100',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    # last_best_perf = evo.best_performances[-1]
    # return last_best_perf
    return sim, evo

def test_entropy(p, n, noshuffle):
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        # '--gen_zfill',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_neurons', str(n), 
        '--perf_func', 'SHANNON_ENTROPY', # OVERLAPPING_STEPS, SHANNON_ENTROPY
        '--agg_func', 'MEAN', # MEAN, MIN
        '--max_gen', '100',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    # last_best_perf = evo.best_performances[-1]
    # return last_best_perf
    return sim, evo

if __name__ == "__main__":    
    # test(2, 1, True)
    # test(2, 2, True)
    # test(1, 2, True)
    # test(1, 2, False)
    test_entropy(2, 2, True)
