from pce.main import main

def test_overlapping(p, a, n, o, noshuffle):
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_agents', str(a),         
        '--num_neurons', str(n),
         '--num_objects', str(o),
        '--perf_func', 'OVERLAPPING_STEPS', # OVERLAPPING_STEPS, SHANNON_ENTROPY
        '--agg_func', 'MIN', # MEAN, MIN
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

def test_entropy(p, a, n, o, noshuffle):
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_agents', str(a),         
        '--num_neurons', str(n),
        '--num_objects', str(o),
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
    test_overlapping(p=1, a=2, n=2, o=2, noshuffle=True)
    # test_entropy(p=1, a=1, n=2, o=0, noshuffle=False)
