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
        '--perf_func', 'OVERLAPPING_STEPS', # OVERLAPPING_STEPS, SHANNON_ENTROPY, MI
        '--agg_func', 'MIN', # MEAN, MIN
        '--max_gen', '20',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    last_best_perf = evo.best_performances[-1][0]
    print('last perf: ', last_best_perf)
    return sim, evo

def test_entropy(p, a, n, o, noshuffle, noshadow=False):
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_agents', str(a),         
        '--num_neurons', str(n),
        '--num_objects', str(o),
        '--perf_func', 'SHANNON_ENTROPY', # OVERLAPPING_STEPS, SHANNON_ENTROPY, MI
        '--agg_func', 'MEAN', # MEAN, MIN
        '--max_gen', '20',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    if noshadow:
        params.append('--noshadow')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    last_best_perf = evo.best_performances[-1][0]
    print('last perf: ', last_best_perf)
    return sim, evo

def test_mi(p, a, n, o, noshuffle):    
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_agents', str(a),         
        '--num_neurons', str(n),
        '--num_objects', str(o),
        '--perf_func', 'MI', # OVERLAPPING_STEPS, SHANNON_ENTROPY, MI
        '--agg_func', 'MEAN', # MEAN, MIN
        '--max_gen', '20',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    last_best_perf = evo.best_performances[-1][0]
    print('last perf: ', last_best_perf)
    return sim, evo

def test_te(p, a, n, o, noshuffle):    
    params = [             
        '--dir', './data/test', 
        '--seed', '1',
        '--num_pop', str(p), 
        '--pop_size', '24',                 
        '--num_agents', str(a),         
        '--num_neurons', str(n),
        '--num_objects', str(o),
        '--perf_func', 'TE', # OVERLAPPING_STEPS, SHANNON_ENTROPY, MI
        '--agg_func', 'MIN', # MEAN, MIN
        '--max_gen', '20',
        '--cores', '5'        
    ]
    if noshuffle:
        params.append('--noshuffle')
    sim, evo = main(params)
    # print(sim.genotype_populations[:,0].tolist())
    last_best_perf = evo.best_performances[-1][0]
    print('last perf: ', last_best_perf)
    return sim, evo

def test_reproducibility():
    # test oerlapping
    _, evo = test_overlapping(p=1, a=2, n=2, o=2, noshuffle=True)
    last_best_perf = evo.best_performances[-1][0]
    assert last_best_perf==13
    print('test overlapping passed\n\n')

    # test entropy
    _, evo = test_entropy(p=2, a=2, n=2, o=0, noshuffle=True)
    last_best_perf = evo.best_performances[-1][0]
    assert abs(last_best_perf-0.43093)<1e-5
    print('test entropy passed\n\n')

    # test mi
    _, evo = test_mi(p=2, a=2, n=2, o=2, noshuffle=True)
    last_best_perf = evo.best_performances[-1][0]
    assert abs(last_best_perf-8.56174)<1e-5
    print('test mi passed\n\n')


if __name__ == "__main__":    
    # test_reproducibility()
    # test_overlapping(p=1, a=2, n=2, o=2, noshuffle=True)
    # test_entropy(p=2, a=2, n=2, o=0, noshuffle=True, noshadow=True)
    # test_mi(p=2, a=2, n=2, o=2, noshuffle=True)
    test_te(p=2, a=2, n=1, o=2, noshuffle=True)
