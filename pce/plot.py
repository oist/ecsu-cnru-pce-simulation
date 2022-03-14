"""
Main plotting functions for visualizing experiment behavior
of a specific simulation seed.
"""

import matplotlib.pyplot as plt
import numpy as np
from pce import utils
from pce import params
from copy import deepcopy

from pce.simulation import test_simulation

def plot_performances(evo, sim, log=False, 
                      only_best=False, 
                      moving_average_window=None):
    """
    Performance over generations.
    """
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle("Agent Performances")
    ax = fig.add_subplot(1, 1, 1)
    if log:
        ax.set_yscale('log')
    # only first population (all the same)
    best_perf = np.array(evo.best_performances)[:,0] 
    if only_best:
        ax.plot(best_perf, label='Best')
    else:
        ax.plot(best_perf, label='Best')
        avg_perf = np.array(evo.avg_performances)[:,0] 
        worse_perf = np.array(evo.worst_performances)[:,0] 
        ax.plot(avg_perf, label='Avg')
        ax.plot(worse_perf, label='Worst')
    if moving_average_window is not None:
        best_perf_mw = utils.moving_average(best_perf, moving_average_window)
        ax.plot(best_perf_mw, label=f'Best(mw-{moving_average_window})')
    plt.legend()
    plt.show()


def plot_data_scatter(data_record, key, trial_idx='all', log=False):
    """
    Plotting data from data_record, specific key
    in a scatter plot.
    """
    exp_data = data_record[key]
    num_trials = len(exp_data) if trial_idx == 'all' else 1
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_', ' ').title() + " (Scattter)"
    fig.suptitle(title)
    for t in range(num_trials):
        trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]
        num_agents = len(trial_data)
        for a in range(num_agents):
            ax = fig.add_subplot(num_agents, num_trials, (a * num_trials) + t + 1)  # projection='3d'
            if log:
                ax.set_yscale('log')
            agent_trial_data = trial_data[a]
            # initial position
            ax.scatter(agent_trial_data[0][0], agent_trial_data[0][1], color='orange', zorder=1)
            ax.plot(agent_trial_data[:, 0], agent_trial_data[:, 1], zorder=0)
    plt.show()


def plot_data_time(data_record, key, trial_idx='all', log=False):
    """
    Line plot of simulation run for a specific key over simulation time steps.
    """
    exp_data = data_record[key]
    num_trials = len(exp_data) if trial_idx == 'all' else 1
    fig = plt.figure(figsize=(10, 6))
    title = key.replace('_', ' ').title() + " (Time)"
    fig.suptitle(title)
    for t in range(num_trials):
        trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]
        if trial_data.ndim == 1:
            ax = fig.add_subplot(1, num_trials, t + 1)
            ax.plot(trial_data)
        else:
            num_agents = len(trial_data)
            for a in range(num_agents):
                ax = fig.add_subplot(num_agents, num_trials, (a * num_trials) + t + 1)
                if log: ax.set_yscale('log')
                agent_trial_data = trial_data[a]
                if agent_trial_data.ndim == 1:
                    agent_trial_data = np.expand_dims(agent_trial_data, -1)
                for n in range(agent_trial_data.shape[1]):
                    ax.plot(agent_trial_data[:, n], label='data {}'.format(n + 1))
                    handles, labels = ax.get_legend_handles_labels()
                    fig.legend(handles, labels, loc='upper right')
    plt.show()


def plot_data_time_multi_keys(data_record, keys, title, log=False):
    """
    Plot several keys in the same plot, e.g. both target and tracker position.
    """
    num_trials = len(data_record[keys[0]])
    fig = plt.figure(figsize=(10, 6))
    if title is not None:
        fig.suptitle(title)
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_trials, t + 1)
        if log: ax.set_yscale('log')
        for k in keys:
            trial_data = data_record[k][t]
            ax.plot(trial_data)
    plt.show()


def plot_scatter_multi_keys(data_record, keys, title, log=False):
    """
    Plot several keys in the same scatter plot.
    """
    num_trials = len(data_record[keys[0]])
    fig = plt.figure(figsize=(10, 6))
    if title is not None:
        fig.suptitle(title)
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_trials, t + 1)
        if log: ax.set_yscale('log')
        for k in keys:
            trial_data = data_record[k][t]
            ax.plot(trial_data[:, 0], trial_data[:, 1])
    plt.show()

def plot_genotype_distance(sim):
    if sim.num_agents != 2:
        return
    genotypes = np.array(sim.genotypes)
    genotypes = utils.linmap(genotypes, params.EVOLVE_GENE_RANGE, (0,1))    
    distance = np.abs(genotypes[0]-genotypes[1])
    distance = distance.reshape(1, -1) # row vector
    cmap_inv = plt.cm.get_cmap('viridis_r')        
    plt.imshow(distance, cmap=cmap_inv)       
    plt.clim(0, 1) 
    plt.colorbar()
    plt.show()

def plot_population_genotype_distance(evo, sim):
    """
    Heatmap of genotype distance within the population.
    """
    from sklearn.metrics.pairwise import pairwise_distances    
    population = evo.population
    genotype_length = len(evo.population[0][0])
    if len(evo.population) > 1:
        population = np.concatenate(evo.population)
    else:
        population = population[0]
        
    dist_norm = utils.genotype_group_distance(population)

    cmap_inv = plt.cm.get_cmap('viridis_r')        
    plt.imshow(dist_norm, cmap=cmap_inv)       
    plt.clim(0, 1) 
    plt.colorbar()
    plt.show()


def plot_results(evo, sim, trial_idx, data_record):
    """
    Main plotting function.
    """

    data_record = deepcopy(data_record)
    
    # original order of axis: trials, steps, agents
    # we need to rever steps and agents

    for k in ['agents_pos', 'agents_vel', 'signal', 'sensor', 'brain_inputs', 'brain_states', 
              'brain_derivatives', 'brain_outputs', 'motors']:
        data_record[k] = np.moveaxis(data_record[k], 1, 2)

    if trial_idx is None:
        trial_idx = 'all'

    if evo is not None:
        plot_performances(evo, sim, log=False)
        # plot_population_genotype_distance(evo, sim)

    # plot_genotype_distance(sim)

    # scatter agents
    if sim.num_neurons == 2:
        plot_data_scatter(data_record, 'brain_outputs', trial_idx)
        # plot_data_scatter(data_record, 'brain_states', trial_idx)

    # time agents
    if sim.num_agents == 2:
        plot_data_time(data_record, 'agents_delta', trial_idx)
    
    plot_data_time(data_record, 'agents_vel', trial_idx)    
    # plot_data_time(data_record, 'agents_pos', trial_idx)
    # plot_data_time(data_record, 'brain_inputs', trial_idx)
    plot_data_time(data_record, 'brain_states', trial_idx)
    plot_data_time(data_record, 'brain_outputs', trial_idx)

    plot_data_time(data_record, 'sensor', trial_idx)
    # plot_data_time(data_record, 'motors', trial_idx)    


def test_plot():
    sim, data_record = test_simulation(
        num_neurons=3,
        num_steps=100,        
        seed=None,        
    )
    plot_results(evo=None, sim=sim, trial_idx=0, data_record=data_record)

if __name__ == "__main__":
    test_plot()



