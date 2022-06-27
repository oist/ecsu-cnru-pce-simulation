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
    ax.set_xlabel("Number of generation")
    ax.set_ylabel("Performance")
    plt.tight_layout()
    plt.savefig("./data/performance") 
    #plt.show()

def plot_exp_performances_box(trials_performances):
    # plotting performance with std of trials. 
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.boxplot(trials_performances, showmeans=True)
    plt.xlabel('Seeds')
    plt.ylabel('Performance')    
    plt.savefig("./data/performance_box") 
    #plt.show()

def plot_data_in_one(data_record, keys, trial_idx='all', log=False):
    """
    Line plot of simulation run for delta, position with shadow objects, signal, brain state, motor, velocities over simulation time steps.
    """
    from matplotlib import gridspec
    fig = plt.figure(figsize=(10, 30))
    spec = gridspec.GridSpec(nrows=9, ncols=1,height_ratios=[3, 3, 1, 1, 3, 3, 3, 3, 3]) # height ratio of figs.
    p_plot = 0 # position of plotting. 
    num_figs = 0 # (number of subtitle)
    for key in keys:
      exp_data = data_record[key]
      if key == 'agents_pos':
        exp_data_shadow = data_record['shadows_pos']
        exp_data_obj = data_record['objs_pos']
      num_trials = len(exp_data) if trial_idx == 'all' else 1
      line_color =['green', 'blue']
      fig_labels={'motors': 'motor ', 'signal':'signal '} # fig legends
      fig_titles={'motors': 'motor ', 'signal':'signal '} # subtitles
      plot_nums = ["(A) ", "(B)", "(C)", "(D)", "(E)", "(F)", "(G)", "(H)", "(I)", "(J)"] # indx of subtitle
      for t in range(num_trials):
          trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]
          if key == 'agents_pos':
            trial_data_shadow = exp_data_shadow[t] if trial_idx == 'all' else exp_data_shadow[trial_idx]
            trial_data_obj = exp_data_obj[t] if trial_idx == 'all' else exp_data_obj[trial_idx]

          if trial_data.ndim == 1: # Delta
              ax = fig.add_subplot(spec[p_plot]) 

              # chage max of trial_data to np.inf (for deliting lines between 0 and env_length)
              if key == "agents_delta_rel":
                  plt.axhline(y=0, color = 'green', linestyle = "dotted") # agent A
                  plt.axhline(y=300/4, color='green', linestyle = "dashdot") # shadow
                  max_idx = np.where(trial_data>=148.0)
                  min_idx = np.where(trial_data<=-148.0)
                  for i in max_idx:
                      trial_data[i] = np.inf 
                  for i in min_idx:
                      trial_data[i] = np.inf 
                  ax.set_xlabel("Time")
                  ax.set_ylabel("relative distance from a")
              else:
                  ax.set_xlabel("Time", fontsize=14)
                  ax.set_ylabel("Delta", fontsize=14)

              ax.set_title(plot_nums[num_figs] + " " + str(key), fontsize=16)
              num_figs+=1
              ax.plot(trial_data)
              p_plot +=1
          else:
              num_agents = len(trial_data)
              if key != 'agents_pos':
                  for a in range(num_agents):
                      ax = fig.add_subplot(spec[p_plot]) 
                      if log: ax.set_yscale('log')
                      agent_trial_data = trial_data[a]
                      agent_trial_data_shadow = trial_data_shadow[a]
                      agent_trial_data_obj = trial_data_obj[a]
                      if agent_trial_data.ndim == 1:
                          agent_trial_data = np.expand_dims(agent_trial_data, -1)
                          if key == 'agents_pos':
                              agent_trial_data_shadow = np.expand_dims(agent_trial_data_shadow, -1)
                              agent_trial_data_obj = np.expand_dims(agent_trial_data_obj, -1)

                      for n in range(agent_trial_data.shape[1]):
                          if key in fig_labels.keys():
                            ax.plot(agent_trial_data[:, n], label=fig_labels[key]+'{}'.format(n + 1))
                          else:
                            ax.plot(agent_trial_data[:, n], label='neuron {}'.format(n + 1))
                          handles, labels = ax.get_legend_handles_labels()
                          if key != 'signal':
                            ax.legend(handles, labels, loc='upper right')
                      if a == 0: # subtitle
                          ax.set_title(plot_nums[num_figs] + " " + str(key) + " (agent 1 (upper), agent 2 (lower))", fontsize=16)
                          num_figs+=1
                      ax.set_xlabel("Time", fontsize = 14)
                      p_plot += 1
                      if key =="agents_vel":
                          ax.set_ylabel("Velocity ", fontsize=14)
                      elif key =="signal":
                          ax.set_ylabel("Inputs", fontsize=14)
                      elif key =="sensor":
                          ax.set_ylabel("Outputs")
                      elif key =="brain_inputs":
                          ax.set_ylabel("Inputs")
                      elif key =="brain_states":
                          ax.set_ylabel("State", fontsize=14)
                      elif key =="brain_outputs":
                          ax.set_ylabel("Outputs")
                      elif key =="motors":
                          ax.set_ylabel("Outputs", fontsize=14)
              elif key == "agents_pos":
                  ax = fig.add_subplot(spec[p_plot]) 
                  if log: ax.set_yscale('log')
                  for an in range(num_agents):
                      agent_trial_data = trial_data[an]
                      agent_trial_data_shadow = trial_data_shadow[an]
                      agent_trial_data_obj = trial_data_obj[an]
                      if agent_trial_data.ndim == 1:
                          agent_trial_data = np.expand_dims(agent_trial_data, -1)
                          agent_trial_data_shadow = np.expand_dims(agent_trial_data_shadow, -1)
                          agent_trial_data_obj = np.expand_dims(agent_trial_data_obj, -1)
                      max_idx = np.where(agent_trial_data>=295.0)
                      min_idx = np.where(agent_trial_data<=5.0)
                      for i in max_idx:
                          agent_trial_data[i] = np.inf 
                      for i in min_idx:
                          agent_trial_data[i] = np.inf 
                      for nn in range(agent_trial_data.shape[1]):
                          ax.plot(agent_trial_data[:, nn], color=line_color[an], label='agent '+str(an+1)) #agent 
                          ax.plot(agent_trial_data_shadow[:, nn], color=line_color[an], linestyle='dotted') #shadow of agent
                          ax.plot(agent_trial_data_obj[:, nn], color=line_color[an], linestyle='dashdot') #shadow of agent
                  handles, labels = ax.get_legend_handles_labels()
                  ax.legend(handles, labels, loc='upper right')
                  ax.set_title(plot_nums[num_figs] + " " + str(key) + " from agent 1", fontsize=16)
                  num_figs+=1
                  ax.set_ylabel("Position ", fontsize=14)
                  ax.set_ylim([-10, 310])
                  p_plot +=1
      plt.tight_layout()
      plt.savefig("./data/all") 
      #plt.show()

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
    line_color = ['green','blue']
    for t in range(num_trials):
        trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]
        num_agents = len(trial_data)
        for a in range(num_agents):
            # plot agent's color.

            ax = fig.add_subplot(num_agents, num_trials, (a * num_trials) + t + 1)  # projection='3d'
            if log:
                ax.set_yscale('log')
            agent_trial_data = trial_data[a]
            # initial position
            ax.scatter(agent_trial_data[:, 0], agent_trial_data[:, 1], color=line_color[a], zorder=1)
            # ax.plot(agent_trial_data[:, 0], agent_trial_data[:, 1], zorder=0)
            ax.set_title("Agent " + str(a+1))
            ax.set_xlabel("Neuro 1")
            ax.set_ylabel("Neuro 2")
    plt.tight_layout()
    plt.savefig("./data/scatter") 
    #plt.show()

def plot_data_time_position(data_record, key, trial_idx='all', log=False):
    """
    Line plot of simulation run for a specific key over simulation time steps.
    """
    exp_data = data_record[key]
    exp_data_shadow = data_record['shadows_pos']
    exp_data_obj = data_record['objs_pos']
    num_trials = len(exp_data) if trial_idx == 'all' else 1
    fig = plt.figure(figsize=(10, 3))
    title = key.replace('_', ' ').title() + " (Time)"
    fig.suptitle(title)
    line_color =['green', 'blue']
    for t in range(num_trials):
        ax = fig.add_subplot(1, num_trials, t + 1)

        trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]
        trial_data_shadow = exp_data_shadow[t] if trial_idx == 'all' else exp_data_shadow[trial_idx]
        trial_data_obj = exp_data_obj[t] if trial_idx == 'all' else exp_data_obj[trial_idx]

        num_agents = len(trial_data)
        for a in range(num_agents):
            if log: ax.set_yscale('log')
            agent_trial_data = trial_data[a]
            agent_trial_data_shadow = trial_data_shadow[a]
            agent_trial_data_obj = trial_data_obj[a]
            if agent_trial_data.ndim == 1:
                agent_trial_data = np.expand_dims(agent_trial_data, -1)
                agent_trial_data_shadow = np.expand_dims(agent_trial_data_shadow, -1)
                agent_trial_data_obj = np.expand_dims(agent_trial_data_obj, -1)

            # chage max of trial_data to np.inf (for deliting lines between 0 and env_length)
            max_idx = np.where(agent_trial_data>=295.0)
            min_idx = np.where(agent_trial_data<=5.0)
            for i in max_idx:
                agent_trial_data[i] = np.inf 
            for i in min_idx:
                agent_trial_data[i] = np.inf 

            max_idx = np.where(agent_trial_data_shadow>=295.0)
            min_idx = np.where(agent_trial_data_shadow<=5.0)
            for i in max_idx:
                agent_trial_data_shadow[i] = np.inf 
            for i in min_idx:
                agent_trial_data_shadow[i] = np.inf 

            ax.set_title("Agent " + str(a+1))
            ax.set_xlabel("Time")
            # plot agent's color.
            for nn in range(agent_trial_data.shape[1]):
                ax.plot(agent_trial_data[:, nn], color=line_color[a], label='agent '+str(a+1)) #agent 
                ax.plot(agent_trial_data_shadow[:, nn], color=line_color[a], linestyle='dotted') #shadow of agent
                ax.plot(agent_trial_data_obj[:, nn], color=line_color[a], linestyle='dashdot') #shadow of agent
                ax.set_ylabel("Position ")
                ax.set_ylim(-10, 310) # 0 - 300
                ax.legend()
  
    plt.tight_layout()
    plt.savefig("./data/posi"+key) 
    #plt.show()

def plot_data_time(data_record, key, trial_idx='all', log=False):
    """
    Line plot of simulation run for a specific key over simulation time steps.
    """
    exp_data = data_record[key]
    num_trials = len(exp_data) if trial_idx == 'all' else 1
    if key =="agents_delta":
      fig = plt.figure(figsize=(10, 3))
    else:
      fig = plt.figure(figsize=(10, 6))
    #title = key.replace('_', ' ').title() + " (Time)"
    #fig.suptitle(title)
    line_colors = ['green','blue']
    fig_labels={'motors': 'motor ', 'signal':'signal '}
    plot_nums = ["(A) ", "(B)"]
    for t in range(num_trials):
        trial_data = exp_data[t] if trial_idx == 'all' else exp_data[trial_idx]

        if trial_data.ndim == 1: # Delta

            ax = fig.add_subplot(1, num_trials, t + 1)

            # chage max of trial_data to np.inf (for deliting lines between 0 and env_length)
            if key == "agents_delta_rel":
                plt.axhline(y=0, color = 'green', linestyle = "dotted") # agent A
                plt.axhline(y=300/4, color='green', linestyle = "dashdot") # shadow
                max_idx = np.where(trial_data>=148.0)
                min_idx = np.where(trial_data<=-148.0)
                for i in max_idx:
                    trial_data[i] = np.inf 
                for i in min_idx:
                    trial_data[i] = np.inf 
                ax.set_xlabel("Time")
                ax.set_ylabel("relative distance from a")
            else:
                ax.set_xlabel("Time", fontsize=14)
                ax.set_ylabel("Delta", fontsize=14)

            ax.plot(trial_data)
        else:
            num_agents = len(trial_data)
            for a in range(num_agents):
                if key == 'agents_pos':
                    ax = fig.add_subplot(num_agents+1, num_trials, (a * num_trials) + t + 1)
                else:
                    ax = fig.add_subplot(num_agents, num_trials, (a * num_trials) + t + 1)
                if log: ax.set_yscale('log')
                agent_trial_data = trial_data[a]
                if agent_trial_data.ndim == 1:
                    agent_trial_data = np.expand_dims(agent_trial_data, -1)

                # chage max of trial_data to np.inf (for deliting lines between 0 and env_length)
                if key == "agents_pos":
                    max_idx = np.where(agent_trial_data>=295.0)
                    min_idx = np.where(agent_trial_data<=5.0)
                    for i in max_idx:
                        agent_trial_data[i] = np.inf 
                    for i in min_idx:
                        agent_trial_data[i] = np.inf 

                for n in range(agent_trial_data.shape[1]):
                    #ax.plot(agent_trial_data[:, n], label='data {}'.format(n + 1))
                    # plot color
                    if key in fig_labels.keys():
                      ax.plot(agent_trial_data[:, n], label=fig_labels[key]+'{}'.format(n + 1))
                    else:
                      ax.plot(agent_trial_data[:, n], label='neuron {}'.format(n + 1))
                    handles, labels = ax.get_legend_handles_labels()
                    if key != 'signal':
                      ax.legend(handles, labels, loc='upper right')
                ax.set_title(plot_nums[a] + "Agent " + str(a+1), fontsize=16)
                ax.set_xlabel("Time", fontsize = 14)
                if key == "agents_pos":
                    if a == 1:
                        ax = fig.add_subplot(num_agents+1, num_trials, (a * num_trials) + t + 2)
                        for an in range(num_agents):
                            # plot agent's color.
                            if an == 0:
                                line_color_ = 'green'
                            elif an == 1:
                                line_color_ = 'blue'

                            agent_trial_data = trial_data[an]
                            if agent_trial_data.ndim == 1:
                                agent_trial_data = np.expand_dims(agent_trial_data, -1)
                            #for i in max_idx:
                            #    agent_trial_data[i] = np.inf 
                            #for i in min_idx:
                            #    agent_trial_data[i] = np.inf 
                            for nn in range(agent_trial_data.shape[1]):
                                if nn == 0:
                                    line_color_ = 'green'
                                elif nn == 1:
                                    line_color_ = 'blue'
                                ax.plot(agent_trial_data[:, nn], color=line_color_, label='neuron {}'.format(nn + 1)) #agent 
                                ax.plot(agent_trial_data[:, nn]-75, color=line_color_, label='neuron {}'.format(nn + 1), linestyle='dotted') #shadow of agent
                    ax.set_ylabel("Position ")
                elif key =="agents_vel":
                    ax.set_ylabel("Velocity ")
                elif key =="signal":
                    ax.set_ylabel("Inputs", fontsize=14)
                elif key =="sensor":
                    ax.set_ylabel("Outputs")
                elif key =="brain_inputs":
                    ax.set_ylabel("Inputs")
                elif key =="brain_states":
                    ax.set_ylabel("State", fontsize=14)
                elif key =="brain_outputs":
                    ax.set_ylabel("Outputs")
                elif key =="motors":
                    ax.set_ylabel("Outputs", fontsize=14)
    
    plt.tight_layout()
    plt.savefig("./data/"+key) 
    #plt.show()


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
    plt.title("Geotype Distace Between Two Agents")
    plt.tight_layout()
    plt.savefig("./data/geno_dist") 
    #plt.show()

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
    plt.title("Population_Genotype Distace (Norm)")
    plt.tight_layout()
    plt.savefig("./data/popu_dist") 
    #plt.show()


def plot_results(evo, sim, trial_idx, data_record):
    """
    Main plotting function.
    """

    data_record = deepcopy(data_record)
    
    # original order of axis: trials, steps, agents
    # we need to rever steps and agents

    #for k in ['agents_pos', 'agents_vel', 'signal', 'sensor', 'brain_inputs', 'brain_states', 
    #          'brain_derivatives', 'brain_outputs', 'motors']:
    for k in ['agents_pos', 'objs_pos', 'shadows_pos', 'agents_vel', 'signal', 'sensor', 'brain_inputs', 'brain_states', 
              'brain_derivatives', 'brain_outputs', 'motors']:
        data_record[k] = np.moveaxis(data_record[k], 1, 2)

    if trial_idx is None:
        trial_idx = 'all'

    if evo is not None:
        plot_performances(evo, sim, log=False)
        plot_exp_performances_box(data_record['trials_performances'])
        plot_population_genotype_distance(evo, sim)

    plot_genotype_distance(sim)

    # scatter agents
    if sim.num_neurons == 2:
        plot_data_scatter(data_record, 'brain_outputs', trial_idx)
        plot_data_scatter(data_record, 'brain_states', trial_idx)

    # time agents
    if sim.num_agents == 2:
        plot_data_time(data_record, 'agents_delta', trial_idx)
        #plot_data_time(data_record, 'agents_delta_rel', trial_idx)

    keys = ['agents_delta', 'agents_pos', 'signal', 'brain_states', 'motors'] #'velocities'
    #keys = ['agents_delta', 'agents_pos', 'shadows_pos', 'objs_pos', 'signal', 'brain_states', 'motors'] #'velocities'
    plot_data_in_one(data_record, keys, trial_idx)    
    plot_data_time_position(data_record, 'agents_pos', trial_idx)    
    plot_data_time(data_record, 'agents_vel', trial_idx)    
    plot_data_time(data_record, 'agents_pos', trial_idx)
    
    plot_data_time(data_record, 'signal', trial_idx)
    plot_data_time(data_record, 'sensor', trial_idx)
    plot_data_time(data_record, 'brain_inputs', trial_idx)
    plot_data_time(data_record, 'brain_states', trial_idx)
    plot_data_time(data_record, 'brain_outputs', trial_idx)
    plot_data_time(data_record, 'motors', trial_idx)    


def test_plot():
    sim, data_record = test_simulation(
        num_neurons=3,
        num_steps=100,        
        seed=None,        
    )
    plot_results(evo=None, sim=sim, trial_idx=0, data_record=data_record)

if __name__ == "__main__":
    test_plot()



