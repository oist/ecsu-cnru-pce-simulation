# PCE Simulation

## Installation

1. Make sure you have `Python 3.7.3` installed (try `python3 -V`).
2. Clone the `pce-simulation` package
    ```
    git clone git@gitlab.com:oist-ecsu/pce-simulation.git
    ```
3. Create and activate python virtual environment, upgrade pip and install requirements.
    ```
    cd pce-simulation
    python3 -m venv .venv
    source .venv/bin/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

## Quick start
### Run simulation

To create a new simulation experiment run `pce.main`.\
Use `--help` to see the list of arguments:
```
python -m pce.main --help
```

### Analyzing results

In order to inspect the result of a simulation experiment saved in a specifc directory, run `pce.run_from_dir`
```
python -m pce.run_from_dir --dir DIR
```

Use `--help` to see the list of arguments:
```
python -m pce.run_from_dir --help
```

Most relevant args are:
```
--dir DIR               Directory path
--gen GEB               Generation number to load. Defaults to the last one.
--genotype_idx IDX      Index (0-based) of agent in population to load. Defaults to 0 (best agent).
--trial TRIAL           Number (1-based) of the trial (defaults to the worst performance)
--viz                   Run visualization of the selected trial
--plot                  Run plots of the selected trial
```

## Tutorial

### Python Environment

Alaways remember to activate the python environemnt first
```
source .venv/bin/activate
```

Make sure that the cosole prompt has the `(.venv)` prefix.
### Run a simulation experiment

The following code runs a simulation experiment: 

```
python -m pce.main --dir ./tutorial --evo_seed 1 --num_pop 2 --pop_size 24 --num_agents 2 --noshuffle --num_neurons 2 --max_gen 100 --cores 5
```

If the output folder (in this case `tutorial`) does not exist, the program will create one for you. If it already exists, it will create (or overwrite) a directory inside it with a name which reflects the list of main arguments being used and a further directory with the seed (in this case `tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001`).

The command parameters can be explained as follows:

- `2 agents` with `2 neurons` each, interact on a PCE game.  We set the evolutionary part of the experiment (using the `pyevolver` library) to use `2 populations` of `24 agents` and `100 generations`.
- We use `evo_seed 1`. This is used **only** in the **evolutionary part of the simulation**, e.g, for determining the initial genotype of the population, and the mutations throughout the generations.
- We use the arguemnt `--noshuffle` which means that agents in the two populations are **not randomly shuffled** before being paired in the simulation. This means that agents in the two populations are always pairwise aligned. Although most agents undergo mutation and crossover during evolution, at least 1 agent in each population (the first and best performing one) is part of the "elite" and will be identical in the following generation.
**This ensures that best performance across subsequent generations will stay identical or will increase (monotonically non-decreasing).**
- Finally `5 cores` are used for the experiemnt.

Other evolutionary and simulation arguments are not specified, therefore the default ones are used. In particular:
   - `--perf_func OVERLAPPING_STEPS`: the performance is based on the number of overlapping steps (percentage of the simulation steps where the two agents overlap).
   - `--agg_func MIN`: among the perfomances of the various trials (10 by default) the MINIMUM value is used as the overall performance of the experiment between two agents.
   - `env_length 300`: the environment length is 300
   - `num_objects 2`: the simultation uses 2 objects
   - `agent_width 4`: the agents (and object) width is 4 units
   - `shadow_delta env_length/4`: shadows are 75 units of distance from their respective agents
   - `alternate_sides False`: by default agents are placed on opposite side of the 1-d space across trials in a fixed arrangemnt: the first agent (GREEN in visualizations) always faced outwards, whereas the second (BLUE) always faces inwards.
   - `objects_facing_agents True`: by default the 2 object are positioned facing their respective agents: one outside the environment facing the first agent (GREEN) and one inside facing the second agent (BLUE).

In addition, it is important to mention that currently, in each trial, **agents and objects are positioned randomly** (uniformally) within the environment (e.g., first agent positioned at 3 o'clock and the second at 6 o'clock). Also keep in mind that those positions are determined by a fixed `seed 0` and are identical for all agents and all generations. This seed cannot be changed when running the experiment (with `pce.main`), but can be modified when rerunning the experiment (with `pce.run_from_dir`) to ensure robustness of results (see `--sim_seed` below).

### Console Output

There are some information being printed in the output console.\
Initially a line specifies how many agents are part of the  `n_elite`, `n_mating`, and `n_filling` (see `pyevolver` library for more details).

```
n_elite, n_mating, n_filling:  1 23 0
```

Next, for each generation, the output of the best/worst/average/variance of the agents performance is displayed:
```
Generation   0: Best: 0.02600|0.02600, Worst: 0.00000|0.00000, Average: 0.02067|0.02067, Variance: 0.00005|0.00005
Generation   1: Best: 0.02600|0.02600, Worst: 0.00000|0.00000, Average: 0.01925|0.01925, Variance: 0.00006|0.00006
Generation   2: Best: 0.02800|0.02800, Worst: 0.00000|0.00000, Average: 0.02058|0.02058, Variance: 0.00006|0.00006
Generation   3: Best: 0.02800|0.02800, Worst: 0.00000|0.00000, Average: 0.02050|0.02050, Variance: 0.00007|0.00007
...
Generation  98: Best: 0.40200|0.40200, Worst: 0.00000|0.00000, Average: 0.04033|0.04033, Variance: 0.00677|0.00677
Generation  99: Best: 0.40200|0.40200, Worst: 0.00000|0.00000, Average: 0.03717|0.03717, Variance: 0.00582|0.00582
Generation 100: Best: 0.40200|0.40200, Worst: 0.00000|0.00000, Average: 0.03592|0.03592, Variance: 0.00586|0.00586
```

We can notice that the performance pairs are identical, because in the simulation the paired agents interact together for all trials and received the same performance.
After 100 generations the experiment produces an agent pair (the first agents in each population) achieving a performance of `~0.40`, meaning that in the worse trial, in about `40%` of the simulation steps the two agents overlap.

### Files in Output

In the output directory we find `10` evolution files `evo_xxx.json`, where `xxx` ranges between `000` (very first random population initialized with random genotipe) and `100` (last generation).\
Each evolution file contains information with the parameters related to the evolutionary part of the experiment, such as `population_size`, `num_populations`, the genotype of the agents (`population`), the agents performances (`performances`).

In addition, we find the file `simulation.json` which list all arguments necessary to replicate the simulation settings of this experiments, such as number of neurons (`num_neurons`), trials (`num_trials`) and simulation steps (`num_steps`).

### Analyzing results

If we want to rerun the simulation we just need to run:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001
```

This would output the following:
```
Agent signature: Xbks7
Performance (in json): 0.402
Performance recomputed: 0.402
Trials Performances: [0.488, 0.496, 0.582, 0.558, 0.518, 0.402, 0.648, 0.526, 0.416, 0.568]
Agent(s) signature(s): ['Xbks7', 'ACjQV']
Non flat neurons: [1 1]
Performance of select trial (6/10): 0.402
```

We can see that the recomputed performance (`0.402`) is the same one listed above (next to generation 100). This is the performance of the 6th trial, being the worst one (remember that by defualt we had `--agg_func MIN`).

We can change the simulation seed (determining the positions of objects and agents across trials) with the `--sim_seed` argument:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --sim_seed 123
```
which procuces the following output:
```
Overriding sim_seed from 0 to 123
Agent signature: Xbks7
Performance (in json): 0.402
Performance recomputed: 0.328
Trials Performances: [0.732, 0.504, 0.368, 0.348, 0.48, 0.684, 0.328, 0.668, 0.506, 0.506]
Agent(s) signature(s): ['Xbks7', 'ACjQV']
Non flat neurons: [1 1]
Performance of select trial (7/10): 0.328
```

We can see that the overall (worse) performance is slighly lower than before, but across the 10 new trials there are higher performances.

### Visualizing results

To see a visualization of this trial add the argument `--viz` (or `--mp4` if you want to save the file):
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --viz
```

![Simulation Video](tutorial/img/pce_overlapping_min_2p_2a_2n_2o_noshuffle_seed_001_t6.mp4)

In `viz` mode you can press `P` for pausing/unpausing the simulation. While the simulation is pause you can move manually to previous/next steps with the `left` and `right` arrows.

In order to see the visualization of the best trial between these two agents, (i.e., the 7th one), use `--trial 7` or `--trial best`:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --trial best --viz
```

### Plotting results

In order to see a set of plots use the '--plot' argument:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --plot
```

![Simulation Video](tutorial/img/plot_01_agent_performances.png)
![Simulation Video](tutorial/img/plot_02_brain_outpupts_scatter.png)
![Simulation Video](tutorial/img/plot_03_agents_delta_time.png)
![Simulation Video](tutorial/img/plot_04_agents_vel_time.png)
![Simulation Video](tutorial/img/plot_06_agents_pos_time.png)
![Simulation Video](tutorial/img/plot_07_agents_signal_time.png)
![Simulation Video](tutorial/img/plot_08_agents_sensor_time.png)
![Simulation Video](tutorial/img/plot_09_agents_brain_input_time.png)
![Simulation Video](tutorial/img/plot_09_agents_brain_states_time.png)
![Simulation Video](tutorial/img/plot_10_agents_brain_outputs_time.png)
![Simulation Video](tutorial/img/plot_11_agents_brain_motors_time.png)

### Ghost simulation

One advanced method for analyzing the experiment is to set an agent in "ghost" mode. This means that the agents, intead of interacting with the other agents, "plays back" the movement of itself in a previously recorded simulation. This allows us to investigate scenarios where the behavior of an agent is realistic (identical to the agent interacting in the task) but without being "sensitive" to possible pertubation of the new simulation (e.g., change of the initial position of the other agent).

In order to run the simulation with the first agent being the 'ghost' we run:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --ghost_index 0
```

This will output:
```
Original performance (without ghost and overriding params): 0.402
Agent signature: Xbks7
Performance (in json): 0.402
Performance recomputed: 0.402
Trials Performances: [0.488, 0.496, 0.582, 0.558, 0.518, 0.402, 0.648, 0.526, 0.416, 0.568]
Agent(s) signature(s): ['Xbks7', 'ACjQV']
Non flat neurons: [1 1]
Performance of selected trial (6/10): 0.402
```

We can notice that the results are identical to the original one. This is because the 'ghost_agent' has been placed in a new simulation which is in fact identical to the original one, resulting in the same results.

In order to use the ghost mode in an effective way, we would need to perturbate the new simulation, for instance, changing the initial position of the other agent. We can do that overriding the seed of the simulation in the ghost mode using the `--sim_seed` argument.
This runs the original simulation with the original `seed 0` (without overriding parameters) and saving all position of the ghost agent. Next, a new simulation is run where the ghost agent positions are played back, whereas the other agent is placed in a different position and behave differently from the original simulation, while responding to the behaviour of the ghost agent.

To run the ghost simulation with "pertubation" run the following:
```
python -m pce.run_from_dir --dir ./tutorial/pce_overlapping_min_2p_2a_2n_2o_noshuffle/seed_001 --sim_seed 123 --ghost_index 0 --trial 6 --viz
```

This will output:
```
Overriding sim_seed from 0 to 123
Original performance (without ghost and overriding params): 0.402
Agent signature: Xbks7
Performance (in json): 0.402
Performance recomputed: 0.004
Trials Performances: [0.426, 0.51, 0.664, 0.03, 0.028, 0.03, 0.024, 0.014, 0.04, 0.004]
Agent(s) signature(s): ['Xbks7', 'ACjQV']
Non flat neurons: [1 1]
Performance of selected trial (6/10): 0.03
Visualizing trial 6/10
```

We can notice that all but 3 trials have a very low performance (less than 0.1).

When visualizing the trial we notice that the green agent (ghost) has the same identical behaviour as in the original simulation. However, the blue agent (non-ghost) get stuck on the shadow of the green agent.

![Simulation Video Ghost](tutorial/img/pce_overlapping_min_2p_2a_2n_2o_noshuffle_seed_001_t6_g0_rs123.mp4)