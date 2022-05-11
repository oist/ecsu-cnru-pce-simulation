# PCE Simulation

Create and activate python virtual environment, and upgrade pip

Install the `pce-simulation` package
```
git clone git@gitlab.com:oist-ecsu/pce-simulation.git
cd pce-simulation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run simulation
```
python -m pce.main --help
```

For instance
```
python -m pce.main --dir ./data/test --seed 1 --num_pop 2 --pop_size 24 --num_agents 2 --num_neurons 2 --num_objects 2 --perf_func OVERLAPPING_STEPS --agg_func MIN --max_gen 20 --cores 5
```