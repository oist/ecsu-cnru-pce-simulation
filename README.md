# PCE Simulation

Create and activate python virtual environment, and upgrade pip

Install the `pce-simulation` package
```
git clone git@gitlab.com:oist-ecsu/pce-simulation.git
cd pce-simulation
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Run simulation
```
python -m pce.main --help
```

(see all the parameters in output)
