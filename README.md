# CSC-52081 — Super Mario Bros RL

Reinforcement learning agents (DQN & PPO) trained on Super Mario Bros using pixel and symbolic (RAM) observations.

## Setup

```fish
python -m venv .venv
source .venv/bin/activate.fish
pip install -r requirements.txt
```

### Apply compatibility patches

The dependencies `gym`, `nes-py`, and `gym-super-mario-bros` were written before NumPy 2.0 and require patching.
`gym`'s `TimeLimit` wrapper also uses the new 5-value step API while `nes-py` still returns the old 4-value API:

```fish
set SITE (python -c "import site; print(site.getsitepackages()[0])")
patch -p0 $SITE/nes_py/_rom.py                   < patches/nes_py_numpy2.patch
patch -p0 $SITE/gym/utils/passive_env_checker.py < patches/gym_bool8_numpy2.patch
patch -p0 $SITE/gym_super_mario_bros/smb_env.py  < patches/smb_env_numpy2.patch
patch -p1 $SITE/gym/wrappers/time_limit.py       < patches/gym_time_limit_compat.patch
```

## Usage

### Visual sanity check

```fish
python tests/visual_test.py
```

### Run tests

```fish
python -m pytest tests/
```

### Training

Open and run the notebooks in order:

| Notebook | Description |
|---|---|
| `notebooks/01_pixel_dqn_train.ipynb` | DQN on pixel observations |
| `notebooks/02_pixel_ppo_train.ipynb` | PPO on pixel observations |
| `notebooks/03_symbolic_dqn_train.ipynb` | DQN on symbolic (RAM) observations |
| `notebooks/04_symbolic_ppo_train.ipynb` | PPO on symbolic (RAM) observations |
| `notebooks/05_evaluation.ipynb` | Evaluate and compare trained agents |
| `notebooks/06_visualizations.ipynb` | Plot results |
