# CSC-52081 — Super Mario Bros Reinforcement Learning

Comparing **PPO with pixel observations (CnnPolicy)** vs **PPO with symbolic/RAM observations (MlpPolicy)** on Super Mario Bros, including transfer learning between levels and analysis of catastrophic forgetting.

> **Authors:** Lucas Martim, Sergio Contente, Leonardo Falabella, Gabriel Corsi, Lara Polachini
> **Course:** CSC-52081 — Reinforcement Learning

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Setup](#setup)
3. [How to Run](#how-to-run)
4. [Training](#training)
5. [Project Structure](#project-structure)
6. [How It Works](#how-it-works)
7. [Results](#results)
8. [References](#references)

---

## Project Overview

### Research Question

Can PPO with a **compact symbolic (RAM-based) observation** compete with or outperform **pixel-based CNN approaches** on Super Mario Bros — while being dramatically faster to train?

### Approach

We compare two observation pipelines under the **same PPO algorithm and reward shaping**:

| | Pixel (CnnPolicy) | Symbolic / RAM (MlpPolicy) |
|---|---|---|
| **Observation** | 84×84 grayscale, 4-frame stack | 13×16 tile grid from NES RAM, 4-frame stack → flattened (833-dim) |
| **Network** | Nature CNN (Mnih et al. 2015) | MLP [512, 512] |
| **Speed** | ~78 fps (Colab T4 GPU) | ~500 fps (CPU, 8 cores) |
| **Preprocessing** | Max-pooling, grayscale, resize | RAM extraction via `smb_grid` |

**Reward shaping** (identical for both, adapted from vietnh1009):

| Event | Reward |
|---|---|
| Score increase | `+score_delta / 40` |
| Flag reached | `+50` |
| Death or timeout | `−50` |
| Normalization | All values divided by 10 |

### Experiments

| # | Experiment | Outcome |
|---|---|---|
| 1 | DQN baseline on 1-1 | Fails to converge in 200k steps → motivates PPO |
| 2 | Pixel PPO on 1-1 | Converges (on Colab GPU) |
| 3 | Symbolic PPO on 1-1 | **100% flag rate, ~500k steps on CPU** |
| 4 | Transfer learning: 1-1 → 1-2 | Learns 1-2 successfully; fine-tuned model scores 0% on 1-1 (catastrophic forgetting) |
| 5 | Curriculum via random starts (all World 1) | Improves level generalization |

---

## Setup

### 1. Create and activate environment

```bash
# Using conda
conda create -n mario python=3.14
conda activate mario

# Or with a virtualenv
python -m venv .venv && source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Apply NumPy 2.0 compatibility patches

The `gym`, `nes-py`, and `gym-super-mario-bros` packages need patching for NumPy ≥ 2.0:

**bash / zsh:**
```bash
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
patch -p0 $SITE/nes_py/_rom.py                   < patches/nes_py_numpy2.patch
patch -p0 $SITE/gym/utils/passive_env_checker.py < patches/gym_bool8_numpy2.patch
patch -p0 $SITE/gym_super_mario_bros/smb_env.py  < patches/smb_env_numpy2.patch
patch -p1 $SITE/gym/wrappers/time_limit.py       < patches/gym_time_limit_compat.patch
```

**fish:**
```fish
set SITE (python -c "import site; print(site.getsitepackages()[0])")
patch -p0 $SITE/nes_py/_rom.py                   < patches/nes_py_numpy2.patch
patch -p0 $SITE/gym/utils/passive_env_checker.py < patches/gym_bool8_numpy2.patch
patch -p0 $SITE/gym_super_mario_bros/smb_env.py  < patches/smb_env_numpy2.patch
patch -p1 $SITE/gym/wrappers/time_limit.py       < patches/gym_time_limit_compat.patch
```

---

## How to Run

### Play manually

Control Mario yourself with a keyboard UI:

```bash
python play_mario.py
```

Controls:

| Key | Action |
|---|---|
| → | Move right |
| ← | Move left |
| Z | Jump (A button) |
| X | Run / Fire (B button) |
| Combinations | Right+Z = run-jump, etc. |

The HUD shows real-time steps, reward, X position, and current action. Edit `ENV_ID` at the top of [play_mario.py](play_mario.py) to play a different level (e.g. `SuperMarioBros-1-2-v3`).

---

### Watch a trained agent play

```bash
# Symbolic PPO agent on World 1-1
python watch_agent.py --model models/symbolic_ppo/final_model --env SuperMarioBros-1-1-v3

# Symbolic PPO agent on World 1-2 (requires running notebook 4 first)
python watch_agent.py --model models/symbolic_ppo_1_2/final_model --env SuperMarioBros-1-2-v3

# Pixel PPO agent (CNN) — requires running notebook 2 first, add --pixel flag
python watch_agent.py --model models/pixel_ppo/final_model --env SuperMarioBros-1-1-v3 --pixel

# Run for N episodes
python watch_agent.py --model models/symbolic_ppo/final_model --env SuperMarioBros-1-1-v3 --episodes 10
```

The window renders the game at 3× scale. Per-episode stats and final flag rate are printed at the end.

---

### Debug the RAM grid

Visualize what the symbolic agent "sees" alongside the actual game:

```bash
python debug_ram.py SuperMarioBros-1-1-v3
```

Split-screen view: game pixels (left) + colored 13×16 grid (right).

| Key | Action |
|---|---|
| V | Toggle numeric overlay on grid |
| P | Pause / unpause |
| R | Reset episode |
| Q / Esc | Quit |

Grid cell encoding: `0` empty, `1` solid tile, `-1` enemy, `2` Mario, `3` powerup, `4` fire-bar bead.

---

### Monitor training with TensorBoard

```bash
tensorboard --logdir logs/symbolic_ppo   # one run
tensorboard --logdir logs                # all runs
```

---

## Training

Training is done through numbered Jupyter notebooks in [notebooks/](notebooks/). Run them in order or pick the experiment you want.

| Notebook | Experiment | Notes |
|---|---|---|
| [1_symbolic_dqn_w1l1_train.ipynb](notebooks/1_symbolic_dqn_w1l1_train.ipynb) | DQN baseline on 1-1 | Motivates switching to PPO |
| [2_pixel_ppo_w1l1_train.ipynb](notebooks/2_pixel_ppo_w1l1_train.ipynb) | Pixel PPO (CnnPolicy) on 1-1 | Best run on Colab T4 GPU |
| [3_symbolic_ppo_w1l1_train.ipynb](notebooks/3_symbolic_ppo_w1l1_train.ipynb) | **Symbolic PPO (MlpPolicy) on 1-1** | Main experiment; runs on CPU |
| [4_symbolic_ppo_transfer_w1l2_train.ipynb](notebooks/4_symbolic_ppo_transfer_w1l2_train.ipynb) | Transfer learning: fine-tune on 1-2 | Loads 1-1 model, continues on 1-2 |
| [5_symbolic_ppo_world1_random_train.ipynb](notebooks/5_symbolic_ppo_world1_random_train.ipynb) | Curriculum via random starts (all World 1) | `RandomStartWrapper` across 1-1, 1-2, 1-3, 1-4 |

Checkpoints are saved every 50,000 steps under `logs/<run_name>/` and final models under `models/`.

---

## Project Structure

```
CSC-52081-Mario/
├── play_mario.py                   # Manual keyboard play with HUD
├── watch_agent.py                  # Visual evaluation of trained agents
├── debug_ram.py                    # Split-screen RAM grid debugger
│
├── src/
│   ├── config.py                   # Centralized hyperparameters (PPOConfig, DQNConfig, EvalConfig)
│   ├── wrappers/
│   │   ├── pixel_wrappers.py       # Pixel pipeline: CustomReward, CustomSkipFrame, make_pixel_vec_env
│   │   └── ram_wrappers.py         # Symbolic pipeline: RAMGridObservation, FrameStackGrid, FlattenGrid,
│   │                               #   RandomStartWrapper, make_symbolic_vec_env, make_symbolic_multitask_vec_env
│   └── utils/
│       ├── callbacks.py            # CheckpointAndLogCallback, CurriculumCallback, PerLevelEvalCallback
│       └── smb_utils.py            # smb_grid: NES RAM → 13×16 tile grid (from yumouwei)
│
├── notebooks/                      # Training notebooks (numbered by experiment)
├── models/                         # Saved model checkpoints and final models
├── logs/                           # TensorBoard training logs
├── patches/                        # NumPy 2.0 compatibility patches
├── presentation/                   # Beamer LaTeX slides
└── requirements.txt
```

---

## How It Works

### Symbolic (RAM) observation pipeline

Instead of processing pixels, we read NES RAM directly to build a compact 13×16 tile grid:

1. **`smb_grid`** (in `src/utils/smb_utils.py`) reads RAM addresses to extract:
   - Tile layout (solid blocks, pipes, etc.)
   - Mario's position (screen X/Y, level X)
   - Enemy positions and types
   - Fire-bar bead positions (angle-based rotation math)
   - Powerup positions

2. **`RAMGridObservation`** wraps the environment so `step()` returns this grid instead of pixels.

3. **`FrameStackGrid`** stacks the last 4 frames along the third axis → shape `(13, 16, 4)`.

4. **`FlattenGrid`** flattens to a 833-dim vector (832 grid values + 1 power-up state flag).

5. **`PPO` with `MlpPolicy`** trains a 2-layer MLP `[512, 512]` on these flattened observations.

### Pixel observation pipeline

1. `CustomSkipFrame` skips 4 frames with max-pooling (anti-flickering), converts to 84×84 grayscale, and stacks 4 frames → shape `(4, 84, 84)`.
2. **`PPO` with `CnnPolicy`** uses the Nature CNN architecture (3 conv layers → FC 512).

### Parallel training

Both pipelines use `SubprocVecEnv` with 8 parallel environments for faster rollout collection.

### Reward shaping

Both pipelines use the same shaping so results are directly comparable:
- Positive score deltas → small positive reward
- Reaching the flag → large positive reward
- Dying or timing out → large negative reward

### Callbacks

- **`CheckpointAndLogCallback`**: Saves model checkpoints every N steps; logs per-episode reward, length, and flag rate to TensorBoard — without needing a `Monitor` wrapper.
- **`CurriculumCallback`**: Gradually increases `random_start_steps` during training so Mario must learn to play from diverse positions.
- **`PerLevelEvalCallback`**: Periodically evaluates the agent on multiple levels and logs per-level metrics separately.