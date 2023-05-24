# Reinforcement Learning with Fast and Forgetful Memory 
Code for paper _Reinforcement Learning with Fast and Forgetful Memory_, where we propose Fast and Forgetful Memory (FFM).

## Quickstart
If you want to use FFM in your project, clone and install the standalone module
```python
git clone <THIS_REPO>
cd ffm/standalone && pip install .
```
Then you can utilize `FFM` or `DropInFFM`. `FFM` is more flexible and has a cleaner interace (and is a little bit faster too), but `DropInFFM` follows the `torch.nn.GRU` interface very closely, so it can function as a drop-in RNN replacement.

```python
from ffm import FFM, DropInFFM
obs_dim = 16
act_dim = 3
batch_size = 4
time_size = 5

ffm = FFM(input_size=obs_dim, hidden_size=32, context_size=4, output_size=act_dim)
markov_states, recurrent_states = ffm(torch.zeros(batch_size, time_size, obs_dim))
next_markov_states, next_recurrent_states = ffm(markov_states, recurrent_states)

diffm = DropInFFM(input_size=obs_dim, hidden_size=32, context_size=4, output_size=act_dim, batch_first=False)
markov_states, recurrent_state = diffm(torch.zeros(time_size, batch_size, obs_dim))
next_markov_states, next_recurrent_state = diffm(markov_states, recurrent_state)
```

## Repository Structure
This repository may be used to reproduce all experiments and plots. Once the review period is over, we will make our Wandb public, but until then we will keep it private to avoid breaking the double-blind review, and so some logging tools may not work.

- `models` contains the FFM models used in all of our experiments
- `ppo.py` runs the POPGym PPO experiments for FFM, including all ablations
- `popgym` contains POPGym at the commit we evaluated against
- `pomdp-baselines` contains POMDP baselines at the commit we evaluated against
- `plotting` contains all the scripts used to generate plots in the paper, only tested on MacOS. These will not work until the raw data is publically released via wandb.
- `standalone` contains a standalone implementation of FFM for use in other projects -- it is pip installable with only torch as a dependency
- `benchmark.py` contains code to benchmark the wall-clock efficiency and memory usage of models
- `aggregations.py` contains various aggregators, such as the one proposed in FFM

## Reproducing POPGym and Ablations
To rerun ablations:
First, go into `popgym` and pip install following the instructions, it should be something like `pip install ".[baselines,navigation]"`. Then:
`python POPGYM_MODELS=RayFFM,RayFFMNoOscillate,RayFFMNoLearnOscillate,RayFFMNoDecay,RayFFMNoLearnDecay,RayFFMNoInGate,RayFFMNoOutGate ppo.py`
Make sure you are running the `ppo.py` in the FFM root directory, not the `ppo.py` in the popgym directory! 

If you run into the error about `tensorflow_probability` not being installed, that is a ray issue. You can either install it, or go into the file where the error occurs and replace `import tensorflow_probability as tfp` with `tfp = None`.

## Reproducing Efficiency Table
To rerun wall-clock time and memory efficiency:
`python benchmark.py`
the results will be saved in `throughput.csv`. This requires that `popgym` is installed and working.

## Reproducing POMDP-Baselines
POPGym and POMDP-Baselines have conflicting python package versions. For this reason, we suggest using separate docker or conda containers for reproducing this. We had trouble installing their original dependencies, but we provide a pip freeze such that you can install the packages necessary to run our portion of the experiments using
`pip install -r pomdp-baselines/pip_freeze.txt`. Then, follow their instructions for setting up correct python paths. We sometimes run into issues installing this, based on the OS and preinstalled packages. If you run into an error with box2d, you will need to `pip install --force-reinstall box2d-kengz`. That said, any others errors we ran into were fixed with a simple google search! If this annoys you, please ask the authors of POMDP-Baselines to update their packages to ones that are still available on pip/conda. Once this works, call
`python pomdp-baselines/gru_vs_ffm.sh <RANDOM_SEED> <CUDA_DEVICE> <CONFIG_GROUP> `
For example,
```bash
python pomdp-baselines/gru_vs_ffm.sh 0 0 CONFIGS_RNN_P
python pomdp-baselines/gru_vs_ffm.sh 0 1 CONFIGS_FFM_P
```
