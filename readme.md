# README

This is a repository modified from [GPG-labeled](https://github.com/arbaazkhan2/gpg_labeled.git) for target defend in reach-avoid games. The model is trained with DDPG with soft updating instead of VPG. This repository contains code for the gym environment `gym_confrontation_game` and gpg-ddpg method `rl_code`.

## Installation

The code is running on gym-0.11.0 and torch-2.4.1+cu118. Please install corresponding dgl version for graph processing. Before running training of testing code please install the gym_confrontation_game environment with commands.

```
cd gym_confrontation_game
pip install -e .
```

## Usage

To use the `gym_confrontation_game` module, import it in your Python code:

```python
import gym_confrontation_game
```

To train or valid the training model:

```
cd rl_code
python3 train.py # for training
python3 test.py # for testing
```

## References

- [GPG-labeled](https://github.com/arbaazkhan2/gpg_labeled.git)
- [Multi_Agent_DDPG](https://github.com/4kasha/Multi_Agent_DDPG.git)


