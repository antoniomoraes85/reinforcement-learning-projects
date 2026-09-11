# First-Visit Monte Carlo Control - GridWorld 4x3

Implementation of a Reinforcement Learning agent using First-Visit Monte Carlo Control.

## Overview

This project implements a Monte Carlo control algorithm in a GridWorld 4x3 environment.

The agent learns an approximate optimal policy by interacting with the environment without knowing its transition dynamics.

## Environment

GridWorld 4x3:

- 12 discrete states
- 4 discrete actions:
  - UP
  - RIGHT
  - DOWN
  - LEFT
- Terminal states:
  - Goal (+1)
  - Pit (-1)
- Configurable:
  - step reward
  - slip probability
  - maximum episode length

## Algorithm

Implemented:

- First-Visit Monte Carlo Control
- Incremental Q-value estimation
- Epsilon-greedy exploration policy

## Results

After training, the agent learns a policy capable of reaching the goal state while avoiding the pit.

## Technologies

- Python
- Gymnasium
- NumPy
- Matplotlib
- Google Colab

## Author

Antonio Moraes
