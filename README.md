# Value Iteration, Policy Iteration, and Prioritized Value Iteration

This repository contains implementations of **Value Iteration**, **Policy Iteration**, and **Prioritized Value Iteration**, along with a detailed analysis of how **living rewards** and **transition probabilities** impact these methods on both small and large maps.

## Project Overview

We provide the following methods and analyses:
- **Value Iteration**: Iteratively computes the optimal value function for a given policy.
- **Policy Iteration**: Alternates between policy evaluation and policy improvement to find the optimal policy.
- **Prioritized Value Iteration**: A variant of Value Iteration that prioritizes states for faster convergence.
- **Living Reward and Transition Probability Analysis**: Examines how different living rewards and transition probabilities influence the behavior and performance of the algorithms.

### Maps
The program runs on two types of grid maps:
- **Small Map**: 20x20 grid.
- **Large Map**: 40x40 grid.

The maps are provided via CSV files, where each entry represents a state, and transitions between states depend on the specified transition probabilities.

## How to Run

Use the following command to run the program:

```bash
python iterations.py <map_size_flag> <csv_map_file> <task_flag>

### Arguments:

**map_size_flag**:
- `1` for small map (20x20)
- `0` for large map (40x40)

**csv_map_file**: Path to the CSV file containing the grid map.

**task_flag**:
- `1` to run **Value Iteration**
- `2` to run **Policy Iteration**
- `3` to run **Prioritized Value Iteration**
- `4` to run **Living Reward Analysis**
- `5` to run **Transition Probability Analysis**

### Example:

```bash
python iterations.py 1 small_map.csv 3
'''

### Analysis

The following analyses are performed:

- **Living Reward Impact**: Assesses how changing the living reward affects the optimal policies and value functions.
- **Transition Probability Impact**: Analyzes the influence of different transition probabilities (S, A, S') on the performance of the algorithms.

All results, including convergence behavior, value functions, and policy visualizations, are presented in the final report.

---
'''
## Report

A detailed report is included, explaining:

- The implementation details of the three algorithms.
- The impact of different parameters on performance.
- Comparative results across different configurations and grid sizes.
