# Smart Elevator Scheduling Using Q-Learning

## Overview

This project demonstrates how Reinforcement Learning (RL) can be used to improve elevator scheduling. The elevator acts as an agent that learns from rewards and penalties to decide which floor to serve next.

The main goal is to reduce passenger waiting time and unnecessary elevator movement through a Q-learning approach.

## Problem Statement

In a multi-floor building, passengers request elevators from different floors. The elevator must decide which request to serve while considering passenger waiting time, travel distance, and the number of pending requests.

A basic scheduling method may not always make efficient decisions. Q-learning can help the elevator learn a better strategy through repeated interactions with its environment.

## Objectives

- Apply Q-learning to a simplified elevator scheduling problem.
- Reduce unnecessary elevator movement.
- Improve passenger service efficiency.
- Understand how an RL agent learns through rewards and penalties.
- Evaluate the scheduling strategy using simulation results.

## Reinforcement Learning Components

### 1. Agent
The agent is the elevator control system that decides which action to take.

### 2. Environment
The environment consists of the building floors, elevator position, and passenger requests.

### 3. State
The state represents the current situation of the elevator, including its current floor and pending requests.

### 4. Actions
Actions represent the possible movements or floor-selection decisions available to the elevator.

### 5. Reward Function
The reward function provides feedback to the agent. Positive rewards can encourage efficient service, while penalties can discourage unnecessary movement and excessive waiting.

### 6. Learning Algorithm
Q-learning is used to estimate the value of actions in different states. Through repeated interaction, the agent improves its decision-making policy.


## Features

- Simulates elevator movement and passenger requests.
- Demonstrates reinforcement learning using Q-learning.
- Allows the agent to learn from rewards and penalties.
- Can be evaluated using waiting time, movement distance, and the number of requests served, if these metrics are recorded by the implementation.

## Technologies Used

- Python
- Reinforcement Learning
- Q-learning
- Simulation

## Requirements

Python 3.10 or later is recommended.

Install any additional libraries imported by your Python program.

## How to Run

1. Download or clone the project.
2. Open the project folder in VS Code or another Python editor.
3. Open the terminal.
4. Run the Python file containing the simulation.

Example:

```bash
python main.py
```

Replace `main.py` with the actual filename of your program if it is different.



## Author Information

- **Haleema Sadia** 
