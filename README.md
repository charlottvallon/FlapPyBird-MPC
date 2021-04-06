Hierarchical Predictive Learning for FlappyBird
===============

This code builds on a [Mixed Integer Programming](https://github.com/philzook58/FlapPyBird-MPC) approach for a [Flappy Bird Clone](https://github.com/sourabhv/FlapPyBird). 

The task is to steer a small flappy bird around a series of pipe obstacles by controlling the timing of its wing flaps.
The pipe obstacles come in pairs from the bottom and top of the screen, leaving a gap for the bird to carefully fly through. As the bird moves through the task, it sees only a fixed distance ahead: the screen only the shows the two upcoming pairs of pipes.
The strategy behind Flappy Bird lies in planning short-term trajectories that are robust to randomness in the heights of the future pipe obstacles still hidden beyond the screen.

We use [Hierarchical Predictive Learning](https://arxiv.org/abs/2005.05948) to solve the game. 
Hierarchical Predictive Learning (HPL) is a data-driven control scheme based on high-level strategies learned from previous task trajectories. Strategies map a forecast of the environment (i.e. size of upcoming pipe obstacles) to a target set (i.e. height above ground) for which the system should aim across a horizon of N steps. 
These strategies are applied to the new task in real-time using a forecast of the upcoming environment, and the resulting output is used as a terminal region by a low-level MIP receding horizon controller. 
Here we use Gaussian Processes trained on previously collected task trajectories to represent the learned strategies.

In a trial of 50 new games tasks, the Hierarchical Predictive Learning controller earned a mean score nearly 6x higher (161) than the original Mixed Integer Program (28). 
This improvement is the result of incorporating the learned strategy as well as additional reachability-based safety constraints. 

Running Code
---------------------------

1. Install pygame, cvxpy, gurobi, numpy. 
2. `python flappy_pred.py`


Citing Work
---------------------------
1. Charlott Vallon and Francesco Borrelli. "Data-driven hierarchical predictive learning in unknown environments." In IEEE CASE (2020).
