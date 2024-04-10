#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:08:03 2024

@author: alex
"""

import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from gymnasium import spaces


#################################################################
# Create a Gym Environment from a Pyro System
#################################################################
class Sys2Gym(gym.Env):
    """Create a Gym Environment from a Pyro System

    Taken from the Pyro system:
    - x0: nominal initial state
    - f: state evolution function xdot = f(x,u,t)
    - g: cost function g(x,u,t) (reward = -g)
    - h: observation function y = h(x,u,t)
    - x_ub, x_lb: state boundaries
    - u_ub, u_lb: control boundaries

    Additionnal parameters of the gym wrapper are:
    - dt: time step for the integration
    - tf: maximum duration of an episode
    - t0: initial time (only relevant if the system is time dependent)
    - render_mode: None or "human" for rendering the system
    - reset_mode: "uniform", "gaussian" or "determinist"
    - clipping_inputs: True if the system clips the control inputs
    - clipping_states: True if the system clips the states
    - x0_lb, x0_ub: boundaries for the initial state (only relevant if reset_mode is "uniform")
    - x0_std: standard deviation for the initial state (only relevant if reset_mode is "gaussian")
    - termination_is_reached: method to define the termination condition of the task (default is always False representing a continous time task)

    """

    metadata = {"render_modes": ["human"]}

    #################################################################
    def __init__(
        self, sys, dt=0.05, tf=10.0, t0=0.0, render_mode=None, reset_mode="uniform"
    ):

        # Boundaries
        self.t0 = t0
        self.tf = tf  # For truncation of episodes
        self.observation_space = spaces.Box(
            sys.x_lb, sys.x_ub
        )  # default is state feedback
        self.action_space = spaces.Box(sys.u_lb, sys.u_ub)

        # Dynamic evolution
        self.sys = sys
        self.dt = dt
        self.clipping_inputs = True  # The sys is assumed to clip out of bounds inputs
        self.clipping_states = False  # The sys is assumed to clip out of bounds states (some gym env do that but this is a very fake behavior not exibiited by real systems, to use with caution)

        # Rendering
        self.render_mode = render_mode

        # Reset parameters (stocasticity of the initial state)
        self.reset_mode = reset_mode
        # self.reset_mode = "uniform"
        self.x0_lb = sys.x0 + 0.1 * sys.x_lb
        self.x0_ub = sys.x0 + 0.1 * sys.x_ub
        # self.reset_mode = "gaussian"
        self.x0_std = 0.1 * (sys.x_ub - sys.x_lb)
        # self.reset_mode = "determinist"

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = t0

        # Init
        self.render_is_initiated = False

        if self.render_mode == "human":
            self.init_render()

    #################################################################
    def reset(self, seed=None, options=None):

        if self.reset_mode == "uniform":

            super().reset(seed=seed)

            self.x = self.np_random.uniform(self.x0_lb, self.x0_ub)
            self.u = self.sys.ubar
            self.t = self.t0

        elif self.reset_mode == "gaussian":

            super().reset(seed=seed)

            self.x = self.np_random.normal(self.sys.x0, self.x0_std)
            self.u = self.sys.ubar
            self.t = 0.0

        else:
            # Deterministic
            self.x = self.sys.x0
            self.u = self.sys.ubar
            self.t = 0.0

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        return y, info

    #################################################################
    def step(self, u):

        # Clipping of inputs
        if self.clipping_inputs:
            u = np.clip(u, self.sys.u_lb, self.sys.u_ub)

        # State and time at the beginning of the step
        x = self.x
        t = self.t
        dt = self.dt

        # Derivatives
        dx = self.sys.f(x, u, t)

        # Euler integration #TODO use a better integrator
        x_new = x + dx * dt
        t_new = t + dt

        # Horrible optionnal hack to avoid the system to go out of bounds
        if self.clipping_states:
            x_new = np.clip(x_new, self.sys.x_lb, self.sys.x_ub)

        # Termination of episodes
        terminated = self.termination_is_reached()

        # Reward = negative of cost function
        if terminated:
            r = -self.sys.cost_funtion.h(x, t)  # Terminal cost
        else:
            r = (
                -self.sys.cost_function.g(x, u, t) * dt
            )  # Instantaneous cost integrated over the time step

        # Truncation of episodes if out of space-time bounds
        truncated = (t_new > self.tf) or (not self.sys.isavalidstate(x_new))

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u  # The memory of the control input is only used for redering

        # Observation
        y = self.sys.h(self.x, self.u, self.t)

        # Info
        info = {"state": self.x, "action": self.u}

        # Rendering
        if self.render_mode == "human":
            self.render()

        return y, r, terminated, truncated, info

    #################################################################
    def init_render(self):

        self.render_is_initiated = True

        self.animator = self.sys.get_animator()
        self.animator.show_plus(self.x, self.u, self.t)
        plt.pause(0.001)

    #################################################################
    def render(self):

        if self.render_mode == "human":
            if not self.render_is_initiated:
                self.init_render()
            self.animator.show_plus_update(self.x, self.u, self.t)
            plt.pause(0.001)

    #################################################################
    def termination_is_reached(self):
        """This method should be overloaded in the child class to define the termination condition for a task that is not time defined in continous time."""

        return False


"""
#################################################################
##################          Main                         ########
#################################################################
"""


if __name__ == "__main__":
    """MAIN TEST"""

    from pyro.dynamic import pendulum

    from stable_baselines3 import PPO

    sys = pendulum.InvertedPendulum()

    # Physical parameters
    sys.gravity = 10.0
    sys.m1 = 1.0
    sys.l1 = 1.0
    sys.lc1 = 0.5 * sys.l1
    sys.I1 = (1.0 / 12.0) * sys.m1 * sys.l1**2

    sys.l_domain = 2 * sys.l1  # graphical domain

    # Min/max state and control inputs
    sys.x_ub = np.array([+3.0 * np.pi, +8])
    sys.x_lb = np.array([-3.0 * np.pi, -8])
    sys.u_ub = np.array([+2.0])
    sys.u_lb = np.array([-2.0])

    # Time constant
    dt = 0.05

    # Cost Function
    # The reward function is defined as: r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
    sys.cost_function.xbar = np.array([0, 0])  # target
    sys.cost_function.R[0, 0] = 0.001 / dt
    sys.cost_function.Q[0, 0] = 1.0 / dt
    sys.cost_function.Q[1, 1] = 0.1 / dt

    sys.x0 = np.array([-np.pi, 0.0])

    gym_env = Sys2Gym(sys, dt=dt, render_mode=None)

    gym_env.clipping_states = True # To reproduce the behavior of gym pendulum

    gym_env.reset_mode = "uniform"
    gym_env.x0_lb = np.array([-np.pi , -1.0])
    gym_env.x0_ub = np.array([+np.pi , +1.0])

    model = PPO("MlpPolicy", gym_env, verbose=1)
    model.learn(total_timesteps=250000)

    gym_env = Sys2Gym(sys, render_mode="human")

    #gym_env.reset_mode = "uniform"

    episodes = 3
    for episode in range(episodes):
        y, info = gym_env.reset()
        terminated = False
        truncated = False

        print("\n Episode:", episode)
        while not (terminated or truncated):
            u, _states = model.predict(y, deterministic=True)
            y, r, terminated, truncated, info = gym_env.step(u)


    # from pyro.control.reinforcementlearning import stable_baseline3_controller

    # ppo_ctl = stable_baseline3_controller(model)


    # ppo_ctl.plot_control_law(sys=sys, n=100)
    # cl_sys = ppo_ctl + sys

    # cl_sys.x0 = np.array([-3.0, -0.0])
    # cl_sys.compute_trajectory(tf=10.0, n=10000, solver="euler")
    # cl_sys.plot_trajectory("xu")
    # cl_sys.animate_simulation()
