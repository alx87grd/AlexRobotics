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

from pyro.dynamic import pendulum

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


class SysEnv(gym.Env):

    def __init__(self, sys, dt=0.1, tf=10.0, render_mode=None):

        # x-y ouputs
        y_ub = np.array([+1, +1, sys.x_ub[1]])
        y_lb = np.array([-1, -1, sys.x_lb[1]])

        self.observation_space = spaces.Box(y_lb, y_ub)
        self.action_space = spaces.Box(sys.u_lb, sys.u_ub)

        self.sys = sys
        self.dt = dt

        self.tf = tf
        self.render_mode = render_mode

        # Memory
        self.x = sys.x0
        self.u = sys.ubar
        self.t = 0.0

        if self.render_mode == "human":

            self.animator = self.sys.get_animator()
            self.animator.show_plus(self.x, self.u, self.t)
            plt.pause(0.001)

    def _get_obs(self):

        theta = self.x[0]
        thetadot = self.x[1]

        y = np.array([np.cos(theta), np.sin(theta), thetadot], dtype=np.float32)

        return y

    def _get_info(self):

        return {"state": self.x, "action": self.u}

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.x = np.random.uniform(np.array([-np.pi, -1]), np.array([np.pi, 1]))
        self.u = self.sys.ubar
        self.t = 0.0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, u):

        u = np.clip(u, self.sys.u_lb, self.sys.u_ub)
        x = self.x
        t = self.t
        dt = self.dt

        # Derivatives
        dx = self.sys.f(x, u, t)

        # Euler integration
        x_new = x + dx * dt
        t_new = t + dt

        x_new[0] = self.angle_normalize(x_new[0])

        # Sat speed --> I hate they do this in gym env
        if x_new[1] > sys.x_ub[1]:
            x_new[1] = sys.x_ub[1]
        if x_new[1] < sys.x_lb[1]:
            x_new[1] = sys.x_lb[1]

        # Cost function
        r = -self.sys.cost_function.g(x, u, t)

        terminated = False  # t > self.tf

        truncated = t > self.tf  # False #not self.sys.isavalidstate(x_new)

        # Memory update
        self.x = x_new
        self.t = t_new
        self.u = u

        y = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return y, r, terminated, truncated, info

    def angle_normalize(self, x):

        return ((x + np.pi) % (2 * np.pi)) - np.pi
    
    def render(self):
        if self.render_mode == "human":
            return self._render_frame()

    def _render_frame(self):

        self.animator.show_plus_update(self.x, self.u, self.t)
        plt.pause(0.001)


sys = pendulum.InvertedPendulum()

# Setting physical parameter to reflect the gym environment

# Physical parameters
sys.gravity = 10.0
sys.m1 = 1.0
sys.l1 = 1.0
sys.lc1 = 0.5 * sys.l1
sys.I1 = (1.0 / 12.0) * sys.m1 * sys.l1**2

sys.l_domain = 2 * sys.l1  # graphical domain

# Min/max state and control inputs
sys.x_ub = np.array([+np.pi, +8])
sys.x_lb = np.array([-np.pi, -8])
sys.u_ub = np.array([+2.0])
sys.u_lb = np.array([-2.0])

# Cost Function
# The reward function is defined as: r = -(theta2 + 0.1 * theta_dt2 + 0.001 * torque2)
sys.cost_function.xbar = np.array([0, 0])  # target
sys.cost_function.R[0, 0] = 0.001
sys.cost_function.Q[0, 0] = 1.0
sys.cost_function.Q[1, 1] = 0.1

# Learning
gym_env = SysEnv(sys, dt=0.05, render_mode=None)
model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=250000)

# Animation of the pendulum with the learned policy
gym_env = SysEnv(sys, render_mode="human")
y, info = gym_env.reset()
episodes = 10
for episode in range(episodes):
    y, info = gym_env.reset()
    terminated = False
    truncated = False
    print("\n Episode:", episode)
    while not (terminated or truncated):
        u, _states = model.predict(y, deterministic=True)
        y, r, terminated, truncated, info = gym_env.step(u)
        
    