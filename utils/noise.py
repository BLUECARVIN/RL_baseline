import numpy as np
import torch


class RandomActionNoise:
    def __init__(self, action_dim, mu=0, theta=0.1, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.x = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.x = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.x)
        dx = dx + self.sigma * np.random.rand(len(self.x))
        self.x = self.x + dx
        return self.x