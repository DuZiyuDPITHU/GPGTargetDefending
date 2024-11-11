import numpy as np
import random
import copy

class NormalNoise:
    def __init__(self, size, seed, mu=0., sigma=0.2):
        self.size = size
        self.mu = mu
        self.sigma = sigma
        self.seed = random.seed(seed)

    def reset(self):
        pass

    def sample(self):
        return np.random.normal(self.mu, self.sigma, self.size)


# cf. http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# cf. https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
# deltat = 1
class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.normal(size=self.mu.shape)
        self.state = x + dx
        return self.state