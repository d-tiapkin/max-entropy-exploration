"""
Baseline for EntGame: computes optimal MVEE policy by regularized Bellman equations given true model
"""
import numpy as np
from numba import jit
from agents.base_agent import BaseAgent
import pandas as pd
import cvxpy as cp

from envs.finitemdp import FiniteMDP
from utils.utils import softmax_sample

class MVEE_Oracle(BaseAgent):
    name: str = "MVEE Policy"
    DELTA: float = 0.1      # fixed value of delta, for simplicity

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float,
                 log_all_episodes: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma)
        self.log_all_episodes = log_all_episodes

    def run(self, total_samples: int) -> pd.DataFrame:
        initial_state = self.env.reset()
        self.reset()
        sample_count = 0
        samples, errors, ucbs = [], [], []

        d = [cp.Variable((self.S, self.A)) for h in range(self.H)]
        
        constraints = [
           d[hh] >= 0 for hh in range(self.H)          # Non-negativity condition
        ] + [
            cp.sum(d[0][ss]) == (ss == initial_state)  for ss in range(self.S)  # Initial conditions
        ]
        for hh in range(self.H-1):
            constraints += [cp.sum(d[hh+1][ss]) == cp.sum(cp.multiply(self.trueP[:,:,ss], d[hh]))  for ss in range(self.S)] 

        objective = cp.Maximize(cp.sum(cp.entr( (sum(d) / self.H))))
        prob = cp.Problem(objective, constraints)
        prob.solve()

        policy = np.zeros((self.H, self.S, self.A))
        for hh in range(self.H):
            policy[hh] = d[hh].value
        policy /= policy.sum(axis=2, keepdims=True)

        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = np.random.choice(self.A, p=policy[hh, state])
                state = self.step(state, action)
            # Log data
            if self.log_all_episodes or sample_count >= total_samples:
                initial_state = self.env.reset()
                samples.append(sample_count)
                ucbs.append(0)
                errors.append(0)
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors,
            "error-ucb": ucbs
        })
