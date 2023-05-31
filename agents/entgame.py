"""
EntGame Algorithm
"""
import numpy as np
from numba import jit
from agents.base_agent import BaseAgent
import pandas as pd

from envs.finitemdp import FiniteMDP
from utils.utils import random_argmax


class EntGame(BaseAgent):
    Q: np.ndarray  # upper bound on Q-function of sampler-player, Q(h,s,a) in the paper
    V: np.ndarray  # upper bound on V-function of sampler-player, V(h,s) = max_a Q(h,s,a) in the paper
    name: str = "EntGame"
    DELTA: float = 0.1      # fixed value of delta, for simplicity
    N0: int = 1             # fixed value of n_0

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, clip: bool, bonus_scale_factor: float,
                 log_all_episodes: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma)
        self.clip = clip
        self.bonus_scale_factor = bonus_scale_factor
        self.log_all_episodes = log_all_episodes

        # compute maximum value function for each step h
        self.v_max = np.zeros(self.H + 1)
        for hh in range(self.H-1, -1, -1):
            self.v_max[hh] = 1 + self.gamma * self.v_max[hh + 1]

    def reset(self) -> None:
        super().reset()
        self.Q = np.zeros((self.H, self.S, self.A))
        self.V = np.zeros((self.H, self.S))

    def beta(self) -> float:
        S, A, H = self.S, self.A, self.H 
        return np.log(4*S*A*H/self.DELTA) + (S-1)*np.log(np.e*(1 + self.N_sa/(S-1)))

    @staticmethod
    @jit(nopython=True)
    def ucb_value_iteration(R: np.ndarray, Q: np.ndarray, V: np.ndarray, P_hat: np.ndarray, 
                            horizon: int, gamma: float, bonus: np.ndarray, vmax: np.ndarray, 
                            clip: bool, bonus_scale_factor: float) -> None:
        S, A = Q[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = 0
                for aa in range(A):
                    q_aa = R[ss, aa] + bonus_scale_factor * vmax[hh] * bonus[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P_hat[ss, aa, :].dot(V[hh+1, :])
                    if aa == 0 or q_aa > max_q:
                        max_q = q_aa
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = max_q
                if clip:
                    V[hh, ss] = min(vmax[hh], V[hh, ss])

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        sample_count = 0
        samples, errors, ucbs = [], [], []
        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = random_argmax(self.Q[hh, state, :])
                state = self.step(state, action)

            # Compute policy of sampler player
            S, A = self.Q[0, :, :].shape
            v_max_scale = np.log(sample_count + S*A)
            R = v_max_scale - np.log(self.N_sa + self.N0)
            bonus = np.sqrt(self.beta()/np.maximum(1, self.N_sa))
            self.ucb_value_iteration(R, self.Q, self.V, self.P_hat, self.H, self.gamma,
                                     bonus, v_max_scale * self.v_max, self.clip, self.bonus_scale_factor)

            # Log data
            if self.log_all_episodes or sample_count >= total_samples:
                initial_state = self.env.reset()
                samples.append(sample_count)
                ucbs.append(self.V[0, initial_state])
                errors.append(self.estimation_error())
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors,
            "error-ucb": ucbs
        })
