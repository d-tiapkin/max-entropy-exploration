"""
UCBVI-Ent Algorithm
"""
import numpy as np
from numba import jit
from agents.mtee_oracle import MTEE_Oracle
import pandas as pd

from envs.finitemdp import FiniteMDP
from utils.utils import softmax_sample
from typing import Tuple, Callable

class UCBVI_Ent(MTEE_Oracle):
    Q: np.ndarray  # upper bound on Q-function of sampler-player, Q(h,s,a) in the paper
    V: np.ndarray  # upper bound on V-function of sampler-player, V(h,s) = max_a Q(h,s,a) in the paper
    name: str = "UCBVI-Ent"
    DELTA: float = 0.1      # fixed value of delta, for simplicity

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float, clip: bool, bonus_scale_factor: float,
                 log_all_episodes: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma, log_all_episodes)
        self.clip = True#clip
        self.bonus_scale_factor = bonus_scale_factor

        # compute maximum value function for each step h
        self.v_max = np.zeros(self.H + 1)
        for hh in range(self.H-1, -1, -1):
            self.v_max[hh] = np.log(self.S * self.A) + self.gamma * self.v_max[hh + 1]

    def reset(self) -> None:
        super().reset()
        self.Q = np.zeros((self.H, self.S, self.A))
        self.V = np.zeros((self.H, self.S))

    def beta_conc(self) -> float:
        S, A, H = self.S, self.A, self.H 
        return np.log(4*S*A*H/self.DELTA) + np.log(4 * np.e * np.maximum(self.N_sa,1) * (2 * self.N_sa + 1))

    def beta_ent_1(self) -> float:
        S, A, H = self.S, self.A, self.H 
        return np.log(4*S*A*H/self.DELTA) + np.log(np.maximum(self.N_sa,1))**2 * np.log(2 * np.maximum(self.N_sa,1) * (self.N_sa + 1))
    
    def beta_ent_2(self) -> float:
        S, A, H = self.S, self.A, self.H 
        return (S-1)*np.log(np.e*(1 + self.N_sa/(S-1))) + 1

    @staticmethod
    @jit(nopython=True)
    def regularized_ucb_value_iteration(R: np.ndarray, Q: np.ndarray, V: np.ndarray, P_hat: np.ndarray, 
                                        horizon: int, gamma: float, bonus: np.ndarray,  bonus_unscaled: np.ndarray,
                                        vmax: np.ndarray, clip: bool, bonus_scale_factor: float) -> None:
        S, A = Q[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = 0
                for aa in range(A):
                    q_aa = R[ss, aa] + vmax[hh] * bonus[ss, aa] + bonus_unscaled[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P_hat[ss, aa, :].dot(V[hh+1, :])
                    if aa == 0 or q_aa > max_q:
                        max_q = q_aa
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = max_q + np.log(np.sum(np.exp(Q[hh, ss, :] - max_q)))
                if clip:
                    V[hh, ss] = min(vmax[hh], V[hh, ss])

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        sample_count = 0
        errors = []

        policy = np.ones((self.H, self.S, self.A)) 
        policy /= policy.sum(axis=2, keepdims=True)
        prev_policy = np.ones((self.H, self.S, self.A))

        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = np.random.choice(self.A, p=policy[hh, state])
                state = self.step(state, action)

            # Compute regularized Bellman equations
            R = -np.sum( self.P_hat * np.log(np.maximum(self.P_hat,1e-10)), axis=2)
            S, A = self.Q[0, :, :].shape

            bonus = np.sqrt(self.beta_conc()/np.maximum(1, self.N_sa)) 
            bonus_unscaled = np.sqrt(self.beta_ent_1() / np.maximum(1, self.N_sa))
            bonus_unscaled += self.beta_ent_2() / np.maximum(1, self.N_sa)

            self.regularized_ucb_value_iteration(R, self.Q, self.V, self.P_hat, self.H, self.gamma,
                                                 bonus, bonus_unscaled, self.v_max, self.clip, self.bonus_scale_factor)
            prev_policy = policy
            policy = np.exp(self.Q - self.Q.max(axis=2, keepdims=True))
            policy /= policy.sum(axis=2, keepdims=True)
            if self.log_all_episodes or sample_count >= total_samples:
                errors.append(self.estimation_error(policy))

        # Reset to generate samples only from the final policy
        self.reset()
        sample_count = 0
        samples = []
        while sample_count < total_samples:
            # Run episode
            state = self.env.reset()
            for hh in range(self.H):
                sample_count += 1
                action = np.random.choice(self.A, p=policy[hh, state])
                state = self.step(state, action)
            # Log data
            if self.log_all_episodes or sample_count >= total_samples:
                samples.append(sample_count)
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors
        })
