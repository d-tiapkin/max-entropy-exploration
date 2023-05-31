"""
Baseline for UCBVI-Ent: computes optimal MTEE policy by regularized Bellman equations given true model
"""
import numpy as np
from numba import jit
from agents.base_agent import BaseAgent
import pandas as pd

from envs.finitemdp import FiniteMDP
from utils.utils import softmax_sample
from typing import Tuple, Callable


class MTEE_Oracle(BaseAgent):
    Q: np.ndarray  # upper bound on Q-function of sampler-player, Q(h,s,a) in the paper
    V: np.ndarray  # upper bound on V-function of sampler-player, V(h,s) = max_a Q(h,s,a) in the paper
    name: str = "MTEE Policy"
    DELTA: float = 0.1      # fixed value of delta, for simplicity

    def __init__(self, env: FiniteMDP, horizon: int, gamma: float,
                 log_all_episodes: bool = False, **kwargs: dict) -> None:
        super().__init__(env, horizon, gamma)
        self.log_all_episodes = log_all_episodes
        
        R = -np.sum( self.trueP * np.log(np.maximum(self.trueP,1e-10)), axis=2)
        self.regularized_value_iteration(R, self.trueQ, self.trueV, self.trueP, self.H, self.gamma)

    def reset(self) -> None:
        super().reset()
        self.Q = np.zeros((self.H, self.S, self.A))
        self.V = np.zeros((self.H, self.S))

    @staticmethod
    @jit(nopython=True)
    def regularized_value_iteration(R: np.ndarray, Q: np.ndarray, V: np.ndarray, P_hat: np.ndarray, 
                                        horizon: int, gamma: float) -> None:
        S, A = Q[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                max_q = 0
                for aa in range(A):
                    q_aa = R[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P_hat[ss, aa, :].dot(V[hh+1, :])
                    if aa == 0 or q_aa > max_q:
                        max_q = q_aa
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = max_q + np.log(np.sum(np.exp(Q[hh, ss, :] - max_q)))

    @staticmethod
    @jit(nopython=True)
    def regularized_policy_evaluation(policy: np.ndarray, R: np.ndarray, P_hat: np.ndarray, 
                                        horizon: int, gamma: float) -> Tuple[np.ndarray, np.ndarray]:
        S, A, _ = P_hat.shape
        Q = np.zeros((horizon, S, A))
        V = np.zeros((horizon, S))
        S, A = Q[0, :, :].shape
        for hh in range(horizon-1, -1, -1):
            for ss in range(S):
                for aa in range(A):
                    q_aa = R[ss, aa]
                    if hh < horizon - 1:
                        q_aa += gamma*P_hat[ss, aa, :].dot(V[hh+1, :])
                    Q[hh, ss, aa] = q_aa
                V[hh, ss] = -np.sum(policy[hh, ss] * np.log(policy[hh, ss] + 1e-10)) + np.sum(policy[hh, ss] * Q[hh, ss])
        return Q, V


    def estimate_policy_value(self, policy) -> Tuple[np.ndarray, np.ndarray]:
        """
        :return: Q_pi, V_pi
        """
        R = -np.sum( self.trueP * np.log(np.maximum(self.trueP,1e-10)), axis=2)
        return self.regularized_policy_evaluation(policy, R, self.trueP, self.H, self.gamma)

    def estimation_error(self, policy) -> float:
        initial_state = self.env.reset()
        Q_pi, V_pi = self.estimate_policy_value(policy)
        return np.abs(self.trueV[0, initial_state] - V_pi[0, initial_state])

    def run(self, total_samples: int) -> pd.DataFrame:
        self.reset()
        sample_count = 0
        samples, errors = [], []

        R = -np.sum( self.trueP * np.log(np.maximum(self.trueP, 1e-10)), axis=2)
        self.regularized_value_iteration(R, self.Q, self.V, self.trueP, self.H, self.gamma)

        policy = np.exp(self.Q - self.Q.max(axis=2, keepdims=True))
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
                errors.append(self.estimation_error(policy))
        return pd.DataFrame({
            "algorithm": [self.name] * len(samples),
            "samples": samples,
            "error": errors,
        })
