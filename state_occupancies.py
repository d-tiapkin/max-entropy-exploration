from itertools import product
from typing import List, Callable
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from agents.random_baseline import RandomBaseline
from agents.rf_ucrl import RF_UCRL
from agents.entgame import EntGame
from agents.entgame_nobonus import EntGame_NoBonus
from agents.ucbvi_ent import UCBVI_Ent
from agents.ucbvi_ent_nobonus import UCBVI_Ent_NoBonus

from agents.mtee_oracle import MTEE_Oracle
from agents.mvee_oracle import MVEE_Oracle


from utils.configuration import load_config_from_args
from utils.utils import plot_occupancies


def main():
    params = load_config_from_args()
    agents = [
        RandomBaseline,
        RF_UCRL,
        MTEE_Oracle,
        EntGame,
        EntGame_NoBonus,
        MVEE_Oracle,
        UCBVI_Ent,
        UCBVI_Ent_NoBonus,
    ]
    show_occupancies(agents, params)


def show_occupancies(agents: List[Callable], params: dict) -> None:
    print("--- State occupancies ---")
    try:
        data = pd.read_csv(params["out"] / 'occupancies_data.csv')
        print("Loaded data from {}.".format(params["out"] / 'occupancies_data.csv'))
        if data.empty:
            raise FileNotFoundError
    except FileNotFoundError:
        output = Parallel(n_jobs=params["n_jobs"], verbose=5)(
            delayed(occupancies)(agent, params) for agent, _ in product(agents, range(params["n_runs"])))
        data = pd.concat(output, ignore_index=True)
        data.to_csv(params["out"] / 'occupancies_data.csv')
    plot_occupancies(data, params["env"], out_dir=params["out"])


def occupancies(agent_class: Callable, params: dict) -> pd.DataFrame:
    agent = agent_class(**params)
    agent.run(params["occupancies_samples"])
    df = pd.DataFrame({"occupancy": agent.N_sa.sum(axis=1),
                       "state": np.arange(agent.N_sa.shape[0])})
    df["algorithm"] = agent.name
    df["samples"] = params["occupancies_samples"]
    return df


if __name__ == "__main__":
    main()
