import argparse
from utils.utils import plot_1d_occupancies
import pandas as pd
from pathlib import Path

from agents.ucbvi_ent import UCBVI_Ent
from agents.ucbvi_ent_nobonus import UCBVI_Ent_NoBonus

from agents.mtee_oracle import MTEE_Oracle

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Path to the experiment parameters")
    args = parser.parse_args()
    path = Path(args.path)
    algorithms = ['Random Policy', 'UCBVI-Ent', 'UCBVI-Ent No Bonus', 'MTEE Policy']
    data = pd.read_csv(path / 'occupancies_data.csv')
    print("Loaded data from {}.".format(path))
    mask = data['algorithm'].isin(algorithms)
    data = data[mask]
    print(f"Filtered algorithms : {algorithms}")
    plot_1d_occupancies(data, path)

if __name__ == '__main__':
    main()