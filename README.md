# Fast Rates for Maximum Entropy Exploration

This repository contains the source code used for the paper *Fast Rates for Maximum Entropy Exploration*.

## Installation

Install the required packages for Python 3 with:

```pip install requirements.txt```

## Instructions

To reproduce the experiments, simply run:

```
python3 state_occupancies.py configs/double_chain.json
```
and
```
python3 state_occupancies.py configs/gridworld.json
```

The results will appear in the `out` directory.
