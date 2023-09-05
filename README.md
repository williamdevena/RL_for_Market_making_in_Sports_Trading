# Reinforcement Learning for Market Making in the Sports Trading market

This repository contains part of the code written for the MSc Thesis titled "Reinforcement Learning for Market Making in Algorithmic Sports Trading". The other part of the code can be found in the repository called "Betfair_historical_data_exploration_and_analysis".
(https://github.com/williamdevena/Betfair_historical_data_exploration_and_analysis)

## Structure

The repository is organized into several directories:

- `environments`: This directory contains three subdirectories, `avellaneda_stoikov`, `gym_env` and `tennis_markov`, which contain Python modules for different models or environments used in the project.

    - `avellaneda_stoikov`: Contains modules to simulate a variation of Avellaneda-Stoikov (AS) framework, adapted to the sport trading environment (the original one was designed to simulate a stocks trading environment). Used to test the baseline models.

    - `gym_env`. Contains modules that regard the custom Gymnasium environment designed to train and tet RL agents, as well as baseline models (it substitutes the modules in `avellaneda_stoikov`).

    - `tennis_markov`: Contains modules to calculate the probabilities in tennis matches, sets, games, and tiebreaks, using a Markov chain model. These modules are used to simulate the price time series of a tennis bet. The price time series that it simulates is used in conjunction with AS framework to simulate a trading environment in sport betting market (in both `gym_env` and `avellaneda_stoikov`).

- `src`: This directory contains a module for data processing and plotting. The `data_processing.py` module involves cleaning, transforming, or otherwise preparing data for use in the project, while the `plotting.py` module contains functions to plot the data generated in other modules.

- `utils`: This directory contains the `pricefileutils.py` module, which handles Betfair price files (used as datasets in this project), and a `setup.py` module used to setup environment variables and random seeds.


In addition, the repository contains two central modules: `training.py` and `testing.py`. These two modules execute the training and testing of different RL agents.

## Installation
Use the following commands to clone the repository and install the necessary requirements:
<pre>
git clone https://github.com/williamdevena/RL_for_Market_making_in_Sports_Trading.git
cd RL_for_Market_making_in_Sports_Trading
pip install -r requirements.txt
</pre>

## Usage

To run the code and conduct various experiments as outlined in the thesis, you can execute the following Python modules located in the main folder:

- **`prices_simulation.py`**: Runs multiple simulations of a tennis match and plots the probability and odds time series for each match.

- **`random_action_simulation.py`**: Runs a single event simulation using a random agent and plots various graphs, including state variables and PnL.

- **`test_single_event.py`**: Executes a single event simulation with fixed environment parameters using a given trained RL agent, and plots relevant graphs.

- **`test_multiple_events.py`**: Conducts multiple simulations using a given trained RL agent and plots distributions of risk and performance metrics like PnL, Sharpe ratio, etc.

- **`test_all_combinations.py`**: Runs simulations on all possible combinations of environment parameters using a trained RL agent and plots distributions of various risk and performance metrics.

- **`test_correlation_state_actions.py`**: Runs simulations on all possible combinations of environment parameters with a given trained RL agent, and calculates and saves plots of the mean correlation matrix between state variables and actions.

Each module can be run with minimal to no modifications to execute parts of the experiments explained in the [thesis](<LINK TO YOUR THESIS>).


## Installation

To install this project, you will need to...

## Contributing

Contributions are welcome! Please read the contributing guidelines before getting started.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out to me at...