# Reinforcement Learning for Market Making in the Sports Trading market

This repository contains part of the code written for the MSc Thesis titled "Reinforcement Learning for Market Making in Algorithmic Sports Trading". The other part of the code can be found in the repository called ["Betfair_historical_data_exploration_and_analysis"](https://github.com/williamdevena/Betfair_historical_data_exploration_and_analysis).

## Thesis Overview

The thesis explores the application of Reinforcement Learning (RL) techniques to market making in the sports trading domain, specifically focusing on tennis betting markets.

The thesis is divided in three key experiments (correspondent to Chapter 4, 5 and 6):

- **`In-depth analysis of sports exchange data`** ([other repo](https://github.com/williamdevena/Betfair_historical_data_exploration_and_analysis)): This experiment focuses on analysing the key features of sports exchange data like volumes, volatility, liquidity, and differences between pre-game and in-play data.

- **`Implementation and testing of baseline models`**: It aims at evaluating baseline models to set a benchmark for market making in the sports market. A crucial part of this experiment includes the design of a novel framework for the simulation of a sports trading environment. In particular, the designed framework represents a version of the Avellaneda-Stoikov (AS) framework adapted to the sports trading market.

- **`Development, training and testing of a novel RL agent`**: This last experiment has the objective of developing an agent, using state-of-the-art RL, with the aim of surpassing the performance of the baseline models. This experiments uses and evaluates three different RL algorithms (DQN, PPO and A2C) and evaluates the perfomance of the agents using several metrics, such as Profit and Loss (PnL), Sharpe Ratio, Sortino Ratio and others..


<!-- It presents an in-depth analysis of betting exchange data ([other repo](https://github.com/williamdevena/Betfair_historical_data_exploration_and_analysis)); designs and develops a framework, inspired by the Avellaneda-Stoikov framework, that simulate a sports trading environment, to train and test RL agents; evaluates the performance of baselines models; and finally trains and tests RL agents using three different RL algorithms (DQN, PPO and A2C). The performance of the baseline models and RL agents is evaluated using several metrics, such as Profit and Loss (PnL), Sharpe Ratio, Sortino Ratio and others. -->


<!-- develops and tests baseline market-making models, and further enhances these models using RL algorithms. The project involves the creation of a custom simulation environment that combines a Tennis Markov model for price simulation with a modified Avellaneda-Stoikov framework to mimic the sports trading mechanics. Various RL algorithms are trained and tested within this environment, and their performance is evaluated based on several metrics including Profit and Loss (PnL), Sharpe Ratio, and Sortino Ratio. -->





## Structure

The repository is organized into several directories:

- **`main directory`**: contains various modules designed to be run by the user to execute experiments, like training, testing and others (more details in section "Usage").

- **`environments`**: This directory contains two subdirectories, `gym_env` and `tennis_markov`.

    <!-- - **`avellaneda_stoikov`**: Contains modules to simulate a variation of Avellaneda-Stoikov (AS) framework, adapted to the sport trading environment (the original one was designed to simulate a stocks trading environment). Used to test the baseline models. -->

    - **`gym_env`**. Contains modules that regard the custom Gymnasium environment designed to train and test both the baseline models and the RL agents, including the variation of the AS framework.
    <!-- It includes the code which implements the framework that simulates the sports trading environemnt. -->

    - **`tennis_markov`**: Contains modules to calculate the probabilities in tennis matches, sets, games, and tiebreaks, using a Markov chain model. These modules are used to simulate the price time series of a tennis bet. The price time series that it simulates is then used by the Gymnasium environment.

- **`src`**: This directory contains a module for data processing and plotting. The `data_processing.py` module involves cleaning, transforming, or otherwise preparing data for use in the project, while the `plotting.py` module contains functions to plot the data generated in other modules.

- **`utils`**: This directory contains the `pricefileutils.py` module, which handles Betfair price files (used as datasets in this project), and a `setup.py` module used to setup environment variables and random seeds.

- **`testing`**: This directory contains modules for testing functionalities.


<!-- In addition, the repository contains two central modules: `training.py` and `testing.py`. These two modules execute the training and testing of different RL agents. -->

## Installation
Use the following commands to clone the repository and install the necessary requirements using pip:
<pre>
git clone https://github.com/williamdevena/RL_for_Market_making_in_Sports_Trading.git
cd RL_for_Market_making_in_Sports_Trading
pip install -r requirements.txt
</pre>
Alternatively, if you prefer conda you can recreate the same environment used by the authors:
<pre>
git clone https://github.com/williamdevena/RL_for_Market_making_in_Sports_Trading.git
cd RL_for_Market_making_in_Sports_Trading
conda env create -f environment.yml
conda activate RL_MM
</pre>


## Usage

All the modules can be run with minimal to no modifications.


### Training RL agents

To execute the training stage of the RL agents run the following command:
<pre>
python training.py
</pre>

To decide which type of agent to train (which RL algorithm), modify the `model_name` variable in the `training` function (choose between 'DQN', 'PPO' and 'A2C').

#### Hyperparameters

To ensure reproducibility, in addition to setting all the random seeds, the hyperparameters of the training have been recorder and are loaded by the **setup_training_hyperparameters** function in the **setup.py** module.

These hyperparameters are laoded and used when the **training.py** is executed. To modify them just modfy their values in the **setup.py** module.

#### Tensorboard logging

The training function automatically logs the hyperparameters and useful metrics in Tensorboard. To visualize the Tensorboard UI during or after the training run the following command in your terminal:
<pre>
tensorboard --logdir PATH_TO_LOG_DIR
</pre>
where PATH_TO_LOG_DIR is specified (and can be changed) in the variable `log_dir` in the main function.


### Testing RL agents

To test the RL agents, execute one of these modules:

- **`test_single_event.py`**: Executes a single event simulation with fixed environment parameters using a given trained RL agent, and plots relevant graphs.

- **`test_multiple_events.py`**: Conducts multiple simulations (multiple matches) using a given trained RL agent and save plots of the distributions of risk and performance metrics like PnL, Sharpe ratio, etc.

- **`test_all_combinations.py`**: Runs simulations on all possible combinations of environment parameters using a trained RL agent and save plots of the distributions of various risk and performance metrics.

- **`test_correlation_state_actions.py`**: Runs simulations on all possible combinations of environment parameters with a given trained RL agent, and calculates and saves plots of the mean correlation matrix between state variables and actions.


### Other useful modules to execute

The following are additional modules that could be useful to execute:

- **`prices_simulation.py`**: Runs multiple simulations of a tennis match and plots the probability and odds time series for each match.

- **`random_action_simulation.py`**: Runs a single event simulation (one tennis match), with fixed environment parameters, using a random agent and plots various graphs, including state variables and PnL.



<!-- to execute parts of the experiments explained in the [thesis](<LINK TO YOUR THESIS>). -->



<!--
## Installation

To install this project, you will need to...

## Contributing

Contributions are welcome! Please read the contributing guidelines before getting started.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out to me at... -->