# Reinforcement Learning for Market Making in Algorithmic Sports Trading

This repository contains part of the code written for the MSc Thesis titled "Reinforcement Learning for Market Making in Algorithmic Sports Trading". The other part of the code can be found in the repository called "Betfair_historical_data_exploration_and_analysis".
(https://github.com/williamdevena/Betfair_historical_data_exploration_and_analysis)

## Structure

The repository is organized into several directories:

- `environments`: This directory contains two subdirectories, `avellaneda_stoikov` and `tennis_markov`, which contain Python modules for different models or environments used in the project.

    - `avellaneda_stoikov`: Contains modules to simulate the Avellaneda-Stoikov (AS) framework, used to simulate a traidng environment and test strategies.

    - `tennis_markov`: Contains modules to calculate the probabilities in tennis matches, sets, games, and tiebreaks, using a Markov chain model. These modules are used to simulate the price time series of a tennis bet. The price time series that it simulates is used in conjunction with AS framework to simulate a trading environment in sport betting market.

- `src`: This directory contains a module for data processing, which involves cleaning, transforming, or otherwise preparing data for use in the project.

- `utils`: This directory contains a module for handling Betfair price files, used as datasets in this project.

## Usage

To use this project, you will need to...

## Installation

To install this project, you will need to...

## Contributing

Contributions are welcome! Please read the contributing guidelines before getting started.

## License

This project is licensed under the terms of the [MIT License](LICENSE).

## Contact

If you have any questions, feel free to reach out to me at...