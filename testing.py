import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_it
from stable_baselines3 import DQN, PPO

from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
from environments.gym_env import sportsTradingEnvironment
from environments.tennis_simulator import tennisSimulator
from src import plotting
from strategy.avellanedaStoikovStrategy import AvellanedaStoikovStrategy
from strategy.fixedOffsetStrategy import FixedOffsetStrategy
from strategy.randomStrategy import RandomStrategy
from utils import setup


def test_rl_agent_all_combinations(model, num_simulations_per_combination, debug=False):
    tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    k_range = range(3, 13)

    # tennis_probs = [0.60, 0.62]
    # k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    list_final_pnl = []
    for a_s, b_s, k in alive_it(possible_combinations):
        env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                b_s=b_s,
                                                                k=k)
        list_pnl = test_rl_agent_multiple_episods(num_episodes=num_simulations_per_combination,
                                       model=model,
                                       env=env,
                                       plotting=False)

        list_final_pnl.extend(list_pnl)
        # mean_return.extend(sim_results['mean_return'])
        # volatility_returns.extend(sim_results['volatility'])
        # min_pnl.extend(sim_results['min_pnl'])
        # max_pnl.extend(sim_results['max_pnl'])
        # sharpe_ratio.extend(sim_results['sharpe_ratio'])
        # sortino_ratio.extend(sim_results['sortino_ratio'])
        # mean_inv_stake.extend(sim_results['mean_inv_stake'])

    df_pnl = pd.DataFrame(list_final_pnl)
    print(df_pnl.describe())
    sns.histplot(list_final_pnl)
    plt.show()







def test_rl_agent_single_episode(model, env, debug=False):
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if debug:
            print(f"Obs: {obs}\nAction: {action}")

    f = plt.figure(figsize=(10, 10))
    f.add_subplot(2, 1, 1)
    plt.plot(env.price, label="mid-price")
    plt.plot(env.back_prices, label="back price")
    plt.plot(env.lay_prices, label="lay price")
    plt.legend()
    #print(len(env.price), len(env.back_prices), len(env.lay_prices))
    f.add_subplot(2, 1, 2)
    #plt.plot(env.price, label="mid-price")
    plt.plot(env.list_pnl, label="PnL")
    plt.plot(env.list_inventory_stake, label="Stake")
    plt.plot(env.list_inventory_odds, label="Odds")
    plt.legend()
    plt.show()



def test_rl_agent_multiple_episods(num_episodes, model, env, plotting=True, debug=False):
    list_final_pnl = []
    #for episode in alive_it(range(num_episodes)):
    for episode in range(num_episodes):
        obs, info = env.reset()
        terminated = False
        while not terminated:
            if model=="random":
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if debug:
                print(f"Obs: {obs}\nAction: {action}")

        list_final_pnl.append(env.list_pnl[-1])

        # f = plt.figure(figsize=(10, 10))
        # f.add_subplot(2, 2, 1)
        # plt.plot(env.price, label="mid-price")
        # plt.plot(env.back_prices, label="mid-price")
        # plt.plot(env.lay_prices, label="mid-price")
        # plt.legend()
        # f.add_subplot(2, 2, 2)
        # plt.plot(env.back_offsets, label="back offset")
        # plt.plot(env.lay_offsets, label="lay offset")
        # plt.legend()
        # f.add_subplot(2, 2, 3)
        # plt.plot(env.list_pnl, label="PnL")
        # plt.legend()
        # f.add_subplot(2, 2, 4)
        # plt.plot(env.list_inventory_stake, label="Stake")
        # plt.plot(env.list_inventory_odds, label="Odds")
        # plt.legend()

    if plotting:
        df_pnl = pd.DataFrame(list_final_pnl)
        print(df_pnl.describe())
        sns.histplot(list_final_pnl)
        plt.show()

    return list_final_pnl



def test_baseline_strategies(plot_path, strategy, num_simulations_per_combination):
    tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
    k_range = range(3, 13)

    # tennis_probs = [0.60, 0.62]
    # k_range = range(3, 5)

    #offset_range = [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    #num_simulations = 5
    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []

    for a_s, b_s, k in alive_it(possible_combinations):
        price_simulator = tennisSimulator.TennisMarkovSimulator(a_s=a_s, b_s=b_s)
        simulator_framework = AvellanedaStoikovFramework(k=k)

        sim_results = simulator_framework.run_single_simulation(price_simulator=price_simulator,
                                                                strategy=strategy,
                                                                num_simulations=num_simulations_per_combination,
                                                                plotting=False)
        final_pnl.extend(sim_results['final_pnl'])
        mean_return.extend(sim_results['mean_return'])
        volatility_returns.extend(sim_results['volatility'])
        min_pnl.extend(sim_results['min_pnl'])
        max_pnl.extend(sim_results['max_pnl'])
        sharpe_ratio.extend(sim_results['sharpe_ratio'])
        sortino_ratio.extend(sim_results['sortino_ratio'])
        mean_inv_stake.extend(sim_results['mean_inv_stake'])


    plotting.plot_results_of_single_strategy_test(plot_path=plot_path,
                                            dict_results={'final_pnl': final_pnl,
                                                          'mean_return': mean_return,
                                                        'volatility': volatility_returns,
                                                        'min_pnl': min_pnl,
                                                        'max_pnl': max_pnl,
                                                        'sharpe_ratio': sharpe_ratio,
                                                        'sortino_ratio': sortino_ratio,
                                                        'mean_inv_stake': mean_inv_stake}
                                            )


def main():
    _ = setup.setup()

    # ## TESTING BASELINE STRATEGIES
    # strategy = FixedOffsetStrategy(offset=0.2)
    # plot_path = "./plots/fixed_02"

    # strategy = FixedOffsetStrategy(offset=0.5)
    # plot_path = "./plots/fixed_05"

    # strategy = FixedOffsetStrategy(offset=0.8)
    # plot_path = "./plots/fixed_08"

    # strategy = RandomStrategy(range_offset=(0, 1))
    # plot_path = "./plots/random"

    # num_simulations_per_combination = 100
    # test_baseline_strategies(plot_path=plot_path,
    #                         strategy=strategy,
    #                         num_simulations_per_combination=num_simulations_per_combination)


    # ### TEST RL ON SINGLE EPISODE
    # a_s = 0.65
    # b_s = 0.65
    # k = 7
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k)
    # model = DQN.load("test_DQN")

    # test_rl_agent_single_episode(model=model, env=env, debug=False)



    # ### TEST MULTIPLE EPISODES
    # a_s = 0.8
    # b_s = 0.65
    # k = 3
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k)
    # model = DQN.load("./model_weights/DQN_1")
    # test_rl_agent_multiple_episods(num_episodes=100,
    #                                        model=model,
    #                                        env=env,
    #                                        debug=False)



    ### TEST ALL POSSIBLE ENV. COMBINATIONS
    model = DQN.load("./model_weights/DQN_1")
    test_rl_agent_all_combinations(model=model,
                                   num_simulations_per_combination=100)



if __name__=="__main__":
    main()