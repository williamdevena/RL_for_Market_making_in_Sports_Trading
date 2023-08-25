import itertools
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from alive_progress import alive_it
from stable_baselines3 import A2C, DQN, PPO

from baselines.avellanedaStoikovStrategy import AvellanedaStoikovStrategy
from baselines.fixedOffsetStrategy import FixedOffsetStrategy
from baselines.randomStrategy import RandomStrategy
from environments.avellaneda_stoikov.avellanedaStoikovFramework import \
    AvellanedaStoikovFramework
from environments.gym_env import sportsTradingEnvironment
from environments.tennis_simulator import tennisSimulator
from src import plotting
from utils import setup


def test_correlations_state_vars_and_actions(model, mode, num_simul_per_combin):
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))
    list_corr_matrices = []

    for a_s, b_s, k in alive_it(possible_combinations):
        for x in range(num_simul_per_combin):
            env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                    b_s=b_s,
                                                                    k=k,
                                                                    mode='testing')

            dict_results = test_rl_agent_single_episode(model=model,
                                        env=env,
                                        debug=False,
                                        plot_results=False)


            # for x, i in dict_results.items():
            #     print(len(i))

            dict_results['price'] = dict_results['price'][:-1]

            df_results = pd.DataFrame.from_dict(dict_results)
            corr_matrix = df_results.corr()
            list_corr_matrices.append(corr_matrix)

    mean_matrix = np.nanmean(list_corr_matrices, axis=0)

    mask = np.triu(mean_matrix)
    f, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(mean_matrix, square=False, annot=True, mask=mask,
                xticklabels=[features for features, _ in dict_results.items()],
                yticklabels=[features for features, _ in dict_results.items()],)
    plt.show()
    plt.close()







def test_rl_agent_all_combinations(model, num_simulations_per_combination, plot_path, mode="long", debug=False):
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))

    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []
    for a_s, b_s, k in alive_it(possible_combinations):
        env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                b_s=b_s,
                                                                k=k,
                                                                mode='testing')
        sim_results = test_rl_agent_multiple_episods(num_episodes=num_simulations_per_combination,
                                       model=model,
                                       env=env,
                                       plot_results=False)

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







def test_rl_agent_single_episode(model, env, debug=False, plot_results=True):
    obs, info = env.reset()
    terminated = False

    while not terminated:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if debug:
            print(f"Obs: {obs}\nAction: {action}")

    spread = [a+b for (a,b) in list(zip(env.back_offsets, env.lay_offsets))]

    if plot_results:
        f = plt.figure(figsize=(10, 10))
        f.add_subplot(2, 2, 1)
        plt.plot(env.price, label="mid-price")
        plt.plot(env.back_prices, label="back price")
        plt.plot(env.lay_prices, label="lay price")
        plt.legend()

        f.add_subplot(2, 2, 2)
        plt.plot(env.back_offsets, label="back offset")
        plt.plot(env.lay_offsets, label="lay offset")
        plt.plot(spread, label="spread")
        plt.legend()

        #print(len(env.price), len(env.back_prices), len(env.lay_prices))
        f.add_subplot(2, 2, 3)
        #plt.plot(env.price, label="mid-price")
        plt.plot(env.list_pnl, label="PnL")
        plt.plot(env.list_inventory_stake, label="Stake")
        plt.plot(env.list_inventory_odds, label="Odds")
        plt.legend()

        f.add_subplot(2, 2, 4)
        #plt.plot(env.price, label="mid-price")
        plt.plot(env.list_volatility_indicator, label="Volat. indicator")
        plt.legend()

        plt.show()

    return {'stake': env.list_inventory_stake,
            'odds': env.list_inventory_odds,
            'price': env.price,
            'momentum': env.list_momentum_indicator,
            'volatility': env.list_volatility_indicator,
            'spread': spread,
            'back_offset': env.back_offsets,
            'lay_offset': env.lay_offsets}




def test_rl_agent_multiple_episods(num_episodes, model, env, plot_results=True, plot_path=None, debug=False):
    final_pnl = []
    mean_return = []
    volatility_returns = []
    min_pnl = []
    max_pnl = []
    sharpe_ratio = []
    sortino_ratio = []
    mean_inv_stake = []
    #for episode in alive_it(range(num_episodes)):
    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            if model=="random":
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if debug:
                print(f"Obs: {obs}\nAction: {action}")


        final_pnl.append(env.list_pnl[-1])
        min_pnl.append(np.min(env.list_pnl))
        max_pnl.append(np.max(env.list_pnl))

        # returns = np.diff(env.list_pnl, prepend=0)

        pnl_series = pd.Series(env.list_pnl)
        returns = pnl_series.pct_change()
        returns = returns.replace(to_replace=[np.inf, np.NINF], value=np.nan)
        returns = returns.interpolate()
        returns = returns.replace(np.nan, 0)

        std_returns = np.std(returns)
        downside_std = np.nanstd(np.clip(returns, np.NINF, 0, out=None))
        mean_ret = np.mean(returns)
        mean_return.append(mean_ret)
        volatility_returns.append(std_returns)

        if mean_ret==0 or std_returns==0:
            #print(mean_ret, std_returns)
            sharpe_ratio.append(0)
        else:
            sharpe_ratio.append(mean_ret/std_returns)


        if mean_ret==0 or downside_std==0:
            #print(mean_ret, std_returns)
            sortino_ratio.append(0)
        else:
            # if downside_std==0:
            #     print(mean_ret, downside_std)
            sortino_ratio.append(mean_ret/downside_std)

        mean_inv_stake.append(np.mean(env.list_inventory_stake))

    if plot_results:
        # df_pnl = pd.DataFrame(final_pnl)
        # print(df_pnl.describe())
        # sns.histplot(final_pnl)

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

        # f = plt.figure(figsize=(10, 10))
        # f.add_subplot(3, 3, 1)
        # sns.histplot(final_pnl, label="Final PnL")
        # plt.legend()
        # f.add_subplot(3, 3, 2)
        # sns.histplot(min_pnl, label="Min PnL")
        # plt.legend()
        # f.add_subplot(3, 3, 3)
        # sns.histplot(max_pnl, label="Max PnL")
        # plt.legend()
        # f.add_subplot(3, 3, 4)
        # sns.histplot(mean_return, label="Mean return")
        # plt.legend()
        # f.add_subplot(3, 3, 5)
        # sns.histplot(volatility_returns, label="Volatility returns")
        # plt.legend()
        # f.add_subplot(3, 3, 6)
        # sns.histplot(sharpe_ratio, label="sharpe ratio")
        # plt.legend()
        # f.add_subplot(3, 3, 7)
        # sns.histplot(sortino_ratio, label="sortino ratio")
        # plt.legend()
        # f.add_subplot(3, 3, 8)
        # sns.histplot(mean_inv_stake, label="Mean inv stake")
        # plt.legend()
        # #plt.show()
        # plt.savefig(plot_path)


    return {'final_pnl': final_pnl,
            'mean_return': mean_return,
            'volatility': volatility_returns,
            'min_pnl': min_pnl,
            'max_pnl': max_pnl,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'mean_inv_stake': mean_inv_stake,}



def test_baseline_strategies(plot_path, strategy, num_simulations_per_combination, mode="long"):
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

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
    # # strategy = FixedOffsetStrategy(offset=0.2)
    # # plot_path = "./plots_correct/fixed_02"

    # # strategy = FixedOffsetStrategy(offset=0.5)
    # # plot_path = "./plots_correct/fixed_05"

    # # strategy = FixedOffsetStrategy(offset=0.8)
    # # plot_path = "./plots_correct/fixed_08"

    # strategy = RandomStrategy(range_offset=(0, 1))
    # plot_path = "./plots_correct/random"

    # num_simulations_per_combination = 100
    # test_baseline_strategies(plot_path=plot_path,
    #                         strategy=strategy,
    #                         num_simulations_per_combination=num_simulations_per_combination)






    # ### TEST RL ON SINGLE EPISODE
    # a_s = 0.65
    # b_s = 0.65
    # k = 4
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k,
    #                                                         mode='testing')
    # #model = DQN.load("model_weights/DQN_4_with_vol_ind")
    # #model = DQN.load("model_weights/DQN_5_with_vol_ind")
    # #model = DQN.load("./model_weights/with_return_reward/DQN_1_return_reward")
    # model = DQN.load("./model_weights/with_k_4/DQN_1_k_4")
    # #model = DQN("MlpPolicy", env)

    # test_rl_agent_single_episode(model=model, env=env, debug=False, plot_results=True)







    # ### TEST MULTIPLE EPISODES
    # a_s = 0.65
    # b_s = 0.65
    # k = 7
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k,
                                                            # mode='testing')

    # # # model = DQN.load("./model_weights/DQN_2")
    # # # plot_path = "plots_RL_single/DQN_2"

    # # # model = DQN.load("./model_weights/DQN_3")
    # # # plot_path = "plots_RL_single/DQN_3"

    # # # model = DQN.load("./model_weights/DQN_4_with_vol_ind")
    # # # plot_path = "plots_RL_single/DQN_4_with_vol_ind"

    # # # model = DQN.load("./model_weights/DQN_5_with_vol_ind")
    # # # plot_path = "plots_RL_single/DQN_5_with_vol_ind"

    # # # model = DQN.load("./model_weights/DQN_6_with_vol_ind")
    # # # plot_path = "plots_RL_single/DQN_6_with_vol_ind"

    # # model = DQN.load("./model_weights/with_return_reward/DQN_1_return_reward")
    # # plot_path = "plots_RL_single/DQN_1_return_reward"

    # # model = DQN.load("./model_weights/with_k_4/DQN_1_k_4")
    # # plot_path = "plots_RL_single/DQN_1_k_4"

    # # model = DQN.load("./model_weights/with_k_4_2/DQN_1_k_4")
    # # plot_path = "plots_RL_single/DQN_1_k_4_2"

    # model = DQN.load("./model_weights/DQN_with_k_2/DQN_1_k_2")
    # plot_path = "plots_RL_single/DQN_1_k_2"

    # # # model = A2C.load("./model_weights/A2C_1")
    # # # plot_path = "plots_RL_single/A2C_1"

    # test_rl_agent_multiple_episods(num_episodes=1000,
    #                                 model=model,
    #                                 env=env,
    #                                 plot_results=True,
    #                                 plot_path=plot_path,
    #                                 debug=False)











    ## TEST ALL POSSIBLE ENV. COMBINATIONS

    ### WITHOUT VOLATILITY INDICATOR
    # model = DQN.load("./model_weights/DQN_2")
    # plot_path = "./plots_RL/DQN_2"

    # model = DQN.load("./model_weights/DQN_3")
    # print(model.policy)
    # plot_path = "./plots_RL/DQN_3"

    # model = A2C.load("./model_weights/A2C_1")
    # # print(model.policy)
    # plot_path = "./plots_RL/A2C_1"


    ### WITH VOL. INDICATOR
    # model_name = "DQN_4_with_vol_ind"
    # model_name = "DQN_5_with_vol_ind"
   # model_name = "DQN_6_with_vol_ind"
    #model_name = "with_k_4/DQN_1_k_4"
    #model_name = "with_k_4_2/DQN_1_k_4"
    #model_name = "with_k_2/DQN_1_k_2"
    #model_name = "random_env/DQN_1_random_env"
    #model_name = "random_env/DQN_2_random_env"
    model_name = "random_env/DQN_3_random_env_3500000_steps"

    model = DQN.load(f"./model_weights/{model_name}")



    # model_name = "PPO_1_with_vol_ind"
    # model = PPO.load(f"./model_weights/{model_name}")

    # model_name = "A2C_1_with_vol_ind"
    # model = A2C.load(f"./model_weights/{model_name}")


    plot_path = f"./plots_RL/{model_name}"
    test_rl_agent_all_combinations(model=model,
                                   num_simulations_per_combination=100,
                                   plot_path=plot_path,
                                   #mode="short",
                                   mode="long"
                                   )






    # ### TESTING CORRELATIONS BETWEEN STATE VARIABLES AND ACTIONS (SPREADS)

    # #model_name = "random_env/DQN_2_random_env"
    # model_name = "random_env/DQN_3_random_env_3500000_steps"

    # model = DQN.load(f"./model_weights/{model_name}")

    # test_correlations_state_vars_and_actions(model=model, mode='long',
    #                                          num_simul_per_combin=10)









    # ### PLOT FINAL PNL DISTRIBUTIONS OF MODELS IN SAME GRAPH
    # # strategy_names = [ "DQN_2", "DQN_3", "A2C_1"]
    # # results_path = "./plots_RL"

    # strategy_names = ["fixed_02", "random", "fixed_05", "fixed_08"]
    # results_path = "./plots"

    # metrics = [
    #     # "final_pnl",
    #     "volatility",
    #     "mean_return",
    #     # "min_pnl",
    #     # "max_pnl",
    #     # "sharpe_ratio",
    #     # "sortino_ratio",
    #     # "mean_inv_stake"
    #     ]

    # plotting.plot_results_of_all_strategies_test(results_path=results_path,
    #                                              strategies_names_list=strategy_names,
    #                                              metrics_list=metrics)




if __name__=="__main__":
    main()