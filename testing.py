"""
This moudle contains the functions to execute the testing of the RL agents.
"""

import copy
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


def test_correlations_state_vars_and_actions(model, mode, num_simul_per_combin, plot_path):
    if mode=="long":
        tennis_probs = [0.60, 0.62, 0.64, 0.66, 0.68, 0.70]
        k_range = range(3, 13)
    else:
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)

    possible_combinations = list(itertools.product(tennis_probs, tennis_probs, k_range))
    list_pearson_corr_matrices = []
    list_kendall_corr_matrices = []

    for a_s, b_s, k in alive_it(possible_combinations):
        for x in range(num_simul_per_combin):
            env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
                                                                    b_s=b_s,
                                                                    k=k,
                                                                    mode='fixed')

            dict_results = test_rl_agent_single_episode(model=model,
                                        env=env,
                                        debug=False,
                                        plot_results=False)
            dict_results['price'] = dict_results['price'][:-1]
            df_results = pd.DataFrame.from_dict(dict_results)
            pearson_corr_matrix = df_results.corr(method='pearson')
            kendall_corr_matrix = df_results.corr(method='kendall')
            list_pearson_corr_matrices.append(pearson_corr_matrix)
            list_kendall_corr_matrices.append(kendall_corr_matrix)

    pearson_mean_matrix = np.nanmean(list_pearson_corr_matrices, axis=0)
    kendall_mean_matrix = np.nanmean(list_kendall_corr_matrices, axis=0)
    pearson_mean_matrix = pearson_mean_matrix[5:]
    kendall_mean_matrix = kendall_mean_matrix[5:]

    mask = copy.deepcopy(pearson_mean_matrix)
    for i in range(len(mask)):
        for j in range(len(mask[i])):
            if j<(i+5):
                mask[i][j] = 0

    f, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(pearson_mean_matrix, square=False, annot=True,
                mask=mask,
                xticklabels=[feature for feature, _ in dict_results.items() if feature!='lay_offset'],
                yticklabels=['spread', 'back_offset', 'lay_offset'],
                vmin=-1,
                vmax=1)
    plt.savefig(plot_path+"_pearson")
    plt.close()
    f, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(kendall_mean_matrix, square=False, annot=True,
                mask=mask,
                xticklabels=[feature for feature, _ in dict_results.items() if feature!='lay_offset'],
                yticklabels=['spread', 'back_offset', 'lay_offset'],
                vmin=-1,
                vmax=1)
    plt.savefig(plot_path+"_kendall")
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
                                                                mode='fixed')
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

        f.add_subplot(2, 2, 3)
        plt.plot(env.list_pnl, label="PnL")
        plt.plot(env.list_inventory_stake, label="Stake")
        plt.plot(env.list_inventory_odds, label="Odds")
        plt.legend()

        f.add_subplot(2, 2, 4)
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

    for episode in range(num_episodes):
        obs, _ = env.reset()
        terminated = False
        while not terminated:
            if model=="random":
                action = env.action_space.sample()
            elif model=='fixed_02':
                action = 22
            elif model=='fixed_05':
                action = 55
            elif model=='fixed_08':
                action = 88
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            if debug:
                print(f"Obs: {obs}\nAction: {action}")


        final_pnl.append(env.list_pnl[-1])
        min_pnl.append(np.min(env.list_pnl))
        max_pnl.append(np.max(env.list_pnl))

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
            sharpe_ratio.append(0)
        else:
            sharpe_ratio.append(mean_ret/std_returns)

        if mean_ret==0 or downside_std==0:
            sortino_ratio.append(0)
        else:
            sortino_ratio.append(mean_ret/downside_std)

        mean_inv_stake.append(np.mean(env.list_inventory_stake))

    if plot_results:
        plotting.plot_results_of_single_strategy_test(plot_path=plot_path,
                                                      dict_results={'final_pnl': final_pnl,
                                                                    'mean_return': mean_return,
                                                                    'volatility': volatility_returns,
                                                                    'min_pnl': min_pnl,
                                                                    'max_pnl': max_pnl,
                                                                    'sharpe_ratio': sharpe_ratio,
                                                                    'sortino_ratio': sortino_ratio,
                                                                    'mean_inv_stake': mean_inv_stake})

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
    elif mode=='short':
        tennis_probs = [0.60, 0.62]
        k_range = range(3, 5)
    elif mode=='single':
        tennis_probs = [0.65]
        k_range = range(4,5)
    else:
        raise ValueError("Wrong value for 'mode' parameter")

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
    # # plot_path = "./plots_k_4/fixed_02"

    # # strategy = FixedOffsetStrategy(offset=0.5)
    # # plot_path = "./plots_k_4/fixed_05"

    # # strategy = FixedOffsetStrategy(offset=0.8)
    # # plot_path = "./plots_k_4/fixed_08"

    # strategy = RandomStrategy(range_offset=(0, 1))
    # plot_path = "./plots_k_4/random"

    # num_simulations_per_combination = 1000
    # test_baseline_strategies(plot_path=plot_path,
    #                         strategy=strategy,
    #                         num_simulations_per_combination=num_simulations_per_combination,
    #                         mode='single')






    # ### TEST RL ON SINGLE EPISODE
    # a_s = 0.65
    # b_s = 0.65
    # k = 4
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k,
    #                                                         mode='fixed')
    # #model = DQN.load("model_weights/DQN_4_with_vol_ind")
    # #model = DQN.load("model_weights/DQN_5_with_vol_ind")
    # #model = DQN.load("./model_weights/with_return_reward/DQN_1_return_reward")
    # #model = DQN.load("./model_weights/with_k_4/DQN_1_k_4")
    # #model = DQN("MlpPolicy", env)


    # #model = DQN.load("./model_weights/with_k_4/DQN/DQN_2_k_4")
    # #model = PPO.load("./model_weights/with_k_4/PPO/PPO_2_k_4")
    # #model = A2C.load("./model_weights/with_k_4/A2C/A2C_2_k_4")
    # model = A2C.load("./model_weights/with_k_4/A2C/A2C_1_k_4")


    # for x in range(10):
    #     test_rl_agent_single_episode(model=model, env=env, debug=False, plot_results=True)







    # ### TEST MULTIPLE EPISODES
    # a_s = 0.65
    # b_s = 0.65
    # k = 4
    # env = sportsTradingEnvironment.SportsTradingEnvironment(a_s=a_s,
    #                                                         b_s=b_s,
    #                                                         k=k,
    #                                                         mode='fixed')

    # # # # model = DQN.load("./model_weights/DQN_2")
    # # # # plot_path = "plots_RL_single/DQN_2"

    # # # # model = DQN.load("./model_weights/DQN_3")
    # # # # plot_path = "plots_RL_single/DQN_3"

    # # # # model = DQN.load("./model_weights/DQN_4_with_vol_ind")
    # # # # plot_path = "plots_RL_single/DQN_4_with_vol_ind"

    # # # # model = DQN.load("./model_weights/DQN_5_with_vol_ind")
    # # # # plot_path = "plots_RL_single/DQN_5_with_vol_ind"

    # # # # model = DQN.load("./model_weights/DQN_6_with_vol_ind")
    # # # # plot_path = "plots_RL_single/DQN_6_with_vol_ind"

    # # # model = DQN.load("./model_weights/with_return_reward/DQN_1_return_reward")
    # # # plot_path = "plots_RL_single/DQN_1_return_reward"

    # # # model = DQN.load("./model_weights/with_k_4/DQN_1_k_4")
    # # # plot_path = "plots_k_4/DQN_1_k_4"

    # # # model = DQN.load("./model_weights/with_k_4_2/DQN_1_k_4")
    # # # plot_path = "plots_RL_single/DQN_1_k_4_2"

    # # # model = DQN.load("./model_weights/DQN_with_k_2/DQN_1_k_2")
    # # # plot_path = "plots_RL_single/DQN_1_k_2"

    # # # # model = A2C.load("./model_weights/A2C_1")
    # # # # plot_path = "plots_RL_single/A2C_1"


    # # # model = "fixed_02"
    # # # plot_path = f"plots_k_4/{model}"

    # # # model = "fixed_05"
    # # # plot_path = f"plots_k_4/{model}"

    # # # model = "fixed_08"
    # # # plot_path = f"plots_k_4/{model}"

    # # # model = "random"
    # # # plot_path = f"plots_k_4/{model}"

    # # # model = PPO.load("./model_weights/with_k_4/PPO/PPO_1_k_4")
    # # # model_name = "PPO_k_4"




    # # # model_name = "PPO_2_k_4"
    # # # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # # # model_name = "DQN_1_k_4"
    # # # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # # # model_name = "DQN_2_k_4"
    # # # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # # # model_name = "A2C_1_k_4"
    # # # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    # # model_name = "A2C_2_k_4"
    # # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

    # model_name = "PPO_3_k_4"
    # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")



    # plot_path = f"plots_k_4/{model_name}"

    # test_rl_agent_multiple_episods(num_episodes=1000,
    #                                 model=model,
    #                                 env=env,
    #                                 plot_results=True,
    #                                 plot_path=plot_path,
    #                                 debug=False)











#     # # TEST ALL POSSIBLE ENV. COMBINATIONS

#     # ## WITHOUT VOLATILITY INDICATOR
#     # model = DQN.load("./model_weights/DQN_2")
#     # plot_path = "./plots_RL/DQN_2"

#     # model = DQN.load("./model_weights/DQN_3")
#     # print(model.policy)
#     # plot_path = "./plots_RL/DQN_3"

#     # model = A2C.load("./model_weights/A2C_1")
#     # # print(model.policy)
#     # plot_path = "./plots_RL/A2C_1"


#     ### WITH VOL. INDICATOR
#     # model_name = "DQN_4_with_vol_ind"
#     # model_name = "DQN_5_with_vol_ind"
#    # model_name = "DQN_6_with_vol_ind"
#     #model_name = "with_k_4/DQN_1_k_4"
#     #model_name = "with_k_4_2/DQN_1_k_4"
#     #model_name = "with_k_2/DQN_1_k_2"
#     #model_name = "random_env/DQN_1_random_env"
#     #model_name = "random_env/DQN_2_random_env"
#     #model_name = "random_env/DQN_3_random_env_3500000_steps"
#     # model_name = "random_env/DQN/DQN_4_random_env"

#     # model = DQN.load(f"./model_weights/{model_name}")



#     # model_name = "PPO_1_with_vol_ind"
#     # model = PPO.load(f"./model_weights/{model_name}")

#     # model_name = "random_env/A2C/A2C_1_random_env"
#     # model = A2C.load(f"./model_weights/{model_name}")


#     # model_name = "fixed_02"
#     # model = model_name

#     # model_name = "fixed_05"
#     # model = model_name

#     # model_name = "fixed_08"
#     # model = model_name

#     # model_name = "random"
#     # model = model_name

#     # model_name = "PPO_1_k_4"
#     # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

#     # model_name = "A2C_1_k_4"
#     # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

#     # model_name = "DQN_1_k_4"
#     # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

#     # model_name = "A2C_2_k_4"
#     # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")

#     # model_name = "DQN_2_k_4"
#     # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

#     # model_name = "PPO_2_k_4"
#     # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

#     model_name = "PPO_3_k_4"
#     model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")



#     plot_path = f"./plots_all_comb/{model_name}"
#     test_rl_agent_all_combinations(model=model,
#                                    num_simulations_per_combination=100,
#                                    plot_path=plot_path,
#                                    #mode="short",
#                                    mode="long"
#                                    )






    # # ### TESTING CORRELATIONS BETWEEN STATE VARIABLES AND ACTIONS (SPREADS)

    # # #model_name = "random_env/DQN_2_random_env"
    # # model_name = "random_env/DQN_3_random_env_3500000_steps"

    # # model = DQN.load(f"./model_weights/{model_name}")




    # # model_name = "PPO_2_k_4"
    # # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # # model_name = "PPO_3_k_4"
    # # model = PPO.load(f"./model_weights/with_k_4/PPO/{model_name}")

    # # model_name = "DQN_2_k_4"
    # # model = DQN.load(f"./model_weights/with_k_4/DQN/{model_name}")

    # model_name = "A2C_2_k_4"
    # model = A2C.load(f"./model_weights/with_k_4/A2C/{model_name}")


    # plot_path = f"corr_matr_actions/{model_name}"

    # test_correlations_state_vars_and_actions(model=model, mode='long',
    #                                          num_simul_per_combin=10,
    #                                          plot_path=plot_path)









    ### PLOT FINAL PNL DISTRIBUTIONS OF MODELS IN SAME GRAPH
    # strategy_names = [ "DQN_2", "DQN_3", "A2C_1"]
    # results_path = "./plots_RL"

    # strategy_names = ["fixed_02", "random", "fixed_05", "fixed_08"]
    # results_path = "./plots"

    strategy_names = ["A2C_2_k_4",
                      "DQN_2_k_4",
                      "PPO_2_k_4",
                      ]


    # results_path = "./plots_all_comb"
    results_path = "./plots_k_4"

    metrics = [
        "final_pnl",
        #"volatility",
        # "mean_return",
        # "min_pnl",
        # "max_pnl",
        # #"sharpe_ratio",
        # "sortino_ratio",
        # "mean_inv_stake"
        ]

    plotting.plot_results_of_all_strategies_test(results_path=results_path,
                                                 strategies_names_list=strategy_names,
                                                 metrics_list=metrics)




if __name__=="__main__":
    main()