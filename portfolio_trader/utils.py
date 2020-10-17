import os
import itertools
import re
import time
from io import StringIO
import requests

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def get_scaler(env):
    # return scikit-learn scaler object to scale the states
    # Note: you could also populate the replay buffer here
    states = []
    for _ in range(env.n_step):
        # --- sample a random value from the action space
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.perform_action(action)
        states.append(state)
        if done:
            break
    # --- create a StandardScaler object and
    #     fit it to the states encountered
    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def run_one_episode(agent, env, scaler, mode):
    state = env.reset_state()
    state = scaler.transform([state])
    done = False

    episode_values = list()  # save the complete vector of episode values
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, info = env.perform_action(action)
        episode_values.append(info['current_value'])
        next_state = scaler.transform([next_state])
        if mode == 'train':
            agent.train_model(state, action, reward, next_state, done)
        state = next_state

    return episode_values


def calculate_sharpe_ratio(prices):
    prices_monthly = prices[::21]
    ret_monthly = np.diff(prices_monthly) / prices_monthly[1:]
    return np.mean(ret_monthly) / np.std(ret_monthly) * np.sqrt(12)
