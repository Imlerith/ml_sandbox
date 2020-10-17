from functools import reduce
import pickle
from datetime import datetime

import matplotlib.pyplot as plt

from portfolio_trader import *

# ----- Set parameters
tickers = ["AAPL", "XOM", "NVDA", "MSFT", "BAC"]
start_date = "2010/01/01"
end_date = "2019/12/31"
models_folder = 'linear_trader_models'
rewards_folder = 'linear_trader_rewards'
num_episodes = 1000
batch_size = 32
initial_investment = 20000
mode = "train"

# ----- Create directories, load data
create_directory(models_folder)
create_directory(rewards_folder)

data_list = list()
for ticker in tickers:
    ticker_df = YahooFinanceHistory(ticker, date_from=start_date, date_to=end_date, timeout=2).get_quote()
    data_list.append(ticker_df.loc[:, ["Date", "Close"]])

data_df = reduce(lambda x, y: pd.merge(x, y, on="Date", how="inner"), data_list)
data_df.drop(columns=["Date"], inplace=True)
data_df.columns = tickers
n_timesteps, n_stocks = data_df.shape
data = data_df.values

# ----- Split into train and test
n_samples_train = (n_timesteps // 3) * 2
train_data = data[:n_samples_train]
test_data = data[n_samples_train:]


# ----- Create instances of the environment and the agent in train/test modes
if mode == 'train':
    env = PortfolioEnvironment(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = RLAgent(state_size, action_size)
    scaler = get_scaler(env)
else:
    # --- create the environment with test data and set the exploration rate
    env = PortfolioEnvironment(test_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)
    agent = RLAgent(state_size, action_size)
    agent.epsilon = 0.01

    # --- load trained weights
    agent.load_model(f'{models_folder}/linear.npz')

    # --- load the scaler
    with open(f'{models_folder}/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)


# ----- Play the episodes num_episodes times and store portfolio values
portfolio_values = list()
for e in range(num_episodes):
    t0 = datetime.now()
    val = run_one_episode(agent, env, scaler, mode)  # portfolio value at the end of the episode
    dt = datetime.now() - t0
    print(f"episode: {e + 1}/{num_episodes}, episode end value: {val[-1]:.2f}, duration: {dt}")
    portfolio_values.append(val)  # save episode end portfolio value

# ----- Save the trained weights
if mode == 'train':
    # --- save the trained model
    agent.save_model(f'{models_folder}/linear.npz')

    # --- save the scaler
    with open(f'{models_folder}/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # --- plot losses
    plt.plot(agent.model.losses)
    plt.show()

# ----- Save portfolio values for each episode
np.save(f'{rewards_folder}/{mode}.npy', portfolio_values)


# ----- Calculate the Sharpe ratios
sharpe_ratios = list()
for prices in portfolio_values:
    sharpe_ratio = calculate_sharpe_ratio(prices)
    sharpe_ratios.append(sharpe_ratio)

