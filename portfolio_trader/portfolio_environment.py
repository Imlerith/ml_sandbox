from portfolio_trader import *


class PortfolioEnvironment:

    """
    Portfolio trading environment for a stock portfolio of size n.

    State: vector of size n_stock * 2 + 1
    - # shares of stock 1 owned
    - # shares of stock 2 owned
    - ...
    - # shares of stock n owned
    - price of stock 1 (daily close price)
    - price of stock 2
    - ...
    - price of stock n
    - cash owned (used to purchase more stocks)

    Action: categorical variable with 3^n possibilities
    - for each stock, can do the following:
    - 0 = sell
    - 1 = hold
    - 2 = buy
    """

    def __init__(self, stock_price_data, init_investment=20000):
        self.stock_price_data = stock_price_data
        self.n_step, self.n_stock = self.stock_price_data.shape
        self.init_investment = init_investment
        self.current_step = None
        self.stock_owned = None
        self.curr_stock_price = None
        self.cash_amount = None
        self.action_space = np.arange(3 ** self.n_stock)
        self.action_list = list(map(list, itertools.product([0, 1, 2], repeat=self.n_stock)))
        self.state_dim = self.n_stock * 2 + 1  # size of the state
        self.reset_state()

    def reset_state(self):
        self.current_step = 1  # point to the first day of stock prices in the dataset
        self.stock_owned = np.zeros(self.n_stock)  # how many shares of stock we own
        self.curr_stock_price = self.stock_price_data[self.current_step]  # current price of each stock
        self.cash_amount = self.init_investment
        return self._get_state()  # return the state vector

    def perform_action(self, action):
        """
        Performs the given action in the environment
        and returns the next state as well as the reward
        :param action: an action from the list of possible actions
        :return: next state, reward, completion flag, current value of the portfolio
        """
        assert action in self.action_space

        # --- get the current portfolio value
        prev_val = self._get_portfolio_value()

        # --- move to to the next day: update the stock prices
        self.current_step += 1
        self.curr_stock_price = self.stock_price_data[self.current_step]

        # --- perform the trade
        self._perform_trade(action)

        # --- get the new portfolio value (after taking the action)
        current_value = self._get_portfolio_value()

        # --- reward is the change in porfolio value (P&L)
        reward = current_value - prev_val

        # --- done if we have reached the end of the time series
        done = self.current_step == self.n_step - 1

        # --- store the current value of the portfolio
        info = {'current_value': current_value}

        return self._get_state(), reward, done, info

    def _get_state(self):
        # --- the state is a vector with 3 parts:
        #     - sub-vector of share numbers
        #     - sub-vector of share prices
        #     - sub-vector of previous share prices
        #     - scalar of cash amount
        state = np.empty(self.state_dim)
        state[:self.n_stock] = self.stock_owned
        state[self.n_stock:2 * self.n_stock] = self.curr_stock_price
        state[-1] = self.cash_amount
        return state

    def _get_portfolio_value(self):
        return self.stock_owned.dot(self.curr_stock_price) + self.cash_amount

    def _perform_trade(self, action):
        """
        Performs a trade given an (index of) the action from the action list
        :param action: action index in the action list
        :return: nothing,
        """

        stock_actions = self.action_list[action]

        # --- determine which stocks to buy or sell
        sell_index = []  # index of the stocks to sell
        buy_index = []  # index of the stocks to buy
        for i, a in enumerate(stock_actions):
            if a == 0:
                sell_index.append(i)
            elif a == 2:
                buy_index.append(i)

        # --- sell any stocks we want to sell,
        #     then buy any stocks we want to buy
        #     (for simplicity, sell ALL shares when selling)
        if sell_index:
            for i in sell_index:
                self.cash_amount += self.curr_stock_price[i] * self.stock_owned[i]
                self.stock_owned[i] = 0
        if buy_index:
            # --- greedily loop through each stock to buy,
            #     and buy one share at a time until running out of cash
            can_buy = True
            while can_buy:
                for i in buy_index:
                    if self.cash_amount > self.curr_stock_price[i]:
                        self.stock_owned[i] += 1  # buy one share
                        self.cash_amount -= self.curr_stock_price[i]
                    else:
                        can_buy = False
