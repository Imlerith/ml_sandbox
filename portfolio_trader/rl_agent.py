from portfolio_trader import *


class RLAgent(object):
    """
    The agent will learn from its past experience and take
    actions such that the future rewards are maximized
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size  # inputs
        self.action_size = action_size  # outputs
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = LinearModel(state_size, action_size)

    def choose_action(self, state):
        """
        Takes in a state and uses epsilon-greedy to
        choose an action based on that state
        :param state:
        :return: index of action to take
        """
        if np.random.rand() <= self.epsilon:
            # --- perform a random action
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        # --- otherwise, perform a greedy action on all the Q values for the input state
        return np.argmax(act_values[0])

    def train_model(self, state, action, reward, next_state, done):
        """
        Train the model: determine the parameters of the value function
        approximation using SGD in an online fashion (using just one training sample)
        :param state: current state
        :param action: action to take
        :param reward: reward (P&L)
        :param next_state: next state (observation)
        :param done: flag to denote end of sample
        :return: nothing, perform one step of SGD
        """
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)  # 2-dim: #samples x #outputs
        # --- the target corresponds only to the action we took;
        #     the target should equal the prediction for other outputs
        target_full[0, action] = target

        # --- one step of SGD
        self.model.sgd(state, target_full)

        # --- decrease the exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load_model(self, name):
        self.model.load_weights(name)

    def save_model(self, name):
        self.model.save_weights(name)
