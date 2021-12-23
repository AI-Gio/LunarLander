import random
import numpy as np
from function_approx import FunctionApprox


class Epsilon_policy:
    """The epsilon greedy policy of the agent. Gives an actions based
    on the best q-values of the network. Has 1-epsilon chance of giving a random action"""
    def __init__(self):
        self.actions = [0, 1, 2, 3]

    def select_action(self, network: FunctionApprox, epsilon: float, state: list) -> int:
        """Select an action through the given neural network"""

        if random.random() > epsilon:
            Q_values = network.q_values(state)[0]
            best_q_value = max(Q_values)
            return np.where(Q_values == best_q_value)[0][0]  # [0][0] to index the first best item
        return random.choice(self.actions)
