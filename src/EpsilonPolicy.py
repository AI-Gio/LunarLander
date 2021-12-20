import random
from function_approx import FunctionApprox
random.seed(7)

class Epsilon_policy:
    """The epsilon soft policy of the agent."""
    def __init__(self, actions: list = [0, 1, 2, 3]):
        self.actions = actions

    def select_action(self, network: FunctionApprox, epsilon: float, state: list) -> int:
        """Select an action through the given neural network"""

        if random.random() > epsilon:
            Q_values = list(network.q_values(state))
            best_q_value = max(Q_values)
            return Q_values.index(best_q_value)
        return random.choice(self.actions)
