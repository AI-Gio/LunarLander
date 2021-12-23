import random
from function_approx import FunctionApprox


class Epsilon_policy:
    """The epsilon soft policy of the agent."""
    def __init__(self):
        self.actions = [0, 1, 2, 3]

    def select_action(self, network: FunctionApprox, epsilon: float, state: list) -> int:
        """Select an action through the given neural network"""

        if random.random() > epsilon:
            Q_values = list(network.q_values(state))
            best_q_value = max(Q_values)
            return Q_values.index(best_q_value)
        return random.choice(self.actions)
