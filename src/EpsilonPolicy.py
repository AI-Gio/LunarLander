import random
from function_approx import FunctionApprox


class Epsilon_policy:
    """The epsilon soft policy of the agent."""
    def __init__(self, actions: list = [0, 1, 2, 3]):
        self.actions = actions

    def select_action(self, network: FunctionApprox, epsilon: float, state: list) -> int:
        """Select an action through the given neural network"""

        if random.random() > epsilon:
            action_values = network.q_values(state)
            best_action_value = max(action_values)
            return action_values.index(best_action_value)
        return random.choice(self.actions)
