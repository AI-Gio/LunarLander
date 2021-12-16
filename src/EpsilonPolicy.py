import random

class Epsilon_policy:
    """The epsilon soft policy of the agent."""
    def __init__(self):
        self.epsilon = 0.1
        self.actions = []
        pass

    def select_action_Q(self, current_state: State) -> tuple:
        """Select an action based on the Q values and the epsilon"""

        state_location = (current_state.x, current_state.y)
        if random.random() > self.epsilon:
            action_values = {direction: self.Q[(state_location, direction)] for direction in self.directions}
            return max(action_values, key=action_values.get)
        return random.choice(self.actions)

    def select_action_double_Q(self, current_state: State, Q1, Q2):
        """Select an action based on two Q values and the epsilon"""

        state_location = (current_state.x, current_state.y)
        if random.random() > self.epsilon:
            action_values = {direction: Q1[(state_location, direction)] + Q2[(state_location, direction)] for direction in self.directions}
            return max(action_values, key=action_values.get)
        return random.choice(self.directions)

    def select_action_pi(self, current_state: State) -> tuple:
        """Select an action based on the pi chances"""
        state_location = (current_state.x, current_state.y)
        return np.random.choice(self.directions, p=self.pi[state_location])[0]

    def print_pi(self):
        """
            Prints out the entire grid and its pi chances, with each coordinate showing all the pi chance of each
            direction.
        """

        print("\033[93mPi Order: ↑, ↓, ←, → \033[35m")
        print_str = "\033[93mPi Policy chances (epsilon soft policy): \033[35m\n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                print_str += "[\033[93m{}\033[35m: {:<28}] ".format((x, y), str(self.pi[(x, y)]))
            print_str += "\n"
        print(print_str)

    def print_Q(self, Q):
        """
            Prints out the entire grid and its Q values, with each coordinate showing all the Q values of each
            direction.
        Parameters
        ----------
            Q (dictionary): The dict containing all the Q values
        """
        print("\033[93mDirection corresponding to each Q value: ↑, ↓, ←, →  \033[35m")
        print_str = "\033[93mQ values: \033[35m\n"
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                print_str += "\033[93m{}\033[35m: ".format(str((x, y)))
                for direction in self.directions:
                    print_str += "{:<4}, ".format(round(Q[((x, y), direction)], 1))
            print_str += "\n"
        print(print_str)

