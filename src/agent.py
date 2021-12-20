import numpy as np
from memory import Memory
from function_approx import FunctionApprox
from EpsilonPolicy import Epsilon_policy


class Agent:
    """
    Agent does cool stuff
    """
    def __init__(self, discount, epsilon, tau, batch_size, epochs, memory_size, learning_rate):
        self.discount = discount
        self.epsilon = epsilon
        self.tau = tau
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.policy_network = FunctionApprox()
        self.target_network = FunctionApprox()
        self.pol = Epsilon_policy([0, 1, 2, 3])
        self.memory = Memory(memory_size)

    def train(self):
        """
        This train function has to calculate y first and then give params
        to function_approx.train()
        :return:
        """

        batch = self.memory.sample(self.batch_size) # lijst aan transitions
        X = []
        Y = []
        for transition in batch:
            state = x = transition.state
            next_state = transition.next_state

            if transition.done:
                q_val_next_state = [0, 0, 0, 0]
            else:
                q_val_next_state = self.policy_network.q_values(next_state)

            argmax_index = np.argmax(q_val_next_state)
            target = transition.reward + self.discount * self.target_network.q_values(next_state)[argmax_index]

            y = self.policy_network.q_values(state)  # this is q_values current state
            y[transition.action] = target

            X.append(np.array(x))
            Y.append(np.array(y))

        X = np.array(X)
        Y = np.array(Y)

        self.policy_network.train(X, Y, self.batch_size, self.epochs, True)  # TODO: wrong input fixed

    def update_t_network(self):
        """
        Update target network
        :return:
        """
        w_policy = self.policy_network.get_weights()
        w_target = self.target_network.get_weights()

        self.target_network.set_weights(self.tau * w_policy + (1 - self.tau) * w_target)

    def choose_action(self, state):
        """
            Lets the agent do an action within the sim
        """
        best_action_index = self.pol.select_action(self.policy_network, self.epsilon, state)
        return best_action_index





