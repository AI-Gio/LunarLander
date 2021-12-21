import numpy as np
from memory import Memory
from function_approx import FunctionApprox
from EpsilonPolicy import Epsilon_policy
import time

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

        batch = self.memory.sample(self.batch_size)  # lijst aan transitions
        states = [transition.state for transition in batch]
        next_states = [transition.next_state for transition in batch]
        rewards = [transition.reward for transition in batch]
        actions = [transition.action for transition in batch]

        done_indexes = [i for i, transition in enumerate(batch) if transition.done]
        q_val_next_state = np.array(self.policy_network.q_values(next_states))
        for i in done_indexes:
            q_val_next_state[i] = [0, 0, 0, 0]

        argmax_indeces = [np.argmax(q_val) for q_val in q_val_next_state]

        q2 = np.array(self.target_network.q_values(next_states))
        targets = rewards + self.discount * np.array([q_val[argmax_indeces[i]] for i, q_val in enumerate(q2)])
        y = self.policy_network.q_values(states)
        for i, q_val in enumerate(y):
            q_val[actions[i]] = targets[i]
        Y = np.array(y)
        X = np.array(states)

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





