import random
from dataclasses import dataclass
import numpy as np
import copy
from collections import namedtuple
from memory import Memory
from function_approx import FunctionApprox
from EpsilonPolicy import Epsilon_policy


class Agent:
    """
    Desc agent
    """
    def __init__(self, discount):
        self.discount = discount
        self.policy_network = FunctionApprox()
        self.target_network = FunctionApprox()
        self.pol = Epsilon_policy([0,1,2,3])
        self.memory = Memory(1000)


    def train(self, batch_size, epochs):
        """
        This train function has to calculate y first and then give params
        to function_approx.train()
        :param batch_size:
        :return:
        """

        batch = self.memory.sample(batch_size) # lijst aan transitions
        X = []
        Y = []
        for transition in batch:
            state = x = transition.state
            next_state = transition.next_state

            q_val_next_state = self.policy_network.q_values(next_state)

            argmax_index = np.argmax(q_val_next_state)
            target = transition.reward + self.discount * self.target_network.q_values(next_state)[argmax_index]

            y = self.policy_network.q_values(state) # this is q_values current state
            y[transition.action] = target

            X.append(x)
            Y.append(y)

        self.policy_network.train(X, Y, batch_size, epochs, True)

    def update_t_network(self, tau: float):
        """
        Update target network
        :return:
        """
        w_policy = self.policy_network.get_weights()
        w_target = self.target_network.get_weights()

        self.target_network.set_weights(tau * w_policy + (1 - tau) * w_target)












            # q-values = target_model(next_state)
            # Q[next_state, best_action] = q-value van best_action
            # target = next_state.reward + self.discount * Q[next_state, best_action]

    def choose_action(self, state, epsilon):
        """
            Lets the agent do an action within the sim
        """
        best_action_index = self.pol.select_action(self.policy_network, epsilon, state)
        return best_action_index





