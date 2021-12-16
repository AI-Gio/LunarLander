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
    def __init__(self):
        self.policy_network = FunctionApprox()
        self.target_network = FunctionApprox()
        self.pol = Epsilon_policy([0,1,2,3])
        self.memory = Memory(1000)


    def train(self, batch_size):
        """
        This train function has to calculate y first and then give params
        to function_approx.train()
        :param batch_size:
        :return:
        """

        batch = self.memory.sample(batch_size)
        for sample in batch:
            next_state = sample.next_state


            # q-values = target_model(next_state)
            # Q[next_state, best_action] = q-value van best_action
            # target = next_state.reward + self.discount * Q[next_state, best_action]

    def choose_action(self, state, epsilon):
        """
            Lets the agent do an action within the sim
        """
        best_action_index = self.pol.select_action(self.policy_network, epsilon, state)
        return best_action_index





