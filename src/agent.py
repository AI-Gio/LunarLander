import random
from dataclasses import dataclass
import numpy as np
import copy
from collections import namedtuple

class Agent:
    """
    Desc agent
    """
    def __init__(self):
        # policy_network = approximator()
        # target_network = approximator()
        pass

    def train(self, batch_size):

        batch = memory.sample(batch_size)
        for sample in batch:
            next_state = sample.next_state
            best_action = Policy.select_action(next_state)
            # q-values = target_model(next_state)
            # Q[next_state, best_action] = q-value van best_action
            # target = next_state.reward + self.discount * Q[next_state, best_action]


    def do_action(self):
        """
            Lets the agent do an action within the grid. Decides the best direction with
            the policy, then uses that direction to make a step within the maze. Updates on
            each step if the agent is on the end state or not.
        """



