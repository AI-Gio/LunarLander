import numpy as np
from memory import Memory
from function_approx import FunctionApprox
from EpsilonPolicy import Epsilon_policy
# todo: better var names
# todo: descriptions
# todo: maybe make train more efficient
# todo: add comments


class Agent:
    """

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
        self.pol = Epsilon_policy()
        self.memory = Memory(memory_size)

    def train_efficient(self):
        """
        This train function has to calculate y first and then give params
        to function_approx.train()
        :return:
        """

        batch = self.memory.sample(self.batch_size)  # list of transitions

        states = [transition.state for transition in batch]
        next_states = [transition.next_state for transition in batch]
        rewards = [transition.reward for transition in batch]
        actions = [transition.action for transition in batch]
        done_indexes = [i for i, transition in enumerate(batch) if transition.done]

        # get q_values of all next states (0 if end state)
        q_val_policy = self.policy_network.q_values(next_states)
        for i in done_indexes:
            q_val_policy[i] = [0, 0, 0, 0]

        # Calculate target
        argmax_indices = [np.argmax(q_val) for q_val in q_val_policy]
        q_val_target = self.target_network.q_values(next_states)
        targets = rewards + self.discount * np.array([q_val[argmax_indices[i]] for i, q_val in enumerate(q_val_target)])

        # calculate current state q_values and replace action q_value with target value
        Y = self.policy_network.q_values(states)
        for i, q_val in enumerate(Y):
            q_val[actions[i]] = targets[i]
        X = np.array(states)

        self.policy_network.train(X, Y, self.batch_size, self.epochs, False)

    def train(self):
        """
        This train function has to calculate y first and then give params
        to function_approx.train()
        :return:
        """
        batch = self.memory.sample(self.batch_size)  # list of transitions
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

        self.policy_network.train(X, Y, self.batch_size, self.epochs, False)  # TODO: wrong input fixed

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
