import random
from collections import deque


class Memory:

    def __init__(self):
        self.size = int
        self.mem = deque([])

    def sample(self, batch_size: int):
        """
        Picks random transitions out of mem list
        :return: list of samples of transitions
        """
        samples = []
        for s in range(batch_size):
            samples.append(random.choice(self.mem))
        return samples

    def record(self, new_mem):
        """
        Adds a new memory to mem list
        :param new_mem: new memory
        """
        self.mem.append(new_mem)
