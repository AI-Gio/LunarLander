import random
from collections import deque
import transition

class Memory:

    def __init__(self, size: int):
        self.size = size
        self.mem = deque([])

    def sample(self, batch_size: int) -> list:
        """
        Picks random transitions out of mem list
        :return: list of samples of transitions
        """
        samples = []
        for s in range(batch_size):
            samples.append(random.choice(self.mem))
        return samples

    def record(self, new_mem: transition.Transition):
        """
        Adds a new memory to mem list
        :param new_mem: new memory
        """
        if len(self.mem) + 1 > self.size:
            self.mem.popleft()
        self.mem.append(new_mem)
