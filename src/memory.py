import random
from collections import deque


class Memory:

    def __init__(self):
        self.size = int
        self.mem = deque([])
        # sample size (batchsize) 64 of 32
        # memory van zo een 5000
        # bij record: als de memory vol zit, oudste eruit
        # hoge learning rate bij hoge batchsize

    def sample(self, batch_size: int) -> list:
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
