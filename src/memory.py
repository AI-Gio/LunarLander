import random
from collections import deque
import transition


class Memory:
    """
    The memory class, containing a series of transitions and handles the memory of the simulation.
    Can add and remove transitions and prioritize transitions. Records the transitions and gives
    a sample of them.
    """
    def __init__(self, size: int):
        self.size = size
        self.mem = deque([])
        self.bad_memory = deque([])
        self.good_memory = deque([])

    def sample(self, batch_size: int) -> list:
        """
        Picks random transitions out of mem list
        :return: list of samples of transitions
        """
        batch = []
        if len(self.mem) < batch_size:  # if there are less memories than the requested batch size
            batch_size = len(self.mem)

        high_batch_size = batch_size // 20  # batch size of the good transitions (~5% of batch size)
        low_batch_size = batch_size // 20  # batch size of the bad transitions (~5% of batch size)

        if len(self.good_memory) < high_batch_size:  # if there are less memories than the requested batch size
            high_batch_size = len(self.good_memory)
        if len(self.bad_memory) < low_batch_size:
            low_batch_size = len(self.bad_memory)

        batch_size -= low_batch_size + high_batch_size  # lower normal batch size by other batch sizes

        batch += random.sample(self.bad_memory, low_batch_size)
        batch += random.sample(self.good_memory, high_batch_size)
        batch += random.sample(self.mem, batch_size)
        return batch

    def record(self, new_mem: transition.Transition):
        """
        Adds a new memory to mem list
        :param new_mem: new memory
        """
        if new_mem.reward >= 20:
            if len(self.good_memory) + 1 > self.size//10:
                self.good_memory.popleft()
            self.good_memory.append(new_mem)
        elif new_mem.reward <= -20:
            if len(self.bad_memory) + 1 > self.size//10:
                self.bad_memory.popleft()
            self.bad_memory.append(new_mem)
        else:
            if len(self.mem) + 1 > self.size:
                self.mem.popleft()
            self.mem.append(new_mem)
