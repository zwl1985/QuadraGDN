import random
import collections

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done, graph):
        self.buffer.append((state, action, reward, next_state, done, graph))

    def sample(self, batch_size):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done, graph = zip(*transitions)
        return state, action, reward, next_state, done, graph

    def size(self):
        return len(self.buffer)

    def __len__(self):
        return len(self.buffer)