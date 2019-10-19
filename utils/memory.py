import random
from collections import deque
import numpy as np


class SimpleMemoryBuffer: # for continous env memory, a simple random buffer
	def __init__(self, size, args):
		self.buffer = deque(maxlen=size)
		self.maxSize = size
		self.len = 0
		self.args = args

	def sample(self, count):
		batch = []
		count = min(count, self.len)
		if self.args.memory_sample == 'FIFO':
			batch = [self.buffer[i] for i in range(count)]
		elif self.args.memory_sample == 'random':
			batch = random.sample(self.buffer, count)

		s_arr = np.float32([arr[0] for arr in batch])	# state
		a_arr = np.float32([arr[1] for arr in batch])	# action
		r_arr = np.float32([arr[2] for arr in batch])	# reward
		s1_arr = np.float32([arr[3] for arr in batch])	# next_state
		done = np.float32([arr[4] for arr in batch])	# if done

		return s_arr, a_arr, r_arr, s1_arr, done

	def len(self):
		return self.len

	def add(self, s, a, r, s1, done):
		# if_done = [1. if done[i] else 0. for i in range(len(done))]
		if_done = 1 if done else 0
		transition = (s, a, r, s1, if_done)
		self.len += 1
		if self.len > self.maxSize:
			self.len = self.maxSize

		self.buffer.append(transition)
