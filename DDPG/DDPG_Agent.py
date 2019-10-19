import sys
import os
sys.path.append('..')

import torch 
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
from torch import optim

import utils
from utils import memory
from utils import noise
from utils.update import soft_update, hard_update

import gym
import numpy as np
from DDPG import DDPG_Model


class DDPGAgent:
	def __init__(self, obs_space, action_space, ram, args):
		self.args = args
		self.obs_dim = obs_space.shape[0]		# observation dimension
		self.act_dim = action_space.shape[0]	# action dimension

		self.action_low = action_space.low 		# the lower bound of action space
		self.action_high = action_space.high 	# the higher bound of action space

		self.ram = ram 	# memory
		self.steps = 0	# steps counter

		self.gamma = args.gamma
		self.batch_size = args.batch_size
		self.start_training = args.start_training
		self.tau = args.tau
		self.critic_lr = args.critic_lr
		self.actor_lr = args.actor_lr
		self.initial_e = args.initial_e
		self.end_e = args.end_e

		self.noise = noise.RandomActionNoise(self.act_dim)
		self.e = self.initial_e

		# initial the learning network and target network
		target_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		learning_net = DDPG_Model.DDPG(self.obs_dim, self.act_dim).cuda()
		hard_update(target_net, learning_net) # make the parameters of two nets be sam

		self.AC = learning_net
		self.AC_t = target_net

		self.actor = self.AC.actor 		# the learning actor
		self.critic = self.AC.critic 	# the learning critic
		self.actor_t = self.AC_t.actor 	# the target actor
		self.critic_t = self.AC_t.critic# the target critic

		self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
		self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

		self.loss_f = nn.MSELoss()

	def save_models(self, name):
		save_path = self.args.save_path + '/params/'
		if not os.path.exists(save_path):
			os.mkdir(save_path)
		torch.save(self.AC_t.state_dict(), save_path + name + '.pt')
		# only save the target net's parameters

	def load_models(self, name):
		load_path = self.args.load_path + '/params/' + name
		self.AC_t.load_state_dict(torch.load(load_path))
		utils.hard_update(self.AC, self.AC_t)
		print("parameters have been loaded")

	def get_exploitation_action(self, state):
		state = Variable(torch.tensor(state)).cuda()
		action = self.action_t(state).detach().cpu().numpy()
		return np.squeeze(action)

	def get_exploration_action(self, state):
		self.steps += 1

		if self.e > self.end_e and self.steps > self.start_training:
			self.e -= (self.initial_e - self.end_e) / self.args.e_decay

		state = Variable(torch.tensor(state)).cuda()
		action = self.actor(state).detach().cpu().numpy()
		action = np.squeeze(action)

		# add random noise
		# noise = self.noise.sample()
		noise = np.random.randn(self.act_dim)
		action_noise = (1 - self.e) * action + self.e * noise
		action_noise = np.clip(action_noise, self.action_low, self.action_high)
		return action_noise

	def optimize(self):
		s1, a1, r1, s2, done = self.ram.sample(self.batch_size)

		s1 = Variable(torch.tensor(s1)).cuda()
		a1 = Variable(torch.tensor(a1)).cuda()
		r1 = Variable(torch.tensor(r1)).cuda()
		s2 = Variable(torch.tensor(s2)).cuda()
		done = Variable(torch.tensor(done)).cuda()

		# optimize critic 
		a2 = self.actor_t(s2).detach()
		r_predict = torch.squeeze(self.critic_t(s2, a2).detach())
		r_predict = self.gamma * (torch.ones(self.batch_size).cuda() - done) * r_predict

		y_j = r1 + r_predict
		r_ = torch.squeeze(self.critic(s1, a1))

		self.critic_optimizer.zero_grad()
		critic_loss = self.loss_f(y_j, r_)
		critic_loss.backward()
		self.critic_optimizer.step()

		# optimize actor

		pred_a1 = self.actor(s1)
		q_p = self.critic(s1, pred_a1)
		actor_loss = torch.mean(-q_p)
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# update net
		soft_update(self.AC_t, self.AC, self.tau)

		return actor_loss.detach().cpu().numpy(), critic_loss.detach().cpu().numpy()
