import numpy as np
from utils.memory import SimpleMemoryBuffer
import gym
import torch


def train(env, args):
	if args.memory == 'simple':
		ram = SimpleMemoryBuffer(args.memory_max_size)

	if args.agent == 'DDPG':
		from DDPG import DDPG_Agent
		agent = DDPG_Agent.DDPGAgent(env.observation_space, env.action_space, ram, args)

	steps_done = 0
	
	for epoch in range(args.max_epoch):
		observation = env.reset()
		total_reward = 0
		epoch_actor_loss = []
		epoch_critic_loss = []

		for r in range(10000):
			actor_loss = 0
			critic_loss = 0
			state = np.float32(observation)

			if steps_done < args.start_training:
				action = env.action_space.sample()
			else:
				action = agent.get_exploration_action(state)

			new_observation, reward, done, _ = env.step(action)

			steps_done += 1
			total_reward += reward

			ram.add(observation, action, reward, new_observation, done)

			if args.render:
				env.render()

			observation = new_observation

			# begin to optimize
			if steps_done > args.start_training:
				actor_loss, critic_loss = agent.optimize()

				epoch_actor_loss.append(actor_loss)
				epoch_critic_loss.append(critic_loss)

			if done:
				break
		print("Total_reward:{}, Mean actor loss:{}, Mean critic loss:{}, total step:{}， epoch:{}".format(
			total_reward, np.mean(epoch_actor_loss), np.mean(epoch_critic_loss), steps_done, epoch))

		if (epoch + 1) % 10 == 0:
			agent.save_models('DDPG')

	env.close()