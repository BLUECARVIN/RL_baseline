import numpy as np
import gym
import torch


def test(env, args):
	if args.agent == 'DDPG':
		from DDPG import DDPG_Agent
		agent = DDPG_Agent.DDPGAgent(env.observation_space,
			env.action_space, None, args)
		agent.load_state_dict(torch.load(args.save_path+"/params/"+args.agent+'.pt'))

	steps_done = 0

	for epoch in range(args.max_epoch):
		observation = env.reset()
		total_reward = 0

		for r in range(10000):
			state = np.float32(observation)

			action = agent.get_exploitation_action(state)
			new_observation, reward, done, _ = env.step(action)

			steps_done += 1
			total_reward += reward

			if args.render:
				env.render()

			observation = new_observation

			if done:
				break

		print("Total_reward:{}, total step:{}, epoch:{}".format(total_reward,
			steps_done, epoch))


	env.close()