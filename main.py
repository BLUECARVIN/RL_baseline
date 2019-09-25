import os
import train
import gym


def main(args):
	env = gym.make(args.env)
	if args.train:
		train(env, args)


if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser()
	parser.add_argument('--env', type=str, default='Walker2d-v2')
	parser.add_argument('--train', type=bool, default=True)
	parser.add_argument('--memory', type=str, default='simple')
	parser.add_argument('--gamma', type=float, default=0.9)
	parser.add_argument('--batch_size', type=int, default=64)
	parser.add_argument('--start_training', type=int, default=1000)
	parser.add_argument('--tau', type=float, default=0.01)
	parser.add_argument('--critic_lr', type=float, default=1e-3)
	parser.add_argument('--actor_lr', type=float, default=1e-3)
	parser.add_argument('--initial_e', type=float, default=0.5)
	parser.add_argument('--end_e', type=float, default=0.01)
	main()