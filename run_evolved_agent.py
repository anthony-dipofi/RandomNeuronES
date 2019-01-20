import random
import copy
import time

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import torch
import torch.multiprocessing as mp

import retro
from baselines.common.atari_wrappers import WarpFrame, FrameStack
import sonic_util

from agent_network import Evolution_Agent_Simple


def env_proc(net, pipe, max_steps = 15*60, game_states=[{'game':'SonicTheHedgehog-Genesis','state':'GreenHillZone.Act1'}], frame_stack = 4, use_cuda = True):
	def format_obs(obs):
		obs = np.array(obs)
		os = []
		for i in range(frame_stack):
			o = obs[:,:,i*3:(i+1)*3]
			o = scipy.misc.imresize(o,(64,64))
			os.append(o)

		obs = np.concatenate(os, axis=2)

		obs = torch.tensor(obs).float()
		obs.requires_grad=False
		obs = obs.permute([2,0,1]).unsqueeze(0)
		if use_cuda:
			obs = obs.cuda()
		return obs		

	game_state_idx = random.randrange(len(game_states))
	env = retro.make(game=game_states[game_state_idx]['game'], state=game_states[game_state_idx]['state'])
	env = sonic_util.SonicDiscretizer(env)
	env = FrameStack(env, frame_stack)
	obs = env.reset()

	obs = format_obs(obs)

	prev_info = None

	t_reward = 0

	i = 0

	max_x = 0


	while(True):
		ins = {}
		ins['obs'] = obs
		a = net(ins)
		
		action = a	
		
		for itr in range(frame_stack):
			obs, rew, done, info = env.step(action)
			env.render()
			if(done):
				break
		obs = format_obs(obs)

		x_loc = info['x'] + info['screen_x']
		if(x_loc > max_x):
			rew = 0.02 * (x_loc - max_x)
			max_x =info['x']+info['screen_x']
		else:
			rew = 0

		if (prev_info is None):
			prev_info = info

		if (info['lives'] < prev_info['lives']):
			done = True

		if i >= max_steps:
			done = True

		t_reward += rew

		if (done == True):
			break
		if(i%50 == 0):
			print(i,t_reward)
		i+=1
		time.sleep(0.016)
	pipe.send({"reward":t_reward})
	return

if __name__ == '__main__':

	p = {}

	p['load_file_name'] = "evolved_agent_16.pth"

	mp.set_start_method('spawn')

	net=torch.load(p['load_file_name']).cuda()


	pipe_main, pipe_sub = mp.Pipe()
	proc = mp.Process(target = env_proc, args = (net, pipe_sub, 15*60*3) )#, [{'game':'SpaceInvaders-v0'}]))
	proc.start()










	
	