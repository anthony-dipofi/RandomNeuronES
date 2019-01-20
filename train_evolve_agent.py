import json
import os
import random
import copy
import time

import retro
import sonic_util
from baselines.common.atari_wrappers import WarpFrame, FrameStack

import numpy as np
import scipy.misc
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
#import torch.multiprocessing as mp
import multiprocessing as mp

from agent_network import Evolution_Agent_Simple

def env_proc(net, pipe, max_steps = 15*60, game_states=[{'game':'SonicTheHedgehog-Genesis','state':'GreenHillZone.Act1'}], frame_stack = 4, use_pixel_distance_reward = True, use_cuda = True):
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
	#env = gym.make(game_states[game_state_idx]['game'])
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
			if(done):
				break

		obs = format_obs(obs)

		x_loc = info['x'] + info['screen_x']
		if(use_pixel_distance_reward):
			if(x_loc > max_x):
				rew = 0.02 * (x_loc - max_x)
				max_x = info['x'] + info['screen_x']
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

		i+=1
	pipe.send({"reward":t_reward})
	return

if __name__ == '__main__':

	p = {}

	p['env_batch_size'] = 12
	p['num_episodes'] = 25
	p['save_file_name'] = "evolved_agent_17"
	p['use_cuda'] = True
	p['game_states'] = [{'game':'SonicTheHedgehog-Genesis','state':'GreenHillZone.Act1'}] #[{'game':'SpaceInvaders-v0'}]
	p['frame_stack'] = 4
	p['random_feature_noise_scale'] = 0.1
	p['initial_noise_scale'] = 0.25
	p['noise_scale'] = 0.15
	p['max_episode_length'] = 15*60*4
	p['use_pixel_distance_reward'] = False

	mp.set_start_method('spawn')

	best_net = None
	best_rew = -1

	net = Evolution_Agent_Simple(linear_features=50, init_noise_scale=p['random_feature_noise_scale'], use_cuda=p['use_cuda'])#.cuda()

	for param in net.parameters():
		param.requires_grad = False


	nets = []
	for i in range(p['env_batch_size']):
		n = copy.deepcopy(net)
		for param in n.parameters():
			param.requires_grad = False
		n.policy_head.weight = torch.nn.Parameter(p['initial_noise_scale'] * torch.randn(n.policy_head.weight.shape))
		n.policy_head.bias = torch.nn.Parameter(p['initial_noise_scale'] * torch.randn(n.policy_head.bias.shape))
		if p['use_cuda']:
			n = n.cuda()
		else:
			n = n.cpu()
		nets.append(n)


	for ep in range(p['num_episodes']):
		start = time.time()
		env_procs = []
		for proc_idx in range(p['env_batch_size']):
			pipe_main, pipe_sub = mp.Pipe()
			proc = mp.Process(target = env_proc, args = (nets[proc_idx], pipe_sub, p['max_episode_length'] , p['game_states'], p['frame_stack'], p['use_pixel_distance_reward'], p['use_cuda']))
			proc.start()
			env_procs.append({'idx': proc_idx, 'pipe': pipe_main, 'proc': proc})

		
		rews = []
		for proc_idx in range(p['env_batch_size']):
			rews.append(env_procs[proc_idx]['pipe'].recv()['reward'])
			
		end = time.time()

		top_rew = -0.1
		top_idx = None
		for i in range(len(rews)):
			if (rews[i] > top_rew):
				top_idx = i
				top_rew = rews[i]
			if( rews[i] > best_rew):
				best_net = copy.deepcopy(nets[i])
				best_rew = rews[i]

		print("ep", ep,"average rew", sum(rews)/p['env_batch_size'])
		print ("time:",end-start, "rewards", rews)
		for i in range(p['env_batch_size']):
			if (i == top_idx):
				continue
			n = copy.deepcopy(nets[top_idx])
			for param in n.parameters():
				param.requires_grad = False
			if(p['use_cuda']):
				n.policy_head.weight += p['noise_scale'] * torch.randn(n.policy_head.weight.shape).cuda()
				n.policy_head.bias  += p['noise_scale'] * torch.randn(n.policy_head.bias.shape).cuda()
				n = n.cuda()
			else:
				n.policy_head.weight += p['noise_scale'] * torch.randn(n.policy_head.weight.shape)
				n.policy_head.bias  += p['noise_scale'] * torch.randn(n.policy_head.bias.shape)
			nets[i] = n




	torch.save(best_net, p['save_file_name']+'.pth')






	
