import torch
import torch.nn as nn
import torch.nn.functional as F

class Evolution_Agent_Simple(nn.Module):
	def __init__(self, vision_out_features = 256, action_space_size = 7, linear_features = 32, use_cuda = False, init_noise_scale = 0.1, clamp = (-10,10)):
		super(Evolution_Agent_Simple, self).__init__()
		self.use_cuda = use_cuda
		self.clamp = clamp
		self.vision_out_features = vision_out_features
		self.action_space_size = action_space_size
		self.lstm_features = linear_features

		self.conv1 = nn.Conv2d( 12,  16, (8, 8), stride = 4)
		self.conv2 = nn.Conv2d( 16,  32, (4, 4), stride = 2)
		self.conv_linear_out = nn.Linear(6*6*32, vision_out_features)

		self.linear1 = nn.Linear(vision_out_features, linear_features)

		self.policy_head = nn.Linear(linear_features, action_space_size)

		for i in range(1,len(list(self.modules()))):
			l = list(self.modules())[i]
			for p in l.parameters():
				p = torch.normal(torch.zeros(p.shape), init_noise_scale*torch.ones(p.shape))


	def forward(self, ins):

		batch_size = len(ins['obs'])

		v = ins['obs']

		img_v = F.relu(self.conv1(v))
		img_v = F.relu(self.conv2(img_v))
		img_v = img_v.view([batch_size, -1])

		img_enc = F.relu(self.conv_linear_out(img_v))

		v = torch.tanh(self.linear1(img_enc))

		p_v = F.softmax(self.policy_head(v), dim = 1)

		action = torch.argmax(p_v)

		return action
