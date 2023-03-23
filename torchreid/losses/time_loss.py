import numpy as np
from torch import nn, tensor
import torch
from torch.autograd import Variable

class time_loss(nn.Module):
	def __init__(self, margin=0.1, dist_type = 'l2'):
		super(time_loss, self).__init__()
		self.margin = margin
		self.dist_type = dist_type
		if dist_type == 'l2':
			self.dist = nn.MSELoss(reduction='sum')
		if dist_type == 'cos':
			self.dist = nn.CosineSimilarity(dim=0)
		if dist_type == 'l1':
			self.dist = nn.L1Loss()
	
	def forward(self, feat, label1):
		feat_size = feat.size()[1]
		feat_num = feat.size()[0]
		label_num =  len(label1.unique())
		feat = feat.chunk(label_num, 0)
		#loss = Variable(.cuda())
		for i in range(label_num):
			center1 = torch.mean(feat[i], dim=0)
			if self.dist_type == 'l2' or self.dist_type == 'l1':
				if i == 0:
					dist = max(0, abs(self.dist(center1, center1)))
				else:
					dist += max(0, abs(self.dist(center1, center1)))
			elif self.dist_type == 'cos':
				if i == 0:
					dist = max(0, 1-self.dist(center1, center1))
				else:
					dist += max(0, 1-self.dist(center1, center1))
		return dist
		