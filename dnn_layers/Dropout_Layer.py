import numpy as np 
import sys, os 
from Module import Module 

class Dropout(Module):
	"""
	Dropout: Randomly 'dropout' neurons while training
	During testing the dropout layer is "transparent" as 
	in it doesn't dropout any neurons

	Args: 
	dropout_param: The probability of dropout
	"""
	def __init__(self, dropout_param=0.5):
		super(ReLU, self).__init__()
		self.dparam = {}
		self.grads = {}
		self.dparam['p'] = dropout_param
		self.dparam['seed'] = 912
		self.cache = None 

	def forward(self, x):
		p = self.dparam['p']
		seed = self.dparam['seed']
		np.random.seed(seed)
		mask = np.random.binomial(1, p, size=*x.shape)
		self.cache = (x, mask)
		out = mask*x 
		return out 


	def backward(self, dout):
		x, mask = self.cache
		dx = dout*mask 
		self.grads['x'] = dx
	 

