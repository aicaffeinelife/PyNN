import numpy as np 
import sys, os 
from Module import Module 

class ReLU(Module):
	"""
	ReLU: Calculates the ReLU activation
	
	Args: 
	inplace: Indicates if the input is changed. 
	Usage: 
	relu = ReLU()
	x = np.random.randn(N,D)
	x_act = relu.forward(x)
	relu.backward(dscore)
	x_grad = relu.grads['x']
	"""
	def __init__(self, inplace=True):
		super(ReLU, self).__init__()
		self._in = inplace 
		self.grads = {}
		self.cache = None 

	def forward(self, x):
		out = None 
		self.cache = x 
		if self._in :
			x = np.maximum(0, x)
			return x 
		else:
			out = np.maximum(0, x)
			return out 


	def backward(self, dout):
		x = self.cache
		dout[dout<=0] = 0 
		dx = dout
		self.grads['x'] = dx 

		

