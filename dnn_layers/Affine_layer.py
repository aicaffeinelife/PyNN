import numpy as np 
import sys, os 
from Module import Module 


class Affine(Module):
	"""
	Affine_Layer: The linear layer
	During forward pass calculates 
	y = Wx+b if bias is True otherwise
	y = Wx 
	
	During the backward pass, the gradients 
	are stored in the grads dict. 

	Args:
	in_feat: Input features to the layer. 
	out_feat: Output features of the layer.


	Usage: 
	layer = Affine()
	x = np.random.randn(N,D)
	y, cache = layer.forward(x)
	layer.backward(dy, cache) 
	print(layer.grads)
	
	"""
	def __init__(self, in_feat, out_feat, bias=True):
		super(Affine, self).__init__()
		self.params = {}
		self.grads = {}
		self._in = in_feat 
		self._out = out_feat
		self.bias = bias
		self.params['W'] = np.random.randn(self._in, self._out)
		if bias:
			self.params['b'] = np.zeros(self._out)

		

	def forward(self, ip):
		out = None 
		cache = None
		if self.bias:
			out = np.matmul(ip, self.params['W']) + self.params['b']
			cache = (ip, self.params['W'], self.params['b'])
		else:
			out = np.matmul(ip, self.params['W'])
			cache = (ip, self.params['W'])
		return out, cache 
		

	def backward(self, dout, cache):
		if self.bias:
			ip, W, b = cache
		else: 
			ip, W = cache  
		dX = np.dot(dout, W.T)
		dW = np.dot(ip.T, dout)
		
		if self.bias:
			dB = np.sum(dout, axis=0)
		else:
			dB = np.zeros(self._out)
		self.grads['X'] = dX
		self.grads['W'] = dW 
		self.grads['b'] = dB 



		

