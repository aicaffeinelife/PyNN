import numpy as np 
import sys, os
from Module import Module 

class Conv2d(Module):
	"""
	Conv2d: Computes a convolution over a 2 
	dimensional spatial map. The forward pass is 
	calculating y = Wx+b at kernel_size x kernel_size
	patches of image. The backward pass is caclulating
	dX, dW and db over the same patch. 

	Note: This version is slow and performance may suffer with large kernel sizes
	or spatial dimensions. 

	Args: 
	in_ch: Input channels to the Convolutional layer 
	out_ch: Output channels of the Convolutional layer 
	kernel_size: An int or a tuple indicating the size of the convolutional kernel
	stride: The stride of the sliding window
	pad: An int representing how many pixels the input spatial map is to be padded with
	on either side.

	Usage: 
	N,C,H,W = 3,3,10,10
	out = 64
	x = np.random.randn(N, C, H, W)
	net = Conv2d(C,out,kernel_size=3,stride=2, pad=1)
	out = net.forward()
	net.backward()
	print(net.grads)

	"""
	def __init__(self, in_ch, out_ch, kernel_size, stride, pad=1):
		super(Conv2d, self).__init__()
		self._in = in_ch
		self._out = out_ch
		kh, kw = self._check_kernel(kernel_size)
		self.params = {}
		self.params['kh'] = kh 
		self.params['kw'] = kw
		self.parmas['stride'] = stride 
		self.params['pad'] = pad 
		self.cache = None 
		self.grads = {}

	
	def _check_kernel(self, kernel):
		_kh = _kw = None
		if not isinstance(kernel, int) or not isinstance(kernel, tuple):
			raise TypeError("Kernel must either be an int or tuple")

		if isinstance(kernel, int):
			_kh = _kw = kernel 
		elif isinstance(kernel, tuple):
			_kh, _kw = kernel
		return _kh, _kw

	def _init_weights(self):
		f = self._out 
		c = self._in 
		hh = self.params['kh']
		ww = self.params['kw']
		W = np.random.randn(f,c,hh,ww)
		b = np.zeros(f)
		self.params['w'] = W 
		self.params['b'] = b 

	def forward(self, x):
		stride = self.params['stride']
		N,c,h,w = x.shape
		F,_, hh,ww = self.params['W'].shape
		W = self.params['W']
		b = self.params['b']
		pad = self.params['pad']
		h_dash = 1 + (H + 2*pad - hh)//stride
		w_dash = 1 + (w + 2*pad - ww)//stride 
		out = np.zeros((N,F,h_dash, w_dash))
		pad_dim = ((0,0), (0,0),(pad,pad),(pad, pad))
		x_padded = np.pad(x, pad_dim, mode='constant', constant_value=0)
		for n in range(N):
			for f in range(F):
				for h_i in range(h_dash):
					for w_i in range(w_dash):
						out[n,f,h_i, w_i] = np.sum(x_padded[n,:, h_i*stride:(hh+ h_i*stride), w_i*stride:ww+w_i*stride]*W[f,:, :, :]) + b[f]
		self.cache = x
		return out 


		

	def backward(self, dout):
		x = self.cache 
		W = self.params['W']
		b = self.params['b']
		dx = np.zeros_like(x)
		dW = np.zeros_like(W)
		db = np.zeros_like(b)
		stride = self.params['stride']
		pad = self.params['pad']
		pad_dim = ((0,0),(0,0),(pad,pad),(pad,pad))
		padded_dx = np.pad(dx, pad_dim, mode='constant', constant_value=0)
		x_pad = np.pad(x, pad_dim, mode='constant',constant_value=0)
		N,c,h,w = x.shape
		F, _, hh,ww = W.shape 
		h_dash, w_dash = dout.shape[2], dout.shape[3]
		for n in range(N):
			for f in range(F):
				for h_i in range(h_dash):
					for w_i in range(w_dash):
						padded_dx[n,:,h_i,w_i] += w[f,:,:,:]*dout[n,f,h_i,w_i]
						dw[f,:,:,:] += x_pad[n,:, h_i*stride:(hh + h_i*stride), w_i*stride:(ww + w_i*stride)]*dout[n,f,h_i, w_i]

		db = np.sum(dout, axis=(0,2,3))
		dx = padded_dx[:, :, pad:-pad, pad:-pad]

		self.grads['x'] = dx 
		self.grads['W'] = dw 
		self.grads['b'] = db 



