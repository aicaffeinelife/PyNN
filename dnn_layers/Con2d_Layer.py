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

	"""
	def __init__(self, in_ch, out_ch, kernel_size, stride, pad=1):
		super(Conv2d, self).__init__()
		self._in = in_ch
		self._out = out_ch
		kh, kw = self._check_kernel(kernel_size)
		self.params = {}
		self.params['kh'] = kh 
		self.params['kw'] = kw 

	
	def _check_kernel(self, kernel):
		_kh = _kw = None
		if not isinstance(kernel, int) or not isinstance(kernel, tuple):
			raise TypeError("Kernel must either be an int or tuple")

		if isinstance(kernel, int):
			_kh = _kw = kernel 
		elif isinstance(kernel, tuple):
			_kh, _kw = kernel
		return _kh, _kw

	def forward(self, x):
		pass 

	def backward(self, dout):
		pass

