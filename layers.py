import numpy as np 

# Defines the layers to be used in the nn

def softmax(x):
	y = np.exp(x)/np.sum(np.exp(x),axis=1, keepdims=True)
	return y

def sigmoid(x):
	return 1./(1.+np.exp(x))

def sigmoid_back(x):
	return sigmoid(x)*sigmoid(1-x)
	

def affine_forward(X,w,b):
	"""
	Implements an affine forward pass of the form
	y = wX + b 
	Dimensions:
	X: NxD, w: Dxh b: hx1
	h: hidden layer dimension
	"""
	y = np.dot(X, w) + b 
	cache = (X,w,b)
	return y, cache


def ReLU_forward(x):
	"""
	Implements ReLU nonlinearity fwd pass
	x: NxC 
	output: NxC
	"""
	out = np.maximum(x,0)
	cache = x 
	return out, cache 

def ReLU_backward(dout, cache):
	"""
	ReLU nonlinearity backward pass
	The backward pass is simply to turn the negative and zero elements to zero
	"""
	x = cache
	dout[dout<=0] = 0
	dx = dout 
	return dx, cache


def affine_backward(dout, cache):
	"""
	Implements the affine backward pass
	Inputs: the gradient of the output, cache containing the original elements 
	Outputs: The gradients dX, dw and db
	Dimensions: 
	X: Nxh; w: Dxh and b: hx1
	"""
	X, w, b = cache
	# dX, dW, db = np.zeros_like(X), np.zeros_like(w), np.zeros_like(b)

	dX = np.dot(dout, w.T) 
	dW = np.dot(X.T, dout)
	db = np.sum(dout, axis=0)
	return dX, dW, db


def affine_relu_forward(x,w,b):
	"""
	Wrapper around affine_forward and relu_forward
	"""
	y, f_cache = affine_forward(x,w,b) 
	y_act, r_cache = ReLU_forward(y)
	cache = (f_cache, r_cache)
	return y_act, cache 

def affine_relu_backward(dout, cache):
	"""
	Wrapper around affine_backward and relu_backward
	"""
	f_cache, r_cache = cache 
	dnrelu = ReLU_backward(dout,r_cache)
	dx, dw, db = affine_backward(dnrelu, f_cache)
	return dx, dw,db
	








