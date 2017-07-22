from layers import * 
from optim import * 
from nn_random import FullyConnectedNet 
import numpy as np
import pprint 

# Layer tests 

# check the error between the actual forward pass 
def relative_err(x, y):
	return np.max(np.abs(x-y)/np.maximum(1e-8, np.abs(x) + np.abs(y)))


def eval_num_grad_array(f, x,dx, h=1e-5):
	"""
	Evaluate the numerical gradient of the backpass 
	f: function to be tested e.g. affine_forward 
	x: input array 
	dx: gradient of the input 
	h: the amount to be tested
	Returns: np array "grad" of the same shape as x containing the gradients 

	Source: cs231n: gradient_check.py
	"""
	grads = np.zeros_like(x)
	it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
	while not it.finished:
		ix = it.multi_index 
		val = x[ix]
		x[ix] = x[ix] + h 
		f_pos = f(x).copy()
		x[ix] = x[ix] - h 
		f_neg = f(x).copy()
		x[ix] = val 
		grads[ix] = np.sum((f_pos - f_neg)*dx)/(2*h)
		it.iternext()
	return grads 



def forward_layer_test():
	num_inputs = 2
	input_shape = 120
	output_dim = 3
	input_size = num_inputs*input_shape
	weight_size =  input_shape*output_dim
	# forward_pass test 
	x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, input_shape)	
	w = np.linspace(-0.2, 0.3, num=weight_size).reshape(input_shape, output_dim)
	b = np.linspace(-0.3,0.1, num=output_dim)
	y1, _ = affine_forward(x,w,b)
	correct_out = np.array([[ 1.49834967,  1.70660132,  1.91485297],
                        [ 3.25553199,  3.5141327,   3.77273342]])
	print("Affine forward test:")
	print("rel_y:{}".format(relative_err(y1, correct_out)))

# backward_pass test
def backward_layer_test():
	x = np.random.randn(10, 6)
	w = np.random.randn(6,5)
	b = np.random.randn(5)
	dout = np.random.randn(10, 5)
	dx_np = eval_num_grad_array(lambda x: affine_forward(x,w,b)[0], x, dout)
	dw_np = eval_num_grad_array(lambda w: affine_forward(x,w,b)[0], w, dout)
	db_np = eval_num_grad_array(lambda b: affine_forward(x,w,b)[0], b, dout)
# print("dx_np:{}, dw_np:{}, db_np:{}".format(dx_np.shape, dw_np.shape, db_np.shape))
	_, cache = affine_forward(x,w,b)
	dx, dw, db = affine_backward(dout, cache)
	rel_x = relative_err(dx_np, dx)
	rel_w = relative_err(dw_np, dw)
	rel_b = relative_err(db_np, db)
	print("relx:{}".format(relative_err(dx_np, dx)))
	print("relw:{}".format(relative_err(dw_np, dw)))
	print("relb:{}".format(relative_err(db_np,db)))


def net_test():
	input_dim = 120
	x = np.random.randn(10,input_dim)
	target = np.random.randint(10, size=(10,))
	hidden_dims = [6,6,6]
	fc = FullyConnectedNet(hidden_dims, mode='train',input_dims=input_dim)
	sgd = SGD(fc.params)
	sgd.zero_grad()
	probs = fc.forward(x,target)
	fc.backprop(probs, target)
	sgd.update(fc.grads)
	
	

def optim_test():
	input_dim = 120
	hidden_dims = [6,6,6]
	fc = FullyConnectedNet(hidden_dims,input_dims=input_dim)
	sgd = SGD(fc.params)
	sgd.zero_grad()
	grads = {}
	grads['W0'] = np.random.randn(120,6)
	grads['b0'] = np.random.randn(6,)
	grads['W1'] = np.random.randn(6,6)
	grads['b1'] = np.random.randn(6,)
	grads['W2'] = np.random.randn(6,6)
	grads['b2'] = np.random.randn(6,)
	grads['W3'] = np.random.randn(6, 10)
	grads['b3'] = np.random.randn(10,)
	sgd.update(grads)
	pprint.pprint(sgd.params)
˙






if __name__ == '__main__':
	net_test()

