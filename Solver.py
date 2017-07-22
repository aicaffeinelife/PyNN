import sys, os 
import matplotlib.pyplot as plt

import numpy as np 
from nn_random import FullyConnectedNet 
from optim import *
from utils import CifarLoader
import argparse
import copy



parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str,required=True, help='cifar10')
parser.add_argument('--epochs', type=int, default=20, help='epochs to train')
parser.add_argument('--print_every', type=int,default=50, help='print after xth iters')
parser.add_argument('--plot_every', type=int, default=100, help='plot the vars after x iters')
parser.add_argument('--save_dir', type=str, help='The location to save the plots in')

args = parser.parse_args()
if not os.path.exists(args.data):
	print("Data path doesn't exists")
	sys.exit(1)

if not os.path.exists(args.save_dir):
	os.makedirs(args.save_dir)


# dataset loading 
train_dset = CifarLoader(args.data, 'train')
test_dset = CifarLoader(args.data, 'test')
 # get the data as a whole 

class Solver(object):
	"""
	Solver: Construct a solver instance with the model and data

	Required args: 
	model : the model to be trained 
	config_dict: Dictionary containg the hyperparam config
	batch_size: The batch size to use


	Optional args:
	
	print_ever, plot_every, check_accuracy

	TODO:
	1. Add checkpointing ability
	2. Expose a plotting interface for easy plotting of monitored params
	"""
	def __init__(self, model, epochs, batch_size, cfg_dict=None, **kwargs):
		super(Solver, self).__init__()
		self.model = model 
		self.epochs = epochs
		self.update_rule = update_rule
		self.cfg_dict = {}
		if cfg_dict is not None:
			self.cfg_dict = copy.deepcopy(cfg_dict)
		self.bs = batch_size

		self.print_every = kwargs.pop('print_every', 100)
		self.plot_every =  kwargs.pop('plot_every', 100)
		self.num_train_samples = kwargs.pop('train_samples', 1000)
		self.num_val_samples = kwargs.pop('val_samples', 1000)
		self.update_rule = kwargs.pop('update_rule', 'adam')

	def _get_optimizer(self):
		optim = None 
		if self.update_rule == 'adam':
			optim = Adam(self.model.params, self.cfg_dict)
		else:
			optim = SGD(self.model.params, self.cfg_dict)
		return optim




	def _init_solver(self):
		"""
		Initialize bookeeping vars for training purposes
		"""
		self.train_loss_history = []
		self.val_loss_history = []
		self.max_val_acc = 0 # best validation accuracy 
		self.val_acc_history = [] 
		self.train_acc_history = []



	def train_step(self):
		num_train = self.num_train_samples
		train_data, train_labels = train_dset.get_complete_dataset()
		train_sample_data = train_data[:num_train]
		train_sample_labels = train_labels[:num_train]
		optim = self._get_optimizer()
		mask = np.random.choice(num_train, self.bs) # randomly pick a batch size training data
		data_batch = train_sample_data[mask]
		label_batch = train_sample_labels[mask]
		self.model.mode = 'train'
		probs, loss = self.model.forward(data_batch, label_batch)
		self.model.backprop(probs, label_batch) # backprop through the NN 
		optim.update(self.model.grads)
		self.model.params = copy.deepcopy(optim.params) # copy back the updated params 
		return loss 


	
	def _check_acc(self, X, y, num_samples=50):
		"""
		Check accuracy of the model on the given num_samples of the data
		"""
		pass 


	def train(self):
		"""
		The actual training step, monitors training loss
		"""
		iters_per_epoch = max(self.num_train_samples//self.bs, 1) 
		self.curr_epoch = 1 
		num_iters = iters_per_epoch * self.epochs 
		for i in range(num_iters):
			loss = self.train_step()
			self.train_loss_history(loss)
			if i%self.print_every == 0:
				print("[%d/%d]: loss: %.3f"%(i+1, num_iters, self.train_loss_history[-1]))

			if i%self.plot_every == 0:
				plt.title('Train loss vs iters')
				plt.plot(i, self.train_loss_history)
				plt.savefig('Train_loss_iters.png')


		epoch_end = (i+1) == iters_per_epoch
		if epoch_end:
			self.curr_epoch += 1 
			self.cfg_dict['lr'] *= self.cfg_dict['lr_decay']
		# TODO: Add accuracy checking and test it on the beginning and end of epochs 
		# TODO: Add mechanism to save only the best parameters obtained 






























		







		