import numpy as np 
import sys, os 

"""
Module: The base class from which all other layers 
inherit. 

Two methods must be overriden by every layer:
forward and backward
"""
class Module(object):
	"""
	Module: Base class
	"""
	def __init__(self):
		super(Module, self).__init__()
	
	def forward(self, input):
		raise NotImplementedError("The forward pass must be defined")

	def backward(self, dout, cache):
		raise NotImplementedError("The backward pass must be defined")


