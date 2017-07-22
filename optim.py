import numpy as np 
import copy
"""
Implementation of the some popular optimization algorithms 
1. SGD with Momentum
2. Adam

Each class requires:
Params: The params dict of the model 
Config: A dictionary containing the hyperparams

# """
class SGD(object):
	"""
	SGD: Implements SGD update with momentum
	SGD + Momentum
	vel_t = mu*vel_t-1 + lr*dw
	w = w - vel_t
	v_t: Param to keep track of moving average of velocity. 
	Note: Uses standard momentum and not Nesterov	
	"""
	def __init__(self, params, cfg = None):
		super(SGD, self).__init__()
		self.params = params 
		self.grads = {}
		if cfg is None:
			self.cfg = {}
			self.cfg.setdefault('lr', 1e-2)
			self.cfg.setdefault('momentum', 0.9)

		else:
			self.cfg = cfg
		
		
	def zero_grad(self):
		for k in self.params.keys():
			self.grads[k] = np.zeros_like(self.params[k])
		


	def update(self, grads):
		if not self.grads:
			self.zero_grad
		else:
			mu  = self.cfg.get('momentum')
			lr = self.cfg.get('lr')
			self.grads = grads 
			for k in self.params.keys():
				print("Updating params:{}".format(k))
				v_t = np.zeros_like(self.grads[k])
				v_t = mu*v_t + lr*self.grads[k]
				self.params[k] -= v_t 
				self.cfg['velocity'] = v_t 



class Adam(object):
	"""
	Adam: Implements the Adam optimizer
	cfg params:
	beta1, beta2: hyperparams for updates 
	epsilon: hyperparam for smoothing the lr
	lr: learning rate
	
	Adam update rule:
	m_t = beta1*m_t-1 + (1-beta1)*dw
	v_t = beta2*v_t-1 + (1-beta2)*dw*dw

	m_cap_t = m_t/(1-beta1**t)
	v_cap_t = v_t/(1-beta2**t)
	wnxt = w - (lr/sqrt(v_cap_t + ep))*m_cap_t 


	"""
	def __init__(self, params, cfg=None):
		super(Adam, self).__init__()
		self.params = params 
		if cfg is None:
			self.cfg = {}
			self.cfg.setdefault('lr',1e-2)
			self.cfg.setdefault('beta1', 0.9)
			self.cfg.setdefault('beta2', 0.99)
			self.cfg.setdefault('ep', 1e-8)
			self.cfg.setdefault('t',1)
		else:
			self.cfg = cfg 
		self.grads = {}

	def zero_grad(self):
		for k in self.params.keys():
			self.grads[k] = np.zeros_like(self.params[k])

	def update(self, grads):
		if not self.grads:
			self.zero_grad()
		else:
			lr = self.cfg.get('lr')
			ep = self.cfg.get('ep')
			beta1 = self.cfg.get('beta1')
			beta2 = self.cfg.get('beta2')
			t = self.cfg.get('t')
			self.grads = copy.deepcopy(grads) # make a deep copy of grad dict 
			for k in self.params.keys():
				m = np.zeros_like(self.params[k])
				v = np.zeros_like(self.params[k])
				m = beta1*m + (1-beta1)*self.grads[k]
				v = beta2*v + (1-beta2)*self.grads[k]
				m_t = m/(1-beta1**t)
				v_t = v/(1-beta2**t)
				t = t+1 
				self.params[k] -= lr*m_t/(np.sqrt(v_t)+ep)
				self.cfg['m'] = m 
				self.cfg['v'] = v 
				self.cfg['t'] = t 
				



		













# def sgd_momentum(w, dw, cfg=None):
# 	"""
	
# 	"""
# 	if cfg is None: cfg = {}
# 	cfg.setdefault('lr', 1e-2)
# 	cfg.setdefault('mu', 0.9)
# 	v_t = cfg.get('velocity', np.zeros_like(w))
# 	v_t = cfg['mu']*v_t + cfg['lr']*dw
# 	w_nxt = w - v_t
# 	cfg['velocity'] = v_t
# 	return w_nxt, cfg



# def adam(x, dx, cfg=None):
# 	"""
# 	Implements Adam update rule
# 	cfg params:
# 	beta1, beta2: hyperparams for updates 
# 	epsilon: hyperparam for smoothing the lr
# 	lr: learning rate
	
# 	Adam update rule:
# 	m_t = beta1*m_t-1 + (1-beta1)*dw
# 	v_t = beta2*v_t-1 + (1-beta2)*dw*dw

# 	m_cap_t = m_t/(1-beta1**t)
# 	v_cap_t = v_t/(1-beta2**t)
# 	wnxt = w - (lr/sqrt(v_cap_t + ep))*m_cap_t 

# 	"""
# 	if cfg is None: cfg = {} 
# 	cfg.setdefault('lr', 1e-2)
# 	cfg.setdefault('beta1', 0.9)
# 	cfg.setdefault('beta2', 0.99)
# 	cfg.setdefault('ep', 1e-8)
# 	cfg.setdefault('m', np.zeros_like(x))
# 	cfg.setdefault('v', np.zeros_like(x))
# 	cfg.setdefault('t', 1)
# 	beta1, beta2 = cfg['beta1'], cfg['beta2']
# 	cfg['m'] = beta1*cfg['m'] + (1-beta1)*dx 
# 	cfg['v'] = beta2*cfg['v'] + (1-beta2)*dx*dx 
# 	m_t = cfg['m']/(1-beta1**cfg['t'])
# 	v_t = cfg['v']/(1-beta2**cfg['t'])
# 	cfg['t'] += 1
# 	x_n = x - cfg['lr']*m_t/(np.sqrt(v_t)+ cfg['ep'])
# 	return x_n, cfg



# # def RMSProp(w, dw, cfg=None):