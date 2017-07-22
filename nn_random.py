import numpy as np 
from layers import *
from optim import *
import pprint

  
class FullyConnectedNet(object):
    """
    FullyConnectedNet: Implements a fully connected net 
    """
  	
    def __init__(self, hidden_dims, mode = 'test', input_dims=3*32*32,  **kwargs):
        super(FullyConnectedNet, self).__init__()
        self.hidden_dims = hidden_dims
        self.n_layers = 1 + len(hidden_dims)
        self.ws = kwargs.pop('weight_scale', 1e-2)
        self.nc = kwargs.pop('num_classes',10)
        self.input_dims = input_dims 
        self.dtype = kwargs.pop('dtype',np.float32)
        self.reg = kwargs.pop('reg',0)
        self.params = self._gen_params(self.input_dims, self.hidden_dims, self.nc, self.ws)
        self.grads = {}
        self.mode = mode
        self.fwd_cache = ()

        for k,v in self.params.items():
            self.params[k] = v.astype(self.dtype)



    def _gen_params(self, input_dim, hidden_dims, num_classes, ws):
        param_dict = {}
        param_dict['W0'] = np.random.randn(input_dim, hidden_dims[0])*ws
        param_dict['b0'] = np.zeros(hidden_dims[0])

        for i in range(len(hidden_dims)):
            if (i+1)%len(hidden_dims) == 0:
                param_dict['W'+str(i+1)] = np.random.randn(hidden_dims[i], num_classes)*ws
                param_dict['b'+str(i+1)] = np.zeros(num_classes)
            else:
                param_dict['W'+str(i+1)] = np.random.randn(hidden_dims[i], hidden_dims[i+1])*ws
                param_dict['b'+str(i+1)] = np.zeros(hidden_dims[i+1])
        return param_dict


    def forward(self, X, target=None, bprop_mode='standard'):
        """
  		Run a normal forward pass and output the scores
  		X,y: Inputs and targets resp
  		mode: Train, test
  		bprop_mode:
  		Standard: The backprop algorithm
  		Random: Random backpropagation

  		Outputs(only in test mode):
  		Probabilities: Softmax probs of outputs

        Output(in train mode):
        NLL Loss, grads computed via the specified mode. 
  	
        """
        loss = 0.0
        cache = ()
        scores = None 
        h0, cache_1 = affine_forward(X,self.params['W0'], self.params['b0'])
        h0_act, _ = ReLU_forward(h0)
        print(h0_act.shape)
        hidden_cache = []
        for i in range(len(self.hidden_dims)):
            w_i = self.params['W'+str(i+1)]
            b_i = self.params['b'+str(i+1)]
            if i == 0:
                h_i, cache_i = affine_forward(h0_act, w_i, b_i)
                h_i_act = ReLU_forward(h_i)
                hidden_cache.append(cache_i)

            elif (i+1)%len(self.hidden_dims) == 0:
                scores, cache = affine_forward(hidden_cache[-1][0], w_i, b_i)
            else:
                h_i, cache_i = affine_forward(hidden_cache[i-1][0], w_i, b_i)
                h_i_act, _ =   ReLU_forward(h_i)
                hidden_cache.append(cache_i)
        
        print(scores.shape)
        # if bprop_mode == 'standard':
        #     grads = self._backprop(scores, y,cache_1,cache,hidden_cache)
        # elif bprop_mode == 'random':
        #     print("Random bprop mode is brewing!")
        probs = softmax(scores)
        corr_probs = -np.log(probs[range(scores.shape[0]), target])
        loss = np.sum(corr_probs)/scores.shape[0]
        if self.mode == 'test':
           return loss 

        if self.mode == 'train':
            self.fwd_cache = (cache_1, hidden_cache, cache)
            return probs, loss

        
      # dscores = np.sum(corr_probs)/y.shape[0]

  		


    def backprop(self, scores ,target):
  		
        grads = self.grads
        ip_cache, hc_cache, op_cache = self.fwd_cache
        print(len(hc_cache))
        dscores = scores 
        dscores[range(scores.shape[0]), target] -= 1
        hg = {} # dict for storing the gradients of the i-1th layer
        if self.mode == 'test':
            print("The neural network is in test mode. Set it to train mode")
            sys.exit(1)
        else:
            dHo_act, dWo, dbo = affine_backward(dscores, op_cache)
            dHo, _ = ReLU_backward(dHo_act, op_cache[0])
            grads['W'+str(len(self.hidden_dims))] = dWo
            grads['b'+str(len(self.hidden_dims))] = dbo 
            hg['H'+str(len(self.hidden_dims))] = dHo 

            for i in range(len(self.hidden_dims)-1, -1, -1):
                if i == 0 :
                    dI_act, dW0, db0 = affine_backward(hg['H'+str(i+1)], ip_cache)
                    dI, _ = ReLU_backward(dI_act, ip_cache[0])
                    grads['W'+str(i)] = dW0
                    grads['b'+str(i)] = db0 
                else:
                    dH_act, dWh, dbh = affine_backward(hg['H'+str(i+1)], hc_cache[i-1])
                    dH, _ = ReLU_backward(dH_act, hc_cache[i-1][0])
                    grads['W'+str(i)] = dWh 
                    grads['b'+str(i)] = dbh 
                    hg['H'+str(i)] = dH
            self.grads = grads 



      #   # outermost layer
      #   dHf_act, dWf,dbf = affine_backward(dy, cache)
      #   grads['W'+str(len(self.hidden_dims)+1)] = dWf
      #   grads['b'+str(len(self.hidden_dims)+1)] = dbf
      #   dHf, _ = ReLU_backward(dHf_act, H_final)
      #   hg = {}
      #   hidden_cache.reverse()
      #   hl = len(self.hidden_dims)

      # # bprop thru outermost hidden layer
      #   dHo_act, dWo, dbo = affine_backward(dHf, hidden_cache[0])
      #   dHo, _ = ReLU_backward(dHo_act, hidden_cache[0][0])
      #   grads['W'+str(len(self.hidden_dims))] = dWo
      #   grads['b'+str(len(self.hidden_dims))] = dbo

      #   hg['H'+str(len(self.hidden_dims))]= dHo
      #   for i in range(len(self.hidden_dims)-1,0,-1):
      #       if (i+1)%(len(self.hidden_dims)) == 0:
      #           dh_act, dw_i, db_i = affine_backward(hg['H'+str(len(self.hidden_dims))],hidden_cache[hl-i])
      #           dh,_ = ReLU_backward(dh_act, hidden_cache[hl-i][0])
      #           grads['W'+str(i)] = dw_i
      #           grads['b'+str(i)] = db_i
      #           hg['H'+str(i)] = dh_i

      #       else:
      #           dH_act, dW, db = affine_backward(hg['H'+str(i+1)],hidden_cache[hl-i])
      #           dH, _ = ReLU_backward(dH_act,hidden_cache[hl-i][0])
      #           grads['W'+str(i)] = dW
      #           grads['b'+str(i)] = db 
      #           hg['H'+str(i)] = dH 

      # #bprop thu input layer 
      #   dI, dW_i, db_i = affine_backward(hg['H'+1], ip_cache)
      #   grads['W'+str(0)] = dW_i
      #   grads['b'+str(0)] = db_i 

      #   self.grads = grads


          




        

        










          


  		




















