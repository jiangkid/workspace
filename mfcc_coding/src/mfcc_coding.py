#coding=utf-8
"""
stacked auto-encoders
modity by jiangwenbin


 Denoising autoencoders are the building blocks for SdA.
 They are based on auto-encoders as the ones used in Bengio et al. 2007.
 An autoencoder takes an input x and first maps it to a hidden representation
 y = f_{\theta}(x) = s(Wx+b), parameterized by \theta={W,b}. The resulting
 latent representation y is then mapped back to a "reconstructed" vector
 z \in [0,1]^d in input space z = g_{\theta'}(y) = s(W'y + b').  The weight
 matrix W' can optionally be constrained such that W' = W^T, in which case
 the autoencoder is said to have tied weights. The network is trained such
 that to minimize the reconstruction error (the error between x and z).

 For the denosing autoencoder, during training, first x is corrupted into
 \tilde{x}, where \tilde{x} is a partially destroyed version of x by means
 of a stochastic mapping. Afterwards y is computed as before (using
 \tilde{x}), y = s(W\tilde{x} + b) and z as s(W'y + b'). The reconstruction
 error is now measured between z and the uncorrupted input x, which is
 computed as the cross-entropy :
      - \sum_{k=1}^d[ x_k \log z_k + (1-x_k) \log( 1-z_k)]


 References :
   - P. Vincent, H. Larochelle, Y. Bengio, P.A. Manzagol: Extracting and
   Composing Robust Features with Denoising Autoencoders, ICML'08, 1096-1103,
   2008
   - Y. Bengio, P. Lamblin, D. Popovici, H. Larochelle: Greedy Layer-Wise
   Training of Deep Networks, Advances in Neural Information Processing
   Systems 19, 2007

"""
import os
import sys
import time
import gzip
import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from dA import dA
from utils import tile_raster_images
import cPickle

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
import pylab

#===============================================================================
# SAE--stacked auto encoder 
#===============================================================================
from scipy.io import loadmat, savemat
class SAE(object):
    
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 layers_sizes=[129, 500, 54, 500, 129],
                 w_list=None,
                 b_list=None,
                 output_folder='out',
                 p_dict=None,
                 sigma_dict=None):
        self.dA_layers = []
        self.dA_train_flags = []
        self.finetune_train_flag = False
        self.n_layers = len(layers_sizes)
        self.n_layers_dA = int(len(layers_sizes)/2)
        self.numpy_rng = numpy_rng
        self.output_folder = output_folder
        
        assert self.n_layers > 0
        if p_dict==None:
            self.p_list = [0]*self.n_layers_dA
            self.sigma_list = [0]*self.n_layers_dA
            self.p = 0
            self.sigma = 0
        elif p_dict!=None and sigma_dict==None:
            assert len(p_dict['p_list']) == self.n_layers_dA
            self.p_list = p_dict['p_list']
            self.sigma_list = [0]*self.n_layers_dA
            self.p = p_dict['p']
            self.sigma = 0 
        elif p_dict!=None and sigma_dict!=None:           
            assert len(p_dict['p_list']) == self.n_layers_dA
            assert len(sigma_dict['sigma_list']) == len(p_dict['p_list'])
            self.p_list = p_dict['p_list']
            self.sigma_list = sigma_dict['sigma_list']
            self.p = p_dict['p']
            self.sigma = sigma_dict['sigma']
            
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        self.x = T.matrix(name='x')
        
        for i in xrange(self.n_layers_dA):            
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.dA_layers[-1].get_active()
                
            if(self.p_list[i] == 0):
                model_file = '%s/L%d.mat'%(output_folder, i)
            else:
                model_file = '%s/L%d_p%g_s%g.mat'%(output_folder, i, self.p_list[i], self.sigma_list[i])
                
            if os.path.isfile(model_file): #this layer has been trained
                model_w, model_b, model_b_prime = load_model_mat(model_file)
                dA_layer = dA(numpy_rng=numpy_rng, input=layer_input, n_visible=layers_sizes[i], n_hidden=layers_sizes[i+1], 
                          W=model_w, bhid=model_b, bvis=model_b_prime)
                self.dA_train_flags.append(True) #set the flag                
            else:
                dA_layer = dA(numpy_rng=numpy_rng, input=layer_input, n_visible=layers_sizes[i], n_hidden=layers_sizes[i+1])
                self.dA_train_flags.append(False) #set the flag
            self.dA_layers.append(dA_layer)
            
        finetune_file = '%s/SAE%s.mat'%(output_folder, self.get_model_file())
        if os.path.isfile(finetune_file): #trained
            self.finetune_train_flag = True
         
    def init_sigmoid_layers(self):
        #sigmoid layers
        self.sigmoid_layers = []        
        self.params = []
        for i in xrange(self.n_layers-1):            
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
                
            if i<(self.n_layers-1)/2:
                dA_layer = self.dA_layers[i]
                n_in = dA_layer.n_visible
                n_out = dA_layer.n_hidden
                W = dA_layer.W
                b = dA_layer.b
            else:#inverse the layer
                dA_layer = self.dA_layers[self.n_layers-2-i]                
                n_in = dA_layer.n_hidden
                n_out = dA_layer.n_visible
                W = theano.shared(numpy.asarray(dA_layer.W.T.eval(), dtype=theano.config.floatX), borrow=True)
                b = dA_layer.b_prime
                
            sigmoid_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=W,
                                        b=b,
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)                                
#         set_trace()
        self.errors = (self.sigmoid_layers[-1].output - self.x) ** 2
                
        self.save_model_mat('%s/SAE_pre%s.mat'%(self.output_folder, self.get_model_file()))
    
    def save_model(self, file_name=None):
        if file_name == None:
            file_name = '%s/SAE%s.mat'%(self.output_folder, self.get_model_file())
        self.save_model_mat(file_name)
    
    def get_model_file(self):
                
        if self.p!=0:
            file_str = '_p%g_s%g'%(self.p, self.sigma)
        else:
            file_str = ''
        for i in range(len(self.p_list)):
            if self.p_list[i] != 0:
                file_str = '%s_(L%g_p%g_s%g)'%(file_str, i , self.p_list[i], self.sigma_list[i])
        return file_str.strip()
    
    
    def save_model_mat(self, file_name):
        params = []
        for item in self.params:            
            param = item.get_value(borrow=True)            
            params.append(param)
#         set_trace()
        savemat(file_name, {'params':params})
        print file_name, 'saved'
        
        
    def get_gaussian_cost(self, x, p=0, sigma=1, mu=0.5):
        x_mean = T.mean(x, axis=0)
        cost = (1 / (sigma * T.sqrt(2 * numpy.pi)) * T.exp(-(x_mean - mu) ** 2 / (2 * sigma ** 2)))
        return p * cost
    
    
    def get_square_cost(self, x, p):        
        x_mean = T.mean(x, axis=0)
        cost = -(x_mean-0.5)**2 + 0.5        
        return p * cost
    
    
    def get_cost(self, p=0, sigma=1):
        # the last layer        
        z = self.sigmoid_layers[-1].output
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        p_idx = len(self.sigmoid_layers)/2 - 1 #penalty layer, the middle layer
        if p == 0:
            cost = T.mean(L)
        elif (p != 0) and (sigma == 0):# for square penalty
            square_cost = self.get_square_cost(self.sigmoid_layers[p_idx].output, p)
            cost = T.mean(L) + T.mean(square_cost)
        elif(p != 0) and (sigma != 0):# for Gaussian penalty                        
            gaussian_cost = self.get_gaussian_cost(self.sigmoid_layers[p_idx].output, p, sigma)
            cost = T.mean(L) + T.mean(gaussian_cost)            
        return cost
    
    def get_cost_updates(self, learning_rate=0.1, p=0, sigma=1):
        
        cost = self.get_cost(p, sigma)
        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]
        return (cost, updates)
 
    def pretraining_functions(self, train_set, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')       
        
        pretrain_fns = []
        for i in range(len(self.dA_layers)):
            # get the cost and the updates list
            cost, updates = self.dA_layers[i].get_cost_updates_p(learning_rate, self.p_list[i], self.sigma_list[i])
            fn = theano.function(
                inputs=[index,
                        learning_rate],
                outputs=cost,
                updates=updates,
                givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}

            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    
    def build_finetune_functions(self, datasets, batch_size):
        
        self.init_sigmoid_layers()#firstly, initial the sigmoid layer
        train_set, valid_set, test_set = datasets        
        n_valid_batches = valid_set.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set.get_value(borrow=True).shape[0] / batch_size
        
        index = T.lscalar('index')  # index to a [mini]batch
        learning_rate = T.scalar('lr')
        
        sae_cost, sae_updates = self.get_cost_updates(learning_rate=learning_rate,
                                                      p=self.p,
                                                      sigma=self.sigma)
    #     set_trace()
        train_fn = theano.function(
                inputs=[index,
                        learning_rate],
                outputs=sae_cost,
                updates=sae_updates,
                givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}
            )
        test_fn = theano.function(
                inputs=[index],
                outputs=sae_cost,
                givens={self.x: test_set[index * batch_size: (index + 1) * batch_size]}
            )
        valid_fn = theano.function(
                inputs=[index],
                outputs=sae_cost,
                givens={self.x: valid_set[index * batch_size: (index + 1) * batch_size]}
            )
        
        # Create a function that scans the entire validation set
        def valid_model():
            return [test_fn(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_model():
            return [valid_fn(i) for i in xrange(n_test_batches)]
        
        return train_fn, valid_model, test_model
    
def train_SAE(datasets,layers_sizes,output_folder,p_dict,sigma_dict):
    
    train_set, valid_set, test_set = datasets
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    pre_epochs=1000
    fine_epochs=1000
    epoch_list = [1, 10]
    finetune_lr = 0.1
    batch_size = 100
    down_epoch = 100
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    logger = mylogger(output_folder + '/log.log')
    logger.log('layer size: '+''.join(['%d '%i for i in layers_sizes]).strip())
    p_list=p_dict['p_list']
    sigma_list=sigma_dict['sigma_list']
    rng = numpy.random.RandomState(123)
    sae = SAE(numpy_rng=rng, layers_sizes=layers_sizes, output_folder=output_folder, p_dict=p_dict, sigma_dict=sigma_dict)
    pretraining_fns = sae.pretraining_functions(train_set, batch_size)
    start_time = strict_time()    
    logger.log('pre_training...')    
    for i in xrange(len(sae.dA_layers)):        
        if sae.dA_train_flags[i] == True:
            logger.log('L%d trained'%(i))
            continue
        pretrain_lr = 0.05
        logger.log('training L%d ...'%(i))
        logger.log('learning rate:%g, p:%g, sigma:%g' % (pretrain_lr, p_list[i], sigma_list[i]))
        for epoch in xrange(1, pre_epochs + 1):
            epoch_time_s = strict_time()
            c = []
            for batch_index in xrange(n_train_batches):
                c.append(pretraining_fns[i](index=batch_index, lr=pretrain_lr))              
            logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
            if epoch % down_epoch == 0:
                pretrain_lr = 0.8 * pretrain_lr            
                logger.log('learning rate: %g' % (pretrain_lr))
        if(p_list[i] == 0):
            model_file = '%s/L%d.mat'%(output_folder, i)
        else:
            model_file = '%s/L%d_p%g_s%g.mat'%(output_folder, i, p_list[i], sigma_list[i])
        sae.dA_layers[i].save_model_mat(model_file)
    
#     train_fn, valid_model, test_model = sae.build_finetune_functions(datasets, batch_size)#must be here! after pre_train
#     return

    p = p_dict['p']
    sigma = sigma_dict['sigma']
    logger.log('fine tuning...')
    if sae.finetune_train_flag == True:
        logger.log('SAE trained')
        return    
    logger.log('learning rate:%g, p:%g, sigma:%g' % (finetune_lr, p, sigma))
    train_fn, valid_model, test_model = sae.build_finetune_functions(datasets, batch_size)#must be here! after pre_train
    
    for epoch in xrange(1, fine_epochs + 1):
        epoch_time_s = strict_time()
        c = []
        for minibatch_index in xrange(n_train_batches):
            err = train_fn(minibatch_index, finetune_lr)
            c.append(err)
        logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
        if epoch % down_epoch == 0:
            finetune_lr = 0.8 * finetune_lr
            logger.log('learning rate: %g' % (finetune_lr))
#         if epoch in epoch_list:
#             sae.save_model('%s/SAE_p%g_s%g_(e_%g).mat'%(output_folder, p, sigma, epoch))
    sae.save_model()
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))

  
def strict_time():
    if sys.platform == "win32":
        return strict_time()
    else:
        return time.time()
    
from data_mat import load_TIMIT, load_model_mat, save_model_mat
from logger import log_init, mylogger
from tools import set_trace

def train_mfcc_coding():
        
    output_folder = '300_layer5'
    if output_folder.find('300') != -1:         
        data_file = '300bps/TIMIT_train_split.mat'
    elif output_folder.find('600') != -1:
        data_file = '600bps/TIMIT_train_dr1_dr4_split.mat'
    elif output_folder.find('1200') != -1:
        data_file = '1200bps/TIMIT_train_dr1_dr2_split.mat'
    elif output_folder.find('2400') != -1:
        data_file = '2400bps/TIMIT_train_dr1_dr2_split.mat'
        
    datasets = load_TIMIT(data_file)
    train_set, valid_set, test_set = datasets
    p_list = [0]
    sigma_list = [0]
    input_dim = train_set.get_value(borrow=True).shape[1]    
#     layers_sizes = [input_dim,2000,1000,500,54,500,1000,2000,input_dim]
#     layers_sizes = [input_dim,2000,2000,54,2000,2000,input_dim]
    layers_sizes = [input_dim,2000,54,2000,input_dim]
    for p in p_list:
        for sigma in sigma_list:
            p_dict = {'p_list': [0, 0], 'p': p}
            sigma_dict = {'sigma_list':[0, 0], 'sigma':sigma}
            train_SAE(datasets,layers_sizes,output_folder,p_dict,sigma_dict)
    
        
if __name__ == '__main__':
    
    train_mfcc_coding()
    
