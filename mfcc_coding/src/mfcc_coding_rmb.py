# coding=utf-8
"""
deep auto-encoder using RBM

"""
import os
import sys
import time
import gzip
import numpy
import collections
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
from learn_rates import LearningRateExpDecay
from gb_rbm import GBRBM
from scipy import io
from scipy.io import loadmat, savemat
from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer
from rbm import RBM
from utils import tile_raster_images
import cPickle

try:
    import PIL.Image as Image
except ImportError:
    import Image
    
import pylab

#===============================================================================
# DAE -- deep auto-encoder 
#===============================================================================
from scipy.io import loadmat, savemat
class deep_auto_encoder(object):
    
    def __init__(self,
                 numpy_rng,
                 theano_rng=None,
                 layers_sizes=[129, 500, 54, 500, 129],
                 output_folder='out',
                 p=0,
                 sigma=0,
                 logger=None):
        
        self.rbm_layers = []
        self.rbm_train_flags = []
        self.finetune_train_flag = False
        self.n_layers = len(layers_sizes)
        self.n_layers_rmb = int(len(layers_sizes) / 2)
        self.numpy_rng = numpy_rng
        self.output_folder = output_folder
        if logger == None:
            self.logger = mylogger(output_folder + '/log.log')
        else:
            self.logger = logger
            
        assert self.n_layers > 0
        self.p = p
        self.sigma = sigma
            
        if not theano_rng:
            self.theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        else:
            self.theano_rng = theano_rng 
        self.x = T.matrix(name='x')
        
        for i in xrange(self.n_layers_rmb):
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.rbm_layers[-1].get_active()
                
            model_file = '%s/L%d.mat' % (output_folder, i)                
            if os.path.isfile(model_file):  # this layer has been trained
                w, hbias, vbias = self.load_RMB_model(i)                
                rbm_layer = GBRBM(numpy_rng=numpy_rng, input=layer_input, n_visible=layers_sizes[i], n_hidden=layers_sizes[i + 1],
                                W=w, hbias=hbias, vbias=vbias)
                self.rbm_train_flags.append(True)  # set the flag                
            else:
                rbm_layer = GBRBM(numpy_rng=numpy_rng, input=layer_input, n_visible=layers_sizes[i], n_hidden=layers_sizes[i + 1])
                self.rbm_train_flags.append(False)  # set the flag
            self.rbm_layers.append(rbm_layer)
            
        finetune_file = '%s/DAE%s.mat' % (output_folder, self.get_model_file())
        if os.path.isfile(finetune_file):  # trained
            self.finetune_train_flag = True
            
            
    def log(self, msg):
        self.logger.log(msg)
    
    
    def save_RBM_model(self, layer_idx, file_name=None):
        rbm_layer = self.rbm_layers[layer_idx]
        w = rbm_layer.W.get_value(borrow=True)
        hbias = rbm_layer.hbias.get_value(borrow=True)
        vbias = rbm_layer.vbias.get_value(borrow=True)
        if file_name == None:
            file_name = '%s/L%g.mat' % (self.output_folder, layer_idx)
        savemat(file_name, {'w':w, 'hbias':hbias, 'vbias':vbias})
        self.log(file_name + ' saved')

        
    def load_RMB_model(self, layer_idx, shared=1):
        file_name = '%s/L%g.mat' % (self.output_folder, layer_idx)
        self.log("Load mat file: " + file_name)
        dataset = io.loadmat(file_name)
        w = dataset['w']
        hbias = dataset['hbias']
        hbias = hbias.reshape(hbias.shape[1])
        vbias = dataset['vbias']
        vbias = vbias.reshape(vbias.shape[1])
        if shared == 1:
            w_shared = theano.shared(numpy.asarray(w, dtype=theano.config.floatX), borrow=True)
            hbias_shared = theano.shared(numpy.asarray(hbias, dtype=theano.config.floatX), borrow=True)
            vbias_shared = theano.shared(numpy.asarray(vbias, dtype=theano.config.floatX), borrow=True)
            ret_val = [w_shared, hbias_shared, vbias_shared]
        else:
            ret_val = [w, hbias, vbias]
        return ret_val
    
         
    def init_sigmoid_layers(self, pre_mat_file=None):
        # sigmoid layers
        p_idx = (self.n_layers - 1) / 2
        self.sigmoid_layers = []
        self.params = []
        self.delta_params = []
         
        if pre_mat_file != None:
            w_list, b_list = self.load_model_mat(pre_mat_file)
#             set_trace()
            assert len(w_list) == len(b_list)
            assert len(w_list) == self.n_layers - 1
            
        for i in xrange(self.n_layers - 1):
            if i == 0:
                layer_input = self.x
            elif(i == p_idx - 1) and (self.p == -1) and (self.sigma != 0):  # add gaussian noise
                gaussian_data = self.theano_rng.normal(size=self.sigmoid_layers[-1].output.shape, std=self.sigma,
                                                    dtype=theano.config.floatX)
                layer_input = self.sigmoid_layers[-1].output + gaussian_data                
                self.log('coding layer: add Gaussian noise')
            elif(i == p_idx) and (self.p == -1) and (self.sigma == 0):  # binary
                layer_input = (self.sigmoid_layers[-1].output >= 0.5)
                self.log('coding layer: force to binary')                
            else:
                layer_input = self.sigmoid_layers[-1].output
            
            if i == self.n_layers - 2:
                activation = None
            else:
                activation = T.nnet.sigmoid
                
            if pre_mat_file == None: #initialize from rbm layers    
                if i < (self.n_layers - 1) / 2:
                    rbm_layer = self.rbm_layers[i]
                    n_in = rbm_layer.n_visible
                    n_out = rbm_layer.n_hidden
                    W = rbm_layer.W
                    b = rbm_layer.hbias
                else:  # inverse the layer
                    rbm_layer = self.rbm_layers[self.n_layers - 2 - i]
                    n_in = rbm_layer.n_hidden
                    n_out = rbm_layer.n_visible
                    W = theano.shared(numpy.asarray(rbm_layer.W.T.eval(), dtype=theano.config.floatX), borrow=True)
                    b = rbm_layer.vbias
            else: # initialize from the data file                
                n_in = w_list[i].shape[0]
                n_out = w_list[i].shape[1]                
                W = theano.shared(numpy.asarray(w_list[i], dtype=theano.config.floatX), borrow=True)
                b = theano.shared(numpy.asarray(b_list[i], dtype=theano.config.floatX), borrow=True)
#             set_trace()
            hidden_layer = HiddenLayer(rng=self.numpy_rng,
                                        input=layer_input,
                                        n_in=n_in,
                                        n_out=n_out,
                                        W=W,
                                        b=b,
                                        activation=activation)
            self.sigmoid_layers.append(hidden_layer)
            self.params.extend(hidden_layer.params)            
            self.delta_params.extend(hidden_layer.delta_params)
        self.errors = (self.sigmoid_layers[-1].output - self.x) ** 2                
        self.save_model_mat('%s/DAE_pre%s.mat' % (self.output_folder, self.get_model_file()))
    
    def save_model(self, file_name=None):
        if file_name == None:
            file_name = '%s/DAE%s.mat' % (self.output_folder, self.get_model_file())
        self.save_model_mat(file_name)
    
    def load_model_mat(self, file_name):
        w_list = []
        b_list = []
        self.log("Load mat file: " + file_name)
        params = io.loadmat(file_name)['params']
        paramSize = params.shape[1]        
        for i in xrange(paramSize/2):
            w_list.append(params[0, i*2])
            b = params[0, i*2+1]            
            b_list.append(b.reshape(b.shape[1]))
        return (w_list, b_list)
    
    def get_model_file(self):
                
        if self.p != 0:
            file_str = '_p%g_s%g' % (self.p, self.sigma)
        else:
            file_str = ''
        return file_str.strip()
    
    
    def save_model_mat(self, file_name):
        params = []
        for item in self.params:            
            param = item.get_value(borrow=True)            
            params.append(param)
#         set_trace()
        savemat(file_name, {'params':params})
        self.log(file_name + ' saved')
               
        
    def get_gaussian_cost(self, x, p=0, sigma=1, mu=0.5):
        x_mean = T.mean(x, axis=0)
        cost = (1 / (sigma * T.sqrt(2 * numpy.pi)) * T.exp(-(x_mean - mu) ** 2 / (2 * sigma ** 2)))
        return p * cost
    
    
    def get_square_cost(self, x, p):        
        x_mean = T.mean(x, axis=0)
        cost = -(x_mean - 0.5) ** 2 + 0.5        
        return p * cost
    
    
    def get_cost(self, p=0, sigma=1):
        # the last layer        
        z = self.sigmoid_layers[-1].output
        L = T.sum((self.x - z) ** 2, axis=1)   
        p_idx = len(self.sigmoid_layers) / 2 - 1  # penalty layer, the middle layer
        if p == 0:
            cost = T.mean(L)
        elif (p != 0) and (sigma == 0):  # for square penalty
            square_cost = self.get_square_cost(self.sigmoid_layers[p_idx].output, p)
            cost = T.mean(L) + T.mean(square_cost)
        elif(p != 0) and (sigma != 0):  # for Gaussian penalty                        
            gaussian_cost = self.get_gaussian_cost(self.sigmoid_layers[p_idx].output, p, sigma)
            cost = T.mean(L) + T.mean(gaussian_cost)            
        return cost
    
    def get_cost_updates(self, learning_rate=0.1, p=0, sigma=1, momentum=0.5):
        cost = self.get_cost(p, sigma)
        gparams = T.grad(cost, self.params)        
        updates = collections.OrderedDict()
        for dparam, gparam in zip(self.delta_params, gparams):
            updates[dparam] = momentum * dparam - gparam * learning_rate
        for dparam, param in zip(self.delta_params, self.params):
            updates[param] = param + updates[dparam]            
        return (cost, updates)
 
    def pretraining_functions(self, train_set, batch_size):
        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')       
        pretrain_fns = []
        for i in range(len(self.rbm_layers)):            
            r_cost, fe_cost, updates = self.rbm_layers[i].get_cost_updates(batch_size=batch_size, lr=learning_rate)
            fn = theano.function(inputs=[index,
                              theano.Param(learning_rate, default=0.0001)],
                              outputs=[r_cost, fe_cost],
                              updates=updates,
                              givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}
                              )           
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    
    def build_finetune_functions(self, datasets, batch_size, pre_mat_file=None):
        
        if pre_mat_file != None:
            pre_mat_file = self.output_folder + '/' + pre_mat_file
        self.init_sigmoid_layers(pre_mat_file)  # firstly, initial the sigmoid layer
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
        cost = self.get_cost()
        test_fn = theano.function(
                inputs=[index],
                outputs=cost,
                givens={self.x: test_set[index * batch_size: (index + 1) * batch_size]}
            )
        valid_fn = theano.function(
                inputs=[index],
                outputs=cost,
                givens={self.x: valid_set[index * batch_size: (index + 1) * batch_size]}
            )
        
        # Create a function that scans the entire validation set
        def valid_model():
            return [test_fn(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_model():
            return [valid_fn(i) for i in xrange(n_test_batches)]
        
        return train_fn, valid_model, test_model
    
def train_DAE(datasets, layers_sizes, output_folder, p, sigma):
    
    train_set, valid_set, test_set = datasets
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    pre_epochs = 10
    fine_epochs = 500
    finetune_lr = 0.01
    batch_size = 100
    down_epoch = 50
    save_epoch = 10
    
    MAX_TRAIN_SET = 100000
    train_set_N = train_set.get_value(borrow=True).shape[0]
    n_train_batches = train_set_N / batch_size if train_set_N < MAX_TRAIN_SET else MAX_TRAIN_SET / batch_size    
    logger = mylogger('%s/log_p%g_s%g.log' % (output_folder, p, sigma))
    logger.log('train_set size: %d'%(n_train_batches*batch_size))
    logger.log('layer size: ' + ''.join(['%d ' % i for i in layers_sizes]).strip())
    rng = numpy.random.RandomState(123)
    dae = deep_auto_encoder(numpy_rng=rng, layers_sizes=layers_sizes, output_folder=output_folder, p=p, sigma=sigma, logger=logger)
    
    pretraining_fns = dae.pretraining_functions(train_set, batch_size)
    start_time = strict_time()    
    logger.log('pre_training...')    
    for i in xrange(len(dae.rbm_layers)):        
        if dae.rbm_train_flags[i] == True:
            logger.log('L%d trained' % (i))
            continue
        pretrain_lr = 0.0001
        logger.log('training L%d ...' % (i))
        logger.log('learning rate:%g' % (pretrain_lr))
        for epoch in xrange(1, pre_epochs + 1):
            epoch_time_s = strict_time()
            r_c, fe_c = [], []
            for batch_index in xrange(n_train_batches):
                [reconstruction_cost, free_energy_cost] = pretraining_fns[i](index=batch_index, lr=pretrain_lr)                
                r_c.append(reconstruction_cost)
                fe_c.append(free_energy_cost)
            logger.log('epoch %d, r_cost %.5f, fe_c %.5f, took %f seconds ' % (epoch, numpy.mean(r_c), numpy.mean(fe_c), (strict_time() - epoch_time_s)))
        dae.save_RBM_model(i)
    
#     train_fn, valid_model, test_model = dae.build_finetune_functions(datasets, batch_size)#must be here! after pre_train

    logger.log('fine tuning...')
    if dae.finetune_train_flag == True:
        logger.log('DAE trained')
        return 
    logger.log('learning rate:%g, p:%g, sigma:%g' % (finetune_lr, p, sigma))    
    train_fn, valid_model, test_model = dae.build_finetune_functions(datasets, batch_size, 'DAE_p-1_s0.3.mat')  # must be here! after pre_train
#     train_fn, valid_model, test_model = dae.build_finetune_functions(datasets, batch_size)  # must be here! after pre_train
    for epoch in xrange(1, fine_epochs + 1):
        epoch_time_s = strict_time()
        c = []
        for minibatch_index in xrange(n_train_batches):
            err = train_fn(minibatch_index, finetune_lr)
            c.append(err)        
#         set_trace()
        logger.log('epoch %d, cost %.5f, took %.2f seconds' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
        if epoch % down_epoch == 0:
            finetune_lr = 0.9 * finetune_lr
            logger.log('learning rate:%g' % (finetune_lr))
        if epoch % save_epoch == 0:
            dae.save_model()
    dae.save_model()
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))

  
def strict_time():
    if sys.platform == "win32":
        return strict_time()
    else:
        return time.time()
    
from data_mat import load_TIMIT, load_model_mat, save_model_mat
from logger import log_init, mylogger
from tools import set_trace

def train_codec():
        
    output_folder = '600_mfs'
    data_file = 'mfs_train_(N4).mat'    
#     if output_folder.find('300') != -1:
#         if output_folder.find('rbm') != -1:
#             data_file = '300bps/rbm_TIMIT_train_split.mat'
#         elif output_folder.find('mfs') != -1:
#             data_file = 'mfs_train_(N8).mat'
#         else:
#             data_file = '300bps/TIMIT_train_split.mat'        
#     elif output_folder.find('600') != -1:
#         data_file = '600bps/TIMIT_train_dr1_dr4_split.mat'
#     elif output_folder.find('1200') != -1:
#         data_file = '1200bps/TIMIT_train_dr1_dr2_split.mat'
#     elif output_folder.find('2400') != -1:
#         data_file = '2400bps/TIMIT_train_dr1_dr2_split.mat'
        
    datasets = load_TIMIT(data_file)
    train_set, valid_set, test_set = datasets
    p_list = [-1]
    sigma_list = [0]
    input_dim = train_set.get_value(borrow=True).shape[1]    
    layers_sizes = [input_dim, 2000, 1000, 500, 108, 500, 1000, 2000, input_dim]
#     layers_sizes = [input_dim,2000,2000,54,2000,2000,input_dim]
#     layers_sizes = [input_dim,500,54,500,input_dim]
    for p in p_list:
        for sigma in sigma_list:
            train_DAE(datasets, layers_sizes, output_folder, p, sigma)    
        
if __name__ == '__main__':
    
    train_codec()
    
