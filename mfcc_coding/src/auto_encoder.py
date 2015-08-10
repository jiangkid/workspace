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
                 input=None,
                 layers_sizes=[129, 500, 256, 500, 129],
                 w_list=None,
                 b_list=None):
        self.sigmoid_layers = []
        self.params = []
        self.n_layers = len(layers_sizes)
        
        assert self.n_layers > 0
        assert (self.n_layers - 1) == len(w_list)  # 0 layer is the input layer, no weight, no bias
        assert len(w_list) == len(b_list)

        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        if input is None:
            self.x = T.matrix(name='input')
        else:
            self.x = input
#         set_trace()
        for i in xrange(1, self.n_layers):  # 0 layer is the input layer
            if i == 1:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=layers_sizes[i - 1],
                                        n_out=layers_sizes[i],
                                        W=w_list[i - 1],
                                        b=b_list[i - 1],
                                        activation=T.nnet.sigmoid)
            self.sigmoid_layers.append(sigmoid_layer)
            self.params.extend(sigmoid_layer.params)
        
        self.errors = (self.sigmoid_layers[-1].output - self.x) ** 2 
    
    
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
    
    
    def get_cost(self, p=0, sigma=1):
        # the last layer        
        z = self.sigmoid_layers[-1].output
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        if p == 0:
            cost = T.mean(L)
        else:
            p_idx = len(self.sigmoid_layers)/2 - 1 #penalty layer, the middle layer            
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
    
    
    def get_hidden_values(self, x, w, b, binary=0):
        """ Computes the values of the hidden layer """        
        hidden = T.nnet.sigmoid(T.dot(x, w) + b)
        if binary == 0:
            ret_val = hidden
        if binary == 1:
            ret_val = hidden>=0.5            
        return ret_val
        
        
    def get_cost_b(self):
        """get the cost with the binary value of the middle layer"""
        m_idx = len(self.sigmoid_layers)/2 - 1 #the middle layer
        z = self.x
        for idx in xrange(len(self.sigmoid_layers)):
            if idx != m_idx:
                z = self.get_hidden_values(z,
                                           self.sigmoid_layers[idx].W,
                                           self.sigmoid_layers[idx].b)
            else:
                z = self.get_hidden_values(z,
                                           self.sigmoid_layers[idx].W,
                                           self.sigmoid_layers[idx].b, 1)
        L = -T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)        
        cost = T.mean(L) 
        return cost
    
    
    def build_finetune_functions(self, datasets, batch_size, learning_rate, p, sigma):
        
        train_set, valid_set, test_set = datasets        
        n_valid_batches = valid_set.get_value(borrow=True).shape[0] / batch_size
        n_test_batches = test_set.get_value(borrow=True).shape[0] / batch_size
        
        
        sae_cost, sae_updates = self.get_cost_updates(learning_rate=learning_rate,
                                                      p=p,
                                                      sigma=sigma)
    #     set_trace()
        index = T.lscalar('index')  # index to a [mini]batch
        
        train_fn = theano.function(
                inputs=[index,
                        learning_rate],
                outputs=sae_cost,
                updates=sae_updates,
                givens={self.x: train_set[index * batch_size: (index + 1) * batch_size]}
            )
        cost_b = self.get_cost_b()
        test_fn = theano.function(
                inputs=[index],
                outputs=cost_b,
                givens={self.x: test_set[index * batch_size: (index + 1) * batch_size]}
            )
        valid_fn = theano.function(
                inputs=[index],
                outputs=cost_b,
                givens={self.x: valid_set[index * batch_size: (index + 1) * batch_size]}
            )
        
        # Create a function that scans the entire validation set
        def valid_model():
            return [test_fn(i) for i in xrange(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_model():
            return [valid_fn(i) for i in xrange(n_test_batches)]
        
        return train_fn, valid_model, test_model
    
        
def strict_time():
    if sys.platform == "win32":
        return strict_time()
    else:
        return time.time()
    
from data_mat import load_TIMIT, load_model_mat, save_model_mat
from logger import log_init, mylogger
from tools import set_trace

data_file_list = (None, '2400bps/TIMIT_train_dr1_split.mat', None, None, None, 
                  'TIMIT_train_dr1.dr2_(N5)_split.mat', None, None, None, None, None,
                  'TIMIT_train_(N11)_split.mat')
layer_list = (None, (70, 500, 54, None), None, None, None,
              (645, 1000, 512, None), None, None, None, None, None,
              (1419, 2000, 1024, None)) 

def train_auto_encoder(L=1, N=1):    
#     N = 1 #frame number
#     L = -1 #layer
    param = {}
    
#     if N == 1:
#         data_file = '600bps/TIMIT_train_dr1_dr4_split.mat'
#         param['pretrain_lr'] = 0.05
#         param['down_epoch'] = 100
#         param['layers_sizes'] = [280, 2000, 500, 54, 500, 500, 280]
#     elif N == 5:
#         param['pretrain_lr'] = 0.05
#         param['down_epoch'] = 50
#         param['layers_sizes'] = [645, 1000, 512, 1000, 512, 645]     
#     elif N == 11:      
#         param['pretrain_lr'] = 0.02
#         param['down_epoch'] = 30
#         param['layers_sizes'] = [1419, 2000, 1024, 2000, 1419]        
    
    data_file = '300bps/TIMIT_train_split.mat'
    param['output_folder'] = '300bps'
    param['pretrain_lr'] = 0.05
    param['down_epoch'] = 100
    param['layers_sizes'] = [560,2000,1000,500,54,500,1000,2000,560]
    param['n_hidden'] = param['layers_sizes'][L]
    
    datasets = load_TIMIT(data_file)
    train_set, valid_set, test_set = datasets
    
    p_list = [0]
    sigma_list = [1]
    if L == 1:
        for p in p_list:      
            for sigma in sigma_list:
                param['item_str'] = 'L1_p%g_s%g' % (p, sigma)
                auto_encoder_Lx(train_set, p, sigma, param)
    elif L == 2:
        L1_p = 0; L1_s = 1
        model_str = 'L1_p%g_s%g'%(L1_p, L1_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)        
        for p in p_list:
            for sigma in sigma_list:
#                 param['item_str'] = 'L2_p%g_s%g_(%s)' % (p, sigma, model_str)
                param['item_str'] = 'L2_p%g_s%g' % (p, sigma)
                auto_encoder_Lx(train_set, p, sigma, param)    
    elif L == 3:
        L1_p = 0; L1_s = 1
        model_str = 'L1_p%g_s%g'%(L1_p, L1_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)        
        L2_p = 0; L2_s = 1        
        model_str = 'L2_p%g_s%g'%(L2_p, L2_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)        
        for p in p_list:
            for sigma in sigma_list:
                param['item_str'] = 'L3_p%g_s%g' % (p, sigma)
                auto_encoder_Lx(train_set, p, sigma, param)
    elif L == 4:
        L1_p = 0; L1_s = 1
        model_str = 'L1_p%g_s%g'%(L1_p, L1_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)
        L2_p = 0; L2_s = 1        
        model_str = 'L2_p%g_s%g'%(L2_p, L2_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)
        L3_p = 0; L3_s = 1        
        model_str = 'L3_p%g_s%g'%(L3_p, L3_s)
        model_file = '%s/%s.mat' %(param['output_folder'], model_str)
        train_set = get_hidden(train_set, model_file)          
        for p in p_list:
            for sigma in sigma_list:
                param['item_str'] = 'L4_p%g_s%g' % (p, sigma)
                auto_encoder_Lx(train_set, p, sigma, param)
    elif L == -1: #finetune
        for p in p_list:      
            for sigma in sigma_list:
                L1_str = 'L1_p0_s1'
                L2_str = 'L2_p0_s1'
                L3_str = 'L3_p0_s1'
                L4_str = 'L4_p0_s1'
                param['item_str'] = 'SAE_p0_s1'
                param['L1_file'] = '%s/%s.mat' %(param['output_folder'], L1_str)
                param['L2_file'] = '%s/%s.mat' %(param['output_folder'], L2_str)
                param['L3_file'] = '%s/%s.mat' %(param['output_folder'], L3_str)
                param['L4_file'] = '%s/%s.mat' %(param['output_folder'], L4_str)  
                auto_encoder_finetune(datasets, 0, 1, param)
    return

   
def auto_encoder_finetune(datasets=None, p=0, sigma=1, param=None, training_epochs=1000):
    if datasets == None:
        datasets = load_TIMIT()
    train_set, valid_set, test_set = datasets
         
    def get_shared(x):
        return theano.shared(numpy.asarray(x, dtype=theano.config.floatX), borrow=False)            
    
    layers_sizes = param['layers_sizes']
    L1_file = param['L1_file']
    L2_file = param['L2_file']
    L3_file = param['L3_file']
    L4_file = param['L4_file']
    output_folder = param['output_folder']
    item_str = param['item_str']
    
    valid_flag = 0
    finetune_lr = 0.1
    batch_size = 100
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar('index') 
    learning_rate = T.scalar('lr') 
    x = T.matrix('x')

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
         
#     pickle_lst = [100,200,300,400,500,600,700,800,900,1000]
    L1_w, L1_b, L1_b_prime = load_model_mat(file_name=L1_file, shared=0)    
    L2_w, L2_b, L2_b_prime = load_model_mat(file_name=L2_file, shared=0)
    L3_w, L3_b, L3_b_prime = load_model_mat(file_name=L3_file, shared=0)
    L4_w, L4_b, L4_b_prime = load_model_mat(file_name=L4_file, shared=0)
    w_list = [get_shared(L1_w), get_shared(L2_w), get_shared(L3_w), get_shared(L4_w), 
              get_shared(L4_w.T), get_shared(L3_w.T), get_shared(L2_w.T), get_shared(L1_w.T)]    
    b_list = [get_shared(L1_b), get_shared(L2_b), get_shared(L3_b), get_shared(L4_b), 
              get_shared(L4_b_prime), get_shared(L3_b_prime), get_shared(L2_b_prime), get_shared(L1_b_prime)]
      
    rng = numpy.random.RandomState(123)
    sae = SAE(numpy_rng=rng,
              input=x,
              layers_sizes=layers_sizes,
              w_list=w_list,
              b_list=b_list)
    print '... building the model'
    train_fn, valid_model, test_model = sae.build_finetune_functions(datasets, batch_size, learning_rate, p, sigma)
    print '... training'    

    logger = mylogger(output_folder + '/' + item_str + '.log')
    logger.log('p:%g, sigma:%g, learning rate:%g' % (p, sigma, finetune_lr))
    
    #===========================================================================
    # start training
    #===========================================================================

    patience = 100 * n_train_batches  # look as this many examples regardless
    patience_increase = 2.  # wait this much longer when a new best is found
    improvement_threshold = 0.999  # a relative improvement of this much is considered significant
    validation_frequency = min(n_train_batches, patience / 2)
        
    best_validation_loss = numpy.inf
    
    test_score = 0.
    start_time = strict_time()

    done_looping = False
    epoch = 0;best_epoch = 0
        
    while (epoch < training_epochs):
        epoch = epoch + 1
        epoch_time_s = strict_time()
        c = []
        for minibatch_index in xrange(n_train_batches):
            err = train_fn(minibatch_index,finetune_lr)
#             err = 0
            c.append(err)
        logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))        
        if epoch % 100 == 0:
            finetune_lr = 0.8 * finetune_lr            
            logger.log('learning rate: %g' % (finetune_lr))
        if valid_flag == 0:
            continue
        validation_losses = numpy.mean(valid_model())
        logger.log('valid %.5f' % (validation_losses))
        if validation_losses < best_validation_loss:
            best_validation_loss = validation_losses
            best_epoch = epoch
            # test it on the test set
            test_losses = numpy.mean(test_model())
            logger.log('test %.5f' % (test_losses))
            sae.save_model_mat(output_folder + '/' + item_str + '.mat')
#     logger.log('best validation %.5f, test error %.5f, on epoch %d'%(best_validation_loss, test_losses, best_epoch))
    sae.save_model_mat(output_folder + '/' + item_str + '.mat')
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))
    return

    for epoch in xrange(1, training_epochs + 1):
        # go through trainng set                
        c = []
        epoch_time_s = strict_time()
        for batch_index in xrange(n_train_batches):            
            err = train_fn(batch_index, finetune_lr)
#             err = 0
            c.append(err)
        logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
        if epoch % 100 == 0:
            finetune_lr = 0.8 * finetune_lr            
            logger.log('learning rate: %g' % (finetune_lr)) 

    sae.save_model_mat(output_folder + '/' + item_str + '.mat')
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))
    
    
def get_L1_out(train_set=None, model_file=None, N=1, shared=1):        
    if model_file == None:
        model_file = 'L1_p0_s1'
    output_folder = 'auto_encoder_out_N%s' %(N)
    model_w, model_b, model_b_prime = load_model_mat(file_name=model_file, output_folder=output_folder, shared=0)
#     set_trace()
    hidden_value = 1 / (1 + numpy.exp(-(numpy.dot(train_set, model_w) + model_b)))
    if shared == 1:
        hidden_value = theano.shared(numpy.asarray(hidden_value, dtype=theano.config.floatX), borrow=True)
    return hidden_value

def get_hidden(train_set=None, model_file=None, shared=1):        
    if model_file == None:
        model_file = 'L1_p0_s1'
    model_w, model_b, model_b_prime = load_model_mat(model_file)
    n_visible,n_hidden = model_w.get_value(borrow=True).shape
    
    rng = numpy.random.RandomState(123)
    da = dA(numpy_rng=rng, input=train_set, n_visible=n_visible, n_hidden=n_hidden, W=model_w, bhid=model_b, bvis=model_b_prime)
#     set_trace()
    hidden_value = da.get_active().eval()
    if shared == 1:
        hidden_value = theano.shared(numpy.asarray(hidden_value, dtype=theano.config.floatX), borrow=True)
    return hidden_value

    
def auto_encoder_Lx(train_set=None, p=0, sigma=1, param=None, training_epochs=1000):
    #===========================================================================
    # train_set: the training data
    # p: penalty; sigma: sigma for Gaussian 
    # N: number of frame; 
    # item_str: string for model and log file 
    # training_epochs: training epochs
    #===========================================================================
    batch_size = 100    
    output_folder = param['output_folder']    
    item_str = param['item_str']
    pretrain_lr = param['pretrain_lr']
    down_epoch = param['down_epoch']
    n_hidden = param['n_hidden']
    
    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set.get_value(borrow=True).shape[0] / batch_size
    n_visible = train_set.get_value(borrow=True).shape[1]
    # start-snippet-2
    # allocate symbolic variables for the data
    index = T.lscalar('index')  # index to a [mini]batch
    learning_rate = T.scalar('lr')  # learning rate to use
    x = T.matrix('x')  # the data is presented as rasterized images
    # end-snippet-2

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
#     pickle_lst = [1, 5, 10, 50, 100, 500, 1000]
    rng = numpy.random.RandomState(123)
    da = dA(numpy_rng=rng, input=x, n_visible=n_visible, n_hidden=n_hidden)            
    cost, updates = da.get_cost_updates_p(learning_rate=learning_rate, p=p, sigma=sigma)             
    print '... building the model'         
    train_fn = theano.function(
        inputs=[index,
                learning_rate],
        outputs=cost,
        updates=updates,
        givens={x: train_set[index * batch_size: (index + 1) * batch_size]}
    )            
    print '... training'
    start_time = strict_time()    
    
    logger = mylogger(output_folder + '/' + item_str + '.log')    
    logger.log('p:%g, sigma:%g, learning rate:%g' % (p, sigma, pretrain_lr))
    for epoch in xrange(1, training_epochs + 1):
        # go through trainng set                
        c = []
        epoch_time_s = strict_time()
        for batch_index in xrange(n_train_batches):            
            err = train_fn(batch_index, pretrain_lr)
#             err = 0
            c.append(err)        
        logger.log('Training epoch %d, cost %.5f, took %f seconds ' % (epoch, numpy.mean(c), (strict_time() - epoch_time_s)))
        if epoch % down_epoch == 0:
            pretrain_lr = 0.8 * pretrain_lr            
            logger.log('learning rate: %g' % (pretrain_lr))  
#         if epoch in pickle_lst:  
#             file_name = "%s_epoch_%d" % (item_str,epoch)
#             save_model_mat(da, file_name)
#             logger.info(file_name+'.mat saved')
    da.save_model_mat(output_folder + '/' + item_str + '.mat')
#     da.save_model_mat("%s_(%d)" %(item_str,training_epochs), output_folder)
    logger.log('ran for %.2f m ' % ((strict_time() - start_time) / 60.))
    
 
import pylab
def test_model(batch_size=100, file_name='da.pkl'):
    
#     datasets = load_data(dataset)
    print '...loading data'
    datasets = load_TIMIT()
    train_set, valid_set, test_set = datasets

    print '...building model'
    
    pickle_lst = [1000]  # , 500, 1000
#     pickle_lst = [1, 10]
    for epoch in pickle_lst:
        print 'epoch: ', epoch
        file_name = "da_epoch_%d" % (epoch)        
        w, b, b_prime = load_model_mat(file_name)
        # allocate symbolic variables for the data
        index = T.lscalar()  # index to a [mini]batch
    
        # generate symbolic variables for input (x and y represent a
        # minibatch)
        x = T.matrix('x')  # data, presented as rasterized images        
        
        rng = numpy.random.RandomState(123)
        theano_rng = RandomStreams(rng.randint(2 ** 30))
        da = dA(
            numpy_rng=rng,
            theano_rng=theano_rng,
            input=x,
            n_visible=129,
            n_hidden=500,
            W=w,
            bhid=b,
            bvis=b_prime
        )
            
#         test_fun = theano.function(
#             inputs=[index],
#             outputs=da.get_reconstructed_out(),
#             givens={
#                 x: test_set_x[index * batch_size:(index + 1) * batch_size]
#             }
#         )
        get_outputs = theano.function(
            inputs=[index],
            outputs=da.get_active(),
            givens={
                x: test_set[index * batch_size:(index + 1) * batch_size]
            }
        )
        
        index = 1
        hidden_value = get_outputs(index)
        plot_data = test_set.get_value(borrow=True)[index * batch_size:(index + 1) * batch_size]
        pylab.figure(); pylab.hist(plot_data.reshape(plot_data.size, 1), 50);
        pylab.figure();pylab.plot(numpy.mean(plot_data, axis=0), '*');pylab.xlim(0, 128);pylab.ylim(0, 1);
        pylab.figure();pylab.hist(hidden_value.reshape(hidden_value.size, 1), 50);
        pylab.figure();pylab.plot(numpy.mean(hidden_value, axis=0), '*');pylab.ylim(0, 1);
        pylab.show()
        set_trace()
#         pylab.title(epoch)
    pylab.show()
        
if __name__ == '__main__':
#     test_SdA()
#     print os.getcwd()
#     auto_encoder(training_epochs=10)
    L_list = [-1]
    for L in L_list:        
        train_auto_encoder(L=L)       
    
#     view_data()
#     test_model()

