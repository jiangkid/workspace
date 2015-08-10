import os
import numpy
import theano
from scipy import io
import platform
from dA import dA
import pylab
from tools import set_trace

def load_TIMIT(mat_file="TIMIT_train_dr1_split.mat", shared=1):
    pre_path = "../data/"
    if platform.system() == "Linux":
        pre_path = "/data/mfcc_coding/"
    data_file = pre_path + mat_file
    print "mat file:", data_file
    dataset = io.loadmat(data_file)
    train_set = dataset['train_set']
    valid_set = dataset['valid_set']    
    test_set = dataset['test_set']  
    if shared == 1:
        shared_train = theano.shared(numpy.asarray(train_set, dtype=theano.config.floatX), borrow=True)    
        shared_valid = theano.shared(numpy.asarray(valid_set, dtype=theano.config.floatX), borrow=True)
        shared_test = theano.shared(numpy.asarray(test_set, dtype=theano.config.floatX), borrow=True)    
        ret_val = [shared_train, shared_valid, shared_test]
    else:
        ret_val = [train_set, valid_set, test_set]
    return ret_val
        

def load_model_mat(file_name=None, shared=1):
    print "Load mat file:", file_name 
    dataset = io.loadmat(file_name)
    w = dataset['w']
    b = dataset['b']
    b = b.reshape(b.shape[1])
    b_prime = dataset['b_prime']
    b_prime = b_prime.reshape(b_prime.shape[1])
    if shared == 1:
        w_shared = theano.shared(numpy.asarray(w, dtype=theano.config.floatX), borrow=True)
        b_shared = theano.shared(numpy.asarray(b, dtype=theano.config.floatX), borrow=True)
        b_prime_shared = theano.shared(numpy.asarray(b_prime, dtype=theano.config.floatX), borrow=True)
        ret_val = [w_shared, b_shared, b_prime_shared]
    else:
        ret_val = [w, b, b_prime]
    return ret_val


def save_model_mat(da, file_name='da', output_folder='auto_encoder_out'):
    os.chdir(output_folder)
    w = da.W.get_value(borrow=True)
    b = da.b.get_value(borrow=True)
    b_prime = da.b_prime.get_value(borrow=True)
    file_name = file_name + '.mat'
    io.savemat(file_name, {'w':w,'b':b,'b_prime':b_prime})
    print file_name, 'saved'
    os.chdir('../')
    
    
if __name__ == '__main__':
    train_set, valid_set, test_set = load_TIMIT()
    