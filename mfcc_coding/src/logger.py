#encoding:utf-8
'''
Created on 

@author: jiangwenbin
'''
import sys
import logging
import logging.handlers

class mylogger(logging.Logger):
    def __init__(self, LOG_FILE='temp.log', print_flag=1):
        self.print_flag = print_flag
        handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)
        fmt = '%(asctime)s - %(message)s'        
        formatter = logging.Formatter(fmt)
        handler.setFormatter(formatter)        
        self.logger = logging.Logger('tst')
        self.logger.addHandler(handler)  # 
        self.logger.setLevel(logging.DEBUG)
        print >> sys.stderr, 'log file: '+LOG_FILE    
        self.logger.info('------log init------')
        
    def log(self, msg):
        if self.print_flag == 1:
            print msg
        self.logger.info(msg)

def log_init(LOG_FILE='temp.log'):    
    handler = logging.handlers.RotatingFileHandler(LOG_FILE, maxBytes=1024 * 1024, backupCount=5)  #
    fmt = '%(asctime)s - %(message)s'
    
    formatter = logging.Formatter(fmt)  #
    handler.setFormatter(formatter)  #
    
#     logger = logging.getLogger('tst')  # 
    logger = logging.Logger('tst');
    logger.addHandler(handler)  # 
    logger.setLevel(logging.DEBUG)
    print >> sys.stderr, 'log file: '+LOG_FILE    
    logger.info('------log init------')
    
    return logger


def test_mylogger():
    import numpy
    logger = mylogger('test_mylogger1.log')
    epoch_list = numpy.arange(10)    
    c = numpy.arange(10)
    for epoch in epoch_list:
        msg = 'Training epoch %d, cost %.5f' %(epoch, numpy.mean(c))
        logger.log(msg)
    del logger
    
    logger = mylogger('test_mylogger2.log')
    epoch_list = numpy.arange(100)    
    c = numpy.arange(100)
    for epoch in epoch_list:
        msg = 'Training epoch %d, cost %.5f' %(epoch, numpy.mean(c))
        logger.log(msg)
        
    
def test_logger():
    import numpy
    logger = log_init('test.log')
    epoch_list = numpy.arange(10)    
    c = numpy.arange(10)
    for epoch in epoch_list:
        msg = 'Training epoch %d, cost %.5f' %(epoch, numpy.mean(c))
        logger.info(msg)
    del logger
    
    logger = log_init('test2.log')
    epoch_list = numpy.arange(100)    
    c = numpy.arange(100)
    for epoch in epoch_list:
        msg = 'Training epoch %d, cost %.5f' %(epoch, numpy.mean(c))
        logger.info(msg)
        
if __name__ == '__main__':
#     test_logger()
    test_mylogger()