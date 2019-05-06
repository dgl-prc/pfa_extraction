
import abc
class DataFactory(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def sequence2tensor(self,sequence,input_tensor_dim):
        '''Transform an input sequence into a tensor, the shape of which should be len(sequence) x 1 x input_tensor_dim'''
        return

    @abc.abstractmethod
    def label2tensor(self,label,num_class):
        '''Transform a label into a tensor, the shape of which should be  1 x num_class'''
        return

    @abc.abstractmethod
    def load_data(self,*inputs):
        pass