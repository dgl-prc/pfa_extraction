import torch.nn as nn
import torch.nn.functional as F
import torch
from my_module import *
from torch.nn.utils.rnn import PackedSequence
import numpy as np
from data_factory.imdb_sentiment.imdb_data_process import MyString

class LSTM(MyModule):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(LSTM, self).__init__()
        self.i2h = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.h2o = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        '''
        :param input: (seq_len, batch, input_size)
        :param hn:  (num_layers * num_directions, batch, hidden_size):
        :param cn:  (num_layers * num_directions, batch, hidden_size)
        :return:
        '''
        output, (hn, cn) = self.i2h(inputs)
        return output, (hn, cn)

    def get_predict_trace(self, input_sequences):
        # PackedSequence is used to support batch training
        output, (hn, cn) = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0]
            pr_dstr = self.output_pr_dstr(output)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()  # (len_seq,1)
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")


class GRU(MyModule):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(GRU, self).__init__()
        self.i2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2o = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output, h_n = self.i2h(input)
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0]  # (len_seq,hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")

class GRU3(MyModule):
    def __init__(self,input_size, hidden_size, num_layers, output_size):
        super(GRU3, self).__init__()
        self.raw_input_size = input_size
        self.i2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.h_shape = (num_layers,1,hidden_size)

    def set_alphabet(self,alphabet):
        self.alphabet = alphabet

    def set_dataProcessor(self,dataProcessor):
        self.dataProcessor = dataProcessor

    def forward(self, input, hx=None):
        if hx is None:
            output, h_n = self.i2h(input)
        else:
            output, h_n = self.i2h(input, hx)
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0]  # (len_seq,hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")

    def classify_word(self, sequence, cuda_num=0,need_split=False):
        '''
        :param word:
        :return:
        '''
        tensor_sequence = self.dataProcessor.sequence2tensor(sequence, self.raw_input_size,need_split=need_split)
        if cuda_num >= 0:
            tensor_sequence = tensor_sequence.cuda(cuda_num)
        output, hx = self.forward(tensor_sequence)
        return self._classify_state(hx)

    def _classify_state(self, hx):
        '''
        make a prediction according to the hidden state
        :param state:
        :return:
        '''
        pr_dstr = self.output_pr_dstr(hx[-1])
        predicts = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
        return bool(predicts[0])


    def get_first_RState(self):
        '''
        returns a continuous vector representation of the network's initial state (an RState, as a list of floats)
        One example of a continuous vector representation of all of a GRU's hidden values would be the concatenation
        of the  h  vectors of each of its layers
        :return:
        '''
        hx = torch.zeros(self.h_shape)
        return self.get_next_RState(self.hx2list(hx),"$")

    def get_next_RState(self, h_t, char):
        '''
        given an RState, returns the next RState the network goes to on input character char
        :param h_t:  a list of floats. `(num_layers * num_directions, batch, hidden_size)
        :param char:
        :return:
        '''
        '''
         given an RState, returns the next RState the network goes to on input character char
        :param char:
        :return:
        '''
        h_t = self.list2hx(h_t)
        tensor_sequence = self.dataProcessor.sequence2tensor(char, self.raw_input_size,is_single_step=True)
        output, hx = self.forward(tensor_sequence,h_t)
        return self.hx2list(hx),self._classify_state(hx)

    def hx2list(self,hx):
        '''
        :param hx: shape (num_layers * num_directions, batch, hidden_size)
        :return: a list of floats
        '''
        hx = hx.squeeze()
        hx = hx.detach().numpy().flatten().tolist()
        return hx

    def list2hx(self,t_list):
        hx=np.array(t_list,dtype='f')
        hx = hx.reshape(self.h_shape)
        hx = torch.from_numpy(hx)
        return hx

class GRU3WrappperDFA(object):
    def __init__(self,rnn):
        super(GRU3WrappperDFA, self).__init__()
        assert isinstance(rnn,GRU3)
        self.rnn = rnn
        self.alphabet = rnn.alphabet

    def eval(self):
        self.rnn.eval()

    def cuda(self,cuda_no):
        self.rnn = self.rnn.cuda(cuda_no)

    def classify_word(self, sequence, cuda_num=0):
        assert isinstance(sequence,MyString)
        tensor_sequence = self.rnn.dataProcessor.sequence2tensor(sequence, self.rnn.raw_input_size,need_split=False)
        if cuda_num >= 0:
            tensor_sequence = tensor_sequence.cuda(cuda_num)
        output, hx = self.rnn.forward(tensor_sequence)

        return self.rnn._classify_state(hx)

    def get_first_RState(self):
        return self.rnn.get_first_RState()

    def get_next_RState(self,h_t,char):
        return self.rnn.get_next_RState(h_t,char)





class GRU2(MyModule):
    def __init__(self, raw_input_size, innder_input_dim, hidden_size, num_layers, num_class, dataProcessor):
        '''
        In this setting, we first map the raw input into a new vector(whose dim is innder_input_dim).
        :param raw_input_size:
        :param innder_input_dim:
        :param hidden_size:
        :param num_layers:
        :param num_class:
        '''
        super(GRU2, self).__init__()
        self.raw_input_size = raw_input_size
        self.dataProcessor = dataProcessor
        self.embedding = nn.Linear(raw_input_size, innder_input_dim)
        self.i2h = nn.GRU(batch_first=True, input_size=innder_input_dim, hidden_size=hidden_size, num_layers=num_layers)
        self.h2o = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)
        self.h_shape = (num_layers,1,hidden_size)
        self.alphabet = list("01")

    def forward(self, input,hx=None):
        new_input = self.embedding(input)
        if hx is None:
            output, h_n = self.i2h(new_input)
        else:
            output, h_n = self.i2h(new_input,hx)
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0]  # (len_seq,hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")

    def classify_word(self, word,cuda_num=0):
        '''
        :param word:
        :return:
        '''
        # if word == "":
        #     return True
        tensor_sequence = self.dataProcessor.sequence2tensor(word, self.raw_input_size)
        if cuda_num >=0:
            tensor_sequence = tensor_sequence.cuda(cuda_num)
        output, hx = self.forward(tensor_sequence)

        return self._classify_state(hx)

    def _classify_state(self,hx):
        '''
        make a prediction according to the hidden state
        :param state:
        :return:
        '''
        pr_dstr = self.output_pr_dstr(hx[-1])
        predicts = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
        return bool(predicts[0])


    def get_first_RState(self):
        '''
        returns a continuous vector representation of the network's initial state (an RState, as a list of floats)
        One example of a continuous vector representation of all of a GRU's hidden values would be the concatenation
        of the  h  vectors of each of its layers
        :return:
        '''
        hx = torch.zeros(self.h_shape)
        # return self.hx2list(hx),self._classify_state(hx)
        return self.hx2list(hx),True

    def get_next_RState(self, h_t, char):
        '''
        given an RState, returns the next RState the network goes to on input character char
        :param h_t:  a list of floats. `(num_layers * num_directions, batch, hidden_size)
        :param char:
        :return:
        '''
        '''
         given an RState, returns the next RState the network goes to on input character char
        :param char:
        :return:
        '''
        h_t = self.list2hx(h_t)
        tensor_sequence = self.dataProcessor.sequence2tensor(char, self.raw_input_size,is_single_step=True)
        output, hx = self.forward(tensor_sequence,h_t)
        return self.hx2list(hx),self._classify_state(hx)

    def hx2list(self,hx):
        '''
        :param hx: shape (num_layers * num_directions, batch, hidden_size)
        :return: a list of floats
        '''
        hx = hx.squeeze()
        hx = hx.detach().numpy().flatten().tolist()
        return hx

    def list2hx(self,t_list):
        hx=np.array(t_list,dtype='f')
        hx = hx.reshape(self.h_shape)
        hx = torch.from_numpy(hx)
        return hx


class MGU(MyModule):
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(MGU, self).__init__()
        pass

    def forward(self, input):
        pass

    def get_predict_trace(self, input_sequences):
        pass
