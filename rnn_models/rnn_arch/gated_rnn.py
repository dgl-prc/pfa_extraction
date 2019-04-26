import torch.nn as nn
import torch.nn.functional as F
import torch
from my_module import *
from torch.nn.utils.rnn import PackedSequence


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
    def __init__(self, input_size, hidden_size, num_layers,num_class):
        super(GRU,self).__init__()
        self.i2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2o = nn.Linear(hidden_size,num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output, h_n = self.i2h(input)
        return output,h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output,PackedSequence):
            output = output if output.dim()==2 else output[0] # (len_seq,hidden_size)
            pr_dstr = self.output_pr_dstr(output) # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr,dim=1).cpu().numpy().tolist()
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")


class MGU(MyModule):
    def __init__(self,input_size,hidden_size, num_layers,  num_class):
        super(MGU,self).__init__()
        pass
    def forward(self, input):
        pass
    def get_predict_trace(self, input_sequences):
        pass