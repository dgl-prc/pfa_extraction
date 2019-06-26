import torch
from torch.autograd import Variable
from torch.nn.utils.rnn import PackedSequence

from my_module import *


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


class GRU(MyModule): # Gated Recurrent Units
    def __init__(self, input_size, hidden_size, num_layers, num_class):
        super(GRU, self).__init__()
        self.i2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2o = nn.Linear(hidden_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output, h_n = self.i2h(input) # output: (seq_len, batch, hidden_size * num_directions) where hidden_size is 10
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0]  # (len_seq, hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_c lass)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            # detach a varible for the tree
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")


class MGU(MyModule):
    def __init__(self, input_size, hidden_size, num_layers,  num_class):
        super(MGU, self).__init__()
        pass
    def forward(self, input):
        pass
    def get_predict_trace(self, input_sequences):
        pass


class GRU2(MyModule): # Gated Recurrent Units

    def __init__(self, input_size, hidden_size, num_layers, num_class, alphabet=18):
        super(GRU2, self).__init__()
        self.input_size = input_size
        self.i2emb = nn.Embedding(alphabet, input_size)
        self.i2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2l = nn.Linear(hidden_size, hidden_size/2)
        self.relu1 = nn.ReLU()
        self.l2o = nn.Linear(hidden_size/2, num_class)
        self.relu2 = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=1)

    def output_pr_dstr(self, hidden_states):
        linear1_out = self.h2l(hidden_states)
        linear1_relu = self.relu1(linear1_out)
        linear2_out = self.l2o(linear1_relu)
        logits = self.relu2(linear2_out)
        pr_dstr = self.softmax(logits)
        return pr_dstr

    def forward(self, imput):
        sequence_tensor = torch.zeros(1, len(imput), self.input_size)
        for i, elm in enumerate(imput):
            elm = int(elm) - 1
            sequence_tensor[0][i] = self.i2emb(Variable(torch.LongTensor([elm])))
        # output: (seq_len, batch, hidden_size * num_directions)
        output, h_n = self.i2h(sequence_tensor)
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0] # (len_seq, hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            # detach a varible for the tree
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")


class GRU3(MyModule):  # Gated Recurrent Units

    def __init__(self, input_size, hidden_size, num_layers, num_class, alphabet=18):
        super(GRU3, self).__init__()
        self.i2emb = nn.Linear(alphabet, input_size)
        self.emb2h = nn.GRU(batch_first=True, input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
        self.h2l = nn.Linear(hidden_size, hidden_size/2)
        self.l2l = nn.Linear(hidden_size/2, input_size)
        self.l2o = nn.Linear(input_size, num_class)
        self.softmax = nn.LogSoftmax(dim=1)
        self.relu = nn.ReLU()

    def output_pr_dstr(self, hidden_states):
        linear1_out = self.h2l(hidden_states)
        linear1_relu = self.relu(linear1_out)
        linear2_out = self.l2l(linear1_relu)
        linear2_relu = self.relu(linear2_out)
        logits = self.l2o(linear2_relu)
        pr_dstr = self.softmax(logits)
        return pr_dstr

    def forward(self, imput):
        emb_out = self.i2emb(imput)
        # output: (seq_len, batch, hidden_size * num_directions)
        output, h_n = self.emb2h(emb_out)
        return output, h_n

    def get_predict_trace(self, input_sequences):
        output, hn = self.forward(input_sequences)
        if not isinstance(output, PackedSequence):
            output = output if output.dim() == 2 else output[0] # (len_seq, hidden_size)
            pr_dstr = self.output_pr_dstr(output)  # (len_seq,num_class)
            predict_trace = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
            # detach a varible for the tree
            return [hn_step.cpu().detach().numpy().tolist() for hn_step in output], predict_trace
        else:
            raise Exception("Batch is not supported at the moment")

