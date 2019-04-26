
import torch.nn as nn

'''
        This model enable both batch training and no-batch training.
        The "forward" fun only return the result of hidden states,and DO NOT
        compute the real label, i.e., classification result, at each time step including the final.
        The "output_pr_dstr" fun will return the probabilistic distribution with a single hidden state or a
        batch of that. This function is important for the loss computation and test the result.
        The "get_predict_trace" fun is designed to gain the inner hidden state as well as the classification
        result based on the hidden state at each time step
'''
class MyModule(nn.Module):

    def output_pr_dstr(self, hidden_states):
        '''
        output the predict probability distribution
        :param hidden_states: (batch,hn_dim),the final hidden state of each sample in a batch
        :return: pr_dstr:(batch,num_class)
        '''
        logits = self.h2o(hidden_states)
        pr_dstr = self.softmax(logits)
        return pr_dstr


    def get_predict_trace(self, input_sequences):
        '''
        return the final predict label
        :param input_sequences:  Tenosor:(batch,seq_len,input_size) batch=1 or PackedSequnces
        :return: labels of each sample in the batch
        '''
        pass