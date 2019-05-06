

import torch
import string
import pickle
class BPProcessor(object):

    def __init__(self):
        self.index_map = {key: i + 1 for i, key in enumerate(string.ascii_lowercase)}
        self.index_map[""]=0
        self.index_map["("] = 27
        self.index_map[")"] = 28

    def sequence2tensor(self, sequence,alphabet_size,is_single_step=False):
        '''
        use one-hot to encode the sequence
        :param sequences: e.g.,"100010100"
        :param alphabet_size: the length of one-hot. since we take "",i.e.,empty string, into account, so the size of
                                 alphabet is 29. empty string,a~z, (, )
        :param is_batch:
        :return:
        '''
        def make_one_hot(ch):
            vector = torch.zeros(alphabet_size)
            vector[self.index_map[ch]] = 1
            return vector

        len_seq = 1 if sequence=="" or is_single_step else len(sequence)+1 # add the empty string
        sequence_tensor = torch.zeros(1, len_seq, alphabet_size)

        if len_seq == 1:
            vector= make_one_hot(sequence)
            sequence_tensor[0][0]=vector
            return sequence_tensor
        else:
            sequence_tensor[0][0] = make_one_hot("")
            for li,ch in enumerate(sequence):
                sequence_tensor[0][li+1] =  make_one_hot(ch)
        return sequence_tensor

    def load_data(self,data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["data"]
