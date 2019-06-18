import torch
import pickle


class TomitaDataProcessor(object):
    def sequence2tensor(self, sequence, input_tensor_dim=3, is_single_step=False):
        '''
        use tensor([1,0]) to denote character "0" and tensor([0,1]) to denote character "1"
        :param sequences: e.g.,"100010100"
        :param input_tensor_dim: the length of one-hot. since we take "",i.e.,empty string, into account, so the size of
                                 alphabet is 3.
        :param is_batch:
        :return:
        '''

        def make_one_hot(ch):
            if ch == "":
                vector = torch.tensor([1, 0, 0])
            elif ch == "0":
                vector = torch.tensor([0, 1, 0])
            else:
                vector = torch.tensor([0, 0, 1])
            return vector

        len_seq = 1 if sequence=="" or is_single_step else len(sequence)+1 # add the empty string
        sequence_tensor = torch.zeros(1, len_seq, input_tensor_dim)

        if len_seq == 1:
            vector = make_one_hot(sequence)
            sequence_tensor[0][0] = vector
            return sequence_tensor
        else:
            sequence_tensor[0][0] = make_one_hot("") # add the empty string
            for li,ch in enumerate(sequence):
                sequence_tensor[0][li+1] = make_one_hot(ch)
        return sequence_tensor


    def load_data(self,data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)
        return data["data"]