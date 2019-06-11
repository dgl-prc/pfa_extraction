
from data_factory.data_factory import *
import torch
class FCSMP_Data_Processor(DataFactory):

    def load_data(self, data_root, data_size=5000):
        train = []
        test = []
        file = data_root + '/pautomac-real-2.train.txt'
        t_file = data_root + '/pautomac-real-2.test.txt'
        with open(file, 'r') as f:
            for i in range(data_size + 1):
                    cur = f.readline().strip()
                    cur = cur.split(' ')
                    if i > 0:
                        train.append((cur[1:-1],int(cur[-1])-1))
        with open(t_file, 'r') as f:
            for j in range(1001):
                    cur = f.readline().strip()
                    cur = cur.split(' ')
                    if j > 0:
                        test.append((cur[1:-1], int(cur[-1]) - 1))
        return train, test

    def sequence2tensor(self, sequence, input_tensor_dim=18):
        sequence_tensor = torch.zeros(1, len(sequence), input_tensor_dim)
        for i, elm in enumerate(sequence):
            elm = int(elm) - 1
            sequence_tensor[0][i][elm] = 1
        return sequence_tensor

    def label2tensor(self,label,num_class):
        pass