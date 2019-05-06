import gensim
import os
import numpy as np
from shutil import copyfile
# from data_process.data_processor import DataProcessor
import torch
import re
import torchvision
import random
from torch.nn.utils.rnn import pack_sequence,pad_packed_sequence



class IMDB_Data_Processor(object):

    def __init__(self, word2vec_model, stop_words_path):
        self.word2vec_model = word2vec_model
        self.reg = '[\-|\.|,|:|!|;|\"]*\s+|[\.]?[\s]?\<br /\>\<br /\>|\.|[\s\"]?'
        self.stop_words_list = []
        with open(stop_words_path, 'r') as f:
            for line in f.readlines():
                self.stop_words_list.append(line.strip())

    # Here, the sequence is a sentence. we need to map each word into a numerical vector.
    def sequence2tensor(self, sequences, input_tensor_dim,is_batch=False):
        '''
        :param sequences:
        :param input_tensor_dim:
        :return: PackedSequence
        '''
        sequences = sequences if isinstance(sequences,list) else [sequences]
        word_sequences = map(self.sequence_purifier,sequences)
        if is_batch:
            tensor_list = []
            for word_sequence in word_sequences:
                sequence_tensor = torch.zeros(len(word_sequence),input_tensor_dim) # batch,len_sequence,input_size
                for li, word in enumerate(word_sequence):
                    try:
                        vector = self.word2vec_model[word]
                    except KeyError as e:
                        continue
                    sequence_tensor[li] = torch.tensor(vector)
                tensor_list.append(sequence_tensor)
            tensor_list.sort(key=lambda x:x.shape[0],reverse=True)
            return pack_sequence(tensor_list)
        else:
            word_sequence = word_sequences[0]
            sequence_tensor = torch.zeros(1, len(word_sequence), input_tensor_dim)  # batch,len_sequence,input_size
            for li, word in enumerate(word_sequence):
                try:
                    vector = self.word2vec_model[word]
                except KeyError as e:
                    # print("{} not in vocabulary".format(word))
                    continue
                sequence_tensor[0][li] = torch.tensor(vector)
            return sequence_tensor

    def label2tensor(self, labels, num_class):
        pass

    def check_char(self,ch):
        if ch == '\'' or str.isalnum(ch):
            return True
        return False

    def sequence_purifier(self, sequence):
        '''refine the sequence. Remove some meaningless words and specific symbol'''
        pure_sequence = []
        new_words = re.split(self.reg, sequence)
        for word in new_words:
            prime_w = word
            word = filter(self.check_char, word)
            if word.strip() == '' or word.strip() == '\'':
                continue
            # remove the the first and the last prime
            try:
                if word[0]=='\'':
                    word=word[1:]
                if word[-1]=='\'':
                    word=word[:-1]
            except IndexError as e:
                print("IndexError:{}=====>{}".format(prime_w,word))

            if not (word.strip().lower() in self.stop_words_list or word.strip() == "" or word.strip().find(
                    'SPOILERS') != -1 or word.strip().isdigit()):
                pure_sequence.append(word)
        return pure_sequence

    def load_data(self,folder_path,random_seed=5566,return_file_name=False,excluded_traces_file=None,data_size=5000):

        sequence_list = []
        file_names = []

        num_pos=int(np.ceil(data_size*0.5))
        num_neg=int(np.ceil(data_size*0.5))

        pos_list = os.listdir(os.path.join(folder_path,'pos'))
        neg_list = os.listdir(os.path.join(folder_path,'neg'))
        ####################################
        # sort according to the file name
        ####################################
        pos_list.sort(key=lambda x:int(x.split("_")[0]))
        neg_list.sort(key=lambda x:int(x.split("_")[0]))

        ########################
        # randomly select file
        #######################
        random.seed(random_seed)
        random.shuffle(pos_list)
        random.seed(random_seed)
        random.shuffle(neg_list)

        pos_selected=pos_list[:num_pos]
        neg_selected=neg_list[:num_neg]

        for file_name in pos_selected:
            with open(os.path.join(folder_path,'pos',file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                sequence_list.append((sequence[0],True))
                file_names.append('pos-'+file_name)

        for file_name in neg_selected:
            with open(os.path.join(folder_path, 'neg',file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                sequence_list.append((sequence[0], False))
                file_names.append('neg-' + file_name)

        #########
        # Shuffle
        #########
        random.seed(random_seed)
        random.shuffle(sequence_list)
        random.seed(random_seed)
        random.shuffle(file_names)
        if return_file_name:
            return sequence_list,file_names
        return sequence_list


class MyString(object):
    def __init__(self, data):
        assert isinstance(data, list)
        self.data = data
        self.p = 0
        self.length = len(self.data)

    def __iter__(self):
        return self

    def __len__(self):
        return self.length

    def next(self):
        if self.p < self.length:
            ele = self.data[self.p]
            self.p += 1
        else:
            self.p = 0
            raise StopIteration()
        return ele

    def __str__(self):
        return " ".join(self.data)

    def __add__(self, other):
        data = self.data + other.data
        return MyString(data)

    def __eq__(self, other):
        return " ".join(self.data) == " ".join(other.data)

    def __hash__(self):
        return hash(self.__str__())

class IMDB_Data_Processor_DFA(IMDB_Data_Processor):
    def __init__(self,word2vec_model, stop_words_path):
        super(IMDB_Data_Processor_DFA, self).__init__(word2vec_model, stop_words_path)

    # overwrite
    def sequence2tensor(self, sequences, input_tensor_dim,need_split=True):
        '''
        Use the dollar as the first input, just like the empty string in regular language
        :param sequences:
        :param input_tensor_dim:
        :return: PackedSequence
        '''
        if need_split:
            sequences = sequences if isinstance(sequences, list) else [sequences]
            word_sequences = map(self.sequence_purifier,sequences)
            word_sequence = word_sequences[0]
        else:
            word_sequence = sequences

        sequence_tensor = torch.zeros(1, len(word_sequence)+1, input_tensor_dim)  # batch,len_sequence,input_size
        sequence_tensor[0][0]=torch.tensor(self.word2vec_model["$"])
        for li, word in enumerate(word_sequence):
            try:
                vector = self.word2vec_model[word]
            except KeyError as e:
                # print("{} not in vocabulary".format(word))
                continue
            sequence_tensor[0][li+1] = torch.tensor(vector)
        return sequence_tensor


    def make_alphabet(self,data_folder):

        alphabet=set()
        # alphabet_count={}
        pos_list = os.listdir(os.path.join(data_folder,'pos'))
        neg_list = os.listdir(os.path.join(data_folder,'neg'))
        for file_name in pos_list:
            with open(os.path.join(data_folder,'pos',file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                word_sequence = map(self.sequence_purifier, sequence)[0]
                alphabet = alphabet|set(word_sequence)
        for file_name in neg_list:
            with open(os.path.join(data_folder, 'neg',file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                word_sequence = map(self.sequence_purifier, sequence)[0]
                alphabet = alphabet | set(word_sequence)

        return alphabet

    def load_clean_data(self, folder_path):
        train_data = self.load_data(folder_path)
        self.load_data(folder_path)
        # clean_data = [] # dict can not be used since list is unhashable type
        clean_data={}
        for data_pair in train_data:
            sequence, label = data_pair
            word_sequence = map(self.sequence_purifier, [sequence])[0]
            assert isinstance(word_sequence, list)
            clean_data[MyString(word_sequence)]=label
        clean_data[MyString(["$"])]=1  # add the "empty string"
        return clean_data


