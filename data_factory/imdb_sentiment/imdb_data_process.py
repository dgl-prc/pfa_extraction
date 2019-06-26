# -*- coding: utf-8 -*-
import email.parser
import random
from shutil import copyfile

import numpy as np
import os
import re
import torch
from torch.nn.utils.rnn import pack_sequence


def copy_files(files_list, src_folder, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    for file_name in files_list:
        copyfile(os.path.join(src_folder, file_name), os.path.join(dest_folder, file_name))


def randomly_choose_train_test_data(random_seed, train_size, test_size, source_folder, dest_folder):
    train_neg_path = os.path.join(source_folder, 'train', 'neg')
    train_pos_path = os.path.join(source_folder, 'train', 'pos')
    test_neg_path = os.path.join(source_folder, 'test', 'neg')
    test_pos_path = os.path.join(source_folder, 'test', 'pos')

    train_neg_files_list = os.listdir(train_neg_path)
    train_pos_files_iist = os.listdir(train_pos_path)
    test_neg_files_list = os.listdir(test_neg_path)
    test_pos_files_iist = os.listdir(test_pos_path)

    rnd_train_pos_path = os.path.join(dest_folder, 'train', 'pos')
    rnd_train_neg_path = os.path.join(dest_folder, 'train', 'neg')
    rnd_test_pos_path = os.path.join(dest_folder, 'test', 'pos')
    rnd_test_neg_path = os.path.join(dest_folder, 'test', 'neg')

    np.random.seed(random_seed)
    random_train_neg = np.random.choice(train_neg_files_list, int(train_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_train_pos = np.random.choice(train_pos_files_iist, int(train_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_test_neg = np.random.choice(test_neg_files_list, int(test_size * 0.5), replace=False)
    np.random.seed(random_seed)
    random_test_pos = np.random.choice(test_pos_files_iist, int(test_size * 0.5), replace=False)

    # do copy
    copy_files(random_train_neg, train_neg_path, rnd_train_neg_path)
    copy_files(random_train_pos, train_pos_path, rnd_train_pos_path)
    copy_files(random_test_neg, test_neg_path, rnd_test_neg_path)
    copy_files(random_test_pos, test_pos_path, rnd_test_pos_path)

    print('DONE!')


class IMDB_Data_Processor():

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
        sequences = sequences if isinstance(sequences, list) else [sequences]
        word_sequences = map(self.sequence_purifier, sequences)
        if is_batch:
            tensor_list = []
            for word_sequence in word_sequences:
                sequence_tensor = torch.zeros(len(word_sequence), input_tensor_dim) # batch,len_sequence,input_size
                for li, word in enumerate(word_sequence):
                    try:
                        vector = self.word2vec_model[word]
                    except KeyError as e:
                        continue
                    sequence_tensor[li] = torch.tensor(vector)
                tensor_list.append(sequence_tensor)
            tensor_list.sort(key=lambda x:x.shape[0],reverse=True) # sorted by the len of the word
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
        if ch == '\'' or str.isalnum(ch): # is alphabet or num
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
                print("IndexError:{}=====>{}".format(prime_w, word))

            if not (word.strip().lower() in self.stop_words_list or word.strip() == "" or word.strip().find(
                    'SPOILERS') != -1 or word.strip().isdigit()):
                pure_sequence.append(word)
        return pure_sequence


    def load_data(self,folder_path,random_seed=5566,return_file_name=False,excluded_traces_file=None,data_size=5000):

        sequence_list = []
        file_names = []

        num_pos=int(np.ceil(data_size*0.5))
        num_neg=int(np.ceil(data_size*0.5))

        pos_list = os.listdir(os.path.join(folder_path, 'pos'))
        neg_list = os.listdir(os.path.join(folder_path, 'neg'))
        ####################################
        # sort according to the file name
        ####################################
        pos_list.sort(key=lambda x: int(x.split("_")[0]))
        neg_list.sort(key=lambda x: int(x.split("_")[0]))

        ########################
        # randomly select file
        #######################
        random.seed(random_seed)
        random.shuffle(pos_list)
        random.seed(random_seed)
        random.shuffle(neg_list)

        pos_selected = pos_list[:num_pos]
        neg_selected = neg_list[:num_neg]

        for file_name in pos_selected:
            with open(os.path.join(folder_path,'pos', file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                sequence_list.append((sequence[0], 1))  # sequence is all the words in text
                file_names.append('pos-'+file_name)

        for file_name in neg_selected:
            with open(os.path.join(folder_path, 'neg', file_name), 'r') as f:
                sequence = f.readlines()
                assert len(sequence) != 0
                sequence_list.append((sequence[0], 0))
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
















