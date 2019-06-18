# -*- coding: utf-8 -*-
import email.parser
import random
from shutil import copyfile

import numpy as np
import os
import re
import torch
from torch.nn.utils.rnn import pack_sequence


class SPAM_Data_Processor():

    def __init__(self, word2vec_model, stop_words_path):
        self.word2vec_model = word2vec_model
        self.reg = r'[\s+\.\!\/_,$%^*(+\"\')]+|[+——()?“”！，。？、~@#￥%……&*<>:-]+'
        self.stop_words_list = ['re']
        with open(stop_words_path, 'r') as f:
            for line in f.readlines():
                self.stop_words_list.append(line.strip())

    def build_data_exp(self, path):
        if os.path.exists(path + "/data_exp"):
            return
        else:
            os.makedirs(path + "/data_exp/train/spam")
            os.makedirs(path + "/data_exp/train/ham")
            os.makedirs(path + "/data_exp/test/spam")
            os.makedirs(path + "/data_exp/test/ham")

        ham_list = []  # 2949
        spam_list = []  #1378
        fi = path +'/SPAMTrain.label'
        with open(fi, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                row = line.split(' ')
                label = int(row[0])
                if label == 0:
                    spam_list.append(row[1])
                else:
                    ham_list.append(row[1])
        spam_len = len(spam_list)
        test_len = int(spam_len / 6)
        for eml in spam_list[:test_len]:
            copyfile(path+'/TRAINING/'+eml, path+"/data_exp/test/spam/"+eml)
        for eml in spam_list[test_len:]:
            copyfile(path+'/TRAINING/'+eml, path+"/data_exp/train/spam/"+eml)
        ham_len = len(ham_list)
        test_len = int(ham_len / 6)

        for eml in ham_list[:test_len]:
            copyfile(path+'/TRAINING/'+eml, path+"/data_exp/test/ham/"+eml)
        for eml in ham_list[test_len:]:
            copyfile(path+'/TRAINING/'+eml, path+"/data_exp/train/ham/"+eml)


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
            tensor_list.sort(key=lambda x: x.shape[0], reverse=True) # sorted by the len of the word
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

    def sequence_purifier(self, sequence):
        pure_sequence = []
        content = re.findall(r'<body[\s\S]*>|<BODY[\s\S]*>', sequence)
        if content:
            sequence = content[0]
            prepurified = re.sub(r'<[\s\S]*?>|http\S*?\s', '', sequence).replace('\n', ' ')
        else:
            prepurified = re.sub(r'<[\s\S]*?>|http\S*?\s', '', sequence).replace('\n', ' ')
        new_words = re.split(self.reg, prepurified)
        for word in new_words:
            word = filter(self.check_char, word)
            if not (word.strip().lower() in self.stop_words_list or word.strip() == ""
                    or word.strip().isdigit() or len(word.strip()) == 1):
                pure_sequence.append(word.strip().lower())
        return pure_sequence

    def label2tensor(self, labels, num_class):
        pass

    def check_char(self,ch):
        if ch == '\'' or str.isalnum(ch): # is alphabet or num
            return True
        return False

    def load_data(self, folder_path, random_seed=5566, return_file_name=False, data_size=2298):

        sequence_list = []
        file_names = []
        # ham, spam = (2458, 1149)
        num_ham = int(np.ceil(data_size*0.5))
        num_spam = int(np.ceil(data_size*0.5))

        ham_list = os.listdir(os.path.join(folder_path, 'ham'))
        spam_list = os.listdir(os.path.join(folder_path, 'spam'))

        ####################################
        # sort according to the file name
        ####################################
        ham_list.sort(key=lambda x: int(re.split(u'[_.]', x)[1]))
        spam_list.sort(key=lambda x: int(re.split(u'[_.]', x)[1]))

        ########################
        # randomly select file
        #######################
        random.seed(random_seed)
        random.shuffle(ham_list)
        random.seed(random_seed)
        random.shuffle(spam_list)

        ham_selected = ham_list[:num_ham]
        spam_selected = spam_list[:num_spam]

        for file_name in ham_selected:
            file_name = os.path.join(folder_path, 'ham', file_name)
            sequence = extract_subject(file_name)
            sequence_list.append((sequence, 1))  # sequence is all the words in text
            file_names.append('ham-'+file_name)

        for file_name in spam_selected:
            file_name = os.path.join(folder_path, 'spam', file_name)
            sequence = extract_subject(file_name)
            sequence_list.append((sequence, 0))
            file_names.append('spam-' + file_name)

        #########
        # Shuffle
        #########
        random.seed(random_seed)
        random.shuffle(sequence_list)
        random.seed(random_seed)
        random.shuffle(file_names)
        if return_file_name:
            return sequence_list, file_names
        return sequence_list


def extract_subject(filename):
    ''' Extract the subject and payload from the .eml file.

    '''
    if not os.path.exists(filename):
        print("ERROR: input file does not exist:", filename)
        return ''
    fp = open(filename)
    msg = email.message_from_file(fp)
    payload = msg.get_payload()

    if isinstance(payload, list):
        payload = payload[0]  # only use the first part of payload

    if not isinstance(payload, str):
        payload = str(payload)

    if 'From nobody' in payload:
        # print filename
        for part in msg.walk():
            if part.get_content_type() == 'text/plain':
                payload = part.get_payload()
                break
            if part.get_content_type() == 'text/html':
                payload = part.get_payload()
                break

    if isinstance(payload, list):
        payload = payload[0]  # only use the first part of payload

    if not isinstance(payload, str):
        payload = str(payload)

    sub = msg.get('subject')
    sub = str(sub)
    return sub + payload



