from __future__ import division
import sys, os
import argparse

sys.path.append('./')
import torch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from rnn_models.rnn_arch.gated_rnn import LSTM, GRU, GRU2, GRU3,GRU3WrappperDFA
from utils.time_util import current_timestamp
import pickle
from utils.constant import *
import numpy as np
import math
from data_factory.tomita.generator import TomitaDataProcessor
from data_factory.fuel_consum.fcsmpDataProcessor import FCSMP_Data_Processor
from data_factory.bp.bp_processor import *
from data_factory.imdb_sentiment.imdb_data_process import *
import gensim
import copy
def adjust_learing_rate(optimizer, lr, epoch, step):
    new_lr = lr * (0.1 ** (epoch // step))
    print("new learning rate:{}".format(new_lr))
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr


# tested
def batch_generator(dataset, batch_size):
    '''
    :param batch_size:
    :param dataset: list(tupe)
    :return: datalist,label_list
    '''
    p = 0
    max_size = len(dataset)
    while p < max_size:
        if p + batch_size <= max_size:
            yield ([ele[0] for ele in dataset[p:p + batch_size]], [ele[1] for ele in dataset[p:p + batch_size]])
        else:
            yield ([ele[0] for ele in dataset[p:max_size]], [ele[1] for ele in dataset[p:max_size]])
        p += batch_size


def forware_pass(model, inputs, RNN_TYPE):
    '''

    :param model:
    :param input:
    :param RNN_TYPE:
    :return: output: all the output,e.g. result of each classification, at each time step
    '''
    if RNN_TYPE == MTYPE_LSTM:
        output, (hn, cn) = model(inputs)
    else:
        output, hn = model(inputs)
    return output, hn


def test_accuracy_batch(rnn, test_data, dataProcessor, input_dim, num_class, use_cuda, RNN_TYPE=MTYPE_SRNN):
    rnn.eval()
    correct = 0
    batch_size = 12
    if use_cuda:
        rnn = rnn.cuda()
    for sequences, labels in batch_generator(test_data, batch_size):
        packed_sequences = dataProcessor.sequence2tensor(sequences, input_dim)
        if use_cuda:
            packed_sequences = packed_sequences.cuda()
        output, hx = forware_pass(rnn, packed_sequences, RNN_TYPE)
        pr_dstr = rnn.output_pr_dstr(hx[-1])
        predicts = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
        correct += sum(np.array(predicts) == np.array(labels))
    rst = "acc:{:.2f}({}/{})".format(correct * 1. / len(test_data), correct, len(test_data))
    return rst

def tets_accuracy_concise(rnn,test_data,cuda_no):
    '''

    :param rnn:
    :param test_data: list(tuple)
    :return:
    '''
    rnn.eval()
    if cuda_no >=0:
        rnn = rnn.cuda(cuda_no)
    data_size = len(test_data)
    cnt = 0
    for words,label in test_data:
        pdt=rnn.classify_word(words,cuda_no)
        if pdt==label:
            cnt+=1
    acc = cnt/data_size
    descr = "acc:{:.2f}({}/{})".format(acc, cnt, data_size)
    return descr, round(acc, 4)

def tets_accuracy_concise_with_clean_data(rnnWrapper,test_data,cuda_no):
    '''

    :param rnn:
    :param test_data: list(tuple)
    :return:
    '''
    rnnWrapper.eval()
    if cuda_no >=0:
         rnnWrapper.cuda(cuda_no)
    data_size = len(test_data)
    cnt = 0
    for words in test_data:
        bkp = copy.deepcopy(words)
        pdt=rnnWrapper.classify_word(words,cuda_no)
        if words==bkp:
            print("No change")
        if pdt==test_data[words]:
            cnt+=1
    acc = cnt/data_size
    descr = "acc:{:.2f}({}/{})".format(acc, cnt, data_size)
    return descr, round(acc, 4)





def test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no, RNN_TYPE=MTYPE_SRNN):
    '''
    this function for the one-loop
    :param rnn:
    :param test_data:
    :param dataProcessor:
    :param input_dim:
    :param num_class:
    :param use_cuda:
    :param RNN_TYPE:
    :return:
    '''
    rnn.eval()
    correct = 0
    batch_size = 1
    if cuda_no != -1:
        rnn = rnn.cuda(cuda_no)

    data_list = test_data.keys() if isinstance(test_data, dict) else test_data
    for sequence in data_list:
        if isinstance(test_data, dict):
            label = test_data[sequence]
        else:
            # tuple
            label = sequence[1]
            sequence = sequence[0]  # list

        tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim,need_split=True)  # Lx1xn_letters
        if cuda_no != -1:
            tensor_sequence = tensor_sequence.cuda(cuda_no)
        output, hx = forware_pass(rnn, tensor_sequence, RNN_TYPE)
        pr_dstr = rnn.output_pr_dstr(hx[-1])
        assert batch_size == 1
        predicts = torch.argmax(pr_dstr, dim=1).cpu().numpy().tolist()
        if label == predicts[0]:
            correct += 1
    acc = correct * 1. / len(test_data)
    descr = "acc:{:.2f}({}/{})".format(correct * 1. / len(test_data), correct, len(test_data))
    return descr, round(acc, 4)


def train_with_optim(rnn, lr, epocs, train_data, test_data=None, dataProcessor=None, cuda_no=-1, input_dim=2, num_class=2,
                     RNN_TYPE=MTYPE_SRNN, lower_bound_acc=0.9, lr_decay=True):

    def ouput_info(train_descr,test_descr):
        train_info = current_timestamp() + '****************======>Init train_acc:' + str(train_descr)
        train_info += 'Test acc:' + str(test_descr) if test_descr is not None else "NO TEST DATA"
        return train_info


    watch_step = 500
    average_loss = 0
    optim = torch.optim.Adam(rnn.parameters(), lr)

    train_descr, train_acc = test_accuracy(rnn, train_data, dataProcessor, input_dim, num_class, cuda_no,
                                           RNN_TYPE=RNN_TYPE)
    if test_data is not None:
        test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
                                             RNN_TYPE=RNN_TYPE)
    else:
        test_descr = None

    train_info=ouput_info(train_descr, test_descr)
    print(train_info)

    rnn.train()
    if cuda_no != -1:
        rnn = rnn.cuda(cuda_no)
    for epoc in range(epocs):
        count = 0
        if lr_decay and lr > 1e-6:
            adjust_learing_rate(optim, lr, epoc, 40)

        data_list = train_data.keys() if isinstance(train_data, dict) else train_data
        for sequence in data_list:
            if isinstance(train_data, dict):
                label = train_data[sequence]
            else:
                # tuple
                label = sequence[1]
                sequence = sequence[0]  # list

            tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim,need_split=True)  # Lx1xn_letters
            ground_truth = torch.LongTensor([label])
            if cuda_no != -1:
                tensor_sequence, ground_truth = tensor_sequence.cuda(cuda_no), ground_truth.cuda(cuda_no)

            optim.zero_grad()
            output, hx = forware_pass(rnn, tensor_sequence, RNN_TYPE)
            pr_dstr = rnn.output_pr_dstr(hx[-1])
            loss = F.nll_loss(pr_dstr, ground_truth)
            loss.backward()
            optim.step()  # update parameters
            count += 1
            average_loss += loss
            if count % watch_step == 0:
                print("epoc:{},sampels:{},loss:{}".format(epoc, count, average_loss / watch_step))
                average_loss = 0

        train_descr, train_acc = test_accuracy(rnn, train_data, dataProcessor, input_dim, num_class, cuda_no,
                                                   RNN_TYPE=RNN_TYPE)
        if test_data is not None:
            test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
                                                 RNN_TYPE=RNN_TYPE)
        else:
            test_descr = None

        train_info = ouput_info(train_descr, test_descr)
        print(train_info)

        rnn.train()
        if train_acc >= lower_bound_acc:
            if test_data is not None:
                if test_acc >= lower_bound_acc:
                    return train_acc, test_acc
            else:
                return train_acc
    test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
                                         RNN_TYPE=RNN_TYPE)
    train_descr, train_acc = test_accuracy(rnn, train_data, dataProcessor, input_dim, num_class, cuda_no,
                                           RNN_TYPE=RNN_TYPE)

    print('Warning:Training Failed!!!!!Can not make model\'s accuracy to be {}!'.format(lower_bound_acc))
    if train_acc >= lower_bound_acc:
        if test_data is not None:
            if test_acc >= lower_bound_acc:
                return train_acc, test_acc
        else:
            return train_acc


def train_with_optim_batch(rnn, lr, epocs, train_data, test_data, dataProcessor,
                           use_cuda=False, input_dim=2, num_class=2, RNN_TYPE=MTYPE_SRNN, batch_size=1):
    watch_step = 100
    average_loss = 0
    optim = torch.optim.Adam(rnn.parameters(), lr)
    test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, use_cuda, RNN_TYPE)
    rnn.train()
    print(current_timestamp() + '****************======>Init acc:' + test_descr)
    if use_cuda:
        rnn = rnn.cuda()
    for epoc in range(epocs):
        count = 0
        for sequences, labels in batch_generator(train_data, batch_size):
            packed_sequences = dataProcessor.sequence2tensor(sequences, input_dim, is_batch=True)  # Lx1xn_letters
            ground_truth = torch.LongTensor(labels)
            if use_cuda:
                packed_sequences, ground_truth = packed_sequences.cuda(), ground_truth.cuda()

            optim.zero_grad()
            output, h_states = forware_pass(rnn, packed_sequences, RNN_TYPE)
            pr_dstr = rnn.output_pr_dstr(h_states[-1])  # h_states[-1] (batch,hn_size) ====> pf_dstr:(batch,num_class)
            loss = F.nll_loss(pr_dstr, ground_truth)
            loss.backward()
            optim.step()  # update parameters
            count += 1
            average_loss += loss
            if count % watch_step == 0:
                print("epoc:{},sampels:{},loss:{}".format(epoc, count, average_loss / watch_step))
                average_loss = 0

        test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, use_cuda,
                                             RNN_TYPE=RNN_TYPE)
        print(current_timestamp() + '###########*************===========>epoc:{},{}'.format(epoc, test_descr))
        rnn.train()


def train_artificial():
    # do_train()
    # print 'ok'
    input_size = 2
    output_size = 2
    hidden_size = 128
    num_layers = 3
    lr, epocs = 0.01, 20

    train_data, val_set, test_data = quadruple_01_data_generator(postitive_ration=0.4);
    dataProcessor = ZeroOneDataProcessor()
    rnn = SRNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)
    train_with_optim(rnn, lr, epocs, train_data, test_data, dataProcessor, use_cuda=False)


def load_train_test(data_root, dataProcessor):
    train_data_path = os.path.join(data_root, 'train')
    test_data_path = os.path.join(data_root, 'test')
    train_data = dataProcessor.load_data(train_data_path)
    test_data = dataProcessor.load_data(test_data_path)
    return train_data, test_data


def train_imdb_group():
    # do_train()
    # print 'ok'
    input_size = 300
    output_size = 2
    hidden_size = 10
    num_layers = 3
    lr, epocs = 0.001, 50
    model_path = 'models/pretrained'
    word2vec_model_path = './models/pretrained/GoogleNews-vectors-negative300.bin'
    stop_words_list_path = './data/pfa_expe1/stopwords.txt'
    print('loading word2vector model....')
    dataProcessor = IMDB_Data_Processor(word2vec_model_path, stop_words_list_path)

    ##################
    #
    ##################
    rnn_list = []
    rnn_list.append(SRNN(input_size=input_size, num_class=output_size, hidden_size=hidden_size, num_layers=num_layers))
    rnn_list.append(LSTM(input_size=input_size, num_class=output_size, hidden_size=hidden_size, num_layers=num_layers))
    rnn_list.append(GRU(input_size=input_size, num_class=output_size, hidden_size=hidden_size, num_layers=num_layers))
    lower_bound_acc_list = [0.74, 0.88, 0.89]
    rnn_type_list = [MTYPE_SRNN, MTYPE_LSTM, MTYPE_GRU]

    for i in range(1, 6):
        print('Data Group:{}.......'.format(i))
        print('loading data set....')
        train_data, test_data = load_train_test(i, dataProcessor)
        for rnn_type, rnn, lower_bound_acc in zip(rnn_type_list, rnn_list, lower_bound_acc_list):
            print('Training {} ....'.format(rnn_type))
            train_acc, test_acc = train_with_optim(rnn, lr, epocs, train_data, test_data, dataProcessor, use_cuda=True,
                                                   input_dim=input_size,
                                                   RNN_TYPE=rnn_type, model_path=model_path,
                                                   lower_bound_acc=lower_bound_acc)
            save_folder = os.path.join(model_path, rnn_type)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder,
                                     rnn_type + str(i) + '-train_acc-' + str(train_acc) + '-test_acc-' + str(
                                         test_acc) + '.pkl')
            with open(save_path, 'wb') as f:
                pickle.dump(rnn, f)

    print('Done!')


def batch_train_imdb(word2vec_model_path, stop_words_list_path, model_path='models/pretrained', data_group=1,
                     cuda_no=-1):
    input_size = 300
    output_size = 2
    hidden_size = 10
    num_layers = 3
    lr, epocs = 0.001, 50
    num_trainig = 5000
    min_acc = 0.87

    # stop_words_list_path = './data/stopwords.txt'
    print('loading word2vector model....')
    dataProcessor = IMDB_Data_Processor(word2vec_model_path, stop_words_list_path)
    rnn_list = []
    rnn_list.append(GRU(input_size=input_size, num_class=output_size, hidden_size=hidden_size, num_layers=num_layers))
    lower_bound_acc_list = [0.87]
    rnn_type_list = [MTYPE_GRU]

    train_data, test_data = load_train_test(data_group=data_group, dataProcessor=dataProcessor, train_size=num_trainig)
    for rnn_type, rnn, lower_bound_acc in zip(rnn_type_list, rnn_list, lower_bound_acc_list):
        print('Training {} ....'.format(rnn_type))
        train_acc, test_acc = train_with_optim(rnn, lr, epocs, train_data, test_data, dataProcessor, cuda_no=cuda_no,
                                               input_dim=input_size,
                                               RNN_TYPE=rnn_type, model_path=model_path,
                                               lower_bound_acc=lower_bound_acc)
        save_folder = os.path.join(model_path, rnn_type, str(num_trainig))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        save_path = os.path.join(save_folder,
                                 rnn_type + str(data_group) + '-train_acc-' + str(train_acc) + '-test_acc-' + str(
                                     test_acc) + '.pkl')
        with open(save_path, 'wb') as f:
            pickle.dump(rnn, f)
    print('Done!')


def train_imdb(params, data_id, cuda_no):
    #########
    # training
    #########
    data_group = params["data_groups"][data_id]
    train_data, test_data = load_train_test(data_root=os.path.join(params["data_root"], data_group),
                                            dataProcessor=params["dataProcessor"])
    print('Training {} with data {}....'.format(params["rnn_type"], data_group))
    if params["rnn_type"] == MTYPE_GRU:
        rnn = GRU(input_size=params["input_size"], num_class=params["output_size"], hidden_size=params["hidden_size"],
                  num_layers=params["num_layers"])
    elif params["rnn_type"] == MTYPE_LSTM:
        rnn = LSTM(input_size=params["input_size"], num_class=params["output_size"], hidden_size=params["hidden_size"],
                   num_layers=params["num_layers"])
    else:
        raise Exception("Unknow rnn type:{}".format(params["rnn_type"]))

    train_acc, test_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, test_data, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=params["input_size"],
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"])
    ########
    # save
    #######
    save_path = os.path.join(params["model_save_root"], params["rnn_type"],
                             data_group + '-train_acc-' + str(train_acc) + '-test_acc-' + str(
                                 test_acc) + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(rnn, f)

    print('Done!')


def train_tomita(params, cuda_no):
    torch.random.manual_seed(199312)
    dataProcessor = TomitaDataProcessor()

    train_data = dataProcessor.load_data(params["train_data_path"])
    test_data = None if params["test_data_path"] is None else dataProcessor.load_data(params["test_data_path"])
    print("Train Size:{},Test Size:{}".format(len(train_data), 0 if test_data is None else len(test_data)))

    rnn = GRU2(raw_input_size=params["alphabet_size"], innder_input_dim=3, num_class=2,
               hidden_size=params["hidden_size"], num_layers=2, dataProcessor=dataProcessor)

    print("Begin Training......")
    train_acc, test_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, test_data, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=params["alphabet_size"],
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"],
                                           lr_decay=params["lr_decay"])

    print("training acc:{},testing acc:{}".format(train_acc, test_acc))

    with open(params["model_save"], "wb") as f:
        pickle.dump(rnn.cpu(), f)
    return rnn


def train_FCSMP(params, cuda_no):
    dataProcessor = FCSMP_Data_Processor()
    train_data, test_data = dataProcessor.load_data(params["data_root"], params["train_number"])

    print('Training {} with Number:{}....'.format(params["rnn_type"], len(train_data)))
    if params["rnn_type"] == MTYPE_GRU:
        rnn = GRU3(raw_input_size=params["raw_input_size"], input_size=params["inner_input_size"],
                   num_class=params["output_size"], hidden_size=params["hidden_size"],
                   num_layers=params["num_layers"])

    else:
        raise Exception("Unknow rnn type:{}".format(params["rnn_type"]))

    train_acc, test_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, test_data, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=params["raw_input_size"],
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"])

    save_path = os.path.join(params["model_save_root"],
                             params["rnn_type"] + '-train_acc-' + str(train_acc) + '-test_acc-' + str(
                                 test_acc) + '.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(rnn, f)

    print('Done!')

def train_bp(params, cuda_no):
    torch.random.manual_seed(199312)
    dataProcessor = BPProcessor()

    train_data = dataProcessor.load_data(params["train_data_path"])
    test_data = None if params["test_data_path"] is None else dataProcessor.load_data(params["test_data_path"])
    print("Train Size:{},Test Size:{}".format(len(train_data), 0 if test_data is None else len(test_data)))

    rnn = GRU2(raw_input_size=params["alphabet_size"], innder_input_dim=3, num_class=2,
               hidden_size=params["hidden_size"], num_layers=2, dataProcessor=dataProcessor)

    print("Begin Training......")
    train_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, test_data, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=params["alphabet_size"],
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"],
                                           lr_decay=params["lr_decay"])

    print("training acc:{}".format(train_acc))

    with open(params["model_save"], "wb") as f:
        pickle.dump(rnn.cpu(), f)
    return rnn

def special_train():
    '''
    Force the RNN making right prediction about empty string
    :return:
    '''

    model_folder = "./rnn_models/pretrained/tomita"
    model_path = os.path.join(model_folder, "test-gru-{}.pkl".format("tomita6"))
    with open(model_path, "rb") as f:
        rnn = pickle.load(f)
    rnn =rnn.cuda()
    ##############################################################
    # force the rnn to make right prediction about empty string
    #############################################################
    optim = torch.optim.Adam(rnn.parameters(), 0.001)
    dataProcessor = TomitaDataProcessor()
    cuda_no = 0
    rnn.train()
    while rnn.classify_word("") is False:
        sequence = ""
        label = True
        tensor_sequence = dataProcessor.sequence2tensor(sequence)  # Lx1xn_letters
        ground_truth = torch.LongTensor([label])
        if cuda_no != -1:
            tensor_sequence, ground_truth = tensor_sequence.cuda(cuda_no), ground_truth.cuda(cuda_no)
        optim.zero_grad()
        output, hx = forware_pass(rnn, tensor_sequence, MTYPE_GRU)
        pr_dstr = rnn.output_pr_dstr(hx[-1])
        loss = F.nll_loss(pr_dstr, ground_truth)
        loss.backward()
        optim.step()  # update parameters

    with open(model_path,"wb") as f:
        pickle.dump(rnn.cpu(),f)

    print(rnn.classify_word("", -1))

def train_imdb_for_exact_learning(train_data_path,model_save_name):
    params = {}
    params["input_size"] = 300
    params["output_size"] = 2
    params["hidden_size"] = 10
    params["num_layers"] = 3
    params["lr"] = 0.001
    cuda_no = 0
    params["rnn_type"] = MTYPE_GRU
    params["min_acc"] = 0.9999
    params["epocs"] = 1000
    params["model_save"] = "./rnn_models/pretrained/imdb_dfa/"+model_save_name

    word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    stop_words_list_path = "/home/dgl/project/pfa-data-generator/data/stopwords.txt"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    dataProcessor = IMDB_Data_Processor_DFA(word2vec_model, stop_words_list_path)
    rnn = GRU3(params["input_size"] , params["hidden_size"], params["num_layers"], params["output_size"])
    train_data = dataProcessor.load_data(train_data_path)
    train_data.append(("$", 1)) # add the "empty string"
    train_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, None, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=params["input_size"],
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"])

    print("training acc:{}".format(train_acc))

    with open(params["model_save"], "wb") as f:
        pickle.dump(rnn.cpu(), f)






def avg_length():
    #Total:400,average length :111.475,alphabet size:11173
    # train_data_path = "/home/dgl/project/pfa-data-generator/data/pfa_expe3/test"
    # Total:99,average length :16.0606060606,alphabet size:936

    train_data_path = "/home/dgl/project/pfa_extraction/data/imdb_for_dfa/dataset3"
    word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    stop_words_list_path = "/home/dgl/project/pfa-data-generator/data/stopwords.txt"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    dataProcessor = IMDB_Data_Processor_DFA(word2vec_model, stop_words_list_path)
    train_data = dataProcessor.load_clean_data(train_data_path, 5000)
    alphabet = dataProcessor.make_alphabet(train_data)
    leg_count = 0
    for seq in train_data:
        leg_count+=len(seq)
    alphabet = dataProcessor.make_alphabet(train_data)
    print("Total:{},average length :{},alphabet size:{}".format(len(train_data),leg_count/len(train_data),len(alphabet)))


if __name__ == "__main__":
    avg_length()


    # dataset1 = "/home/dgl/project/pfa_extraction/data/imdb_for_dfa"
    # dataset2="/home/dgl/project/pfa_extraction/data/imdb_for_dfa/dataset2"
    # dataset3="/home/dgl/project/pfa_extraction/data/imdb_for_dfa/dataset3"
    # model_save_name = "gru3.pkl"
    # train_imdb_for_exact_learning(dataset3,model_save_name)

    # with open("./rnn_models/pretrained/imdb_dfa/gru.pkl","r") as f:
    #     rnn=pickle.load(f)
    #
    # word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    # stop_words_list_path = "/home/dgl/project/pfa-data-generator/data/stopwords.txt"
    # folder_path = "/home/dgl/project/pfa_extraction/data/imdb_for_dfa"
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    # dataProcessor = IMDB_Data_Processor_DFA(word2vec_model, stop_words_list_path)
    # rnn.set_dataProcessor(dataProcessor)
    #
    # # rnn.eval()
    # # test_data = dataProcessor.load_data(folder_path=folder_path)
    # # dscb,acc=tets_accuracy_concise(rnn, test_data,0)
    # # print(dscb)
    #
    # clean_data = dataProcessor.load_clean_data(folder_path=folder_path)
    # dscb,acc=tets_accuracy_concise_with_clean_data(GRU3WrappperDFA(rnn), clean_data, 0)
    # print(dscb)
    # reload_clean_data = dataProcessor.load_clean_data(folder_path=folder_path)
    # dscb, acc = tets_accuracy_concise_with_clean_data(GRU3WrappperDFA(rnn), clean_data, 0)
    # print(dscb)

    # input_dim = 300
    # num_class = 2
    # cuda_no = 0
    # RNN_TYPE = MTYPE_GRU
    # train_descr, train_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
    #                                        RNN_TYPE=RNN_TYPE)

    # params = {}
    ########################
    # Trainig Tomita Data
    ########################
    # for grammar in [3,6,5,1,2,4,7]:
    #     print("training Tomita{}".format(grammar))
    #     params["lr"] = 0.001
    #     params["epocs"] = 100
    #     params["rnn_type"] = MTYPE_GRU
    #     params["min_acc"] = 0.99999
    #     params["hidden_size"] = 100
    #     params["train_data_path"] = "./data/tomita/training/tomita" + str(grammar) + ".pkl"
    #     params["test_data_path"] = "./data/tomita/test/tomita" + str(grammar) + ".pkl"
    #     params["model_save"] = "./rnn_models/pretrained/tomita/gru-tomita" + str(grammar) + ".pkl"
    #     params["lr_decay"] = True
    #     params["alphabet_size"] = 3
    #     print("Params:{}\n".format(params))
    #     rnn = train_tomita(params, cuda_no=-1)
    #     break
    # rnn = rnn.cpu()
    #
    ########################
    # Trainig BP Data
    ########################
    # print("training BP languages....")
    # params["lr"] = 0.001
    # params["epocs"] = 100
    # params["rnn_type"] = MTYPE_GRU
    # params["min_acc"] = 0.9999
    # params["hidden_size"] = 50
    # params["alphabet_size"] = 29 # empty string, a-z, (, )
    # params["train_data_path"] = "./data/bp/gru_bp.pkl"
    # params["test_data_path"] = None
    # params["model_save"] = "./rnn_models/pretrained/bp/gru-gru_bp.pkl"
    # params["lr_decay"] = True
    #
    # print("Params:{}\n".format(params))
    # rnn = train_bp(params, cuda_no=0)

    # special_train()


