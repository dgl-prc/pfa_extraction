import sys, os
import argparse
sys.path.append('./')
import torch
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.gated_rnn import LSTM, GRU,GRU2
from data_process.imdb_sentiment.imdb_data_process import IMDB_Data_Processor
from utils.time_util import current_timestamp
import pickle
from utils.constant import *
import numpy as np
import math
from data_process.tomita.generator import TomitaDataProcessor



def adjust_learing_rate(optimizer,lr,epoch,step):
    new_lr = lr*(0.1**(epoch//step))
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
    for sequence in test_data.keys():
        label = test_data[sequence]
        tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim)  # Lx1xn_letters
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


def train_with_optim(rnn, lr, epocs, train_data, test_data, dataProcessor, cuda_no=-1, input_dim=2, num_class=2,
                     RNN_TYPE=MTYPE_SRNN, lower_bound_acc=0.9,lr_decay=False):
    watch_step = 100
    average_loss = 0
    optim = torch.optim.Adam(rnn.parameters(), lr)

    test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
                                         RNN_TYPE=RNN_TYPE)
    train_descr, train_acc = test_accuracy(rnn, train_data, dataProcessor, input_dim, num_class, cuda_no,
                                           RNN_TYPE=RNN_TYPE)
    print(current_timestamp() + '****************======>Init train_acc:' + str(train_descr) + 'Init test acc:' + str(
        test_acc))

    rnn.train()
    if cuda_no != -1:
        rnn = rnn.cuda(cuda_no)
    for epoc in range(epocs):
        count = 0
        if lr_decay and lr>1e-5:
            adjust_learing_rate(optim, lr, epoc, 40)
        for sequence in train_data.keys():
            label = train_data[sequence]
            tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim)  # Lx1xn_letters
            ground_truth = torch.LongTensor([label])
            if cuda_no != -1:
                tensor_sequence, ground_truth = tensor_sequence.cuda(), ground_truth.cuda(cuda_no)

            optim.zero_grad()
            # output, states = forware_pass(rnn, tensor_sequence, RNN_TYPE)
            # pr_distr = output[0][-1].unsqueeze(0)
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

        test_descr, test_acc = test_accuracy(rnn, test_data, dataProcessor, input_dim, num_class, cuda_no,
                                             RNN_TYPE=RNN_TYPE)
        train_descr, train_acc = test_accuracy(rnn, train_data, dataProcessor, input_dim, num_class, cuda_no,
                                               RNN_TYPE=RNN_TYPE)
        print(current_timestamp() + '****************======>train_acc:' + str(
            train_descr) + 'test acc:' + str(test_descr))
        rnn.train()
        if test_acc >= lower_bound_acc and train_acc >= lower_bound_acc:
            return train_acc, test_acc
    print('Warning:Training Failed!!!!!Can not make model\'s accuracy to be {}!'.format(lower_bound_acc))
    return train_acc, test_acc


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

    with open(params["datapath"],"rb") as f:
        data = pickle.load(f)
    pos_data,pos_labels = data["pos"]
    neg_data,neg_labels = data["neg"]

    total_data=np.concatenate((pos_data,neg_data))
    total_label=np.concatenate((pos_labels,neg_labels))

    #shuffle
    idcies=range(len(total_data))
    np.random.shuffle(idcies)
    total_data = total_data[idcies]
    total_label = total_label[idcies]

    # split  5:1 train to test
    import math
    train_size = int(math.floor(0.8*len(total_data)))
    train_data = total_data[:train_size]
    train_label = total_label[:train_size]

    test_data = total_data[train_size:]
    test_label = total_label[train_size:]

    test_data = {sequence:label for sequence,label in zip(test_data,test_label)}
    train_data =  {sequence:label for sequence,label in zip(train_data,train_label)}

    print("Training Data:pos({})".format(sum(train_label)))
    print("Testing Data:pos({})".format(sum(test_label)))

    with open( params["split_data_path"],"wb") as f:
        pickle.dump({"train":train_data,"test":test_data},f)

    torch.random.manual_seed(199312)
    dataProcessor = TomitaDataProcessor()

    rnn=GRU2(raw_input_size=2,innder_input_dim=3,num_class=2,hidden_size=params["hidden_size"],num_layers=2,dataProcessor=dataProcessor)

    descr, acc=test_accuracy(rnn, test_data, dataProcessor, 2, 2, -1,RNN_TYPE=MTYPE_GRU)
    print("Before Learing:{}".format(descr))


    print("Begine Training......")
    train_acc, test_acc = train_with_optim(rnn, params["lr"], params["epocs"], train_data, test_data, dataProcessor,
                                           cuda_no=cuda_no,
                                           input_dim=2,
                                           RNN_TYPE=params["rnn_type"],
                                           lower_bound_acc=params["min_acc"],
                                           lr_decay=params["lr_decay"])

    print("training acc:{},testing acc:{}".format(train_acc,test_acc))

    with open(params["model_save"],"wb") as f:
        pickle.dump(rnn.cpu(),f)

    return rnn

if __name__ == "__main__":
    # import gensim
    #
    # ###############
    # # load word2vec
    # ###############
    # word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    # stop_words_list_path = "/home/dgl/project/pfa-data-generator/data/stopwords.txt"
    # print('loading word2vector model....')
    # word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    #     word2vec_model_path, binary=True)
    # dataProcessor = IMDB_Data_Processor(word2vec_model, stop_words_list_path)
    #
    params = {}
    # ###############
    # # model setting
    # ###############
    # params["input_size"] = 300
    # params["output_size"] = 2
    # params["hidden_size"] = 10
    # params["num_layers"] = 3
    # params["lr"] = 0.001
    # params["epocs"] = 5pfa_ijcai
    #
    # ####################
    # # artifacts setting
    # ####################
    # params["dataProcessor"] = dataProcessor
    # params["model_save_root"] = '/home/dgl/project/pfa-data-generator/models/pretrained/exp_ijcai19/5000'
    # params["data_root"] = "/home/dgl/project/pfa-data-generator/data/exp_ijcai19/5000"
    # params["data_groups"] = {1: "pfa_expe1", 2: "pfa_expe2", 3: "pfa_expe3", 4: "pfa_expe4", 5: "pfa_expe5"}
    #
    # #################
    # # training env settings
    # #################
    # cuda_no = 0
    # params["rnn_type"] = MTYPE_GRU
    # params["min_acc"] = 0.86
    # for data_id in range(1, 6):
    #     train_imdb(params, data_id, cuda_no)

    for grammar in [6]:
        print("training Tomita{}".format(grammar))
        params["lr"] = 0.001
        params["epocs"] = 1000
        params["rnn_type"] = MTYPE_GRU
        params["min_acc"] = 0.9999
        params["hidden_size"]=100
        params["datapath"] = "./data/tomita/tomita"+str(grammar)+".pkl"
        params["split_data_path"] = "./data/tomita/split_tomita"+str(grammar)+".pkl_icml"
        params["model_save"] = "./models/pretrained/tomita/gru-tomita"+str(grammar)+".pkl_icml"
        params["lr_decay"]=True
        rnn = train_tomita(params, cuda_no=0)
    rnn = rnn.cpu()
    # print(rnn.classify_word("11000"))
    # print(rnn.classify_word("1100010"))

    # with open("./models/pretrained/tomita/gru-tomita"+str(1)+".pkl_icml","rb") as f:
    #     rnn=pickle.load(f).cpu()
    # # print(rnn.classify_word("11000"))
    # # print(rnn.classify_word("1100"))
    # str = "1100"
    # dataProcessor=TomitaDataProcessor()
    # input_tensor = dataProcessor.sequence2tensor(str)
    # output, h_n = rnn(input_tensor)
    # print(output)
    # print("ouput shape {}".format(output.shape))
    # print(rnn.hx2list(h_n))
    # print("=========")
    # h_0 = rnn.get_first_RState()
    # for char in str:
    #     h_0=rnn.get_next_RState(h_0, char)
    #     print(h_0)
