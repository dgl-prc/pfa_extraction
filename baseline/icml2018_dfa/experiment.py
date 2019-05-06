from __future__ import division
import sys
sys.path.append("../../")
from Extraction import extract
from rnn_models.rnn_arch.gated_rnn import *
import pickle
import os
import string
import pickle
from LSTM import LSTMNetwork
from GRU import GRUNetwork
from RNNClassifier import RNNClassifier
from Training_Functions import make_train_set_for_target, mixed_curriculum_train
from Tomita_Grammars import tomita_1, tomita_2, tomita_3, tomita_4, tomita_5, tomita_6, tomita_7
from Extraction import extract
from data_factory.imdb_sentiment.imdb_data_process import IMDB_Data_Processor_DFA
import gensim

class Wrapper:
    def __init__(self, rnn, alphabet, target, name, train_set):
        self.rnn = rnn
        self.alphabet = alphabet
        self.target = target
        self.name = name
        words = sorted(list(train_set.keys()), key=lambda x: len(x))
        short_pos = next((w for w in words if target(w) == True), None)
        short_neg = next((w for w in words if target(w) == False), None)
        self.starting_examples = [w for w in [short_pos, short_neg] if not None == w]
        self.train_set = train_set
        self.dfas = []

    def __repr__(self):
        return self.name


class RNNPathWrapper:
    '''
    Since I wrongly defined the get_first_RState(self) in the orginal RNN and do not want to train them again, I just
    use this class to reload the function.
    '''

    def __init__(self, rnn):
        self.rnn = rnn
        self.alphabet = rnn.alphabet

    def get_first_RState(self):
        Rstate, label = self.rnn.get_first_RState()
        return self.rnn.get_next_RState(Rstate, "")

    def get_next_RState(self, h_t, char):
        return self.rnn.get_next_RState(h_t, char)

    def classify_word(self, word, cuda_num=0):
        return self.rnn.classify_word(word, cuda_num)


def test_accuracy(fa_model, dataset):
    '''
    :param model:
    :param dataset:
    :return:
    '''
    correct = 0
    for key in dataset.keys():
        label = dataset[key]
        pdt = fa_model.classify_word(key)
        if pdt == label:
            correct += 1
    return correct / len(dataset)


def test_accuracy_rnn(fa_model, dataset):
    '''
    :param model:
    :param dataset:
    :return:
    '''
    correct = 0
    for key in dataset.keys():
        label = dataset[key]
        pdt = fa_model.classify_word(key, -1)
        if pdt == label:
            correct += 1
    return correct / len(dataset)


def test_fidelity(fa_model, rnn, dataset):
    count = 0
    for key in dataset.keys():
        fa_pdt = fa_model.classify_word(key)
        rnn_pdt = rnn.classify_word(key, -1)
        if fa_pdt == rnn_pdt:
            count += 1
    return count / len(dataset)


def get_start_samples(dataset, rnn):
    all_words = sorted(list(dataset.keys()), key=lambda x: len(x))
    pos = next((w for w in all_words if rnn.classify_word(w, -1) == True), None)
    neg = next((w for w in all_words if rnn.classify_word(w, -1) == False), None)
    starting_examples = [w for w in [pos, neg] if not None == w]
    return starting_examples

# def get_start_samples_real_data(dataset, rnn):
#
#     all_words = sorted(dataset, key=lambda x: len(x[0]))
#     pos = next((w for w,label in all_words if rnn.classify_word(" ".join(w), -1) == True), None)
#     neg = next((w for w,label in all_words if rnn.classify_word(" ".join(w), -1) == False), None)
#     starting_examples = [w for w in [pos, neg] if not None == w]
#     return starting_examples


def extract_on_tomita():
    model_folder = "../../rnn_models/pretrained/tomita"
    data_folder = "../../data/tomita/training"
    import pickle
    import os
    grammars = ["tomita1", "tomita2", "tomita3", "tomita4","tomita5",  "tomita6", "tomita7"]
    for grammar in grammars:
        print("Testing {}....\n".format(grammar))
        model_path = os.path.join(model_folder, "test-gru-{}.pkl".format(grammar))
        data_path = os.path.join(data_folder, "{}.pkl".format(grammar))
        with open(model_path, "rb") as f:
            rnn = pickle.load(f)
            rnn = RNNPathWrapper(rnn)
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        train_set = dataset["data"]

        starting_examples = get_start_samples(train_set, rnn)
        dfa = extract(rnn, time_limit=None, initial_split_depth=10, starting_examples=starting_examples)
        acc = test_accuracy(dfa, train_set)
        fdlt = test_fidelity(dfa, rnn, train_set)
        print("DFA,Accuracy:{},Fidelity:{}".format(acc, fdlt))


def extract_tomita_with_build_in_rnn():
    alphabet = "01"
    data_folder = "../../data/tomita/training"
    grammars = ["tomita1", "tomita3", "tomita4", "tomita7", "tomita6", "tomita5", "tomita1", "tomita2"]
    for grammar in grammars:
        print("Testing {}....\n".format(grammar))
        data_path = os.path.join(data_folder, "{}.pkl".format(grammar))
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        train_set = dataset["data"]
        rnn = RNNClassifier(alphabet, num_layers=2, hidden_dim=100, RNNClass=GRUNetwork)
        mixed_curriculum_train(rnn, train_set, stop_threshold=0.0005)
        # with open("./icml-gru-{}.pkl".format(grammar),"wb") as f:
        #     pickle.dump(rnn,f)
        starting_examples = get_start_samples(train_set, rnn)
        dfa = extract(rnn, time_limit=None, initial_split_depth=10, starting_examples=starting_examples)
        acc = test_accuracy(dfa, train_set)
        fdlt = test_fidelity(dfa, rnn, train_set)
        print("DFA,Accuracy:{},Fidelity:{}".format(acc, fdlt))


def extract_on_bp():
    model_folder = "../../rnn_models/pretrained/bp"
    data_folder = "../../data/bp/"

    print("Testing {balanced parentheses}....\n")
    model_path = os.path.join(model_folder, "gru-bp.pkl")
    data_path = os.path.join(data_folder, "bp.pkl")
    with open(model_path, "rb") as f:
        rnn = pickle.load(f).cpu()
        rnn = RNNPathWrapper(rnn)
    with open(data_path, "rb") as f:
        dataset = pickle.load(f)

    train_set = dataset["data"]
    rnn.alphabet = list("()" + string.ascii_lowercase)

    starting_examples = get_start_samples(train_set, rnn)
    dfa = extract(rnn, time_limit=50, initial_split_depth=10, starting_examples=starting_examples)
    acc = test_accuracy(dfa, train_set)
    fdlt = test_fidelity(dfa, rnn, train_set)
    print("DFA,Accuracy:{},Fidelity:{}".format(acc, fdlt))


def extract_on_imdb():
    model_path = "../../rnn_models/pretrained/imdb_dfa/gru.pkl"
    data_folder = "../../data/imdb_for_dfa/"
    stop_words_list_path = "../../data/imdb_for_dfa/stopwords.txt"
    print("Testing {IMDB dataset}....\n")

    #############
    # load RNN
    #############
    word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_model_path, binary=True)
    dataProcessor = IMDB_Data_Processor_DFA(word2vec_model, stop_words_list_path)
    alphabet = dataProcessor.make_alphabet(data_folder=data_folder)
    with open(model_path, "rb") as f:
        rnn = pickle.load(f)
    rnn.set_alphabet(alphabet)
    print("size of alphabet:{}".format(len(alphabet)))
    rnn.set_dataProcessor(dataProcessor)

    rnnWrapper = GRU3WrappperDFA(rnn)
    rnnWrapper.eval()

    ####################
    # load train_data
    ###################
    train_set=dataProcessor.load_clean_data(data_folder)
    starting_examples = get_start_samples(train_set, rnnWrapper)
    dfa = extract(rnnWrapper, time_limit=None, initial_split_depth=10, starting_examples=starting_examples,real_sense=True)
    acc = test_accuracy(dfa, train_set)
    fdlt = test_fidelity(dfa, rnnWrapper, train_set)
    print("DFA,Accuracy:{},Fidelity:{}".format(acc, fdlt))




def debug_Mylist(train_set, rnnWrapper):
    cnt = 0
    for w in train_set:
        pdt =  rnnWrapper.classify_word(w, -1)
        label = train_set[w]
        if pdt==label:
            cnt+=1
    print cnt/len(train_set)


def check_accuracy():
    model_folder = "../../rnn_models/pretrained/tomita"
    data_folder = "../../data/tomita/training"

    grammars = ["tomita1", "tomita2", "tomita3", "tomita4","tomita5",  "tomita6", "tomita7"]
    for grammar in grammars:
        print("Testing {}....\n".format(grammar))
        model_path = os.path.join(model_folder, "test-gru-{}.pkl".format(grammar))
        data_path = os.path.join(data_folder, "{}.pkl".format(grammar))
        with open(model_path, "rb") as f:
            rnn = pickle.load(f)
        with open(data_path, "rb") as f:
            dataset = pickle.load(f)
        train_set = dataset["data"]

        # acc = test_accuracy_rnn(rnn, train_set)
        print("{},empty:{}".format(grammar, rnn.classify_word("", -1)))








if __name__ == "__main__":
    # extract_on_tomita()
    # check_accuracy()
    extract_on_imdb()

    # extract_tomita_with_build_in_rnn()

    # extract_on_bp()

    # model_folder = "../../rnn_models/pretrained/tomita"
    # data_folder = "../../data/tomita/training"
    #
    # model_path = os.path.join(model_folder, "test-gru-{}.pkl".format("tomita6"))
    # data_path = os.path.join(data_folder, "{}.pkl".format("tomita6"))
    # with open(model_path, "rb") as f:
    #     rnn = pickle.load(f)
    # with open(data_path, "rb") as f:
    #     dataset = pickle.load(f)
    # dataset = dataset["data"]
    #
    # correct = 0
    # for key in dataset.keys():
    #     label = dataset[key]
    #     pdt = rnn.classify_word(key,-1)
    #     if pdt==label:
    #         correct+=1
    # print correct/len(dataset)
