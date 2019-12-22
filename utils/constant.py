import os

def get_path(r_path):
    return os.path.join(PROJECT_ROOT,r_path)

PROJECT_ROOT="/home/dgl/project/pfa_extraction"
WORD2VEC_PATH = "data/wordvec/GoogleNews-vectors-negative300.bin"
GLOVE2VEC_PATH = "data/wordvec/glove_word2vec_f.txt"
SENTENCE_ENCODER_PATH = "textbugger/universal-sentence-encoder"
STOP_WORDS_PATH = "data/stopwords.txt"


MTYPE_SRNN='SRN'
MTYPE_LSTM='LSTM'
MTYPE_GRU='GRU'



class ModelPath:
    class BP:
        GRU  = ""
        LSTM = ""
    class TOMITA:
        GRU  = ""
        LSTM = ""
    class IMDB:
        GRU  = "rnn_models/pretrained/imdb_dfa/GRU/pfa_expe3-train_acc-0.953-test_acc-0.853.pkl"
        LSTM = "rnn_models/pretrained/imdb_dfa/LSTM/pfa_expe3-train_acc-0.9112-test_acc-0.834.pkl"
    class SPAM:
        GRU  = ""
        LSTM = ""

class DataPath:
    class BP:
        TRAIN = ""
        TEST = ""

    class TOMITA:
        TRAIN = ""
        TEST = ""

    class IMDB:
        TRAIN = "data/imdb/pfa_expe3/train"
        TEST = "data/imdb/pfa_expe3/test"

    class SPAM:
        TRAIN = ""
        TEST = ""

    class MR:
        RAW_DATA = "data/mr/raw"
        PROCESSED_DATA = "data/mr/processed_mr.pkl"
        WV_MATRIX = "data/mr/mr_wv_matrix.pkl"



