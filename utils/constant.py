project_root="/home/dgl/project/pfa_extraction"

MTYPE_SRNN='SRN'
MTYPE_LSTM='LSTM'
MTYPE_GRU='GRU'


word2vec_path = "rnn_models/pretrained/GoogleNews-vectors-negative300.bin"

class ModelPath:
    class BP:
        GRU  = ""
        LSTM = ""
    class TOMITA:
        GRU  = ""
        LSTM = ""
    class IMDB:
        GRU  = "rnn_models/pretrained/imdb_dfa/pfa_expe3-train_acc-0.953-test_acc-0.853.pkl"
        LSTM = "rnn_models/pretrained/imdb_dfa/pfa_expe3-train_acc-0.9112-test_acc-0.834.pkl"
    class SPAM:
        GRU  = ""
        LSTM = ""

