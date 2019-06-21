import os
from utils.save_function import DataPersistence
# from data_factory.spam_filtering.spam_data_process import extract_subject
import pickle
# DataPersistence('./')
from tempfile import mkdtemp



if __name__ == '__main__':
    # dir = '../data/spam/data_exp/test/spam/'  # TRAIN_02622.eml
    # a = []
    # for file in os.listdir(dir):
    #     # print dir + file
    #     a.append(extract_subject(dir + file) == extract_subject(dir + file))
    # print False in a, a
    # with open('/home/leor/project/ase2019_exper/pfa_extractor/rnn_models/pretrained/bp/gru-bp.pkl', 'r') as f:
    #     rnn = pickle.load(f)
    #     print rnn
    persistence = DataPersistence('/tmp/test')
    persistence.save_output([[1, 2, 3, 4, 5, 6, 7]], './storage/bp/outcome/bp_hier_test')
