import sys

sys.path.append('../')
from utils.save_function import DataPersistence
from utils.constant import *
from utils.time_util import *
from data_factory.imdb_sentiment.imdb_data_process import *
from pfa_extractor.trace_processor import *
from pfa_build.abs_trace_extractor import AbstractTraceExtractor
from pfa_build.pfa import build_pfa
from pfa_build.test_acc_pfa import *
import os
import gensim
import pickle
from models.gated_rnn import *


class PathConstant:
    PROJECT_PATH = "/home/dgl/project/pfa_extraction"
    DATAID = 'pfa_expe3'
    MODEL_TYPE = "GRU"
    WORD2VEC_PATH = os.path.join(PROJECT_PATH, "data/GoogleNews-vectors-negative300.bin")
    VARIABLES_PATH = os.path.join(PROJECT_PATH, 'experiments/variables.txt')
    # PFA output dir
    PFA_SAVE_DIR = os.path.join(PROJECT_PATH, 'storage/imdb/pfa_construction/hier', DATAID, MODEL_TYPE)
    RNN_MODEL_PATH = os.path.join(PROJECT_PATH,
                                  'rnn_models/pretrained/imdb_dfa/GRU/pfa_expe3-train_acc-0.953-test_acc-0.853.pkl')
    root_path = os.path.join(PROJECT_PATH, 'storage/imdb/traces_data/hier')
    STOP_WORDS_PATH = os.path.join(PROJECT_PATH, 'data/imdb//stopwords.txt')
    DATA_PATH = os.path.join(PROJECT_PATH, 'data/imdb/pfa_expe3/test')


if __name__ == '__main__':

    '''
    Only for GRU, IMDB, pfa_expe3 usage.
    '''
    train_size = 500
    random_seed = 11921002
    num_clusters = 2
    input_dim = 300
    max_length = 60000

    if not os.path.exists(PathConstant.PFA_SAVE_DIR):
        os.makedirs(PathConstant.PFA_SAVE_DIR)

    print('loading word2vec model....')
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        PathConstant.WORD2VEC_PATH, binary=True)

    extractor = AbstractTraceExtractor()
    data_processor = IMDB_Data_Processor(word2vec_model, PathConstant.STOP_WORDS_PATH)
    model_path = PathConstant.RNN_MODEL_PATH
    ##############
    # loading data
    #############
    train_data = data_processor.load_data(PathConstant.DATA_PATH)

    ################
    # loading model
    ################
    with open(model_path, 'r') as f:
        rnn = pickle.load(f)

    trace_processor = TraceProcessor(extractor, rnn, train_data, data_processor, input_dim)
    trace_processor.init_hier_parttiion(num_clusters)
    input_traces_pfa = trace_processor.get_pfa_input_trace()
    persistence = DataPersistence(os.path.join(PathConstant.PFA_SAVE_DIR, "depth-" + str(num_clusters)))

    ###########################
    #save traces_outputs_list, ori_traces_size_list,
    #words_traces, predict_ground_list, ori_points
    ###########################
    persistence.save_train_data(*(trace_processor.get_train_data()))
    output_path = os.path.join(persistence.root_path, "dtmc")
    input_trace_path = persistence.save_pfa_input_trace(input_traces_pfa)
    pfa, used_traces_path = build_pfa(input_trace_path, PathConstant.MODEL_TYPE, max_length,
                                      output_path, PathConstant.VARIABLES_PATH)
    (acc_a, fdlt_a, rnn_acc_a), (acc_w, fdlt_w, rnn_acc_w) = test_pfa_acc(pfa, used_traces_path, persistence)
    print("depth:{}, acc_action:{}, fdlt_action:{}, acc_words:{}, fdlt_words:{}".format(num_clusters, acc_a, fdlt_a,
                                                                                        acc_w, fdlt_w))
    trace_processor.tmp_clear()
