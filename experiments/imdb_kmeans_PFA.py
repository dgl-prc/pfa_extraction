import sys
sys.path.append('.')
from utils.logger import Logger
from utils.save_function import *
from utils.constant import *
from utils.time_util import *
from data_factory.imdb_sentiment.imdb_data_process import *
from pfa_extractor.trace_processor import *
from pfa_build.abs_trace_extractor import AbstractTraceExtractor
from pfa_build.pfa import build_pfa
from pfa_build.test_acc_pfa import *
import os
import gensim


if __name__ == '__main__':
    sys.stdout = Logger('./logs/imdb/imdb_kmeans_out.log', sys.stdout)
    sys.stderr = Logger('./logs/imdb/imdb_kmeans_err.log', sys.stderr)
    variables_path = './variables.txt'
    root_path = '../storage/bp/traces_data/hier_refine'

    print('loading word2vec model....')
    word2vec_model_path = "../rnn_models/pretrained/GoogleNews-vectors-negative300.bin"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=True)

    max_deepth = 20
    n_init = 10
    train_size = 500
    random_seed = 5566
    k_cluster = 2
    input_dim = 300
    max_length = 11921002
    pfa_save_root = '../storage/imdb/pfa_construction/kmeans'
    models_root = '../rnn_models/pretrained/imdb_dfa'
    data_path = '../data/imdb/pfa_expe3/test'
    data = 'pfa_expe3'
    models_type = {MTYPE_GRU:'pfa_expe3-train_acc-0.953-test_acc-0.853.pkl',
                   MTYPE_LSTM:'pfa_expe3-train_acc-0.9112-test_acc-0.834.pkl'}
    stop_words_list_path = '../data/imdb/stopwords.txt'
    max_iters = 20
    extractor = AbstractTraceExtractor()
    data_processor = IMDB_Data_Processor(word2vec_model, stop_words_list_path)
    output_list = []
    for rnn_type in [MTYPE_GRU, MTYPE_LSTM]:
        print('==============RNN:{}=====DATA:{}================'.format(rnn_type, data))
        model_path = os.path.join(models_root, rnn_type, models_type[rnn_type])
        output_path = os.path.join(pfa_save_root, 'pfa_expe3')

        print('=====================pfa learning with hierarchical cluster to start!===================')
        persistence = DataPersistence(os.path.join(root_path, rnn_type))
        train_data = data_processor.load_data(data_path)

        with open(model_path, 'r') as f:
            rnn = pickle.load(f)
        print('Doing abstract initial with k={}....'.format(k_cluster))
        time = Time()
        trace_processor = TraceProcessor(extractor, rnn, train_data, data_processor, input_dim)
        trace_processor.init_kmeans_refine_parttition(k_cluster)
        input_traces_pfa = trace_processor.get_pfa_input_trace()
        persistence.save_train_data(*(trace_processor.get_train_data()))
        deepth = 2
        while deepth <= max_deepth:
            pfa, used_traces_path = build_pfa(input_traces_pfa, rnn_type, max_length,
                                             output_path, persistence.trace_path, variables_path)
            print("=====Training ACC of deepth:{}=======".format(deepth))
            acc, fdlt, rnn_acc = get_pfa_acc_v2(pfa, used_traces_path, persistence)
            elasped = time.time_counter()
            print elasped, deepth
            output_list.append([data, rnn_type, deepth, acc, fdlt, rnn_acc, elasped]); deepth += 1
            input_traces_pfa = trace_processor.kmeans_input_update(pfa, used_traces_path,
                                                                        persistence.trace_path)
        persistence.save_output(output_list, '../storage/imdb/outcome/expe_3/imdb_kmeans_refine_' + rnn_type)



