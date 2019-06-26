import sys
sys.path.append('.')
from utils.logger import Logger
from utils.save_function import *
from utils.constant import *
from utils.time_util import *
from data_factory.tomita.tomita_processor import TomitaDataProcessor
from pfa_extractor.trace_processor import *
from pfa_build.abs_trace_extractor import AbstractTraceExtractor
from pfa_build.pfa import build_pfa
from pfa_build.pfa_predict import *
import os


if __name__ == '__main__':
    sys.stdout = Logger('./logs/tomita/tomita_hier_refine_out.log', sys.stdout)
    sys.stderr = Logger('./logs/tomita/tomita_hier_refine_err.log', sys.stderr)
    variables_path = './variables.txt'
    root_path = '../storage/tomita/traces_data/hier_refine'
    max_deepth = 20
    n_init = 10
    train_size = 500
    random_seed = 5566
    k_cluster = 2
    input_dim = 3
    max_length = 60000
    pfa_save_root = '../storage/tomita/pfa_construction/hier_refine'
    models_root = '../rnn_models/pretrained/tomita'
    data_root = '../data/tomita/training'
    models_type = {MTYPE_GRU:'test-gru-tomita1.pkl', MTYPE_LSTM:'test-lstm-tomita1.pkl'}
    max_iters = 20
    data = 'tomita1.pkl'
    extractor = AbstractTraceExtractor()
    data_processor = TomitaDataProcessor()
    output_list = []
    for rnn_type in [MTYPE_LSTM]: # MTYPE_GRU,
        print('==============RNN:{}=====DATA:{}================'.format(rnn_type, data))
        train_path = os.path.join(data_root, data)
        model_path = os.path.join(models_root, models_type[rnn_type])
        output_path = os.path.join(pfa_save_root, data.split('.')[0])

        print('=====================pfa learning with hierarchical cluster to start!===================')
        persistence = DataPersistence(os.path.join(root_path, rnn_type))
        train_data = data_processor.load_data(train_path)

        with open(model_path, 'r') as f:
            rnn = pickle.load(f).cuda()
        print('Doing abstract initial with k={}....'.format(k_cluster))
        time = Time()
        trace_processor = TraceProcessor(extractor, rnn, train_data, data_processor, input_dim)
        trace_processor.init_hier_refine_parttiion(k_cluster)
        input_traces_pfa = trace_processor.get_pfa_input_trace(null_added=True)
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
            input_traces_pfa = trace_processor.hier_refine_input_update(pfa, used_traces_path,
                                                                        persistence.trace_path, null_added=True)
        persistence.save_output(output_list, '../storage/bp/outcome/bp_hier_refine_' + rnn_type)



