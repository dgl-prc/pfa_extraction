import sys

sys.path.append("../")
import torch
from models.gated_rnn import *
from utils.constant import *
import pickle
from pfa_build.abs_trace_extractor import *
import os
from utils.save_function import *


################
# NOTE:
# 1. the number of samples used should be 500.
# 2. For the pfa, we also should specific the number of samples from the cluster stage, instead of build pfa.
################
def classify_state(abs_traces_list, traces_outputs_list, num_states=2, num_class=2):
    '''
    state is 0-index,so it does not include the init state
    :param abs_traces_list:
    :param traces_outputs_list:
    :param num_states:
    :param num_class:
    :return:
    '''

    ########
    #        label1 label2
    # state1 #count #count
    # state2 #count #count
    #
    #########
    countMatrix = np.zeros((num_states, num_class), dtype=float)
    for abs_trace, output_trace in zip(abs_traces_list, traces_outputs_list):
        for state, label in zip(abs_trace, output_trace):
            countMatrix[state][label] += 1.

    print countMatrix

    state2label = []  # index denotes the state.
    for row_id, row in enumerate(countMatrix):
        countMatrix[row_id] /= np.sum(row)
        state2label.append(np.argmax(countMatrix[row_id]))

    return state2label


def getStateOutout(currentState, state2label):
    '''
    currentState is 1-index
    :param currentState:
    :param state2label:
    :return:
    '''
    return state2label[currentState - 1]


def make_trans_matrix(abs_traces_list, words_traces, num_clusters):
    '''
    let 0 as the init state, so the cluster 0 should be state 1, and cluster 1 should be state 2.
    :param abs_traces_list:
    :param words_traces:
    :return:
    '''
    #####################
    # count word trigger
    ####################
    words_trans_matrix = {}
    for abs_trace, words_trace in zip(abs_traces_list, words_traces):
        current_state = 0
        for next_state, word in zip(abs_trace, words_trace):
            next_state += 1  # change into 1-index.
            if word not in words_trans_matrix.keys():
                words_trans_matrix[word] = np.zeros((num_clusters + 1, num_clusters + 1), dtype=int)
                words_trans_matrix[word][current_state, next_state] = 1
            else:
                words_trans_matrix[word][current_state, next_state] += 1
            current_state = next_state
    return words_trans_matrix


def predict(word_sequence, transMatix, state2label):
    currentState = 0
    stateTrace = [currentState]
    isTerminate = False
    for word in word_sequence:
        if not word in transMatix.keys():
                isTerminate = True
                break
        candidates = transMatix[word][currentState]
        if np.sum(candidates) == 0:
            isTerminate = True
            break
        currentState = np.argmax(candidates)
        stateTrace.append(currentState)
    prdt = getStateOutout(currentState, state2label)
    return prdt, stateTrace, isTerminate


def get_fsa_acc_fdlt(words_traces, predict_ground_list, transMatix, state2label):
    acc_count = 0
    fdlt_count = 0
    rnn_acc_check = 0
    num_terminate = 0
    total_samples = len(words_traces)
    for word_sequence, prdt_grnd in zip(words_traces, predict_ground_list):
        fsa_prdt, stateTrace, isTerminate = predict(word_sequence, transMatix, state2label)
        rnn_prdt, grnd = prdt_grnd
        if isTerminate:
            num_terminate += 1
        if int(fsa_prdt) == int(grnd):
            acc_count += 1.
        if int(fsa_prdt) == int(rnn_prdt):
            fdlt_count += 1.
        if int(rnn_prdt) == int(grnd):
            rnn_acc_check += 1.

    print("total_samples:{},acc_count:{},fdlt_count:{},terminate:{}".format(total_samples, acc_count, fdlt_count,
                                                                            num_terminate))
    acc = acc_count / total_samples
    fdlt = fdlt_count / total_samples
    rnn_acc = rnn_acc_check / total_samples
    return acc, fdlt, rnn_acc


def save_excepted_samples(words_traces, predict_ground_list, transMatix, state2label, file_names):
    '''
    Save two types samples:
        1. Samples that be predicted by RNN but FSA
        2. Samples that the predict results of FSA and RNN are not consistent.

    :param words_traces:
    :param predict_ground_list:
    :param transMatix:
    :param state2label:
    :return:
    '''

    acc_count = 0
    fdlt_count = 0

    notCorrectWordsTrace = []
    notCorrectStateTrace = []
    notCorrectFielName = []

    notConsistentWordsTrace = []
    notConsistentStateTrace = []
    notConsistentFielName = []

    ##############
    # debug
    ##############
    # sequnece_path = "/home/dgl/ijcai2019/casestudy/pfa/weRight/words_trace/neg-8591_4.txt"
    # word_trace = []
    # with open(sequnece_path, 'r') as f:
    #     for i, line in enumerate(f.readlines()):
    #         word_trace.append(line.strip())
    # fsa_prdt, stateTrace = predict(word_trace, transMatix, state2label)
    #

    for word_sequence, prdt_grnd, file_name in zip(words_traces, predict_ground_list, file_names):
        fsa_prdt, stateTrace, isTerminate = predict(word_sequence, transMatix, state2label)
        rnn_prdt, grnd = prdt_grnd
        if int(fsa_prdt) == int(grnd):
            acc_count += 1.
        elif int(fsa_prdt) != int(rnn_prdt):
            notCorrectWordsTrace.append(word_sequence)
            notCorrectStateTrace.append(stateTrace)
            notCorrectFielName.append(file_name)

        if int(fsa_prdt) == int(rnn_prdt):
            fdlt_count += 1.
        else:
            notConsistentWordsTrace.append(word_sequence)
            notConsistentStateTrace.append(stateTrace)
            notConsistentFielName.append(file_name)

    ##############
    # save data
    ##############
    save_folder = "/home/dgl/ijcai2019/casestudy/fsa"

    notCorrectSavePath = os.path.join(save_folder, "notCorrect")
    notCorrectWordsPath = os.path.join(notCorrectSavePath, "words_trace")
    notCorrectStatePath = os.path.join(notCorrectSavePath, "state_trace")
    save_word_trace(notCorrectWordsPath, notCorrectWordsTrace, notCorrectFielName)
    save_state_trans(notCorrectStatePath, notCorrectStateTrace, notCorrectFielName)
    #
    #
    #
    # notConsistentSavePath = os.path.join(save_folder,"notConsistent")
    # notConsistentWordsPath = os.path.join(notConsistentSavePath,"words_trace")
    # notConsistentStatePath = os.path.join(notConsistentSavePath,"state_trace")
    # save_word_trace(notConsistentWordsPath, notConsistentWordsTrace, notConsistentFielName)
    # save_state_trans(notConsistentStatePath, notConsistentStateTrace, notConsistentFielName)
    #
    # with open(os.path.join(save_folder,"namelist.pkl"),'w') as f:
    #     pickle.dump(notCorrectFielName,f)
    #     pickle.dump(notConsistentFielName,f)


def learn_fsa(params, root_path):
    if os.path.exists(root_path):
        shutil.rmtree(root_path)
    os.makedirs(root_path)

    extractor = AbstractTraceExtractor()
    word2vec_model = params["word2vec_model"]
    stop_words_list_path = params["stop_words_list_path"]
    n_init = params["n_init"]
    train_size = params["train_size"]
    k_cluster = params["k_cluster"]
    train_path = params["train_path"]  # original train data path
    model_path = params["model_path"]
    rnn_type = params["rnn_type"]
    random_seed = params["random_seed"]
    input_dim = 300

    time_stamp = current_timestamp()
    data_processor = IMDB_Data_Processor(word2vec_model, stop_words_list_path)

    train_data, file_names = data_processor.load_data(train_path, random_seed=random_seed, data_size=train_size,
                                                      return_file_name=True)  # return the filenames to check the data
    with open(model_path, 'r') as f:
        rnn = pickle.load(f)
    print('Doing abstract with k={}....'.format(k_cluster))

    ###################################################
    # The original trace: hidden states.
    # *****This code is copied from refienPFA.py******.
    ###################################################
    traces_list, traces_outputs_list, predict_ground_list, words_traces, _ = extractor.collect_hidden_state_seq(rnn,
                                                                                                                train_data,
                                                                                                                data_processor,
                                                                                                                input_dim)

    ###################################################
    # Clustering.  len(abs_trace)==len(words_trace)
    # *****This code is copied from refinePFA.py******.
    ###################################################
    ori_points, ori_traces_size_list = extractor.tracesList2vectorsList(traces_list)
    clustering_labels, cluster_centers, kmeans = extractor.state_partition(ori_points, k_cluster, n_init)
    abs_trace_list = extractor.vectorsList2newTraceList(clustering_labels, ori_traces_size_list)

    ######################
    #  FSA
    ######################
    words_trans_matrix = make_trans_matrix(abs_trace_list, words_traces, num_clusters=k_cluster)
    state2label = classify_state(abs_trace_list, traces_outputs_list, num_states=k_cluster, num_class=2)

    #####################
    # just for case study
    #####################
    # save_excepted_samples(words_traces, predict_ground_list, words_trans_matrix, state2label, file_names)
    # exit(2)

    acc, fdlt, rnn_acc = get_fsa_acc_fdlt(words_traces, predict_ground_list, words_trans_matrix, state2label)
    print('Accuracy of RNN:{}\nAccuracy of PFA:{}\nFidelity of PFA:{}'.format(rnn_acc, acc, fdlt))

    test_path = os.path.join(data_root, data_group, "train")
    test_data, file_names = data_processor.load_data(test_path, random_seed=random_seed, data_size=100,
                                                      return_file_name=True)
    _, _, t_predict_ground_list, t_words_traces, _ = extractor.collect_hidden_state_seq(rnn, test_data,
                                                                                        data_processor,
                                                                                        input_dim)
    acc, fdlt, rnn_acc = get_fsa_acc_fdlt(t_words_traces, t_predict_ground_list, words_trans_matrix, state2label)
    print('TESTing: Accuracy of RNN:{}\nAccuracy of PFA:{}\nFidelity of PFA:{}'.format(rnn_acc, acc, fdlt))


if __name__ == "__main__":
    import os
    import gensim

    max_length = 50000
    root_path = os.path.join("./tmp/", str(max_length))
    trace_path = os.path.join(root_path, "input_trace")
    word_traces_path = os.path.join(root_path, "words_trace")
    test_word_traces_path = os.path.join(root_path, "test_words_trace")
    rnn_prdct_grnd_path = os.path.join(root_path, "rnn_prdct_grnd.pkl")
    test_rnn_prdct_grnd_path = os.path.join(root_path, "test_rnn_prdct_grnd.pkl")
    variables_path = "./variables.txt"
    ori_points_path = os.path.join(root_path, "ori_points.pkl")
    hn_trace_info_path = os.path.join(root_path, "hn_trace_info.pkl")
    func_path = os.path.join(root_path, "func.pkl")
    import shutil

    params = {}
    print('loading word2vec model....')
    word2vec_model_path = "/home/dgl/project/pfa-data-generator/models/pretrained/GoogleNews-vectors-negative300.bin"
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        word2vec_model_path, binary=True)

    params["word2vec_model"] = word2vec_model
    params["stop_words_list_path"] = "../data/stopwords.txt"
    params["n_init"] = 10
    params["train_size"] = 500
    params["random_seed"] = 11921002

    #############
    #
    #############
    data_groups = {1: "pfa_expe1", 2: "pfa_expe2", 3: "pfa_expe3", 4: "pfa_expe4", 5: "pfa_expe5"}
    dtmc_save_root = '/home/dgl/project/pfa-data-generator/dtmc_results/ijcai19'
    models_root = '/home/dgl/project/pfa-data-generator/models/pretrained/exp_ijcai19/5000'
    data_root = "/home/dgl/project/pfa-data-generator/data/exp_ijcai19/5000"
    lstm_models = ["pfa_expe1-train_acc-0.8818-test_acc-0.843.pkl", "pfa_expe2-train_acc-0.9144-test_acc-0.856.pkl",
                   "pfa_expe3-train_acc-0.9112-test_acc-0.834.pkl", "pfa_expe4-train_acc-0.887-test_acc-0.833.pkl",
                   "pfa_expe5-train_acc-0.8924-test_acc-0.869.pkl"]
    gru_models = ["pfa_expe1-train_acc-0.9054-test_acc-0.861.pkl",
                  "pfa_expe2-train_acc-0.93-test_acc-0.854.pkl",
                  "pfa_expe3-train_acc-0.953-test_acc-0.853.pkl",
                  "pfa_expe4-train_acc-0.9008-test_acc-0.874.pkl",
                  "pfa_expe5-train_acc-0.8912-test_acc-0.88.pkl"]

    ################
    # global setting
    ################
    params["max_iters"] = 1
    params["max_length"] = max_length

    models = {MTYPE_GRU: gru_models, MTYPE_LSTM: lstm_models}

    for rnn_type in [MTYPE_LSTM, MTYPE_GRU]:
        ######################
        # fix to datagroup 1
        ######################
        data_group = data_groups[1]
        model_name = models[rnn_type][0]
        print("========RNN:{}===data:{}============================".format(rnn_type, data_group))
        for k in range(2, 3):
            ###############################################
            # each model corresponding to each data group
            ###############################################
            params["train_path"] = os.path.join(data_root, data_group, "test")
            params["model_path"] = os.path.join(models_root, rnn_type, model_name)
            params["rnn_type"] = rnn_type
            params["k_cluster"] = k
            output_path = os.path.join(dtmc_save_root, rnn_type, data_group,
                                       "k={},length={}".format(params["k_cluster"], params["max_length"]))
            print("RNN:{},data:{},model:{}".format(rnn_type, data_group, model_name))

            ###########
            # training
            ###########
            print("=========k:{}=======".format(params["k_cluster"]))
            learn_fsa(params, root_path)
