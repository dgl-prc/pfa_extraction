import os

'''
The following codes are derived based on function 'refinePFA' which supports two types of prediction strategy. 
'''


def predict_words(pfa, trans, persistence):
    ''' make prediction with word trace.

    In this fashion, we make prediction with the word trace directly,i.e, we can give a prediction of sentence without the information of RNN.

    Parameters
    -----------
    pfa: an instance of PFA class.
    trans:
    persistence:
    '''
    terminate_count = 0
    acc_count = 0
    fdlt_count = 0
    rnn_acc_check = 0
    t_word_traces_path = persistence.word_traces_path
    prdct_grnd_list = pfa.load_prdct_grnd_pairs(persistence.rnn_prdct_grnd_path)
    total_samples = len(prdct_grnd_list)
    for word_traces_file in os.listdir(t_word_traces_path):
        word_trace = pfa.load_word_sequence(os.path.join(t_word_traces_path, word_traces_file))
        #############
        # DIFFERENCE2:
        #############
        sample_idx = int(word_traces_file.split(".")[0])
        prdct, grnd = prdct_grnd_list[sample_idx]  # get the RNN prediction and the ground truth.
        pfa_prdct, path_prob, is_terminate, state_trans_path = pfa.predict_word_trace(word_trace, trans)
        if is_terminate:
            terminate_count += 1
            # print("trace_file:{}".format(word_traces_file))
        if int(pfa_prdct) == int(grnd):
            acc_count += 1.
        if int(pfa_prdct) == int(prdct):
            fdlt_count += 1.
        if int(prdct) == int(grnd):
            rnn_acc_check += 1.

    print("total_samples:{},acc_count:{},fdlt_count:{}".format(total_samples, acc_count, fdlt_count))
    acc = acc_count / total_samples
    fdlt = fdlt_count / total_samples
    rnn_acc = rnn_acc_check / total_samples

    print(
        'Terminate Data:{}\nAccuracy of RNN:{}\nAccuracy of PFA:{}\nFidelity of PFA:{}'.format(terminate_count, rnn_acc,
                                                                                               acc, fdlt))
    return acc, fdlt, rnn_acc


def predict_actions(pfa, persistence, action_trace_paths):
    ''' make prediction with the abstracted action trace.

    In this fashion, we make prediction with the abstract action trace,which is obtained with the hidden state vectors of RNN.

    Parameters
    -----------
    pfa: an instance of PFA class.
    action_trace_paths:
    persistence:
    '''
    #########
    # predict
    #########
    terminate_count = 0
    acc_count = 0
    fdlt_count = 0
    rnn_acc_check = 0

    prdct_grnd_list = pfa.load_prdct_grnd_pairs(persistence.rnn_prdct_grnd_path)
    total_samples = len(action_trace_paths)

    for action_trace_path in action_trace_paths:
        _, tail = os.path.split(action_trace_path)
        tail = tail.strip()
        sample_idx = int(tail.split(".")[0])
        prdct, grnd = prdct_grnd_list[sample_idx]
        #################################################################################################
        # what the fuck? Actually, the result is obtained according to word trace instead of action trace
        ##################################################################################################
        action_trace = pfa.load_action_trace(action_trace_path)
        pfa_prdct, path_prob, is_terminate, state_trans_path = pfa.predict_with_abs_trace(action_trace=action_trace)
        if is_terminate:
            terminate_count += 1
            # print("trace_file:{}".format(tail))
        if int(pfa_prdct) == int(grnd):
            acc_count += 1.
        if int(pfa_prdct) == int(prdct):
            fdlt_count += 1.
        if int(prdct) == int(grnd):
            rnn_acc_check += 1.

    print("total_samples:{},acc_count:{},fdlt_count:{}".format(total_samples, acc_count, fdlt_count))
    acc = acc_count / total_samples
    fdlt = fdlt_count / total_samples
    rnn_acc = rnn_acc_check / total_samples

    print(
        'Terminate Data:{}\nAccuracy of RNN:{}\nAccuracy of PFA:{}\nFidelity of PFA:{}'.format(terminate_count, rnn_acc,
                                                                                               acc, fdlt))
    return acc, fdlt, rnn_acc


def test_pfa_acc(pfa, used_traces_path, persistence, t_word_traces_path=None, t_prdct_grnd_path=None):
    '''

    :param pfa:
    :param used_traces_path:
    :param persistence:
    :param t_word_traces_path:
    :param t_prdct_grnd_path:
    :return:
    '''
    # TODO: replae the following code with pfa.make_trans_matrix

    ###########################
    # Build trans action matrix
    ###########################
    trans = []
    for i in range(pfa.num_states):  # row
        i_row = []
        for j in range(pfa.num_states):
            i_row.append({})  # word:counts of trigger
        trans.append(i_row)
    with open(used_traces_path, 'r') as f:
        action_trace_paths = f.readlines()
    action_trace_paths = [action_trace_path.strip() for action_trace_path in action_trace_paths]
    exception_count = 0
    for action_trace_path in action_trace_paths:
        if exception_count > 10:
            raise Exception("ERROR!!!!")
        _, tail = os.path.split(action_trace_path)
        tail = tail.strip()
        # to be continued
        action_trace = pfa.load_action_trace(os.path.join(persistence.action_traces_path, tail))
        word_trace = pfa.load_word_sequence(os.path.join(persistence.word_traces_path, tail))
        predict, accumu_prob, is_terminate, state_transition = pfa.predict_with_abs_trace(action_trace)

        if not is_terminate:
            assert len(state_transition) - len(word_trace) == 2
        else:
            print("trace path:{}".format(os.path.join(persistence.trace_path, tail)))
            print("word_trace_path:{}".format(os.path.join(persistence.word_traces_path, tail)))
        for i, start_state in enumerate(state_transition):
            start_state = start_state[0]
            next_state = state_transition[i + 1][0]
            if i == 0:
                trigger_word = "<>"
            else:
                trigger_word = word_trace[i - 1]
            if trigger_word in trans[start_state][next_state].keys():
                trans[start_state][next_state][trigger_word] += 1
            else:
                trans[start_state][next_state][trigger_word] = 1
            if i + 1 == len(state_transition) - 1:
                break

    trans = pfa.make_words_trans_matrix(used_traces_path=used_traces_path, trace_path=persistence.action_traces_path,
                                        word_traces_path=persistence.word_traces_path)

    return predict_actions(pfa, persistence, action_trace_paths), predict_words(pfa, trans, persistence)
