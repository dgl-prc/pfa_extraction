
import os

#TODO: move this funciton to pfa class
def make_trans_matrix(pfa,used_traces_path,trace_path,word_traces_path):
    ############
    # Build trans action matrix
    #############
    trans = []
    for i in range(pfa.num_states):  # row
        i_row = []
        for j in range(pfa.num_states):
            i_row.append({})  # word:counts of trigger
        trans.append(i_row)
    with open(used_traces_path, 'r') as f:
        action_trace_paths = f.readlines()
    exception_count = 0
    for action_trace_path in action_trace_paths:
        if exception_count > 10:
            raise Exception("ERROR!!!!")
        _, tail = os.path.split(action_trace_path)
        tail = tail.strip()
        action_trace = pfa.load_action_trace(os.path.join(trace_path, tail))
        word_trace = pfa.load_word_sequence(os.path.join(word_traces_path, tail))
        predict, accumu_prob, is_terminate, state_transition = pfa.predict_with_abs_trace(action_trace)

        if not is_terminate:
            assert len(state_transition) - len(word_trace) == 2
        else:
            print("trace path:{}".format(os.path.join(trace_path, tail)))
            print("word_trace_path:{}".format(os.path.join(word_traces_path, tail)))
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
    return trans


#TODO: move to PFA
def predict_word_trace_v2(pfa, word_trace, trans):
    is_terminate = False
    curent_states = [(2, 1.0)]
    stateTrans={(2, 1.0):[(1,1.0),(2,1.0)]}
    for trigger_word in word_trace:
        next_states = []
        cur2next={}
        for (curent_state, cur_accum_prob) in curent_states:
            next_state_candidates = get_next_states(curent_state, trigger_word, trans)
            cur2next[(curent_state, cur_accum_prob)]=[]
            for next_state in next_state_candidates:
                if next_state != -1:
                    trans_prob = pfa.pfa_graph[curent_state, next_state]
                    next_states.append((next_state, cur_accum_prob * trans_prob))
                    cur2next[(curent_state, cur_accum_prob)].append((next_state,trans_prob))
        if len(next_states) == 0:
            is_terminate = True
            break
        else:
            curent_states = pfa.state_merge(copy.deepcopy(next_states))

            ###############
            # update path
            ################
            for key in cur2next.keys():
                current_path = stateTrans[key]
                cur_accum_prob = key[1]
                for next_state, trans_prob in cur2next[key]:
                    newKey = (next_state, trans_prob * cur_accum_prob)
                    newCurPath = copy.deepcopy(current_path)
                    newCurPath.append((next_state, trans_prob))
                    stateTrans[newKey] = newCurPath
                stateTrans.pop(key)

            ###############
            # state merge
            ##############
            keys = stateTrans.keys()
            keys.sort(key=lambda x: x[1], reverse=True)
            key_dict = {}
            for key in keys:
                state = key[0]
                accumProb = key[1]
                if state in key_dict.keys():
                    stateTrans.pop(key)
                else:
                    key_dict[state] = accumProb



    curent_states.sort(key=lambda x: x[1], reverse=True)
    final_state = curent_states[0][0]
    predict = pfa.get_state_output(final_state)

    #######################
    # get final stateTrans
    #######################
    trans_path=stateTrans[curent_states[0]]

    return predict, is_terminate,trans_path


def predict(pfa,t_word_traces_path=None,t_prdct_grnd_path=None):
    #########
    # predict
    #########
    terminate_count = 0
    acc_count = 0
    fdlt_count = 0
    rnn_acc_check = 0

    if t_prdct_grnd_path is not None and t_word_traces_path is not None:
        prdct_grnd_list = pfa.load_prdct_grnd_pairs(t_prdct_grnd_path)
        total_samples = len(prdct_grnd_list)
        for word_traces_file in os.listdir(t_word_traces_path):
            word_trace = pfa.load_word_sequence(os.path.join(t_word_traces_path, word_traces_file))
            #############
            # DIFFERENCE2:
            #############
            sample_idx = int(word_traces_file.split(".")[0])
            prdct, grnd = prdct_grnd_list[sample_idx]
            pfa_prdct, is_terminate, trans_path = predict_word_trace_v2(pfa, word_trace, trans)

            if is_terminate:
                terminate_count += 1
                # print("trace_file:{}".format(word_traces_file))

            if int(pfa_prdct) == int(grnd):
                acc_count += 1.
            if int(pfa_prdct) == int(prdct):
                fdlt_count += 1.
            if int(prdct) == int(grnd):
                rnn_acc_check += 1.
    else:
        prdct_grnd_list = pfa.load_prdct_grnd_pairs(prdct_grnd_path)
        total_samples = len(action_trace_paths)
        for action_trace_path in action_trace_paths:
            _, tail = os.path.split(action_trace_path)
            tail = tail.strip()
            sample_idx = int(tail.split(".")[0])
            prdct, grnd = prdct_grnd_list[sample_idx]
            word_trace = pfa.load_word_sequence(os.path.join(word_traces_path, tail))
            #############
            # DIFFERENCE2:
            #############
            pfa_prdct, is_terminate, trans_path = predict_word_trace_v2(pfa, word_trace, trans)
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