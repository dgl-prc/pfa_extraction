import re
from scipy.sparse import coo_matrix
import os
import graphviz as gv
from IPython.display import Image
from IPython.display import display
import copy


class PFA():
    def __init__(self, pm_file_path, label_path):
        '''

        :param pm_file_path:
        :param label_path: the label for each dtmc state.
        :param integrate_files_path:
        '''
        self.pm_file_path = pm_file_path
        self.label_path = label_path
        self.pfa_graph, self.graphviz_data, self.num_states = self.recover_pfa()
        self.label_state_dict, self.state_label_dict = self.load_label()
        self.pfa_graph_actoins = self.abs_trans_actions()
        # self.integrate_files_path = integrate_files_path  # the file contains:....

    def load_label(self):
        '''
        maintain two dict to map each state into its label.
        :return: label_state_dict. dict
                 state_label_dict. dict
        '''
        label_state_dict = {}
        state_label_dict = {}
        max_state_id = -1  # 1-index
        with open(self.label_path, 'r') as f:
            for line in f.readlines():
                if line.strip() == '':
                    continue
                else:
                    state_label = re.search(r'id\:[\s]*([\d]+)\,[\s]*label\:[\s]*\[(.*)\]', line)
                    state_id = int(state_label.group(1))
                    label = tuple(map(str.strip, state_label.group(2).split(',')))
                    label_state_dict[label] = state_id
                    state_label_dict[state_id] = label
                    max_state_id = state_id

        return label_state_dict, state_label_dict

    def state2action(self, state_id):
        if state_id in [0, 1]:
            return "-1-1"
        label = list(self.state_label_dict[state_id])
        action = label[-1]
        return action

    def _extrac_dest_val(self, raw_dest):
        '''
        :param raw_dest: e.g. "0.3333333333333333 :(s'=6)"
        :return: the dest and the weight, e.g. (6,0.3333333333333333)
        '''
        prob = float(raw_dest.split(':')[0].strip())
        dest = int(re.search('\(.*=([\d]+)\)', raw_dest).group(1))
        return dest, prob

    def _extract_coordinate(self, line):
        '''
        :param line:
        :return: triples list : triple: (row,col,value)
        '''
        triple_list = []
        start = int(re.search(r'\[\]s=([\d]+)\s->', line.strip()).group(1))
        dests = [dest.strip() for dest in re.split(r'\+|\-\>', line)[1:]]
        tuples = map(self._extrac_dest_val, dests)
        triples = [(start, dest, prob) for dest, prob in tuples]
        return triples

    def recover_pfa(self):
        '''
        This function will recover the pfa from the .pm file.
        The PFA is recovered as a coo_matrix in which each element is the transition probability responding to (row-->col)
        :return: coo_matrix. the PFA
                 triple. the first is the rows, the second is the cols, and the last is the trans prob. This is used to
                         visualize the pfa.
        '''
        rows = []
        cols = []
        trans_prob = []
        with open(self.pm_file_path, 'r') as f:
            for line in f.readlines():
                if line.strip() == '' or line.strip() == 'dtmc':
                    continue
                if 'endmodule' in line.strip():
                    print('End recovering pfa....')
                    break
                if 'module' in line.strip():
                    print('Begin recovering pfa....')
                    continue
                if 'init' in line.strip():
                    num_states = re.search(r'\[0\.\.([\d]+)\]', line.strip()).group(1)
                    num_states = int(num_states) + 1
                    print('Total states:{}'.format(num_states))
                if line.strip().startswith("[]s="):
                    triples = self._extract_coordinate(line)
                    rows.extend([triple[0] for triple in triples])
                    cols.extend([triple[1] for triple in triples])
                    trans_prob.extend([triple[2] for triple in triples])

        return coo_matrix((trans_prob, (rows, cols)), shape=(num_states, num_states)).tocsr(), (
        rows, cols, trans_prob), num_states

    def generate_grpah(self, save_path, view, layout):

        def action2NodeLabel(action):
            if action == "-1-1":
                return "-1/-1"
            output = action[-1]
            cluster = action[:-1]
            return cluster + "/" + output

        mygraph = gv.Digraph('DTMC', engine=layout, graph_attr={"rankdir": "LR"})
        for start, end, weight in zip(self.graphviz_data[0], self.graphviz_data[1], self.graphviz_data[2]):
            mygraph.node(str(start), label=str(start) + "," + action2NodeLabel(self.state2action(start)),
                         shape="circle")
            mygraph.node(str(end), label=str(end) + "," + action2NodeLabel(self.state2action(end)), shape="circle")
            mygraph.edge(str(start), str(end), str(round(weight, 5)))
        display(Image(filename=mygraph.render(save_path, view=view)))
        print('DONE!')

    def _get_action_state_list(self, next_state_ids):
        '''
        Get the action to jump the corresponding state
        :param next_state_ids:
        :return:
        '''
        act_state_dcit = {}
        for state_id in next_state_ids:
            if state_id == 1:
                continue
            action = self.state2action(state_id)
            #############
            # repair the pfa bug
            #############
            if len(action) > 2 and action[0] == '0':
                action = action[1:]
            act_state_dcit[action] = state_id
        return act_state_dcit

    def make_words_trans_matrix(self, used_traces_path, trace_path, word_traces_path):
        ############
        # Build trans action matrix
        #############
        trans = []
        # structure of trans matrix
        for i in range(self.num_states):  # row
            i_row = []
            for j in range(self.num_states):
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
            action_trace = self.load_action_trace(os.path.join(trace_path, tail))
            word_trace = self.load_word_sequence(os.path.join(word_traces_path, tail))

            # get the state transition of each action trace which from the training set.
            predict, accumu_prob, is_terminate, state_transition = self.predict_with_abs_trace(action_trace)

            if not is_terminate:
                assert len(state_transition) - len(word_trace) == 2
            else:
                print("Bad action trace: trace path:{},word_trace_path:{}".format(os.path.join(trace_path, tail),
                                                                                  os.path.join(word_traces_path, tail)))
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

    def abs_trans_actions(self):
        '''
        For each state, we maintain a dict to find the next state under a specific action.
        :return: abs_trans_actions. list(dict)
        '''
        abs_trans_actions = {}
        for state_id in range(0, self.num_states):
            next_state_ids = self.pfa_graph.getrow(state_id).indices
            act_state_dcit = self._get_action_state_list(next_state_ids)  # Map < action, nextState_id >
            abs_trans_actions[state_id] = act_state_dcit
        return abs_trans_actions

    def get_state_output(self, state):
        current_label = self.state_label_dict[state]
        if current_label[-1] == "-1-1":
            return -1
        else:
            current_output = current_label[-1][-1]
        return current_output

    def get_state_cluster(self, state):
        current_label = self.state_label_dict[state]
        if current_label[-1] == "-1-1":
            current_cluster = -1
        else:
            current_cluster = current_label[-1][:-1]
        return current_cluster

    def get_next_states(self, curent_state, trigger_word, trans):
        possibe_next = []
        for next_state, dict in enumerate(trans[curent_state]):
            if dict.has_key(trigger_word):
                possibe_next.append((next_state, dict[trigger_word]))
        if len(possibe_next) == 0:
            return [-1]
        return [state[0] for state in possibe_next]

    def state_merge(self, nodes):
        new_states = []
        nodes.sort(key=lambda x: x[1], reverse=True)
        keys = {}
        for node in nodes:
            if keys.has_key(node[0]):
                continue
            keys[node[0]] = node[1]
            new_states.append(node)
        return new_states

    def predict_with_abs_trace(self, action_trace):
        '''
        output the label and its probability
        :param action_trace: a list of action, the element of which is the concatenation of the values of all variables
                ,e.g. var1var2
        :return: label, probability
        '''
        state_trans_path = []
        is_terminate = False
        current_state = 1
        curren_prob = 1.0
        current_output = -1
        path_prob = 1.0  # accumulate probability
        for action in action_trace:
            state_trans_path.append((current_state, curren_prob, current_output))
            act_state_dcit = self.pfa_graph_actoins[current_state]
            if act_state_dcit.has_key(action):
                next_state = act_state_dcit[action]
                curren_prob = self.pfa_graph[current_state, next_state]
                path_prob *= curren_prob
                current_state = next_state
                current_output = self.get_state_output(current_state)
            else:
                is_terminate = True
                break;
        state_trans_path.append((current_state, curren_prob, current_output))
        predict = self.get_state_output(
            current_state)  # the last variable's value of the last element of the current label
        return predict, path_prob, is_terminate, state_trans_path

    def predict_word_trace(self, word_trace, trans):
        '''
        :param pfa:
        :param word_trace:
        :param trans:
        :return:
        '''
        is_terminate = False
        curent_states = [(2, 1.0, 0)]
        # stateTrans keys
        stateTrans = {(2, 1.0, 0): [(1, 1.0), (2, 1.0)]}
        for trigger_word in word_trace:
            next_states = []
            cur2next = {}
            #######################################
            # get the all the possible next states
            #######################################
            for (curent_state, cur_accum_prob, cur_layer) in curent_states:
                next_state_candidates = self.get_next_states(curent_state, trigger_word, trans)
                cur2next[(curent_state, cur_accum_prob, cur_layer)] = []
                for next_state in next_state_candidates:
                    if next_state != -1:
                        trans_prob = self.pfa_graph[curent_state, next_state]
                        next_states.append((next_state, cur_accum_prob * trans_prob, cur_layer + 1))
                        cur2next[(curent_state, cur_accum_prob, cur_layer)].append((next_state, trans_prob))
            if len(next_states) == 0:
                is_terminate = True
                break
            else:
                # merge the state. Turn the next states to current states."next states" contains the path probability
                curent_states = self.state_merge(copy.deepcopy(next_states))
                ########################
                # update the trans path
                ########################
                for key in cur2next.keys():
                    current_path = stateTrans[key]
                    cur_accum_prob = key[1]
                    cur_layer = key[2]
                    for next_state, trans_prob in cur2next[key]:
                        newKey = (next_state, trans_prob * cur_accum_prob, cur_layer + 1)
                        newCurPath = copy.deepcopy(current_path)
                        newCurPath.append((next_state, trans_prob))
                        stateTrans[newKey] = newCurPath
                    stateTrans.pop(key)
                ###############
                # state merge
                ###############
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
        path_prob = curent_states[0][1]
        final_state = curent_states[0][0]
        predict = self.get_state_output(final_state)

        #######################
        # get final stateTrans
        #######################
        state_trans_path = stateTrans[curent_states[0]]
        return predict, path_prob, is_terminate, state_trans_path

    ##########
    # just of test
    #########
    def predict_first_version(self, action_trace):
        '''
        output the label and its probability
        :param action_trace: a list of action, the element of which is the concatenation of the values of all variables
                ,e.g. var1var2
        :return: label, probability
        '''
        current_state = 1
        curren_prob = 1.0
        current_label = ('',)
        for i in range(len(action_trace)):
            label = tuple(action_trace[:i + 1])
            if self.label_state_dict.has_key(label):
                next_state = self.label_state_dict[label]
                curren_prob *= self.pfa_graph[current_state, next_state]
                current_state = next_state
                current_label = label

        predict = current_label[-1][-1]  # the last variable's value of the last element of the current label
        return predict, curren_prob

    def load_action_trace(self, trace_path):
        action_trace = []
        with open(trace_path, 'r') as f:
            for line in f.readlines():
                if "state output" in line:
                    continue
                else:
                    action = "".join(line.strip().split())
                    action_trace.append(action)
        return action_trace

    def load_word_sequence(self, sequnece_path):
        word_trace = []
        with open(sequnece_path, 'r') as f:
            for i, line in enumerate(f.readlines()):
                word_trace.append(line.strip())
        return word_trace

    def load_prdct_grnd_pairs(self, prdct_grnd_path):
        # prdct_grnd_list = []
        # with open(prdct_grnd_path,'r') as f:
        #     for line  in f.readlines():
        #         prdct_grnd_list.append(tuple(line.strip().split(',')))
        # return  prdct_grnd_list
        import pickle
        with open(prdct_grnd_path, 'r') as f:
            rnn_prdct_grnd = pickle.load(f)
        return rnn_prdct_grnd


def build_pfa(input_trace_path, rnn_type, max_length, output_path, variables_path):
    '''
    :param input_traces_pfa:
    :return: pfa
    '''
    import shutil
    from subprocess import call
    if os.path.exists(output_path):
        shutil.rmtree(output_path)

    model_name = rnn_type
    # .pm shows the state transformation route with responding probability
    pm_file_path = os.path.join(output_path, model_name + str(max_length) + ".pm")
    pfa_label_path = os.path.join(output_path, model_name + str(max_length) + "_label.txt")
    actual_used_abs_trace = os.path.join(output_path, "used_trace_list.txt")
    print("Building PFA with JAR....")
    rstcode = call(["bash", "../shell/learn_with_java.sh", model_name, input_trace_path, output_path, variables_path,
                    str(max_length)])  # java: command not found
    if rstcode > 0:
        print("Fail to Building PFA with JAR")
        return

    pfa = PFA(pm_file_path, pfa_label_path)

    return pfa, actual_used_abs_trace
