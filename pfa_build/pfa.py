import re
from scipy.sparse import coo_matrix
import os
import graphviz as gv
from trans_matrix import TransMatrix
import copy

class PFA():
    def __init__(self, pm_file_path, label_path,integrate_files_path):
        '''

        :param pm_file_path:
        :param label_path: the label for each dtmc state.
        :param integrate_files_path:
        '''
        self.pm_file_path = pm_file_path
        self.label_path = label_path
        self.pfa_graph,self.graphviz_data,self.num_states = self.recover_pfa()
        self.label_state_dict, self.state_label_dict = self.load_label()
        self.pfa_graph_actoins = self.abs_trans_actions()
        self.integrate_files_path = integrate_files_path # the file contains:....

    def load_label(self):
        '''
        maintain two dict to map each state into its label.
        :return: label_state_dict. dict
                 state_label_dict. dict
        '''
        label_state_dict = {}
        state_label_dict = {}
        max_state_id = -1 # 1-index
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

    def state2action(self,state_id):
        if state_id in [0,1]:
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

        return coo_matrix((trans_prob, (rows, cols)), shape=(num_states, num_states)).tocsr(),(rows,cols,trans_prob),num_states

    def generate_grpah(self,save_path,view,layout):

        def action2NodeLabel(action):
            if action == "-1-1":
                return "-1/-1"
            output = action[-1]
            cluster = action[:-1]
            return cluster+"/"+output

        mygraph = gv.Digraph('DTMC',engine=layout)
        for start,end,weight in zip(self.graphviz_data[0],self.graphviz_data[1],self.graphviz_data[2]):
            mygraph.node(str(start),label=str(start)+","+action2NodeLabel(self.state2action(start)))
            mygraph.node(str(end),label=str(end)+","+action2NodeLabel(self.state2action(end)))
            mygraph.edge(str(start),str(end),str(round(weight,5)))
        mygraph.render(save_path,view=view)
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
            if len(action)>2 and action[0]=='0':
                action = action[1:]
            act_state_dcit[action] = state_id
        return act_state_dcit

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

    def get_state_output(self,state):
        current_label = self.state_label_dict[state]
        if current_label[-1] == "-1-1":
            return -1
        else:
            current_output = current_label[-1][-1]
        return current_output

    def get_state_cluster(self,state):
        current_label = self.state_label_dict[state]
        if current_label[-1] == "-1-1":
            current_cluster =  -1
        else:
            current_cluster = current_label[-1][:-1]
        return current_cluster

    def state_merge(self,nodes):
        new_states = []
        nodes.sort(key=lambda x: x[1], reverse=True)
        keys={}
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
        state_transition = []
        is_terminate = False
        current_state = 1
        curren_prob = 1.0
        current_output = -1
        accumu_prob = 1.0 # accumulate probabilistic
        for action in action_trace:
            state_transition.append((current_state, curren_prob, current_output))
            act_state_dcit = self.pfa_graph_actoins[current_state]
            if act_state_dcit.has_key(action):
                next_state = act_state_dcit[action]
                curren_prob = self.pfa_graph[current_state, next_state]
                accumu_prob *=curren_prob
                current_state = next_state
                current_output = self.get_state_output(current_state)
            else:
                is_terminate = True
                break;
        state_transition.append((current_state, curren_prob, current_output))
        predict = self.get_state_output(current_state)  # the last variable's value of the last element of the current label
        return predict, accumu_prob,is_terminate,state_transition

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
            label = tuple(action_trace[:i+1])
            if self.label_state_dict.has_key(label):
                next_state= self.label_state_dict[label]
                curren_prob *= self.pfa_graph[current_state,next_state]
                current_state = next_state
                current_label = label

        predict = current_label[-1][-1] # the last variable's value of the last element of the current label
        return predict,curren_prob

    def load_action_trace(self,trace_path):
        action_trace = []
        with open(trace_path, 'r') as f:
            for line in f.readlines():
                if "state output" in line:
                    continue
                else:
                    action = "".join(line.strip().split())
                    action_trace.append(action)
        return action_trace

    def load_word_sequence(self,sequnece_path):
        word_trace = []
        with open(sequnece_path, 'r') as f:
            for i,line in enumerate(f.readlines()):
                word_trace.append(line.strip())
        return word_trace

    def load_prdct_grnd_pairs(self,prdct_grnd_path):
        # prdct_grnd_list = []
        # with open(prdct_grnd_path,'r') as f:
        #     for line  in f.readlines():
        #         prdct_grnd_list.append(tuple(line.strip().split(',')))
        # return  prdct_grnd_list
        import pickle
        with open(prdct_grnd_path,'r') as f:
            rnn_prdct_grnd = pickle.load(f)
        return rnn_prdct_grnd