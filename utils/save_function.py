import os
import pickle
import pandas as pd
import shutil

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        pkl_obj = pickle.load(f)
    return pkl_obj

def save_pickle(file_path, obj, protocol=3):
    parent_path = os.path.split(file_path)[0]
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, protocol=protocol)

def save_word_trace(save_path, words_traces, file_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for word_trace,file_name in zip(words_traces, file_names):
        with open(os.path.join(save_path, file_name), 'w') as f:
            for word in word_trace:
                f.write("{}\n".format(word))

def save_state_trans(save_path, stateTranslist, file_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for state_trans,file_name in zip(stateTranslist, file_names):
        with open(os.path.join(save_path, file_name), 'w') as f:
            for state in state_trans:
                f.write("{}\n".format(str(state)))


class DataPersistence:
    def __init__(self, root_path, trace="input_trace", word_traces="words_trace",
                 test_word_traces="test_words_trace", rnn_prdct_grnd="rnn_prdct_grnd.pkl",
                 test_rnn_prdct_grnd="test_rnn_prdct_grnd.pkl", ori_points="ori_points.pkl",
                 hn_trace_info="hn_trace_info.pkl", func="func.pkl"):
        self.root_path = os.path.abspath(root_path)
        if os.path.exists(self.root_path):
            shutil.rmtree(self.root_path)
        os.makedirs(self.root_path)
        self.action_traces_path = os.path.join(root_path, trace)  # abstract traces like 0.txt
        self.word_traces_path = os.path.join(root_path, word_traces)
        self.test_word_traces_path = os.path.join(root_path, test_word_traces)
        self.rnn_prdct_grnd_path = os.path.join(root_path, rnn_prdct_grnd)
        self.test_rnn_prdct_grnd_path = os.path.join(root_path, test_rnn_prdct_grnd)
        self.ori_points_path = os.path.join(root_path, ori_points)
        self.hn_trace_info_path = os.path.join(root_path, hn_trace_info)
        self.func_path = os.path.join(root_path, func)
        self.table_head = ['data_group', 'rnn_type', 'deepth', 'acc', 'fdlt', 'rnn_acc', 'running_time']

    def save_train_data(self, traces_outputs_list, ori_traces_size_list,
                        words_traces, predict_ground_list, ori_points):
        with open(self.hn_trace_info_path, "w") as f:
            pickle.dump(traces_outputs_list, f)
            pickle.dump(ori_traces_size_list, f)

        if not os.path.exists(self.word_traces_path):
            os.makedirs(self.word_traces_path)
            for i, word_trace in enumerate(words_traces):
                with open(os.path.join(self.word_traces_path, str(i) + '.txt'), 'w') as f:
                    for word in word_trace:
                        f.write("{}\n".format(word))
                i += 1

        with open(self.rnn_prdct_grnd_path, 'w') as f:
            pickle.dump(predict_ground_list, f)

        with open(self.ori_points_path, 'w') as f:
            pickle.dump(ori_points, f)

    def save_test_data(self, words_traces, predict_ground_list):
        with open(self.test_rnn_prdct_grnd_path, 'w') as f:
            pickle.dump(predict_ground_list, f)

        if not os.path.exists(self.test_word_traces_path):
            os.makedirs(self.test_word_traces_path)
            for i, word_trace in enumerate(words_traces):
                with open(os.path.join(self.test_word_traces_path, str(i) + '.txt'), 'w') as f:
                    for word in word_trace:
                        f.write("{}\n".format(word))

    def save_function(self, cluster_model, extractor):
        with open(self.func_path, "w") as f:
            pickle.dump(cluster_model, f)
            pickle.dump(extractor, f)

    def save_output(self, output_list, output_name):
        """
        caveat : the len of output list must equal to 7 ('data_group', 'rnn_type', 'deepth',
        'acc', 'fdlt', 'rnn_acc', 'running_time')
        :param output_list: data to be store as .csv
        :param output_name: the name of the output.csv
        :return:
        """
        path = os.path.dirname(output_name)
        if not os.path.exists(path):
            os.makedirs(path)
        if os.path.exists(output_name):
            df = pd.DataFrame(output_list)
            df.to_csv(output_name, mode='a', index=None, header=None)
        else:
            if len(output_list[0]) != len(self.table_head):
                raise Exception('Colum number not matched !')
            df = pd.DataFrame(output_list, columns=self.table_head)
            df.to_csv(output_name, index=None)

    def save_pfa_input_trace(self,input_traces_pfa):

        if os.path.exists(self.action_traces_path):
            shutil.rmtree(self.action_traces_path)
        ##############################
        # save pfa input trace
        ##############################
        os.makedirs(self.action_traces_path)
        for i, input_trace in enumerate(input_traces_pfa):
            with open(os.path.join(self.action_traces_path, str(i) + '.txt'), 'w') as f:
                f.write('state output\n')
                for state, output in input_trace:
                    f.write("{} {}\n".format(state, output))
            i += 1

        return self.action_traces_path

