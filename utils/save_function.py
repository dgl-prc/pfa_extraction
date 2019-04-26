import os

def save_word_trace(save_path,words_traces,file_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for word_trace,file_name in zip(words_traces,file_names):
        with open(os.path.join(save_path, file_name), 'w') as f:
            for word in word_trace:
                f.write("{}\n".format(word))

def save_state_trans(save_path,stateTranslist,file_names):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for state_trans,file_name in zip(stateTranslist,file_names):
        with open(os.path.join(save_path, file_name), 'w') as f:
            for state in state_trans:
                f.write("{}\n".format(str(state)))