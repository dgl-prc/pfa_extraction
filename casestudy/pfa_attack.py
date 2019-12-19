import sys
sys.path.append("../")
from pfa_build.pfa import *


pm_file_path = ""
label_path = ""
used_traces_path=""
trace_path=""
word_traces_path=""


pfa = PFA(pm_file_path, label_path)
trans = pfa.make_trans_matrix(used_traces_path, trace_path, word_traces_path)
input_word_trace = None

predict, path_prob, is_terminate, state_trans_path = pfa.predict_word_trace(input_word_trace, trans)
