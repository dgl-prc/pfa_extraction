import sys
sys.path.append('.')
import cluster
from pfa_build.abs_trace_extractor import AbstractTraceExtractor
from cluster import *


class TraceProcessor:
    def __init__(self, extractor, rnn, train_data, data_processor, input_dim, is_stacked=False):
        self.extractor = extractor
        if is_stacked:
            self.traces_list, self.traces_outputs_list, \
            self.predict_ground_list,self. words_traces, self.indices = \
                self.extractor .collect_hidden_stack_seq(rnn, train_data, data_processor,input_dim)
        else:
            self.traces_list, self.traces_outputs_list, \
            self.predict_ground_list, self.words_traces, self.indices = \
                self.extractor.collect_hidden_state_seq(rnn, train_data, data_processor,input_dim)

    def flatten_traces(self):
        self.ori_points, self.ori_traces_size_list = self.extractor.tracesList2vectorsList(self.traces_list)

    def init_kmeans_refine_parttition(self, n_cluster):
        self.labels, self.cluster_centers, self.kmeans = self.extractor.kmeans_state_partition(self.ori_points, n_cluster)
        self.cluster_model = AdditiveKMeans(self.hier_cluster)

    def init_hier_refine_parttiion(self, n_cluster):
        self.labels, self.hier_cluster = self.extractor.state_partition(self.ori_points, n_cluster)
        self.cluster_model = CustomAgglomerativeClustering(self.hier_cluster)

    def init_hier_parttiion(self, n_cluster):
        self.hier_cluster, cachedir = self.extractor.hier_state_partition(self.ori_points, n_cluster)
        self.cluster_model = self.hier_cluster
        self.cluster_model.cachedir = cachedir

    def get_pfa_bpInput_trace(self):
        self.abs_trace_list = self.extractor.vectorsList2newTraceList(self.labels, self.ori_traces_size_list)
        input_traces_pfa = []
        for trace, trace_outputs in zip(self.abs_trace_list, self.traces_outputs_list):
            input_trace_pfa = [("-1", "-1")]
            for state, output in zip(trace, trace_outputs):
                input_trace_pfa.append((str(state), str(int(output))))
            input_traces_pfa.append(input_trace_pfa)
        for trace in input_traces_pfa:
            del trace[1]
        self.input_traces_pfa = input_traces_pfa
        return input_traces_pfa

    def get_pfa_input_trace(self):
        self.abs_trace_list = self.extractor.vectorsList2newTraceList(self.labels, self.ori_traces_size_list)
        input_traces_pfa = []
        for trace, trace_outputs in zip(self.abs_trace_list, self.traces_outputs_list):
            input_trace_pfa = [("-1", "-1")]
            for state, output in zip(trace, trace_outputs):
                input_trace_pfa.append((str(state), str(int(output))))
            input_traces_pfa.append(input_trace_pfa)
        self.input_traces_pfa = input_traces_pfa
        return input_traces_pfa

    def get_train_data(self):
        return self.traces_outputs_list, self.ori_traces_size_list, self.words_traces, \
               self.predict_ground_list, self.ori_points

    def hier_input_update(self):
        self.cluster_model.set_params(n_clusters=self.cluster_model.n_clusters + 1)
        clustering_labels = self.cluster_model.fit(self.ori_points)
        self.labels = clustering_labels
        input_traces_pfa = self.get_pfa_input_trace()
        return input_traces_pfa

    def hier_refine_input_update(self):
        pass

    def kmeans_input_update(self):
        pass