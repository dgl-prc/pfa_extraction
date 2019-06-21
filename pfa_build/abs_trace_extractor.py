from sklearn.cluster import KMeans, AgglomerativeClustering
from tempfile import mkdtemp
from joblib import Memory
import numpy as np
import pickle
# from data_process.name_classifiction.rnnexamples import DataProcessor
import string
import os
from data_factory.imdb_sentiment.imdb_data_process import IMDB_Data_Processor
from data_factory.imdb_sentiment.vocabulary import Vob
from utils.time_util import current_timestamp
from utils.constant import *
import pickle
from pfa_build.pfa import PFA
import torch
'''
using k-means to clusteing
'''
# class PartitionFunction(object):
#     def __init__(self,fun,**params):
#         pass
#     def partition

'''
input: model,dataset
output: trace_list
'''

class MODE():
    WL = 1  # wrong_labeld_only
    VOB = 2 # build vocabulary,which maintains the frequence for each word as pos and neg
    NORMAL = 3


class AbstractTraceExtractor():

    def collect_hidden_state_seq(self, rnn, dataset, dataProcessor, input_dim, use_cuda=False, mode=MODE.NORMAL,
                                 with_neuter=True):
        '''
        :param rnn:
        :param dataset:
        :param dataProcessor:
        :param input_dim:
        :param use_cuda:
        :param is_inspect:
        :return: traces_list: the hidden state vector traces
                 outputs_list: the output label at each time step
                 predict_ground_list: the pairs of predict label and the true label
                 wl_samples: the pured samples which are wrongly predicted.
        '''
        traces_list = []
        outputs_list = []
        predict_ground_list = []
        samples = []
        indices = []
        index = 0
        vob = Vob()
        if not torch.cuda.is_available():
            use_cuda = False
        for sequence, label in dataset:
            tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim)
            # filter the sequence which is too long that is more than 1000 words
            if len(tensor_sequence[0]) > 1000: continue
            if use_cuda:
                tensor_sequence = tensor_sequence.cuda()
            hn_trace, label_trace = rnn.get_predict_trace(tensor_sequence)
            if mode == MODE.VOB:
                pure_sequence = dataProcessor.sequence_purifier(sequence)
                vob.add_word(pure_sequence)
                if with_neuter:
                    vob.parse_trace(pure_sequence, label_trace)
                else:
                    vob.parse_trace_without_neuter(pure_sequence, label_trace)
            else:
                if mode == MODE.WL:
                    if label != label_trace[-1]:
                        # for spam we use payload_purifier, otherwise using sequence_purifier
                        pure_sequence = dataProcessor.sequence_purifier(sequence)
                        samples.append(pure_sequence)
                        indices.append(index)
                        traces_list.append(hn_trace)  # tensor to numpy
                        outputs_list.append(label_trace)
                        predict_ground_list.append((label_trace[-1], label))
                if mode == MODE.NORMAL:
                    pure_sequence = dataProcessor.sequence_purifier(sequence)
                    samples.append(pure_sequence)
                    indices.append(index)
                    traces_list.append(hn_trace)  # tensor to numpy
                    outputs_list.append(label_trace)
                    # assert len(pure_sequence) == len(hn_trace) and len(hn_trace) == len(label_trace)
                    predict_ground_list.append((label_trace[-1], label))
            index += 1
            if index % 1000 == 0:
                print("Handling {}/{}".format(index, len(dataset)))

        if mode == MODE.VOB:
            return vob
        else:
            return traces_list, outputs_list, predict_ground_list, samples, indices

    def collect_hidden_stack_seq(self, rnn, dataset, dataProcessor, input_dim, use_cuda=False, mode=MODE.NORMAL, with_neuter=True):
        """

        :param rnn:
        :param dataset:
        :param dataProcessor:
        :param input_dim:
        :param use_cuda:
        :param mode:
        :param with_neuter:
        :return: traces_list: the hidden state vector concatenated stack traces
                 outputs_list: the output label at each time step
                 predict_ground_list: the pairs of predict label and the true label
                 wl_samples: the pured samples which are wrongly predicted.
        """
        traces_list = []
        outputs_list = []
        predict_ground_list = []
        samples = []
        indices = []
        index = 0
        vob = Vob()
        for sequence, label in dataset:

            # compare
            tensor_sequence = dataProcessor.sequence2tensor(sequence, input_dim)
            if use_cuda:
                tensor_sequence = tensor_sequence.cuda()
            hn_trace2, label_trace2 = rnn.get_predict_trace(tensor_sequence)

            # get the hidden stacks
            h0, _ = rnn.get_first_RState()
            hn_trace = []; label_trace = []
            h_cur, label_cur = rnn.get_next_RState(h0, '')
            hn_trace.append(h_cur)
            label_trace.append(label_cur)
            for chr in sequence:
                h_cur, label_cur = rnn.get_next_RState(h_cur, chr)
                hn_trace.append(h_cur)
                label_trace.append(label_cur)

            if mode == MODE.VOB:
                pure_sequence = dataProcessor.sequence_purifier(sequence)
                vob.add_word(pure_sequence)
                if with_neuter:
                    vob.parse_trace(pure_sequence, label_trace)
                else:
                    vob.parse_trace_without_neuter(pure_sequence, label_trace)
            else:
                if mode == MODE.WL:
                    if label != label_trace[-1]:
                        # for spam we use payload_purifier, otherwise using sequence_purifier
                        pure_sequence = dataProcessor.sequence_purifier(sequence)
                        samples.append(pure_sequence)
                        indices.append(index)
                        traces_list.append(hn_trace)  # tensor to numpy
                        outputs_list.append(label_trace)
                        predict_ground_list.append((label_trace[-1], label))
                if mode == MODE.NORMAL:
                    pure_sequence = dataProcessor.sequence_purifier(sequence)
                    samples.append(pure_sequence)
                    indices.append(index)
                    traces_list.append(hn_trace)  # tensor to numpy
                    outputs_list.append(label_trace)
                    # assert len(pure_sequence) == len(hn_trace) and len(hn_trace) == len(label_trace)
                    predict_ground_list.append((label_trace[-1], label))
            index += 1
            if index % 1000 == 0:
                print("Handling {}/{}".format(index, len(dataset)))

        if mode == MODE.VOB:
            return vob
        else:
            return traces_list, outputs_list, predict_ground_list, samples, indices

    def tracesList2vectorsList(self,traces_list):
        '''
        organize all the state vector as a single list, in which each state vector can be regarded a point in the clustering
        :param traces_list:
        :return: points, an 2d-array which hold all the state vector in each row.
                 traces_size_list, each element in which denotes the length
                 of a trace.
        '''
        vectorsList = []
        traces_size_list = []
        for trace in traces_list:
            traces_size_list.append(len(trace))
            for state_vector in trace:
                vectorsList.append(state_vector)
        points = np.squeeze(np.array(vectorsList))
        return points, np.array(traces_size_list)


    def vectorsList2newTraceList(self,labels, traces_size_list):
        '''
        :param labels: array-like ,the label(a.k.a the new state) of each hidden state vector
        :param traces_size_list: the length of each observed state trace in RNN
        :return: a new trace list, in which,every state denoted by a integer.
        '''
        start = 0
        newTraceList = []
        for trace_len in traces_size_list:
            trace = labels[start:trace_len + start]
            newTraceList.append(trace)
            start += trace_len
        return newTraceList

    def kmeans_state_partition(self, vectorsList, n_clusters, n_init=10):
        '''
        :param traces_list: list, each element in which is a couple of vectors
        :param params:
        :return:
        '''
        kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(vectorsList)
        labels, cluster_centers = kmeans.labels_, kmeans.cluster_centers_
        return labels, cluster_centers, kmeans

    def hier_state_partition(self, vectorsList, n_clusters, is_refined=False):
        if not is_refined:
            cachedir = mkdtemp()
            memory = Memory(cachedir=cachedir, verbose=0)
            Aggmeans = AgglomerativeClustering(memory=memory, n_clusters=n_clusters,
                                               compute_full_tree=True).fit(vectorsList)
            labels = Aggmeans.labels_
            return labels, Aggmeans, cachedir
        else:
            Aggmeans = AgglomerativeClustering(n_clusters=n_clusters).fit(vectorsList)
            labels = Aggmeans.labels_
            return labels, Aggmeans

    def save_txt(self, trace_folder, trace_list, traces_outputs_list, traces_size_list, predict_ground_list):
        ##########
        # save to txt
        ##########
        if not os.path.exists(trace_folder):
            os.makedirs(trace_folder)

        print('traces....')
        output_path = os.path.split(trace_folder)[0]
        if not os.path.exists(trace_folder):
            os.makedirs(trace_folder)
        i = 0
        for trace, trace_outputs in zip(trace_list, traces_outputs_list):
            with open(os.path.join(trace_folder, str(i) + '.txt'), 'w') as f:
                f.write('state output\n')
                f.write("{} {}\n".format('-1', '-1'))  # the start state of PFA
                for state, output in zip(trace, trace_outputs):
                    f.write("{} {}\n".format(state, output))
            i += 1

        print('saving trace size info....')
        trace_size_path = os.path.join(output_path, 'real_traces_size_list.txt')
        with open(trace_size_path, 'w') as f:
            for trace_size in traces_size_list:
                f.write(str(trace_size) + '\n')

        print('saving predict-ground info....')
        predict_ground_path = os.path.join(output_path, 'predict-ground_list.txt')
        with open(predict_ground_path, 'w') as f:
            for predict, ground in predict_ground_list:
                f.write(str(predict) + ',' + str(ground) + '\n')


    def trace_extract(self,rnn, train_data, test_data, data_processor, output_path, input_dim, k, n_init):
        traces_list, traces_outputs_list, predict_ground_list,_,_ = self.collect_hidden_state_seq(rnn, train_data, data_processor,
                                                                                         input_dim)
        points, traces_size_list = self.tracesList2vectorsList(traces_list)
        clustering_labels, cluster_centers, kmeans = self.state_partition(points, k, n_init)
        new_trace_list = self.vectorsList2newTraceList(clustering_labels, traces_size_list)

        ##########
        # Get the clustering result of test data
        ##########
        traces_list_test, traces_outputs_list_test, predict_ground_list_test,_,_ = self.collect_hidden_state_seq(rnn, test_data,
                                                                                                        data_processor,
                                                                                                        input_dim)
        points_test, traces_size_list_test = self.tracesList2vectorsList(traces_list_test)
        clustering_labels_test = kmeans.predict(points_test)
        new_trace_list_test = self.vectorsList2newTraceList(clustering_labels_test, traces_size_list_test)

        self.save_txt(trace_folder=os.path.join(output_path, 'train'), trace_list=new_trace_list,
                 traces_outputs_list=traces_outputs_list,
                 traces_size_list=traces_size_list, predict_ground_list=predict_ground_list)

        self.save_txt(trace_folder=os.path.join(output_path, 'test'), trace_list=new_trace_list_test,
                 traces_outputs_list=traces_outputs_list_test,
                 traces_size_list=traces_size_list_test, predict_ground_list=predict_ground_list_test)

        with open(os.path.join(output_path, 'KMEANS.pkl'), 'w') as f:
            pickle.dump(kmeans, f)