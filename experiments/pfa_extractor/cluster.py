import numpy as np
import os


class AdditiveKMeans(object):
    def __init__(self, kmeans):
        self.myKMeans = kmeans

    def center_split(self, old_center_index, new_centers):
        assert len(new_centers) == 2
        self.myKMeans.cluster_centers_ = np.delete(self.myKMeans.cluster_centers_, old_center_index, 0)
        self.myKMeans.cluster_centers_ = np.vstack((self.myKMeans.cluster_centers_, new_centers))
        self.myKMeans.n_clusters += 1

    def predict(self, points):
        return self.myKMeans.predict(points)


class CustomAgglomerativeClustering(object):
    def __init__(self, AgglomerativeClustering):
        self.myClustering = AgglomerativeClustering
        self.n_clusters = AgglomerativeClustering.n_clusters
        self.labels = AgglomerativeClustering.labels_

    def cluster_split(self, ori_points, splitedCluster):
        ori_points = np.array(ori_points)
        subCluster_index = np.where(self.labels == splitedCluster)
        subCluster_list = ori_points[subCluster_index]
        predict_list = self.myClustering.fit_predict(subCluster_list)
        predict_list[np.where(predict_list == 0)] = splitedCluster
        predict_list[np.where(predict_list == 1)] = self.n_clusters
        for i, index in enumerate(subCluster_index[0]):
            self.labels[index] = predict_list[i]
        self.n_clusters += 1
        return self.labels


def get_spurious_cluster(pfa, used_traces_path, trace_path):
    '''
    calculate the concrete one-step transition probability.
    :param pfa. PFA
    :param traces_list. each trace is the abstract trace, element of which consists of a cluster state and the output label
    :return: the spurious state
    '''
    trans_count = np.zeros((pfa.num_states, pfa.num_states), dtype=int)  # 1-index
    with open(used_traces_path, 'r') as f:
        action_trace_paths = f.readlines()
    for action_trace_path in action_trace_paths:
        _, tail = os.path.split(action_trace_path)
        tail = tail.strip()
        action_trace = pfa.load_action_trace(os.path.join(trace_path, tail))
        current_state = 1
        for action in action_trace:
            act_state_dcit = pfa.pfa_graph_actoins[current_state]
            if act_state_dcit.has_key(action):
                next_state = act_state_dcit[action]
                trans_count[current_state][next_state] += 1
                current_state = next_state
            else:
                break;
    real_trans_prob = trans_count * 1.0 / np.sum(trans_count)

    # state:0,1,2 should be ignored since the trans probability of both 0-->1, 1-->2 are 1.0
    # For state 2,whose label is "-1-1",so it should not be selected as a spurious state.
    differences = pfa.pfa_graph.toarray() - real_trans_prob
    differences[0] *= 0.
    differences[1] *= 0.
    differences[2] *= 0.
    # find the most spurious trans. tuple:(out_state,in_state)
    ind = np.unravel_index(np.argmax(differences), differences.shape)
    spurious_state = ind[0]  # the out state
    # find the spurious cluster
    spurious_cluster = pfa.get_state_cluster(spurious_state)
    return int(spurious_cluster)