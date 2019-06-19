import numpy as np



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