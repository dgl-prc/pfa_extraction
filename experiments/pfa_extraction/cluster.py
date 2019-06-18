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