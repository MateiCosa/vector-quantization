import os
import numpy as np
from utils import ImgData
from recolor import Lloyd

class Simplifier():
                                      
    def __init__(self, img, k, thresh=263668099):
        self.img = img
        self.unique_col_tensor = self.img.get_unique_col_tensor()
        self.unique_col_counts = self.img.get_unique_col_counts()
        self.unique_points_num = self.img.get_unique_col_num()
        self.n = self.unique_col_tensor.shape[0]
        self.k = k
        self.max_cluster_size = self.n - self.k + 1
        self.center_candidates = set()
        self.power = np.full(3, 2)
        self.n_iters = 0
        self.thresh = thresh
        self.centers = None
        self.num_centers = 0
        self.path = os.path.dirname(os.path.realpath(__file__))

    def _find_center(self, cluster_ids):
        center = np.zeros(3, dtype=np.uint8)
        center = np.sum(self.unique_col_tensor[cluster_ids] * self.unique_col_counts[cluster_ids][:, np.newaxis], axis=0) // np.sum(self.unique_col_counts[cluster_ids])
        return center

    def _compute_cost(self, cluster_ids, center=None):
        if center is None:
            center = self._find_center(cluster_ids)
        return np.sum(((self.unique_col_tensor[cluster_ids] - center) ** self.power) * self.unique_col_counts[cluster_ids][:, np.newaxis])

    def _gen_clusters(self, verbose=True, eps=1e-6):
        stack = [([], 0, self.unique_points_num)]
        while stack:
            self.n_iters += 1
            if verbose and self.n_iters % 100000 == 0:
                print(f"Completed {self.n_iters} iterations.")
            curr_cluster, start, end = stack.pop()
            if len(curr_cluster) == self.unique_points_num - self.k + 1:
                continue
            for i in range(start, end):
                new_cluster = curr_cluster + [i]
                center = self._find_center(new_cluster)
                if tuple(center) in self.center_candidates:
                    continue
                cost = self._compute_cost(new_cluster, center)
                if cost - self.thresh <= eps:
                    self.center_candidates.add(tuple(center))
                stack.append((new_cluster, i+1, end))
    
    def run(self, verbose=True, output_file = "data/params/centers.csv", eps = 1e-6):
        self.n_iters = 0
        self.center_candidates = set()

        if verbose:
            print("Initiating simplifying procedure...")

        self._gen_clusters(verbose, eps)
        self.num_centers = len(self.center_candidates)

        if verbose:
            print(f"Procedure ended after {self.n_iters} iterations.")
            print(f"Number of center candidates found: {self.num_centers}.\n")

        self.centers = np.array([np.array(self.center_candidates.pop(), dtype=np.uint8) for _ in range(self.num_centers)])

        np.savetxt(self.path + "/../" + output_file, self.centers, fmt="%i", delimiter=",")

    def get_center_candidates(self):
        return self.centers
    
class Threshold_Generator():
    
    def __init__(self, img, k, seeds=None):
        self.img = img
        self.k = k
        self.thresh = np.inf
        if seeds is None:
            seeds = range(100)
        self.seeds = seeds

    def _get_optimal_cost_thresh(self, seeds, verbose=True):
        for seed in seeds:
            kmeans = Lloyd(self.img, self.k, seed=seed)
            kmeans.run(save_img=False)
            self.thresh = min(self.thresh, kmeans.get_cost())
            if verbose:
                print(f"seed {seed} coverged in {kmeans.get_n_iters()} steps.")
    
    def run(self, seeds=range(100), verbose=True):
        self.thresh = np.inf
        self._get_optimal_cost_thresh(seeds, verbose)
        print(f"Threshold: {self.get_threshold()}.\n")

    def get_threshold(self):
        return self.thresh

