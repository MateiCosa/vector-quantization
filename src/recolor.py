import sys
from utils import ImgData
import numpy as np

class Lloyd():

    def __init__(self, img, k, init_config=None, seed=0):
        self.img = img
        self.unique_col_tensor = self.img.get_unique_col_tensor()
        self.unique_col_counts = self.img.get_unique_col_counts()
        self.n = self.unique_col_tensor.shape[0]
        self.p = self.unique_col_tensor.shape[1]
        self.k = k
        self.centers = np.zeros(self.k)
        self.clusters = np.zeros(self.n)
        self.dist = np.zeros((self.n, self.k))
        self.cost = np.inf
        self.converged = False
        self.n_iters = 0
        self.power = 2 * np.ones(self.p)
        self.id_range = np.arange(self.n)
        self.seed = seed
        self.unique_col_num = self.unique_col_tensor.shape[0]
        self._init_config(init_config)
        self.output_tensor = None
        
    def _init_centers(self, center_vals=None, in_place=True):
        if center_vals is None:
            np.random.seed(self.seed)
            if self.n <= self.k:
                self.converged = True
                center_vals = self.unique_col_tensor[:min(self.n, self.k)]
            else:
                center_vals = self.unique_col_tensor[np.random.choice(self.n, self.k, replace=False)]

        if in_place:
            self.centers = center_vals
        else:
            return center_vals
    
    def _compute_dist(self, in_place=True):
        dist = np.zeros((self.n, self.k))
        for j, center in enumerate(self.centers):
            dist[:, j] = np.sum(np.fabs(self.unique_col_tensor - center) ** self.power, axis=1)
        if in_place:
            self.dist = dist
        else:
            return dist
        
    def _assign_to_cluster(self, in_place=True):
        assignment = np.argmin(self.dist, axis=1)
        if in_place:
            self.clusters = assignment
        else:
            return assignment

    def _compute_cost(self, in_place=True):
        cost = sum([self.dist[i, self.clusters[i]] * self.unique_col_counts[i] for i in range(self.n)])
        if in_place:
            self.cost = cost
        else:
            return cost
    
    def _init_config(self, init_config):
        self._init_centers(init_config)
        self._compute_dist()
        self._assign_to_cluster()
        self._compute_cost()

    def _update_centers(self, in_place=True):
        new_centers = np.zeros((self.k, 3))
        for i in range(self.k):
            points_from_cluster = (self.clusters == i)
            if points_from_cluster.sum():
                cluster_ids = self.id_range[points_from_cluster]
                new_centers[i] = np.sum(self.unique_col_tensor[cluster_ids] * self.unique_col_counts[cluster_ids][:, np.newaxis], axis=0) / np.sum(self.unique_col_counts[cluster_ids])
            else:
                new_centers[i] = self.unique_col_tensor[np.random.choice(self.n, 1)]
        if in_place:
            self.centers = new_centers
        else:
            return new_centers
    
    def _compute_output_tensor(self):
        
        if not self.converged:
            print("Algorithm did not converge.")
            exit(1)
        
        output_tensor = self.img.get_tensor().copy()
        color_to_center = {tuple(self.unique_col_tensor[i]): self.centers[self.clusters[i]] for i in range(self.n)}
        flattened_tensor = output_tensor.reshape(-1, output_tensor.shape[-1])  
        flattened_tuples = [tuple(row) for row in flattened_tensor]
        updated_flattened = np.array([color_to_center.get(row, row) for row in flattened_tuples])
        output_tensor = updated_flattened.reshape(output_tensor.shape)
        self.output_tensor = output_tensor.astype(np.uint8)

    def run(self, eps=1e-6, save_img=True, verbose=False):
        if verbose:
            print(f"Number of unique colors: {self.unique_col_num}")
            print(f"({self.n_iters}) cost: {self.cost}")
        while not self.converged:
            self.n_iters += 1
            self._update_centers()
            self._compute_dist()
            self._assign_to_cluster()
            new_cost = self._compute_cost(in_place=False)
            if verbose:
                print(f"({self.n_iters}) cost: {new_cost}")
            if np.fabs(self.cost - new_cost) < eps:
                self.converged = True
            self.cost = new_cost
        if verbose:
            print(f"Converged in {self.n_iters} steps.")

        self._compute_output_tensor()
        self.img.set_output(self.output_tensor)
        if save_img:
            self.img.save_output()
    
    def get_centers(self):
        return self.centers
    
    def get_clusters(self):
        return self.clusters
    
    def get_compressed_data(self):
        return self.output_tensor
    
    def get_cost(self):
        return self.cost
    
    def get_n_iters(self):
        return self.n_iters

if __name__ == "__main__":

    input_file = sys.argv[1]
    output_file = sys.argv[2]
    k = int(sys.argv[3])

    img = ImgData(input_file, output_file)
    kmeans = Lloyd(img, k)
    kmeans.run(verbose=True)
