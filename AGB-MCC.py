import time
import numpy as np
from munkres import Munkres
from scipy.spatial.distance import cdist
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import cKDTree

class NNSearch:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.dis_mat = self.calculate_distance_matrix()
    def calculate_distance_matrix(self):
        # Calculate the distance matrix for all pairs of points using cdist
        return cdist(self.coordinates, self.coordinates)
    def get_dis_index(self):
        dis_mat = self.dis_mat
        # Sort the entire distance matrix
        dis = np.sort(dis_mat, axis=1)
        index = np.argsort(dis_mat, axis=1)
        return dis, index
    def natural_search(self):
        n = self.dis_mat.shape[0]
        dis, index = self.get_dis_index()
        nb = np.zeros(n, dtype=int)
        t = 1
        num_1 = 0
        num_2 = 0
        all_neighbors = [[] for _ in range(n)]
        while t < n:
            # Batch update neighbor information
            y = index[:, t]
            for i in range(n):
                all_neighbors[i].append(y[i])
            nb += np.bincount(y, minlength=n)
            num_1 = np.count_nonzero(nb == 0)
            if num_1 == num_2:
                break
            num_2 = num_1
            t += 1
        # Prepare to plot the points and neighbors
        Point = np.full(n, -1, dtype=int)
        gb_list = []
        for i in range(n):
            if Point[i] == -1:
                all_indices = [i] + all_neighbors[i]
                gb_list.append(all_indices)
                Point[all_indices] = 1

        print(f'Number of natural neighbors: {t}')
        return np.array(gb_list)

def calculate_center_and_radius(gb):
    dataGb = line[gb]
    center = dataGb.mean(axis=0)
    radius = np.max((((dataGb - center) ** 2).sum(axis=1) ** 0.5))
    return radius, center

def calculate_c_and_r(gb):
    data_no_label = gb[:, :] 
    center = data_no_label.mean(axis=0)
    radius = np.max((((data_no_label - center) ** 2).sum(axis=1) ** 0.5))
    return center, radius

def get_avg_comp(gb_list, radius):
    avg_n_div_maxr = sum([len(x) / radius[i] for i, x in enumerate(gb_list)]) / len(gb_list)
    return avg_n_div_maxr

def get_CNGB_NCNGB(gb_list, Radius, par_A, ave_comp):
    NCNGB = [i for i, x in enumerate(gb_list) if len(x) / Radius[i] < par_A * ave_comp]
    CNGB = [i for i, x in enumerate(gb_list) if len(x) / Radius[i] >= par_A * ave_comp]
    return np.array(CNGB), np.array(NCNGB)

def cluster_CNGB_1(CNGB, Centers, cngb_radius):
    Centers = np.array(Centers)  
    CNGB_N = len(CNGB)  

    unvisited = [i for i in range(CNGB_N)]  
    cluster = [-1 for i in range(CNGB_N)]
    K = -1
    cluster_particles = {} 
    while len(unvisited) > 0:
        p = unvisited[0]  
        unvisited.remove(p)  
        neighbors = [] 
        for i in range(CNGB_N):
            if i != p:
                dis = ((Centers[CNGB[i]] - Centers[CNGB[p]]) ** 2).sum(axis=0) ** 0.5  # Euclidean distance between granular balls
                if dis <= (cngb_radius[i] + cngb_radius[p]):                      neighbors.append(i)

        K = K + 1 
        cluster[p] = K  
    
        if K not in cluster_particles:
            cluster_particles[K] = []
        cluster_particles[K].append(CNGB[p])  
       
        for pi in neighbors:
            if pi in unvisited:
                unvisited.remove(pi)  
                neighbors_pi = [] 
 
                for j in range(CNGB_N):
                    if j != pi:
                        dis_pi = ((Centers[CNGB[j]] - Centers[CNGB[pi]]) ** 2).sum(axis=0) ** 0.5  # Calculate distance
                        if dis_pi <= (cngb_radius[j] + cngb_radius[pi]):                              neighbors_pi.append(j)

                for t in neighbors_pi:
                    if t not in neighbors:
                        neighbors.append(t)

            if cluster[pi] == -1:
                cluster[pi] = K
                cluster_particles[K].append(CNGB[pi])  
    return cluster, cluster_particles

def cluster_CNGB_2(CNGB, cluster_p, Centers, K, cluster):
    cluster_count = {}
    for cluster_label, particles in cluster_p.items():
        cluster_count[cluster_label] = len(particles)
    sorted_cluster_count = sorted(cluster_count.items(), key=lambda x: x[1], reverse=True)
    top_k_clusters = [cluster_label for cluster_label, _ in sorted_cluster_count[:K]]
    included = [i for i in range(len(CNGB)) if cluster[i] in top_k_clusters]
    excluded = [i for i in range(len(CNGB)) if cluster[i] not in top_k_clusters]
    for excluded_index in excluded:
        min_distance = float('inf')  
        closest_cluster_label = None
        excluded_center = Centers[CNGB[excluded_index]]
        for included_index in included:
            included_center = Centers[CNGB[included_index]]

            distance = np.linalg.norm(excluded_center - included_center)

            if distance < min_distance:
                min_distance = distance
                closest_cluster_label = cluster[included_index]
        cluster[excluded_index] = closest_cluster_label
    return cluster

def cluster_NCNGB(NCBS, point_cluster, gb_list, line):
    max_unlabeled = sum(len(gb_list[i]) for i in NCBS)
    unlabeled_indices = np.empty(max_unlabeled, dtype=int)
    unlabeled_count = 0
    for i in NCBS:
        for j in range(gb_list[i].shape[0]):
            point_index = gb_list[i][j]
            if point_cluster[point_index] == -1:
                unlabeled_indices[unlabeled_count] = point_index
                unlabeled_count += 1
    unlabeled_indices = unlabeled_indices[:unlabeled_count]
    if unlabeled_count == 0:
        return point_cluster
    labeled_indices = np.where(point_cluster != -1)[0]
    labeled_points = line[labeled_indices]
    tree = cKDTree(labeled_points)
    for point_index in unlabeled_indices:
        _, nearest_index = tree.query(line[point_index])
        nearest_original_index = labeled_indices[nearest_index]
        point_cluster[point_index] = point_cluster[nearest_original_index]
    return point_cluster

def evaluation(y_true, y_pred):
    nmi = nmi_score(y_true, y_pred, average_method='arithmetic')
    ari = ari_score(y_true, y_pred)
    y_true = y_true - np.min(y_true)
    l1 = list(set(y_true))
    num_class1 = len(l1)
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    ind = 0
    if num_class1 != num_class2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1
    l2 = list(set(y_pred))
    num_class2 = len(l2)
    if num_class1 != num_class2:
        print('error')
        return
    cost = np.zeros((num_class1, num_class2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        c2 = l2[indexes[i][1]]
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c
    acc = accuracy_score(y_true, new_predict)
    f1 = f1_score(y_true, new_predict, average='macro')
    return acc, nmi, ari, f1

if __name__ == '__main__':

    file_name = ['D1']
    par_A_val = [0.5]
    for i, file in enumerate(file_name):
        df = np.loadtxt(f'{file}.txt')
        line = df[:, :-1]
        line_label = df[:, -1]
        par_A = par_A_val[i] 
        unique_labels, label_counts = np.unique(line_label, return_counts=True)
        K = len(unique_labels)

        print(f'File name: {file}')
        print(f'Data size: {len(line)}')
        print(f'Average concentration: {par_A}')
        print(f'Number of clusters: {K}')

        scaler = MinMaxScaler(feature_range=(0, 1))
        line = scaler.fit_transform(line)

        startTime = time.time()
        nn_search = NNSearch(line)
        gb_list = nn_search.natural_search()
        
        Radius = []
        Centers = []
        for gb in gb_list:
            radius, center = calculate_center_and_radius(gb)
            Radius.append(radius)
            Centers.append(center)

        avg_comp = get_avg_comp(gb_list, Radius)
        CNGB, NCNGB = get_CNGB_NCNGB(gb_list, Radius, par_A, avg_comp)

        cngb_radius = []
        for i in CNGB:
            cngb_radius.append(Radius[i])

        clusterA, cluster_p = cluster_CNGB_1(CNGB, Centers, cngb_radius)

        point_cluster = np.ones(line.shape[0])
        point_cluster = -1 * point_cluster
        for i1 in range(len(CNGB)):
            for j1, cb_point_index in enumerate(gb_list[CNGB[i1]]):
                point_cluster[cb_point_index] = clusterA[i1]

     
        cluster = cluster_CNGB_2(CNGB, cluster_p, Centers, K, clusterA)

        point_cluster = np.ones(line.shape[0])
        point_cluster = -1 * point_cluster
        
        for i1 in range(len(CNGB)):
            for j1, cb_point_index in enumerate(gb_list[CNGB[i1]]):
                point_cluster[cb_point_index] = cluster[i1]

        point_cluster = cluster_NCNGB(NCNGB, point_cluster, gb_list, line)
        endTime = time.time()
        times = endTime - startTime 

        # Remove noise points and calculate ACC and NMI
        valid_indices = line_label != -1
        line_label = line_label[valid_indices]
        point_clusterA = point_cluster[valid_indices]
        acc, nmi, ari, f1 = evaluation(line_label, point_clusterA)
        
        print(f"ACC: {acc:.4f}\nNMI: {nmi:.4f}\nARI: {ari:.4f}\nF1: {f1:.4f}")
        print(f'Run time: {times:.4f} s')
        print('\n')