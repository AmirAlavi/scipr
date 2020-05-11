from abc import ABC, abstractmethod
import math
import datetime
from collections import defaultdict

from scipy import spatial
from scipy.optimize import linear_sum_assignment
import numpy as np


class Match(ABC):
    @abstractmethod
    def match(self, A, B, kd_tree_B):
        pass

    def __call__(self, A, B, kd_tree_B):
        return self.match(A, B, kd_tree_B)


class Closest(Match):
    # TODO: add 'prune_matches' option in a constructor here

    def match(self, A, B, kd_tree_B):
        distances, B_indices = kd_tree_B.query(A)
        A_indices = np.arange(A.shape[0])
        return A_indices, B_indices, distances


class Hungarian(Match):
    def __init__(self, frac_to_match=1.0):
        self.frac_to_match = frac_to_match

    def match(self, A, B, kd_tree_B):
        t0 = datetime.datetime.now()
        dist_mat = spatial.distance.cdist(A, B)
        A_indices, B_indices = linear_sum_assignment(dist_mat)
        t1 = datetime.datetime.now()
        time_str = pretty_tdelta(t1 - t0)
        print('hungarian matching took ' + time_str)
        distances = dist_mat[A_indices, B_indices]
        if self.frac_to_match < 1.0:
            n_to_match = math.floor(self.frac_to_match * min(A.shape[0],
                                                             B.shape[0]))
            idx = np.argsort(distances)
            distances = distances[idx][:n_to_match]
            A_indices = A_indices[idx][:n_to_match]
            B_indices = B_indices[idx][:n_to_match]
        assert(len(np.unique(A_indices)) == len(A_indices))
        assert(len(np.unique(B_indices)) == len(B_indices))
        return A_indices, B_indices, distances


class Greedy(Match):
    def __init__(self, alpha=0.5, beta=2):
        self.alpha = alpha
        self.beta = beta

    def match(self, A, B, kd_tree_B):
        dist_mat = spatial.distance.cdist(A, B)
        # Select pairs that should be matched between set A and B,
        # iteratively building up a mask that selects those matches
        mask = np.zeros(dist_mat.shape, dtype=np.float32)
        # sort the distances by smallest->largest
        sorted_idx = np.stack(np.unravel_index(np.argsort(dist_mat.ravel()),
                                               dist_mat.shape), axis=1)
        target_matched_counts = defaultdict(int)
        source_matched = set()
        for i in range(sorted_idx.shape[0]):
            # A tuple, match_idx[0] is index of the pair in set A,
            # match_idx[1] " " B
            match_idx = sorted_idx[i]
            if target_matched_counts[match_idx[1]] < self.beta and \
               match_idx[0] not in source_matched:
                # if the target point in this pair hasn't been matched to too
                # much, and the source point in this pair has never been
                # matched to, then select this pair
                mask[match_idx[0], match_idx[1]] = 1
                target_matched_counts[match_idx[1]] += 1
                source_matched.add(match_idx[0])
            if len(source_matched) > self.alpha * dist_mat.shape[0]:
                # if matched enough of the source set, then stop
                break
        A_indices, B_indices = np.where(mask == 1)
        distances = dist_mat[A_indices, B_indices]
        return A_indices, B_indices, distances


class MNN(Match):
    def __init__(self, k=10):
        self.k = k

    def match(self, A, B, kd_tree_B):
        t0 = datetime.datetime.now()
        kd_A = spatial.cKDTree(A)

        distances_a2b, B_nb_indices = kd_tree_B.query(A, self.k)
        distances_b2a, A_nb_indices = kd_A.query(B, self.k)
        A_indices, B_indices, distances = [], [], []
        for a_idx, b_neighbors in enumerate(B_nb_indices):
            for b_idx, a_neighbors in enumerate(A_nb_indices):
                if a_idx in a_neighbors and b_idx in b_neighbors:
                    A_indices.append(a_idx)
                    B_indices.append(b_idx)
                    distances.append(kd_tree_B.query(A[a_idx])[0])

        t1 = datetime.datetime.now()
        time_str = pretty_tdelta(t1 - t0)
        print('mnn matching took ' + time_str)
        return np.array(A_indices), np.array(B_indices), np.array(distances)


def pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
