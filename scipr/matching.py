from abc import ABC, abstractmethod
import math
import datetime
from collections import defaultdict
import logging

from scipy import spatial
from scipy.optimize import linear_sum_assignment
import numpy as np


class Match(ABC):
    """Base class for all matching function objects.

    Your matching functions should also subclass this class.
    """
    @abstractmethod
    def match(self, A, B, kd_tree_B):
        """ Find matching pairs of cells between two batches.

        Parameters
        ----------
        A : numpy.ndarray
            The "source" batch of cells to align. Dimensions are
            (cellsA, genes).

        B : numpy.ndarray
            The "target" (or "reference") batch data to align to. Dimensions
            are (cellsB, genes).

        kd_tree_B : scipy.spatial.ckdtree.cKDTree
            A KD tree of the ``B`` batch for fast queries, since ``B`` is the
            stationary "reference" batch which does not move, SCIPR computes
            once at the beginning and passes it along to the matching algorithm
            at each step.

        Returns
        -------
        A_indices : numpy.ndarray
            Index array into ``A``, selecting the matched cells in ``A``. Of
            length S.

        B_indices : numpy.ndarray
            Index array into ``B``, selecting the matched cells in ``B``. Also
            of length S. Each element in this array corresponds to its match in
            ``A_indices``.

        distances : numpy.ndarray
            The distances between the pairs in ``A_indices`` and ``B_indices``,
            also of length S.
        """
        pass

    def __call__(self, A, B, kd_tree_B):
        return self.match(A, B, kd_tree_B)


class Closest(Match):
    """Use the classic "closest" strategy to assign pairs.

    For each cell in source batch ``A``, pair it with the closest cell to it in
    the target batch ``B``.
    """
    # TODO: add 'prune_matches' option in a constructor here

    def match(self, A, B, kd_tree_B):
        log = logging.getLogger(__name__)
        t0 = datetime.datetime.now()
        distances, B_indices = kd_tree_B.query(A)
        A_indices = np.arange(A.shape[0])
        t1 = datetime.datetime.now()
        time_str = _pretty_tdelta(t1 - t0)
        log.info('Closest matching took ' + time_str)
        return A_indices, B_indices, distances


class Hungarian(Match):
    """Use the Hungarian algorithm for the assignment problem to assign pairs.

    The "Hungarian" method is an efficient algorithm to solve the assignment
    problem, where finding pairs between two sets ``A`` and ``B`` is treated
    as a bipartite matching problem.

    Parameters
    ----------
    frac_to_match : float
        If not 1.0, then this is the fraction of the matches to keep. All of
        the matches are sorted from smallest distance to largest, and then only
        the top ``n`` are returned, where ``n`` is the ``frac_to_match``
        portion of the smaller of the two sets of cells.
    """
    def __init__(self, frac_to_match=1.0):
        self.frac_to_match = frac_to_match

    def match(self, A, B, kd_tree_B):
        log = logging.getLogger(__name__)
        t0 = datetime.datetime.now()
        dist_mat = spatial.distance.cdist(A, B)
        A_indices, B_indices = linear_sum_assignment(dist_mat)
        t1 = datetime.datetime.now()
        time_str = _pretty_tdelta(t1 - t0)
        log.info('Hungarian matching took ' + time_str)
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
    """Use the greedy matching algorithm from the SCIPR paper.

    First all pairings between the two sets are sorted by distance from
    smallest to largest, and then selecting pairings proceeds down the list.
    The selection of pairs stops when we have assigned ``alpha`` fraction of
    the source set to pairs. When we are considering a pair, if the ``target``
    point in that pair has already participated in ``beta`` pairs, we do not
    pick it.

    Parameters
    ----------
    alpha : float
        The ``alpha`` hyperparameter in the above algorithm.

    beta : int
        The ``beta`` hyperparameter in the above algorithm.
    """
    def __init__(self, alpha=0.5, beta=2):
        self.alpha = alpha
        self.beta = beta

    def match(self, A, B, kd_tree_B):
        log = logging.getLogger(__name__)
        t0 = datetime.datetime.now()
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
        t1 = datetime.datetime.now()
        time_str = _pretty_tdelta(t1 - t0)
        log.info('Greedy matching took ' + time_str)
        return A_indices, B_indices, distances


class MNN(Match):
    """Use the Mutual Nearest Neighbors strategy to assign pairs.

    For any given cell ``a`` in a set ``A``, if a cell ``b`` in a set ``B`` is
    in the set of nearest neighbors of ``a`` among ``B``, and ``a`` is in the
    set of nearest neighbors of ``b`` among A, then pair ``(a, b)`` is added to
    the set of pairs.

    Parameters
    ----------
    k : int
        The number of neighbors of each cell to consider when finding mutual
        nearest neighbors.
    """
    def __init__(self, k=10):
        self.k = k

    def match(self, A, B, kd_tree_B):
        log = logging.getLogger(__name__)
        t0 = datetime.datetime.now()
        kd_A = spatial.cKDTree(A)

        _, B_nb_indices = kd_tree_B.query(A, self.k)
        _, A_nb_indices = kd_A.query(B, self.k)
        if self.k == 1:
            B_nb_indices = np.expand_dims(B_nb_indices, 1)
            A_nb_indices = np.expand_dims(A_nb_indices, 1)
        A_indices, B_indices, distances = [], [], []
        for a_idx, b_neighbors in enumerate(B_nb_indices):
            for b_idx, a_neighbors in enumerate(A_nb_indices):
                if a_idx in a_neighbors and b_idx in b_neighbors:
                    A_indices.append(a_idx)
                    B_indices.append(b_idx)
                    distances.append(kd_tree_B.query(A[a_idx])[0])

        t1 = datetime.datetime.now()
        time_str = _pretty_tdelta(t1 - t0)
        log.info('MNN matching took ' + time_str)
        return np.array(A_indices), np.array(B_indices), np.array(distances)


def _pretty_tdelta(tdelta):
    hours, rem = divmod(tdelta.seconds, 3600)
    mins, secs = divmod(rem, 60)
    return '{:02d}:{:02d}:{:02d}'.format(hours, mins, secs)
