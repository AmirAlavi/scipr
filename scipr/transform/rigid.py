import numpy as np

from .base import Transformer


class Rigid(Transformer):
    """Use a rigid transformation function to align the pairs of cells.

    Rigid trasformations are constrained to the operations of rotation,
    reﬂection, translation, and combinations of these.
    """
    def __init__(self):
        super().__init__()

    def fit(self, A, B):
        # See http://nghiaho.com/?page_id=671
        # center
        A_centroid = np.mean(A, axis=0)
        B_centroid = np.mean(B, axis=0)

        H = np.dot((A - A_centroid).T, B - B_centroid)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        detR = np.linalg.det(R)
        x = np.identity(Vt.T.shape[1])
        x[x.shape[0]-1, x.shape[1]-1] = detR
        R = np.linalg.multi_dot([Vt.T, x, U.T])

        t = B_centroid.T - np.dot(R, A_centroid.T)

        model = {
            'R': R,
            't': t
        }
        return model

    def transform(self, model, A):
        return np.dot(model['R'], A.T).T + model['t']

    def chain(self, model, step_model, step_number):
        # For rigid transforms, since we can learn the final function simply
        # by learning the optimal rotation from the original A to the final A,
        # we don't need to maintain an overall model during fitting. See
        # finalize.
        return None

    def finalize(self, model, A_orig, A_final):
        return self.fit(A_orig, A_final)
