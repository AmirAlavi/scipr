# iterative point set registration methods for scRNA-seq alignment
from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
from scipy import spatial


class SCIPR(object):
    def __init__(self, match_algo, transform_algo, n_iter=20,
                 input_normalization='l2'):
        self.match_algo = match_algo
        self.transform_algo = transform_algo
        self.n_iter = n_iter
        self.input_normalization = input_normalization
        self.fitted = False

    def fit(self, A, B):
        A = self.apply_input_normalization(A)
        B = self.apply_input_normalization(B)
        kd_B = spatial.cKDTree(B)
        A_orig = A
        for i in range(self.n_iter):
            a_idx, b_idx, distances = self.match_algo(A, B, kd_B)
            step_model = self.transform_algo.fit_step(A[a_idx], B[b_idx], i)
            A = self.transform_algo.transform(step_model, A)
        self.transform_algo.finalize(A_orig, A)
        self.fitted = True

    def transform(self, A):
        if not self.fitted:
            raise RuntimeError('Must call "fit" before "transform"!')
        A = self.apply_input_normalization(A)
        return self.transform_algo._transform(A)

    def apply_input_normalization(self, X):
        if self.input_normalization == 'std':
            print('Applying Standard Scaling')
            scaler = StandardScaler().fit(X)
            return scaler.transform(X)
        elif self.input_normalization == 'l2':
            print('Applying L2 Normalization')
            return normalize(X)
        elif self.input_normalization == 'log':
            print('Applying log normalization')
            return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
