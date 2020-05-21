# iterative point set registration methods for scRNA-seq alignment
import logging

from sklearn.preprocessing import StandardScaler, normalize
import numpy as np
from scipy import spatial


class SCIPR(object):
    """Single Cell Iterative Point set Registration (SCIPR).

    Alignment of scRNA-seq data batches using an adaptation of the Iterative
    Closest Points (ICP) algorithm. SCIPR's core steps are matching the points
    and learning the transform function, and what strategy to use for each of
    these are specified by the user.

    Parameters
    ----------
    match_algo : scipr.matching.Match
        Which matching strategy to use, an instance of a :class:`Match`.

    transform_algo : scipr.transform.Transformer
        Which transformation strategy to use, an instance of a
        :class:`Transformer`.

    n_iter : int
        Number of steps of SCIPR to run. Each step is a matching phase,
        followed by updating the transformation function.

    input_normalization : {'l2', 'std', 'log'}
        Which input normalization to apply to data before aligning with SCIPR.
         - 'l2' : Scale each cell's count vector to have unit norm (vector
           length).
         - 'std' : Scale each gene to have zero mean and unit variance.
         - 'log' : Apply Seurat-style log normalization to each cell's count
           vector (as in Seurat's ``normalize`` function).
    """
    def __init__(self, match_algo, transform_algo, n_iter=5,
                 input_normalization='l2'):
        self.match_algo = match_algo
        self.transform_algo = transform_algo
        self.n_iter = n_iter
        self.input_normalization = input_normalization
        self.fitted = False

    def _log_hparams(self, tboard):
        tboard.add_hparams(hparam_dict={
            'match_algo': str(self.match_algo),
            'transform_algo': str(self.transform_algo),
            'n_iter': self.n_iter,
            'input_normalization': self.input_normalization},
                           metric_dict={})

    def fit(self, A, B, tensorboard=False, tensorboard_dir=None):
        """Fit the model to align to a reference batch.

        Parameters
        ----------
        A : numpy.ndarray
            The "source" batch of cells to align. Dimensions are
            (cellsA, genes).

        B : numpy.ndarray
            The "target" (or "reference") batch data to align to. ``A`` is
            aligned onto ``B``, where ``B`` is unchanged, and remains a
            stationary "reference". Dimensions are (cellsB, genes).

        tensorboard : bool
            If True, enable tensorboard logging of SCIPR algorithm metrics.

        tensorboard_dir : None or str
            If None, will use an automatically generated folder to store
            tensorboard event files. If specified, will place event files in
            the specified directory (creates it if it doesn't already exist).
        """
        log = logging.getLogger(__name__)
        if tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            tboard = SummaryWriter(log_dir=tensorboard_dir)
            if tensorboard_dir is None:
                log.warning('tensorboard_dir is not specified, using auto ' +
                            f'generated folder: {tboard.get_logdir()}')
            self._log_hparams(tboard)
        A = self._apply_input_normalization(A)
        B = self._apply_input_normalization(B)
        kd_B = spatial.cKDTree(B)
        A_orig = A
        for i in range(self.n_iter):
            a_idx, b_idx, distances = self.match_algo(A, B, kd_B)
            avg_distance = np.mean(distances)
            log.info(f'SCIPR step {i}, n-pairs: {len(a_idx)}, ' +
                     f'avg distance: {avg_distance}')
            if tensorboard:
                tboard.add_scalar('num_pairs', len(a_idx), i)
                tboard.add_scalar('avg_distance', avg_distance, i)
            step_model = self.transform_algo._fit_step(A[a_idx], B[b_idx], i)
            A = self.transform_algo.transform(step_model, A)
        self.transform_algo._finalize(A_orig, A)
        self.fitted = True

    def transform(self, A):
        """Apply alignment to a batch of cells.

        Cells are transformed to be aligned to the same "reference" batch from
        the :meth:`fit` method.

        Parameters
        ----------
        A : numpy.ndarray
            The batch of cells to align. Dimensions are (cellsA, genes).

        Returns
        -------
        numpy.ndarray
            The aligned batch of cells, same shape as input ``A``.

        Raises
        ------
        RuntimeError
            If this method is called before :meth:`fit` method.
        """
        if not self.fitted:
            raise RuntimeError('Must call "fit" before "transform"!')
        A = self._apply_input_normalization(A)
        return self.transform_algo._transform(A)

    def _apply_input_normalization(self, X):
        log = logging.getLogger(__name__)
        if self.input_normalization == 'std':
            log.info('Applying Standardization normalization')
            scaler = StandardScaler().fit(X)
            return scaler.transform(X)
        elif self.input_normalization == 'l2':
            log.info('Applying L2 normalization')
            return normalize(X)
        elif self.input_normalization == 'log':
            log.info('Applying log normalization')
            return np.log1p(X / X.sum(axis=1, keepdims=True) * 1e4)
