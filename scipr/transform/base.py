from abc import ABC, abstractmethod


class Transformer(ABC):
    """Base class for all transformation function objects.

    Your transformation functions should also subclass this class.
    """
    def __init__(self):
        self._model = None

    def _fit_step(self, A, B, step_number):
        step_model = self.fit(A, B)
        self._model = self.chain(self._model, step_model, step_number)
        return step_model

    @abstractmethod
    def fit(self, A, B):
        """Return a model to transform A onto B.

        Each point in A correspods to the point in B at the same index (as
        chosen by the matching algorithm of SCIPR). The Transformer is fitted
        to learn a function to move points in A closer to their corresponding
        point in B.

        Parameters
        ----------
        A : numpy.ndarray
            The selected "source" cells to align. Dimensions are
            (cells, genes).
        B : numpy.ndarray
            The "target" cells which correspond to each of the "source" cells
            in ``A``. Dimensions are the same as ``A``, so that each row in
            ``B`` is a cell that is paired up with the same row (cell) in
            ``A``.

        Returns
        -------
        model : dict
            The fitted model parameters (state) of the transformation function.
            For example, weights and biases.
        """
        pass

    @abstractmethod
    def transform(self, model, A):
        """Use the given model to transform the data.

        Parameters
        ----------
        model : dict
            The fitted model parameters (state) of the transformation function
            to use to transform ``A``.

        A : numpy.ndarray
            The cells to transform (i.e. to "align"), dimensions are
            (cells, genes).

        Returns
        -------
        numpy.ndarray
            The trasformation of ``A``, same shape as input ``A``.

        """
        pass

    def _transform(self, A):
        return self.transform(self._model, A)

    @abstractmethod
    def chain(self, model, step_model, step_number):
        """Update the overall model.

        Update the transformation function's parameters with the fitted
        parameters of the latest step. The overall alignment function we are
        learning is the composition of the transformation functions learned at
        each step, and you need to provide the logic for how to compose these
        functions here.

        Parameters
        ----------
        model : dict
            The current state of the overall model parameters, before fitting
            the latest step.
        step_model : dict
            The fitted model parameters (state) of the transformation function
            from the latest step.
        step_number : int
            The number of the current step in the SCIPR algorithm.

        Returns
        -------
        model : dict
            The updated overall model that is the fitted weights from all of
            prior steps, composed with the weights from the current latest
            step.
        """
        pass

    @abstractmethod
    def finalize(self, model, A_orig, A_final):
        """Finalize the overall model at the end of SCIPR.

        If there are any final operations necessary to fit the overall model,
        take them here.

        Parameters
        ----------
        model : dict
            The current state of the overall model parameters, after fitting
            and updating at all of the steps of SCIPR.
        A_orig : numpy.ndarray
            The original "source" batch before the first step of SCIPR, which
            we are fitting to align.
        A_final : numpy.ndarray
            The final state of the "source" batch after the last step of SCIPR,
            the result of transforming ``A_orig`` at each step.

        Returns
        -------
        model : dict
            The final overall model that is the fitted weights from all of the
            steps of the SCIPR algorithm.
        """
        pass

    def _finalize(self, A_orig, A_final):
        self._model = self.finalize(self._model, A_orig, A_final)
