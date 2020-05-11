from abc import ABC, abstractmethod


class Transformer(ABC):
    def __init__(self):
        self.model = None

    def fit_step(self, A, B, step_number):
        step_model = self.fit(A, B)
        self.chain(step_model, step_number)
        return step_model

    @abstractmethod
    def fit(self, A, B):
        """Return a fitted (step) model"""
        pass

    @abstractmethod
    def transform(self, model, A):
        """Use the given model to transform the data"""
        pass

    def _transform(self, A):
        self.transform(self.model, A)

    @abstractmethod
    def chain(self, step_model, step_number):
        """Update the overall (from all steps) model"""
        pass

    @abstractmethod
    def finalize(self, A_orig, A_final):
        """Finalize the overall (from all steps) model"""
        pass
