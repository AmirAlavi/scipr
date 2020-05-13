import numpy as np

import scipr
from scipr.matching import Closest, Hungarian, Greedy, MNN
from scipr.transform import Affine, Rigid, StackedAutoEncoder

np.random.seed(1817)
A = np.random.random((10, 100))
B = np.random.random((20, 100))

matchers = [Closest(), Greedy(), Hungarian(), MNN()]
for match in matchers:
    transform = StackedAutoEncoder()

    model = scipr.SCIPR(match, transform, n_iter=2)

    model.fit(A, B)

    A_new = model.transform(A)

    assert(np.all(np.not_equal(A, A_new)))