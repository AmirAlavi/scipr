#!/usr/bin/env python

"""Tests for `scipr` package."""


import unittest

import numpy as np

from scipr import scipr


class TestScipr(unittest.TestCase):
    """Tests for `scipr` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        np.random.seed(1817)
        self.A = np.random.random((10, 100))
        self.B = np.random.random((20, 100))

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_constructor(self):
        """Test model constructors."""
        scipr_affine = scipr.AffineSCIPR()
        scrpr_ae = scipr.StackedAutoEncoderSCIPR()
    
    def test_001_fit_affine_closest(self):
        """Test affine fitting with the 'closest' algorithm"""
        scipr_affine = scipr.AffineSCIPR(n_iter=1, n_epochs_per_iter=1, matching_algo='closest')
        scipr_affine.fit(self.A, self.B)
    
    def test_002_fit_affine_mnn(self):
        """Test affine fitting with the 'mnn' algorithm"""
        scipr_affine = scipr.AffineSCIPR(n_iter=1, n_epochs_per_iter=1, matching_algo='mnn')
        scipr_affine.fit(self.A, self.B)
    
    def test_003_fit_affine_greedy(self):
        """Test affine fitting with the 'greedy'"""
        scipr_affine = scipr.AffineSCIPR(n_iter=1, n_epochs_per_iter=1, matching_algo='greedy')
        scipr_affine.fit(self.A, self.B)