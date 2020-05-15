#!/usr/bin/env python

"""Tests for `scipr` package."""


import unittest
import io
from contextlib import redirect_stdout

import numpy as np

import scipr
from scipr.matching import Closest
from scipr.transform import Rigid, Affine


class TestScipr(unittest.TestCase):
    """Tests for `scipr` package."""

    def setUp(self):
        """Set up test fixtures, if any."""
        np.random.seed(1817)
        self.A = np.random.random((10, 100))
        self.B = np.random.random((20, 100))
        # Example Match and Transform objects,
        # useful to instantiate basic SCIPR objects
        self.match = Closest()
        self.rigid_transform = Rigid()
        self.affine_transform = Affine()

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_l2_normalization(self):
        """Test L2 input normalization."""
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.rigid_transform,
                            input_normalization='l2')
        normalized = model._apply_input_normalization(self.A)
        norm = np.linalg.norm(normalized, axis=1)
        self.assertTrue(np.allclose(norm, 1.0))

    def test_001_std_normalization(self):
        """Test standard scaling input normalization"""
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.rigid_transform,
                            input_normalization='std')
        normalized = model._apply_input_normalization(self.A)
        means = np.mean(normalized, axis=0)
        stds = np.std(normalized, axis=0)
        self.assertTrue(np.allclose(means, 0.))
        self.assertTrue(np.allclose(stds, 1.))

    def test_002_log_normalization(self):
        """Test Seurat-style log input normalization"""
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.rigid_transform,
                            input_normalization='log')
        normalized = model._apply_input_normalization(self.A)
        feature_counts = np.expand_dims(np.sum(self.A, axis=1), 1)
        correct = np.log1p((self.A / feature_counts) * 10000)
        self.assertTrue(np.allclose(normalized, correct))

    def test_003_raise_unfitted(self):
        """Test raises Runtime Exception if transform called before fit"""
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.rigid_transform,
                            input_normalization='l2')
        with self.assertRaises(RuntimeError):
            model.transform(self.A)

    def test_004_test_logging(self):
        with self.assertLogs(level='INFO') as cm:
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.rigid_transform,
                                input_normalization='l2')
            model.fit(self.A, self.B)
        self.assertEqual(cm.output[:2],
                         ['INFO:scipr.scipr:Applying L2 normalization',
                          'INFO:scipr.scipr:Applying L2 normalization'])

    def test_005_no_stdout_in_normal_conditions(self):
        f = io.StringIO()
        with redirect_stdout(f):
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.affine_transform,
                                input_normalization='l2',
                                n_iter=2)
            model.fit(self.A, self.B)
            model.transform(self.A)
        self.assertEqual(f.getvalue(), '')
