#!/usr/bin/env python

"""Tests for `scipr` package."""


import unittest

import numpy as np
from scipy import spatial

# import scipr
from scipr.matching import Closest, MNN
# from scipr.transform import Rigid, Affine


class TestMatching(unittest.TestCase):
    """Tests for Match algorithms."""

    def setUp(self):
        """Set up test fixtures, if any."""
        self.A = np.array([
            [0., 0.],
            [-0.5, -0.5]
        ])
        self.B = np.array([
            [1., 1.],
            [-1.0, -0.5]
        ])
        self.kd_B = spatial.cKDTree(self.B)
        self.kd_A = spatial.cKDTree(self.A)

    def tearDown(self):
        """Tear down test fixtures, if any."""

    def test_000_closest_A2B(self):
        """Test closest matching strategy A -> B"""
        match = Closest()
        a_idx, b_idx, distances = match(self.A, self.B, self.kd_B)
        self.assertTrue(np.array_equal(a_idx, [0, 1]))
        self.assertTrue(np.array_equal(b_idx, [1, 1]))

    def test_001_closest_B2A(self):
        """Test closest matching strategy B -> A"""
        match = Closest()
        a_idx, b_idx, distances = match(self.B, self.A, self.kd_A)
        self.assertTrue(np.array_equal(a_idx, [0, 1]))
        self.assertTrue(np.array_equal(b_idx, [0, 1]))

    def test_002_MNN_A2B(self):
        """Test MNN matching strategy A -> B"""
        match = MNN(k=2)
        a_idx, b_idx, distances = match(self.A, self.B, self.kd_B)
        print(a_idx)
        self.assertTrue(np.array_equal(a_idx, [1]))
        self.assertTrue(np.array_equal(b_idx, [1]))

    def test_003_MNN_B2A(self):
        """Test MNN matching strategy B -> A"""
        match = MNN(k=2)
        a_idx, b_idx, distances = match(self.B, self.A, self.kd_A)
        self.assertTrue(np.array_equal(a_idx, [1]))
        self.assertTrue(np.array_equal(b_idx, [1]))