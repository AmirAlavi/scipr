#!/usr/bin/env python

"""Tests for `scipr.matching` module."""


import unittest

import numpy as np
from scipy import spatial

from scipr.matching import Closest, MNN, Greedy


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
        self.C = np.array([
            [0., 0.],
            [0., 1.],
            [0., -1.1]
        ])
        self.D = np.array([
            [1., 0.],
            [1., 0.1],
            [1., 0.2]
        ])
        self.kd_A = spatial.cKDTree(self.A)
        self.kd_B = spatial.cKDTree(self.B)
        self.kd_C = spatial.cKDTree(self.C)
        self.kd_D = spatial.cKDTree(self.D)

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
        b_idx, a_idx, distances = match(self.B, self.A, self.kd_A)
        self.assertTrue(np.array_equal(b_idx, [0, 1]))
        self.assertTrue(np.array_equal(a_idx, [0, 1]))

    def test_002_MNN_A2B(self):
        """Test MNN matching strategy A -> B"""
        match = MNN(k=1)
        a_idx, b_idx, distances = match(self.A, self.B, self.kd_B)
        self.assertTrue(np.array_equal(a_idx, [1]))
        self.assertTrue(np.array_equal(b_idx, [1]))

    def test_003_MNN_B2A(self):
        """Test MNN matching strategy B -> A"""
        match = MNN(k=1)
        b_idx, a_idx, distances = match(self.B, self.A, self.kd_A)
        self.assertTrue(np.array_equal(b_idx, [1]))
        self.assertTrue(np.array_equal(a_idx, [1]))

    def test_004_greedy_C2D(self):
        """Test greedy matching strategy C -> D"""
        match = Greedy(alpha=1, beta=2)
        c_idx, d_idx, distances = match(self.C, self.D, self.kd_D)
        self.assertTrue(np.array_equal(c_idx, [0, 1, 2]))
        self.assertTrue(np.array_equal(d_idx, [0, 2, 0]))

    def test_005_greedy_D2C(self):
        """Test greedy matching strategy D -> C"""
        match = Greedy(alpha=1, beta=2)
        d_idx, c_idx, distances = match(self.D, self.C, self.kd_C)
        self.assertTrue(np.array_equal(d_idx, [0, 1, 2]))
        self.assertTrue(np.array_equal(c_idx, [0, 0, 1]))

    def test_006_greedy_C2D_beta_1(self):
        """Test greedy matching strategy C -> D with beta=1"""
        match = Greedy(alpha=1, beta=1)
        c_idx, d_idx, distances = match(self.C, self.D, self.kd_D)
        self.assertTrue(np.array_equal(c_idx, [0, 1, 2]))
        self.assertTrue(np.array_equal(d_idx, [0, 2, 1]))

    def test_007_greedy_D2C_beta_1(self):
        """Test greedy matching strategy D -> C with beta=1"""
        match = Greedy(alpha=1, beta=1)
        d_idx, c_idx, distances = match(self.D, self.C, self.kd_C)
        self.assertTrue(np.array_equal(d_idx, [0, 1, 2]))
        self.assertTrue(np.array_equal(c_idx, [0, 2, 1]))

    def test_008_greedy_C2D_alpha_05(self):
        """Test greedy matching strategy C -> D with alpha=0.5"""
        match = Greedy(alpha=0.5, beta=2)
        c_idx, d_idx, distances = match(self.C, self.D, self.kd_D)
        self.assertTrue(np.array_equal(c_idx, [0, 1]))
        self.assertTrue(np.array_equal(d_idx, [0, 2]))

    def test_009_greedy_D2C_alpha_05(self):
        """Test greedy matching strategy D -> C with alpha=0.5"""
        match = Greedy(alpha=0.5, beta=2)
        d_idx, c_idx, distances = match(self.D, self.C, self.kd_C)
        self.assertTrue(np.array_equal(d_idx, [0, 1]))
        self.assertTrue(np.array_equal(c_idx, [0, 0]))
