#!/usr/bin/env python

"""Tests for `scipr` package."""


import unittest
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
import tempfile

import numpy as np
import anndata

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

    def test_004_test_transform_same_shape(self):
        """Test tranforming returns data of same shape as input"""
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.rigid_transform,
                            input_normalization='l2')
        model.fit(self.A, self.B)
        transformed = model.transform(self.A)
        self.assertEqual(self.A.shape, transformed.shape)

    def test_005_test_logging(self):
        """Test fit function emits logging information"""
        with self.assertLogs(level='INFO') as cm:
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.rigid_transform,
                                input_normalization='l2')
            model.fit(self.A, self.B)
        self.assertEqual(cm.output[:2],
                         ['INFO:scipr.scipr:Applying L2 normalization',
                          'INFO:scipr.scipr:Applying L2 normalization'])

    def test_006_no_stdout_in_normal_conditions(self):
        """Test no output (silent) from SCIPR methods under normal
        conditions (no warnings)"""
        f = io.StringIO()
        with redirect_stdout(f):
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.affine_transform,
                                input_normalization='l2',
                                n_iter=2)
            model.fit(self.A, self.B)
            model.transform(self.A)
        self.assertEqual(f.getvalue(), '')

    def _does_folder_contain_event_file(self, folder):
        for f in folder.iterdir():
            if 'events.out.tfevents' in f.name:
                return True
        return False

    def test_007_tensorboard_auto_directory(self):
        """Test auto creates directory for event files if not specified"""
        f = io.StringIO()
        with redirect_stderr(f):
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.rigid_transform,
                                input_normalization='l2',
                                n_iter=2)
            model.fit(self.A, self.B, tensorboard=True)
        stdout = f.getvalue()
        self.assertIn('tensorboard_dir is not specified', stdout)
        tboard_path = Path(stdout.split()[-1])
        self.assertTrue(tboard_path.exists())
        self.assertTrue(self._does_folder_contain_event_file(tboard_path))

    def test_008_tensorboard_custom_directory(self):
        """Test creates specified directory for event files"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tboard_path = Path(tmp_dir)
            model = scipr.SCIPR(match_algo=self.match,
                                transform_algo=self.rigid_transform,
                                input_normalization='l2',
                                n_iter=2)
            model.fit(self.A, self.B, tensorboard=True,
                      tensorboard_dir=tboard_path)
            self.assertTrue(tboard_path.exists())
            self.assertTrue(self._does_folder_contain_event_file(tboard_path))

    def _test_anndata_helper(self):
        model = scipr.SCIPR(match_algo=self.match,
                            transform_algo=self.affine_transform,
                            n_iter=3,
                            input_normalization='l2')
        adata = anndata.AnnData(X=np.concatenate([self.A, self.B], axis=0),
                                obs={
                                    'batch': (['A'] * self.A.shape[0]) +
                                             (['B'] * self.B.shape[0])
                                },
                                dtype=self.A.dtype)
        model.fit_adata(adata, 'batch', 'A', 'B')
        return adata, model

    def test_009_anndata_transform(self):
        """Test fit and transform with AnnData is equivalent to numpy"""
        adata, model = self._test_anndata_helper()
        transformed_adata, idx = model.transform_adata(adata, 'batch', 'A')
        self.assertEqual(transformed_adata.shape[0], np.sum(idx))
        transformed_np = model.transform(self.A)
        self.assertTrue(np.array_equal(transformed_adata, transformed_np))

    def test_010_anndata_transform_inplace(self):
        """Test inplace transform of AnnData is equivalent to numpy"""
        adata, model = self._test_anndata_helper()
        model.transform_adata(adata, 'batch', 'A', inplace=True)
        transformed_adata = adata[adata.obs['batch'] == 'A'].X
        transformed_np = model.transform(self.A)
        self.assertTrue(np.array_equal(transformed_adata, transformed_np))
