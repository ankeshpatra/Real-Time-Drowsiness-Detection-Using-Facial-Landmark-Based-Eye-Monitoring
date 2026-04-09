"""
test_ear.py — Unit tests for the EAR calculation and utility functions.

Run with:
    python test_ear.py
"""

import unittest
import numpy as np

from utils import eye_aspect_ratio


class TestEyeAspectRatio(unittest.TestCase):
    """Test the EAR formula against known geometric configurations."""

    def test_wide_open_eye(self):
        """A perfectly round eye should give EAR ≈ 0.5."""
        # Simulate a wide-open eye:
        #   p1=(0,0), p2=(1,1), p3=(3,1), p4=(4,0), p5=(3,-1), p6=(1,-1)
        eye = np.array([
            [0, 0],    # p1 — left corner
            [1, 1],    # p2 — upper-left lid
            [3, 1],    # p3 — upper-right lid
            [4, 0],    # p4 — right corner
            [3, -1],   # p5 — lower-right lid
            [1, -1],   # p6 — lower-left lid
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        # A = ||p2-p6|| = ||(1,1)-(1,-1)|| = 2.0
        # B = ||p3-p5|| = ||(3,1)-(3,-1)|| = 2.0
        # C = ||p1-p4|| = ||(0,0)-(4,0)|| = 4.0
        # EAR = (2+2) / (2*4) = 0.5
        self.assertAlmostEqual(ear, 0.5, places=5)

    def test_fully_closed_eye(self):
        """When vertical distances collapse to zero, EAR should be 0."""
        eye = np.array([
            [0, 0],
            [1, 0],
            [3, 0],
            [4, 0],
            [3, 0],
            [1, 0],
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        self.assertAlmostEqual(ear, 0.0, places=5)

    def test_half_open_eye(self):
        """EAR with half the vertical opening of wide-open."""
        eye = np.array([
            [0, 0],
            [1, 0.5],
            [3, 0.5],
            [4, 0],
            [3, -0.5],
            [1, -0.5],
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        # A = 1.0, B = 1.0, C = 4.0
        # EAR = (1+1) / (2*4) = 0.25
        self.assertAlmostEqual(ear, 0.25, places=5)

    def test_degenerate_horizontal_zero(self):
        """If p1 == p4, the denominator is zero; should return 0.0."""
        eye = np.array([
            [2, 0],
            [2, 1],
            [2, 1],
            [2, 0],
            [2, -1],
            [2, -1],
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        self.assertEqual(ear, 0.0)

    def test_asymmetric_eye(self):
        """Test with non-symmetric landmark positions."""
        eye = np.array([
            [0, 0],    # p1
            [1, 2],    # p2
            [3, 1],    # p3
            [4, 0],    # p4
            [3, -1],   # p5
            [1, -1],   # p6
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        # A = ||p2-p6|| = ||(1,2)-(1,-1)|| = 3.0
        # B = ||p3-p5|| = ||(3,1)-(3,-1)|| = 2.0
        # C = ||p1-p4|| = 4.0
        # EAR = (3+2) / (2*4) = 0.625
        self.assertAlmostEqual(ear, 0.625, places=5)

    def test_returns_float(self):
        """Ensure the function always returns a Python float."""
        eye = np.array([
            [0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]
        ], dtype=float)
        result = eye_aspect_ratio(eye)
        self.assertIsInstance(result, float)


class TestThresholdLogic(unittest.TestCase):
    """Verify that typical EAR values fall in the documented ranges."""

    def test_open_above_threshold(self):
        """An open eye's EAR should be above the default threshold (0.25)."""
        # Simulate open eye
        eye = np.array([
            [0, 0], [1, 1], [3, 1], [4, 0], [3, -1], [1, -1]
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        self.assertGreater(ear, 0.25)

    def test_closed_below_threshold(self):
        """A closed eye's EAR should be below the default threshold."""
        eye = np.array([
            [0, 0], [1, 0.05], [3, 0.05], [4, 0], [3, -0.05], [1, -0.05]
        ], dtype=float)
        ear = eye_aspect_ratio(eye)
        self.assertLess(ear, 0.25)


if __name__ == "__main__":
    unittest.main(verbosity=2)
