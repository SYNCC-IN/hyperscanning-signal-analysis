"""
Unit tests for the secore_loader module.

Tests compute_signal_lag and fix_and_interpolate_ibi with synthetic inputs
to verify correct behavior without requiring real data files.
"""
import numpy as np
import pytest

from src.secore_loader import compute_signal_lag, fix_and_interpolate_ibi


class TestComputeSignalLag:
    """Tests for compute_signal_lag."""

    def test_zero_lag_identical_signals(self):
        """Identical signals should return lag of 0."""
        rng = np.random.default_rng(0)
        s = rng.standard_normal(200)
        lag = compute_signal_lag(s, s)
        assert lag == 0

    def test_known_positive_lag(self):
        """Signal2 leading signal1 by k samples should return lag of +k."""
        rng = np.random.default_rng(1)
        s = rng.standard_normal(300)
        k = 20
        # np.roll(s, -k) shifts s left by k: s2 leads s1 by k samples
        s_shifted = np.roll(s, -k)
        lag = compute_signal_lag(s, s_shifted)
        assert lag == k

    def test_known_negative_lag(self):
        """Signal2 lagging signal1 by k samples should return lag of -k."""
        rng = np.random.default_rng(2)
        s = rng.standard_normal(300)
        k = 15
        # np.roll(s, k) shifts s right by k: s2 lags s1 by k samples
        s_shifted = np.roll(s, k)
        lag = compute_signal_lag(s, s_shifted)
        assert lag == -k

    def test_returns_int(self):
        """Return value should be a plain Python int (or numpy integer)."""
        rng = np.random.default_rng(3)
        s = rng.standard_normal(100)
        lag = compute_signal_lag(s, s)
        assert isinstance(lag, (int, np.integer))

    def test_2d_input_flattened(self):
        """2-D column-vector inputs should be handled via flatten()."""
        rng = np.random.default_rng(4)
        s = rng.standard_normal((100, 1))
        lag = compute_signal_lag(s, s)
        assert lag == 0


class TestFixAndInterpolateIbi:
    """Tests for fix_and_interpolate_ibi."""

    def _make_synthetic_ibi(self, n=200, mean_ibi_ms=800.0, fs_out=4, seed=0):
        """Return (ibi_cum_s, stage) for a synthetic regular IBI sequence."""
        rng = np.random.default_rng(seed)
        ibi_ms = mean_ibi_ms + rng.standard_normal(n) * 20.0
        ibi_ms = np.clip(ibi_ms, 400.0, 1500.0)
        ibi_cum_s = np.cumsum(ibi_ms) / 1000.0
        stage = np.ones(n, dtype=int)
        return ibi_cum_s, stage

    def test_output_tuple_length(self):
        """Function should return a 6-tuple."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        result = fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=4, window_size=5)
        assert len(result) == 6

    def test_output_arrays_same_length(self):
        """t_interp, ibi_interp, stage_interp, and rmssd_interp must share length."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        t_interp, ibi_interp, stage_interp, nn_ms, t_nn, rmssd_interp = (
            fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=4, window_size=5)
        )
        assert t_interp.shape == ibi_interp.shape
        assert t_interp.shape == stage_interp.shape
        assert t_interp.shape == rmssd_interp.shape

    def test_t_interp_uniform_spacing(self):
        """t_interp should be a uniform grid with step 1/fs_out."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        fs_out = 4
        t_interp, *_ = fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=fs_out, window_size=5)
        diffs = np.diff(t_interp)
        np.testing.assert_allclose(diffs, 1.0 / fs_out, rtol=1e-6)

    def test_rmssd_non_negative(self):
        """RMSSD values should all be >= 0."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        *_, rmssd_interp = fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=4, window_size=5)
        assert np.all(rmssd_interp >= 0.0)

    def test_rmssd_no_nan(self):
        """RMSSD should have no NaN values after interpolation/fill."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        *_, rmssd_interp = fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=4, window_size=5)
        assert not np.any(np.isnan(rmssd_interp))

    def test_stage_interp_values_are_integers(self):
        """stage_interp should only contain integer values."""
        ibi_cum_s, stage = self._make_synthetic_ibi()
        _, _, stage_interp, *_ = fix_and_interpolate_ibi(ibi_cum_s, stage, fs_out=4, window_size=5)
        assert stage_interp.dtype.kind in ("i", "u"), "stage_interp should have integer dtype"
