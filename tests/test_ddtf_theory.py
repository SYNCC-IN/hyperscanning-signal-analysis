"""Theoretical properties of dDTF: partial coherence of a cascade's indirect pair is exactly zero for known coefficients."""

import numpy as np

from src.sim_estimation import connectivity_cube, freq_matrices
from src.sim_generators import build_coupling

FS = 2.5
FREQS = np.linspace(0.02, 1.23, 100)


def cascade_model():
    """Cascade node1 -> node0 -> node2 plus an unconnected node 3 (AR(2) self terms)."""
    return build_coupling([0.55, 0.55, 0.86, 0.86], [0.35] * 4, FS, [(1, 0, 1, 0.4), (0, 2, 1, 0.4)], 3), np.eye(4)


def test_indirect_pair_has_zero_partial_coherence_and_zero_ddtf():
    coupling, variance = cascade_model()
    ddtf = connectivity_cube(coupling, variance, FREQS, FS, "dDTF")
    ffdtf = connectivity_cube(coupling, variance, FREQS, FS, "ffDTF")
    assert ffdtf[2, 1].max() > 1e-3          # the indirect total flow 1 -> 2 exists
    assert ddtf[2, 1].max() < 1e-12          # but the direct measure is zero
    assert ddtf[0, 1].min() > 0              # direct edge present


def test_partial_coherence_matches_spectral_matrix_inverse():
    coupling, variance = cascade_model()
    a = freq_matrices(coupling, FREQS, FS)
    h = np.linalg.inv(a)
    s_inv = np.linalg.inv(h @ variance @ np.conj(h.transpose(0, 2, 1)))
    diag = np.real(np.einsum("fii->fi", s_inv))
    kappa = np.abs(s_inv) / np.sqrt(diag[:, :, None] * diag[:, None, :])
    ddtf = connectivity_cube(coupling, variance, FREQS, FS, "dDTF")
    ffdtf = connectivity_cube(coupling, variance, FREQS, FS, "ffDTF")
    assert np.allclose(ddtf, ffdtf * kappa.transpose(1, 2, 0), atol=1e-12)
