"""Tests for src.group_model.convergence_row."""
import arviz as az
import numpy as np

from src.group_model import convergence_row


def test_convergence_row_uses_unrounded_rhat():
    """R-hat between 1.005 and 1.01 must pass the < 1.01 gate (arviz.summary would round it to 1.01)."""
    rng = np.random.default_rng(0)
    draws = rng.standard_normal((4, 4000))
    for shift in np.linspace(0.0, 0.2, 201):                 # find a chain offset giving R-hat in [1.005, 1.01)
        theta = draws + shift * np.arange(4)[:, None]
        rhat = float(az.rhat(az.from_dict(posterior={"theta": theta})).theta)
        if 1.005 <= rhat < 1.01:
            break
    idata = az.from_dict(posterior={"theta": theta},
                         sample_stats={"diverging": np.zeros((4, 4000), dtype=bool)},
                         log_likelihood={"y": rng.standard_normal((4, 4000, 20)) - 1.0})
    row = convergence_row("m", idata, 20, 0, rhat_max=1.01, ess_min=400, pareto_k_max=np.inf)
    assert 1.005 <= row["max_rhat"] < 1.01
    assert row["max_rhat"] == rhat
    assert row["pass_l9"]
