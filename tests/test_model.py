import numpy as np
import pytest

import best


@pytest.fixture
def mock_trace():
    np.random.seed(0)
    S = 10000
    return best.model.BestResultsOne(None, {'Means': np.random.randn(S)}), S


def test_posterior_prob(mock_trace):
    br, S = mock_trace

    def error(p):
        return (p * (1 - p) / S) ** 0.5 * 3

    assert br.posterior_prob('Means', -1, 1) == pytest.approx(0.683, abs=error(0.683))
    assert br.posterior_prob('Means', low=1) == pytest.approx(0.159, abs=error(0.159))
    assert br.posterior_prob('Means', high=-1) == pytest.approx(0.159, abs=error(0.159))
    assert br.posterior_prob('Means', low=1, high=-1) == 0
    assert br.posterior_prob('Means') == 1


def test_hpd(mock_trace):
    br, _ = mock_trace
    assert br.hpd('Means', 0.05) == pytest.approx((-1.96, 1.96), abs=0.1)
