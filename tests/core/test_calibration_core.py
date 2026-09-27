import math

import pytest

from metanoia.calibration import softmax


def test_softmax_is_stable_and_normalized() -> None:
    p = softmax([1000.0, 999.0, -1000.0])
    assert math.isclose(sum(p), 1.0)
    assert math.isclose(p[0] / p[1], math.e)


def test_temperature_flattens_without_reordering() -> None:
    sharp, flat = softmax([2.0, 1.0, 0.0]), softmax([2.0, 1.0, 0.0], temperature=10.0)
    assert sorted(range(3), key=sharp.__getitem__) == sorted(range(3), key=flat.__getitem__)
    assert max(flat) < max(sharp)
    with pytest.raises(ValueError):
        softmax([1.0], temperature=0.0)
