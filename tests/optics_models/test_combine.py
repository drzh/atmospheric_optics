from __future__ import annotations

import pytest

from optics_models.combination import combine_log
from optics_models.probability import sigmoid


def test_combine_log_increases_with_stronger_component_support() -> None:
    weak = combine_log(0.1, 0.1, 0.1)
    medium = combine_log(0.4, 0.4, 0.4)
    strong = combine_log(1.0, 1.0, 1.0)

    assert weak < medium < strong
    assert strong == pytest.approx(1.0, abs=0.001)


def test_combine_log_penalizes_missing_component_support() -> None:
    assert combine_log(0.0, 1.0, 1.0) < 0.01
    assert combine_log(1.0, 0.0, 1.0) < 0.01
    assert combine_log(1.0, 1.0, 0.0) < 0.01


@pytest.mark.parametrize("components", [(0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)])
def test_physical_gates_return_exact_zero(components) -> None:
    assert combine_log(*components) == 0.0


def test_sigmoid_saturates_without_overflow() -> None:
    assert sigmoid(-1e6, k=8.0) == 0.0
    assert sigmoid(1e6, k=8.0) == 1.0
