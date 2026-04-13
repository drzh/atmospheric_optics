from __future__ import annotations

import pytest

from models.combine import combine_log


def test_combine_log_increases_with_stronger_component_support() -> None:
    weak = combine_log(0.1, 0.1, 0.1)
    medium = combine_log(0.4, 0.4, 0.4)
    strong = combine_log(1.0, 1.0, 1.0)

    assert weak < medium < strong
    assert strong == pytest.approx(0.5, abs=0.001)


def test_combine_log_penalizes_missing_component_support() -> None:
    assert combine_log(0.0, 1.0, 1.0) < 0.01
    assert combine_log(1.0, 0.0, 1.0) < 0.01
    assert combine_log(1.0, 1.0, 0.0) < 0.01
