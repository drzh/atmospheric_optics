"""Lunar-specific atmospheric optics rules."""

from __future__ import annotations

from typing import Mapping

from optics_models.combination import ModelComponents
from optics_models.corona import CoronaParameters, predict_corona
from optics_models.probability import unit_feature


LunarCoronaParameters = CoronaParameters


DEFAULT_LUNAR_CORONA_PARAMETERS = LunarCoronaParameters()


def predict_lunar_corona(
    features: Mapping[str, float],
    parameters: LunarCoronaParameters = DEFAULT_LUNAR_CORONA_PARAMETERS,
    *,
    return_components: bool = False,
) -> float | ModelComponents:
    """Predict the probability of a lunar corona."""

    lunar_features = dict(features)
    lunar_features["source_visible"] = unit_feature(features, "source_visible", default=unit_feature(features, "moon_visible"))
    return predict_corona(lunar_features, parameters, return_components=return_components)
