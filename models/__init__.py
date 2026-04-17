"""Rule-based models for atmospheric optics prediction."""

from .combine import combine_log
from .cza_model import predict_cza
from .halo_model import predict_halo, predict_parhelia
from .rainbow_model import predict_rainbow
from .ice_crystal_model import (
    predict_circumhorizontal_arc,
    predict_sun_pillar,
    predict_upper_tangent_arc,
)
from .lunar_model import predict_lunar_corona
from .scattering_model import predict_crepuscular_rays, predict_fogbow

__all__ = [
    "combine_log",
    "predict_cza",
    "predict_circumhorizontal_arc",
    "predict_crepuscular_rays",
    "predict_fogbow",
    "predict_halo",
    "predict_lunar_corona",
    "predict_parhelia",
    "predict_rainbow",
    "predict_sun_pillar",
    "predict_upper_tangent_arc",
]
