"""Rule-based models for atmospheric optics prediction."""

from .circumzenithal_arc import predict_cza
from .combination import combine_log
from .halo import predict_halo, predict_parhelia
from .ice_crystals import (
    predict_circumhorizontal_arc,
    predict_sun_pillar,
    predict_upper_tangent_arc,
)
from .lunar import predict_lunar_corona
from .rainbow import predict_rainbow
from .scattering import predict_crepuscular_rays, predict_fogbow

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
