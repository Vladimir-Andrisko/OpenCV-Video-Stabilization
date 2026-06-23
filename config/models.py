from dataclasses import dataclass, field
from enum import Enum

class Feature(Enum):
    ShiTomasi = 0
    FAST = 1
    ORB = 2

class Filter(Enum):
    MoovingAverage = 0
    Gauss = 1
    Savgol = 2
    LowPass = 3

@dataclass
class Config:
    zoom: float = 1.05
    max_translation: int = 30
    max_rotation: float = 0.3

@dataclass
class featureDetectionConfig:
    max_corners: int = 200
    quality_level: float = 0.01
    min_distance: float = 20.0
    block_size: int = 3
    gradient_size: int = 3
    use_harris_detector: bool = False

@dataclass
class filterConfig:
    moving_average_radius: int = 1
    gauss_radius: int = 1
    gauss_sigma: float = 0.5
    savgol_window: int = 21
    savgol_poly: int = 3

@dataclass
class fastConfig:
    threshold: int = 25

@dataclass
class orbConfig:
    nfeatures: int = 100
    scaleFactor: float = 1.2
    nlevels: int = 8
    edgeThreshold: int = 31
    firstLevel: int = 0
    WTA_K: int = 2
    patchSize: int = 31
    fastThreshold: int = 20

@dataclass
class pipelineConfig:
    feature_type: Feature = Feature.ShiTomasi
    filter_type: Filter = Filter.MoovingAverage

    config: Config = field(
        default_factory=Config
    )
    featureDetection: featureDetectionConfig = field(
        default_factory=featureDetectionConfig
    )
    filter: filterConfig = field(
        default_factory=filterConfig
    )
    fast: fastConfig = field(
        default_factory=fastConfig
    )
    orb: orbConfig = field(
        default_factory=orbConfig
    )