from dataclasses import dataclass, field

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
    zoom: float = 1.05

@dataclass
class fastConfig:
    threshold: int = 25

@dataclass
class pipelineConfig:
    featureDetection: featureDetectionConfig = field(
        default_factory=featureDetectionConfig
    )
    filter: filterConfig = field(
        default_factory=filterConfig
    )
    fast: fastConfig = field(
        default_factory=fastConfig
    )