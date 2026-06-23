import json

from config.models import (
    featureDetectionConfig,
    filterConfig,
    fastConfig,
    orbConfig,
    pipelineConfig
)

def load_config(file: str):
    with open(file, 'r') as f:
        data = json.load(f)
    
    return pipelineConfig(
        featureDetection = featureDetectionConfig(**data.get("featureDetection", {})),
        filter = filterConfig(**data.get("filter", {})),
        fast = fastConfig(**data.get("fast", {})),
        orb = orbConfig(**data.get("ORBConfig", {}))
    )