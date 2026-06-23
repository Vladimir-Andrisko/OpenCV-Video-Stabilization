import json
from config.models import (
    Config_offline,
    Config_optimal,
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
        configOffline = Config_offline(**data.get("CONFIG_offline", {})),
        configOptimal = Config_optimal(**data.get("CONFIG_optimal", {})),
        featureDetection = featureDetectionConfig(**data.get("featureDetection", {})),
        filter = filterConfig(**data.get("filter", {})),
        fast = fastConfig(**data.get("fast", {})),
        orb = orbConfig(**data.get("ORBConfig", {}))
    )