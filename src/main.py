import cv2
import numpy as np
from . import offlineVideoStabilization
from config.loader import load_config
from matplotlib import pyplot as plt

INPUT_VIDEO = "unstable_videos/videoplayback.mp4"
OUTPUT_VIDEO = "output/output_fast1.mp4"
cfg = load_config("config/config.json")

INPUT_PICTURE = "harris_test.jpg"

def main():
    print("OpenCV version: " + cv2.__version__)
    # videoStabilization.simpleStabilize(INPUT_VIDEO, OUTPUT_VIDEO, cfg, debug=True)
    offlineVideoStabilization.fastStabilize(INPUT_VIDEO, OUTPUT_VIDEO, cfg, False)

if __name__ == "__main__":
    main()