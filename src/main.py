import cv2
import numpy as np
from . import offlineVideoStabilization as offVS
from config.loader import load_config
from matplotlib import pyplot as plt

INPUT_VIDEO = "unstable_videos/eurobot_memristor.mp4"
OUTPUT_VIDEO = "output/output_test2.avi"
cfg = load_config("config/config.json")

def main():
    print("OpenCV version: " + cv2.__version__)
    offVS.stabilize(INPUT_VIDEO, OUTPUT_VIDEO, cfg, feature_detection=offVS.featureDetection.FAST, filter=offVS.Filter.Gauss)

if __name__ == "__main__":
    main()