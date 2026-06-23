import cv2
import numpy as np
from . import offlineVideoStabilization as offVS, optimizationVideoStabilization as optVS
from config.loader import load_config
from matplotlib import pyplot as plt

INPUT_VIDEO = "unstable_videos/videoplayback.mp4"
OUTPUT_VIDEO = "output/output_optimization2.avi"
cfg = load_config("config/config.json")

def main():
    print("OpenCV version: " + cv2.__version__)
    #offVS.stabilize(INPUT_VIDEO, OUTPUT_VIDEO, cfg, feature_detection=offVS.featureDetection.FAST, filter=offVS.Filter.Gauss)
    optVS.stabilize(INPUT_VIDEO, OUTPUT_VIDEO, cfg, feature_detection=optVS.featureDetection.ORB)

if __name__ == "__main__":
    main()