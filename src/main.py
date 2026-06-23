from . import offlineVideoStabilization as offVS, optimizationVideoStabilization as optVS
from config.loader import load_config

INPUT_DIR = "unstable_videos/"
OUTPUT_DIR = "output/"
cfg = load_config("config/config.json")

def main():
    with open('videosConfig.txt', 'r') as f:
	    input_videos = f.read().split()

    for input_video in input_videos:
        output_video = "out_" + input_video
        stabilize(str(INPUT_DIR+input_video), str(OUTPUT_DIR+output_video), cfg, cfg.feature_type.ORB, cfg.filter_type.Gauss, optimal=True)
        

def stabilize(input, output, cfg, feature, filter, optimal=False):
    print(f"Offline stabilization of {input} with: {filter}, filter and {feature} featureDetection")
    offVS.stabilize(input, output, cfg, feature_detection=feature, filter=filter)
    if optimal:
        print(f"Optimal stabilization of {input} with: {feature} featureDetection")
        optVS.stabilize(input, output, cfg, feature_detection=feature)

if __name__ == "__main__":
    main()

