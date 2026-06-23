import numpy as np
from scipy.signal import savgol_filter, butter, filtfilt
import matplotlib.pyplot as plt
import cv2

def movingAverage(curve, radius):
    window_size = 2*radius + 1

    # kernel
    f = np.ones(window_size)/window_size

    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def gaussianSmoothing(curve, radius, sigma=None):
    window_size = 2*radius + 1

    if sigma is None:
        sigma = radius / 2
    x = np.arange(window_size) - radius

    # Gaussian kernel
    f = np.exp(-(x**2) / (2 * sigma**2))
    f = f / np.sum(f)

    curve_pad = np.pad(curve, (radius, radius), mode='edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed
    

def create_kalman():
    kalman = cv2.KalmanFilter(6, 3)

    # measurement: dx, dy, angle
    kalman.measurementMatrix = np.zeros((3, 6), np.float32)
    kalman.measurementMatrix[0, 0] = 1
    kalman.measurementMatrix[1, 1] = 1
    kalman.measurementMatrix[2, 2] = 1

    kalman.transitionMatrix = np.eye(6, dtype=np.float32)
    dt = 1.0

    # position += velocity
    kalman.transitionMatrix[0, 3] = dt
    kalman.transitionMatrix[1, 4] = dt
    kalman.transitionMatrix[2, 5] = dt

    kalman.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4
    kalman.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2
    kalman.errorCovPost = np.eye(6, dtype=np.float32)

    kalman.statePost = np.zeros((6, 1), dtype=np.float32)

    return kalman

def smoothAverage(trajectory, r):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=r)
    
    return smoothed_trajectory

def smoothGauss(trajectory, r, sigma=None):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:,i] = gaussianSmoothing(trajectory[:,i], radius=r, sigma=sigma)
    
    return smoothed_trajectory

def smoothSavgol(trajectory, window=11, poly=3):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = savgol_filter(trajectory[:, i], window_length=window, polyorder=poly, mode='nearest')

    return smoothed_trajectory

def fixBorder(frame, zoom):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, zoom)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def lowpass(x, cutoff=0.1):
    b, a = butter(2, cutoff, btype='low')
    return filtfilt(b, a, x, axis=0)

def calculateM(dx, dy, da):
    m = np.zeros((2,3), np.float32)

    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    return m

def plotTrajectory(trajectory, smooth_trajectory, plotname):
    labels = ["dx", "dy", "da"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for i in range(3):
        axes[i].plot(trajectory[:, i], label="Original", linewidth=1)
        axes[i].plot(smooth_trajectory[:, i], label="Smoothed", linewidth=2)

        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Value")

        axes[i].grid(True)
        axes[i].legend()

        plt.tight_layout()

    plt.savefig(str(plotname + ".png"), dpi=300, bbox_inches="tight")

def setupORB(config):
    orb_nFeatures = config.orb.nfeatures
    orb_scaleFactor = config.orb.scaleFactor
    orb_nlevels = config.orb.nlevels
    orb_edgeThreshold = config.orb.edgeThreshold
    orb_firstLevel = config.orb.firstLevel
    orb_WTA_K = config.orb.WTA_K
    orb_patchSize = config.orb.patchSize
    orb_fastThreshold = config.orb.fastThreshold

    orb = cv2.ORB_create(
        nfeatures=orb_nFeatures, scaleFactor=orb_scaleFactor, nlevels=orb_nlevels,
        edgeThreshold=orb_edgeThreshold, firstLevel=orb_firstLevel, WTA_K=orb_WTA_K,
        patchSize=orb_patchSize, fastThreshold=orb_fastThreshold
    )
    return orb

def setupShiTomasi(config):
    pass

def setupFast(config):
    threshold = config.fast.threshold
    return cv2.FastFeatureDetector_create(threshold=threshold)