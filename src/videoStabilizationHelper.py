import numpy as np
import cv2

def movingAverage(curve, radius):
    window_size = 2*radius + 1

    f = np.ones(window_size)/window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]

    return curve_smoothed

def smooth(trajectory, r):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=r)
    
    return smoothed_trajectory

def fixBorder(frame, zoom):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, zoom)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def calculateM(dx, dy, da):
    m = np.zeros((2,3), np.float32)

    m[0,0] = np.cos(da)
    m[0,1] = -np.sin(da)
    m[1,0] = np.sin(da)
    m[1,1] = np.cos(da)
    m[0,2] = dx
    m[1,2] = dy

    return m