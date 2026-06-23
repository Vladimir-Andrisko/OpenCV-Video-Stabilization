import cv2
import numpy as np
from . import videoStabilizationHelper
from enum import Enum
PRINT_DELAY = 500

class featureDetection(Enum):
    ShiTomasi = 0,
    FAST = 1,
    ORB = 2

class Filter(Enum):
    MoovingAverage = 0,
    Gauss = 1,
    Savgol = 2,
    Kalman = 3

def stabilize(input, output, config, debug=False, feature_detection=featureDetection.ShiTomasi, filter=Filter.MoovingAverage):
    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Wrong input file name")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_period = int(1000/fps)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(output, fourcc, fps, (width, height))

    if not writer.isOpened():
        print("Can not create output video file!")
        cap.release()
        return
    
    ret, prev = cap.read()
    if not ret:
        cap.release()
        writer.release()
        return

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)
    transforms = np.zeros((n_frames-1, 3), np.float32)
    debug_points = []

    ## ShiTomasi
    max_corners = config.featureDetection.max_corners 
    quality_level = config.featureDetection.quality_level
    min_distance = config.featureDetection.min_distance
    block_size = config.featureDetection.block_size

    ## FAST
    threshold = config.fast.threshold

    ## ORB
    orb_nFeatures = config.orb.nfeatures
    orb_scaleFactor = config.orb.scaleFactor
    orb_nlevels = config.orb.nlevels
    orb_edgeThreshold = config.orb.edgeThreshold
    orb_firstLevel = config.orb.firstLevel
    orb_WTA_K = config.orb.WTA_K
    orb_patchSize = config.orb.patchSize
    orb_fastThreshold = config.orb.fastThreshold

    ## Filter
    moovingRadius = config.filter.moving_average_radius
    gaussRadius = config.filter.gauss_radius
    gaussSigma = config.filter.gauss_sigma
    savgolWindow = config.filter.savgol_window
    savgolPoly = config.filter.savgol_poly

    zoom = config.filter.zoom

    fast = cv2.FastFeatureDetector_create(threshold=threshold)
    orb = cv2.ORB_create(
        nfeatures=orb_nFeatures, scaleFactor=orb_scaleFactor, nlevels=orb_nlevels,
        edgeThreshold=orb_edgeThreshold, firstLevel=orb_firstLevel, WTA_K=orb_WTA_K,
        patchSize=orb_patchSize, fastThreshold=orb_fastThreshold
    )

    kalman = videoStabilizationHelper.create_kalman()
    # transform_smooth = np.zeros((n_frames, 3), dtype=np.float32)

    for i in range(n_frames-2):
        if feature_detection == featureDetection.ShiTomasi:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, 
                maxCorners=max_corners, qualityLevel=quality_level, 
                minDistance=min_distance, blockSize=block_size
            )
        elif feature_detection == featureDetection.ORB:
            kp_prev = orb.detect(prev_gray, None)
            prev_pts = cv2.KeyPoint_convert(kp_prev)
            prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        elif feature_detection == featureDetection.FAST:
            kp_prev = fast.detect(prev_gray, None)
            prev_pts = cv2.KeyPoint_convert(kp_prev)
            prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        else:
            print("Wrong feature detection")
            return

        success, curr = cap.read()
        if not success:
            break
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        assert curr_pts.shape == prev_pts.shape

        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if debug:
            debug_points.append([])
            for pt in curr_pts:
                x, y = pt.ravel().astype(int)
                debug_points[i].append((x, y))

        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)
        # m, inliers = cv2.findHomography(prev_pts, curr_pts, method=cv2.RANSAC)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [float(dx), float(dy), float(da)]
        prev_gray = curr_gray

        if i % int(fps+1) == 0:
            print("Frame: " + str(i) +  "/" + str(n_frames) + " - Tracked points : " + str(len(prev_pts)))

        trajectory = np.cumsum(transforms, axis=0)

        if filter == Filter.MoovingAverage:
            smooth_trajectory = videoStabilizationHelper.smoothAverage(trajectory, moovingRadius)
        elif filter == Filter.Gauss:
            smooth_trajectory = videoStabilizationHelper.smoothGauss(trajectory, gaussRadius, gaussSigma)
        elif filter == Filter.Savgol:
            smooth_trajectory = videoStabilizationHelper.smoothSavgol(trajectory, savgolWindow, savgolPoly)
        elif filter == Filter.Kalman:
            pass
        else:
            print("Wrong filter input!")
            return

        difference = smooth_trajectory - trajectory
        transform_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames-2):
        success, frame = cap.read()
        if not success:
            break

        dx = transform_smooth[i, 0]
        dy = transform_smooth[i, 1]
        da = transform_smooth[i, 2]

        m = videoStabilizationHelper.calculateM(dx, dy, da)

        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized = videoStabilizationHelper.fixBorder(frame_stabilized, zoom)
 
        if i % PRINT_DELAY == 0:
            print(f"Written frame: {i}/{n_frames}")

        if debug:
            for (x, y) in debug_points[i]:
                cv2.circle(frame_stabilized, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        writer.write(frame_stabilized)

    videoStabilizationHelper.plotTrajectory(trajectory, smooth_trajectory)
    
    cap.release()
    writer.release()

    cv2.destroyAllWindows()