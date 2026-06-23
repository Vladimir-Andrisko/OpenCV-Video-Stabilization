import cv2
import numpy as np
from . import videoStabilizationHelper

PRINT_DELAY = 500

def stabilize(input, output, config, feature_detection, filter, debug=False):
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

    ## Filter
    moovingRadius = config.filter.moving_average_radius
    gaussRadius = config.filter.gauss_radius
    gaussSigma = config.filter.gauss_sigma
    savgolWindow = config.filter.savgol_window
    savgolPoly = config.filter.savgol_poly

    zoom = config.config.zoom
    max_translation = config.config.max_translation
    max_rotation = config.config.max_rotation

    fast = videoStabilizationHelper.setupFast(config)
    orb = videoStabilizationHelper.setupORB(config)

    for i in range(n_frames-2):
        if feature_detection == config.feature_type.ShiTomasi:
            prev_pts = cv2.goodFeaturesToTrack(prev_gray, 
                maxCorners=max_corners, qualityLevel=quality_level, 
                minDistance=min_distance, blockSize=block_size
            )
        elif feature_detection == config.feature_type.ORB:
            kp_prev = orb.detect(prev_gray, None)
            prev_pts = cv2.KeyPoint_convert(kp_prev)
            prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)
        elif feature_detection == config.feature_type.FAST:
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
        if curr_pts is None:
            continue

        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if debug:
            debug_points.append([])
            for pt in curr_pts:
                x, y = pt.ravel().astype(int)
                debug_points[i].append((x, y))

        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [float(dx), float(dy), float(da)]
        prev_gray = curr_gray

        if i % int(PRINT_DELAY) == 0:
            print("Frame: " + str(i) +  "/" + str(n_frames) + " - Tracked points : " + str(len(prev_pts)))

        trajectory = np.cumsum(transforms, axis=0)

        xy = trajectory[:, :2]   # dx, dy
        rot = trajectory[:, 2:]  # da

        if filter == config.filter_type.MoovingAverage:
            smooth_xy = videoStabilizationHelper.smoothAverage(xy, moovingRadius)
            smooth_angle = videoStabilizationHelper.smoothAverage(rot, moovingRadius)
        elif filter == config.filter_type.Gauss:
            smooth_xy = videoStabilizationHelper.smoothGauss(xy, gaussRadius, gaussSigma)
            smooth_angle = videoStabilizationHelper.smoothGauss(rot, gaussRadius, gaussSigma)
        elif filter == config.filter_type.Savgol:
            smooth_xy = videoStabilizationHelper.smoothSavgol(xy, savgolWindow, savgolPoly)
            smooth_angle = videoStabilizationHelper.smoothSavgol(rot, savgolWindow, savgolPoly)
        elif filter == config.filter_type.LowPass:
            smooth_xy = videoStabilizationHelper.lowpass(xy)
            smooth_angle = videoStabilizationHelper.lowpass(rot)
        else:
            print("Wrong filter input!")
            return

        smooth_trajectory = np.hstack([smooth_xy, smooth_angle])

        difference = smooth_trajectory - trajectory

        difference[:,0] = np.clip(difference[:,0], -max_translation, max_translation)
        difference[:,1] = np.clip(difference[:,1], -max_translation, max_translation)
        difference[:,2] = np.clip(difference[:,2], -max_rotation, max_rotation)

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

        frame_stabilized = cv2.warpAffine(frame, m, (width, height), borderMode=cv2.BORDER_REFLECT)

        gray = cv2.cvtColor(frame_stabilized, cv2.COLOR_BGR2GRAY)
        mask = (gray == 0).astype(np.uint8) * 255
        frame_stabilized = cv2.inpaint(frame_stabilized, mask, 3, cv2.INPAINT_TELEA)
        frame_stabilized = videoStabilizationHelper.fixBorder(frame_stabilized, zoom)
 
        if i % PRINT_DELAY == 0:
            print(f"Written frame: {i}/{n_frames}")

        if debug:
            for (x, y) in debug_points[i]:
                cv2.circle(frame_stabilized, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        writer.write(frame_stabilized)

    output = output.split("/")[1]
    output = output.split(".")[0]
    output = "plots/" + output
    videoStabilizationHelper.plotTrajectory(trajectory, smooth_trajectory, output)
    
    cap.release()
    writer.release()

    cv2.destroyAllWindows()