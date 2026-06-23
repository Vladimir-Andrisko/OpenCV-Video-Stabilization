import cv2
import numpy as np
from . import videoStabilizationHelper
from enum import Enum
import cvxpy as cp
import matplotlib.pyplot as plt

PRINT_DELAY = 500

class featureDetection(Enum):
    ShiTomasi = 0,
    FAST = 1,
    ORB = 2

def stabilize(input, output, config, debug=False, feature_detection=featureDetection.ShiTomasi):
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
    transforms = [[], [], []]
    debug_points = []

    ## ShiTomasi
    max_corners = config.featureDetection.max_corners 
    quality_level = config.featureDetection.quality_level
    min_distance = config.featureDetection.min_distance
    block_size = config.featureDetection.block_size

    zoom = config.config.zoom
    max_translation = config.config.max_translation
    max_rotation = config.config.max_rotation

    fast = videoStabilizationHelper.setupFast(config)
    orb = videoStabilizationHelper.setupORB(config)

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

        transforms[0].append(dx)
        transforms[1].append(dy)
        transforms[2].append(da) 
        prev_gray = curr_gray

        if i % int(fps+1) == 0:
            print("Frame: " + str(i) +  "/" + str(n_frames) + " - Tracked points : " + str(len(prev_pts)))

    lbd1 = 10000
    lbd2 = 1000

    trajectory = np.cumsum(np.array(transforms), axis=1)

    # Optimal values:
    fx = cp.Variable(np.size(trajectory[0]))
    fy = cp.Variable(np.size(trajectory[1]))
    fa = cp.Variable(np.size(trajectory[2]))

    constraints = [cp.abs(fx-trajectory[0]) <= max_translation, 
                   cp.abs(fy-trajectory[1]) <= max_translation,
                   cp.abs(fa-trajectory[2]) <= max_rotation]

    obj = 0																																																																																								
    for i in range(np.size(trajectory[0])):
        obj = (cp.sum_squares(fx - trajectory[0]) + cp.sum_squares(fy - trajectory[1]) + cp.sum_squares(fa - trajectory[2]))

    # DP1
    for i in range(np.size(trajectory[0])-1):
        obj += lbd1 * (cp.norm1(fx[1:] - fx[:-1]) + cp.norm1(fy[1:] - fy[:-1]) + cp.norm1(fa[1:] - fa[:-1]))

    # DP2
    for i in range(np.size(trajectory[0])-2):
        obj += lbd2 * (cp.norm1(fx[2:] - 2*fx[1:-1] + fx[:-2]) + cp.norm1(fy[2:] - 2*fy[1:-1] + fy[:-2]) + cp.norm1(fa[2:] - 2*fa[1:-1] + fa[:-2]))

    prob = cp.Problem(cp.Minimize(obj), constraints)
    print("Startet solving the optimization problem")
    prob.solve(solver=cp.OSQP)
    print("Optimization problem solved")

    smoothTrajectory = np.array([fx.value, fy.value, fa.value])
    difference = trajectory - smoothTrajectory
    transform_smooth = transforms - difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(n_frames-2):
        success, frame = cap.read()
        if not success:
            break

        dx = transform_smooth[0, i]
        dy = transform_smooth[1, i]
        da = transform_smooth[2, i]

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

    labels = ["dx", "dy", "da"]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    for i in range(3):
        axes[i].plot(trajectory[i], label="Original", linewidth=1)
        axes[i].plot(smoothTrajectory[i], label="Smoothed", linewidth=2)

        axes[i].set_title(labels[i])
        axes[i].set_xlabel("Frame")
        axes[i].set_ylabel("Value")

        axes[i].grid(True)
        axes[i].legend()

        plt.tight_layout()

    plt.savefig("optimal_plot.png", dpi=300, bbox_inches="tight")

    cap.release()
    writer.release()

    cv2.destroyAllWindows()