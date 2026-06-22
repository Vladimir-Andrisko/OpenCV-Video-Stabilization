import cv2
import numpy as np
from . import videoStabilizationHelper
PRINT_DELAY = 500

def simpleStabilize(input, output, config, debug=False):
    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Wrong input file name")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_period = int(1000/fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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

    ## config
    max_corners = config.featureDetection.max_corners 
    quality_level = config.featureDetection.quality_level
    min_distance = config.featureDetection.min_distance
    block_size = config.featureDetection.block_size
    use_harris_detector = config.featureDetection.use_harris_detector
    radius = config.filter.moving_average_radius
    zoom = config.filter.zoom

    for i in range(n_frames-2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, 
            maxCorners=max_corners, 
            qualityLevel=quality_level, 
            minDistance=min_distance, 
            blockSize=block_size, 
            useHarrisDetector=use_harris_detector
        )

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

        dx = m[0, 2]
        dy = m[1, 2]
        dangle = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [float(dx), float(dy), float(dangle)]
        prev_gray = curr_gray

        if i % PRINT_DELAY == 0:
            print("Frame: " + str(i) +  "/" + str(n_frames) + " - Tracked points : " + str(len(prev_pts)))

        trajectory = np.cumsum(transforms, axis=0)
        smooth_trajectory = smooth(trajectory, radius)

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

        m = np.zeros((2,3), np.float32)

        m[0,0] = np.cos(da)
        m[0,1] = -np.sin(da)
        m[1,0] = np.sin(da)
        m[1,1] = np.cos(da)
        m[0,2] = dx
        m[1,2] = dy

        frame_stabilized = cv2.warpAffine(frame, m, (width, height))
        frame_stabilized = fixBorder(frame_stabilized, zoom)
 
        if i % PRINT_DELAY == 0:
            print(f"Written frame: {i}/{n_frames}")

        if debug:
            for (x, y) in debug_points[i]:
                cv2.circle(frame_stabilized, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        writer.write(frame_stabilized)
    
    cap.release()
    writer.release()

    cv2.destroyAllWindows()


def fastStabilize(input, output, config, debug=False):
    cap = cv2.VideoCapture(input)

    if not cap.isOpened():
        print("Wrong input file name")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_period = int(1000/fps)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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

    transforms = np.zeros((n_frames-1, 3), np.float32)
    debug_points = []

    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    threshold = config.fast.threshold
    radius = config.filter.moving_average_radius
    zoom = config.filter.zoom

    fast = cv2.FastFeatureDetector_create(threshold=threshold)

    for i in range(n_frames-2):
        kp_prev = fast.detect(prev_gray, None)

        success, curr = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        
        prev_pts = cv2.KeyPoint_convert(kp_prev)
        prev_pts = prev_pts.reshape(-1, 1, 2).astype(np.float32)

        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        assert prev_pts.shape == curr_pts.shape

        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        if debug:
            debug_points.append([])
            for point in prev_pts:
                x, y = point.ravel().astype(int)
                debug_points[i].append((x, y))

        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])

        transforms[i] = [float(dx), float(dy), float(da)]
        prev_gray = curr_gray

        trajectory = np.cumsum(transforms, axis=0)
        smooth_trajectory = videoStabilizationHelper.smooth(trajectory, radius)

        difference = smooth_trajectory - trajectory
        transform_smooth = transforms + difference
        if i % int(fps+1) == 0:
            print("Frame: " + str(i) +  "/" + str(n_frames) + " - Tracked points : " + str(len(kp_prev)))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    for i in range(n_frames-2):
        success, frame = cap.read()
        if not success:
            break

        dx = transform_smooth[i, 0]
        dy = transform_smooth[i, 1]
        da = transform_smooth[i, 2]

        m = videoStabilizationHelper.calculateM(dx, dy, da)

        frame_stabilized = cv2.warpAffine(frame, m, (width, height), flags=cv2.INTER_LANCZOS4)
        frame_stabilized = videoStabilizationHelper.fixBorder(frame_stabilized, zoom)
 
        if i % PRINT_DELAY == 0:
            print(f"Written frame: {i}/{n_frames}")

        if debug:
            for (x, y) in debug_points[i]:
                cv2.circle(frame_stabilized, (x, y), 3, (0,255,0), -1)

        writer.write(frame_stabilized)
    
    cap.release()
    writer.release()

    cv2.destroyAllWindows()