import cv2
import numpy as np

INPUT_VIDEO = "unstable_videos/videoplayback.mp4"
OUTPUT_VIDEO = "output.mp4"
SMOOTHING_RADIUS = 3

def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size)/window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed

def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)

    for i in range(3):
        smoothed_trajectory[:,i] = movingAverage(trajectory[:,i], radius=SMOOTHING_RADIUS)
    
    return smoothed_trajectory

def fixBorder(frame):
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1]/2, s[0]/2), 0, 1.05)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame

def main():
    
    cap = cv2.VideoCapture(INPUT_VIDEO)

    if not cap.isOpened():
        print("Wrong input file name")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    frame_period = int(1000/fps)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = cv2.VideoWriter("output.avi", fourcc, fps, (width, height))

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

    for i in range(n_frames-2):
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=400, qualityLevel=0.1, minDistance=20, blockSize=9)

        success, curr = cap.read()
        if not success:
            break
        
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

        assert curr_pts.shape == prev_pts.shape

        idx = np.where(status==1)[0]
        prev_pts = prev_pts[idx]
        curr_pts = curr_pts[idx]

        m, inliers = cv2.estimateAffinePartial2D(prev_pts, curr_pts, method=cv2.RANSAC)

        dx = m[0, 2]
        dy = m[1, 2]

        dangle = np.arctan2(m[1, 0], m[0, 0])

        # print(f"dx: {dx}, dy: {dy}, angle: {dangle}")
        transforms[i] = [float(dx), float(dy), float(dangle)]
        prev_gray = curr_gray

        print("Frame: " + str(i) +  "/" + str(n_frames) + " -  Tracked points : " + str(len(prev_pts)))
        trajectory = np.cumsum(transforms, axis=0)
        smooth_trajectory = smooth(trajectory)

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
        frame_stabilized = fixBorder(frame_stabilized)
 
        # Write the frame to the file
        # frame_out = cv2.hconcat([frame, frame_stabilized])

        # if frame_out.shape[1] > 1920:
        #     frame_out = cv2.resize(frame_out, (frame_out.shape[1]//2, frame_out.shape[0]//2));

        print("writer expects:", (width, height))
        print("actual frame:", frame_stabilized.shape)
        print(f"Written frame: {i}, out of {n_frames}")
        writer.write(frame_stabilized)

    
    cap.release()
    writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()