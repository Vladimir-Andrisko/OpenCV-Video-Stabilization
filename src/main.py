import cv2

INPUT_VIDEO = "videoplayback.mp4"
OUTPUT_VIDEO = "output.mp4"

def main():
    
    capture = cv2.VideoCapture(INPUT_VIDEO)

    if not capture.isOpened():
        print("Wrong input file name")
        return

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"FPS: {fps}")
    print(f"Resolution: {width}x{height}")

    frame_period = int(1000/fps)

    writer = cv2.VideoWriter("output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    if not writer.isOpened():
        print("Can not create output video file!")
        capture.release()
        return

    while True:
        ret, frame = capture.read()

        if not ret:
            print("End of video.")
            break

        stabilized_frame = frame
        cv2.imshow("Original Video", stabilized_frame)
        writer.write(stabilized_frame)

        key = cv2.waitKey(frame_period) & 0xFF

        if key == ord('q'):
            print("Video interrupted by user.")
            break
    
    capture.release()
    writer.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()