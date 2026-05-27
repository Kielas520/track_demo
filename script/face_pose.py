import cv2
from src.detector.face_pose import FacePoseDetector
from src.Kalmanfilter.Kalmanfilter6D import FaceKalmanFilter6D


def main():
    cap = cv2.VideoCapture(0)

    window_name = "Face 6DOF Pose + KF"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = FacePoseDetector(dis_mode=1, max_faces=1)
    kf = FaceKalmanFilter6D()
    kf_initialized = False

    fyaw, fpitch, froll = 0.0, 0.0, 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        if detections and len(detections) > 0:
            face = detections[0]
            tx, ty, tz = face["tx"], face["ty"], face["tz"]
            roll, pitch, yaw = face["roll"], face["pitch"], face["yaw"]

            if not kf_initialized:
                kf.init_state(tx, ty, tz, roll, pitch, yaw)
                kf_initialized = True

            dt = 0.033
            kf.predict(dt)
            kf.update([tx, ty, tz, roll, pitch, yaw])

            fx, fy, fz = kf.get_filtered_pos()
            froll, fpitch, fyaw = kf.get_filtered_rot()

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        output = detector.draw()

        if output is None:
            output = frame

        if kf_initialized:
            cv2.putText(output, f"KF Y:{fyaw:6.1f} P:{fpitch:6.1f} R:{froll:6.1f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow(window_name, output)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
