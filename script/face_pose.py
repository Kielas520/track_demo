import cv2
import numpy as np
from src.detector.face_pose import FacePoseDetector, _euler_to_rvec
from src.Kalmanfilter.Kalmanfilter6D import FaceKalmanFilter6D


def _draw_kf_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=80):
    axis_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ])
    pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = pts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    cv2.line(img, origin, tuple(pts[1]), (0, 255, 255), 3)  # X: yellow
    cv2.line(img, origin, tuple(pts[2]), (0, 255, 255), 3)  # Y: yellow
    cv2.line(img, origin, tuple(pts[3]), (0, 255, 255), 3)  # Z: yellow
    cv2.circle(img, origin, 5, (0, 255, 255), -1)


def main():
    cap = cv2.VideoCapture(0)

    window_name = "Face 6DOF Pose + KF"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = FacePoseDetector(dis_mode=1, max_faces=1)
    kf = FaceKalmanFilter6D()
    kf_initialized = False

    fx, fy, fz = 0.0, 0.0, 0.0
    froll, fpitch, fyaw = 0.0, 0.0, 0.0

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
            output = frame.copy()

        if kf_initialized and detections and len(detections) > 0:
            rvec_kf = _euler_to_rvec(froll, fpitch, fyaw)
            tvec_kf = np.array([[fx], [fy], [fz]], dtype=np.float64)
            _draw_kf_axis(output, detector.camera_matrix,
                          detector.dist_coeffs, rvec_kf, tvec_kf, length=80)

            cv2.putText(output, f"KF Y:{fyaw:6.1f} P:{fpitch:6.1f} R:{froll:6.1f}",
                        (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow(window_name, output)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
