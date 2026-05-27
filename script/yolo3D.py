import cv2
import numpy as np
from src.detector.detector3d import Detector3D


def nothing(x):
    pass


def main():
    cap = cv2.VideoCapture(0)

    window_name = "YOLO Pose Estimation"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    detector = Detector3D(model_path="yolo11n-pose.pt", device="mps", dis_mode=1, conf_thres=0.5)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        output = detector.draw()

        if output is None:
            output = frame

        fps = 1000.0 / (cv2.getTickCount() - getattr(main, "_tick", cv2.getTickCount())) * cv2.getTickFrequency() if hasattr(main, "_tick") else 0
        main._tick = cv2.getTickCount()

        person_count = len(detections) if detections else 0
        cv2.putText(output, f"Persons: {person_count} | FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow(window_name, output)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
