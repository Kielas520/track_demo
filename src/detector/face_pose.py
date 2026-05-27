import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import RunningMode
import os


LANDMARK_INDICES = [1, 33, 263, 61, 291, 199]

DEFAULT_MODEL = "models/face_landmarker.task"


def _mat_to_euler(R):
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    if sy < 1e-6:
        yaw = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    else:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    return roll, pitch, yaw


def _euler_to_rvec(roll, pitch, yaw):
    r, p, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec


class FacePoseDetector:
    def __init__(self, dis_mode=1, max_faces=1,
                 min_detection_confidence=0.5, min_tracking_confidence=0.5,
                 model_path=None):
        self.dis_mode = dis_mode

        if model_path is None:
            model_path = DEFAULT_MODEL
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    os.path.dirname(__file__), "..", "..", DEFAULT_MODEL)

        base_opts = base_options.BaseOptions(model_asset_path=model_path)
        opts = vision.FaceLandmarkerOptions(
            base_options=base_opts,
            running_mode=RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_presence_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

        self.img_raw = None
        self.img_res = None
        self.res = None
        self._frame_counter = 0
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    def detect(self, frame):
        if frame is None:
            return None

        self.img_raw = frame.copy()
        h, w = frame.shape[:2]

        if self.camera_matrix is None:
            self.camera_matrix = np.array([
                [w, 0, w / 2],
                [0, w, h / 2],
                [0, 0, 1],
            ], dtype=np.float64)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        self._frame_counter += 1
        result = self.landmarker.detect_for_video(mp_image, self._frame_counter)

        self.res = []

        if result is None or not result.face_landmarks:
            return self.res

        transforms = (result.facial_transformation_matrixes
                      if result.facial_transformation_matrixes else [])

        for i, face_landmarks in enumerate(result.face_landmarks):
            if i >= len(transforms):
                continue

            T = np.array(transforms[i], dtype=np.float64)
            R_mat = T[:3, :3]
            tvec = T[:3, 3].reshape(3, 1)
            rvec, _ = cv2.Rodrigues(R_mat)

            roll, pitch, yaw = _mat_to_euler(R_mat)

            image_points = np.array([
                [face_landmarks[idx].x * w, face_landmarks[idx].y * h]
                for idx in LANDMARK_INDICES
            ], dtype=np.float64)

            self.res.append({
                "rvec": rvec,
                "tvec": tvec,
                "yaw": float(np.degrees(yaw)),
                "pitch": float(np.degrees(pitch)),
                "roll": float(np.degrees(roll)),
                "image_points": image_points,
                "tx": float(tvec[0][0]),
                "ty": float(tvec[1][0]),
                "tz": float(tvec[2][0]),
            })

        return self.res

    def draw(self, res=None):
        if res is None:
            res = self.res

        if res is None or self.img_raw is None:
            return None

        if self.dis_mode == 0:
            return None

        self.img_res = self.img_raw.copy()

        for face in res:
            cv2.drawFrameAxes(
                self.img_res, self.camera_matrix, self.dist_coeffs,
                face["rvec"], face["tvec"], 50,
            )

            for pt in face["image_points"]:
                cv2.circle(self.img_res, (int(pt[0]), int(pt[1])),
                           3, (0, 255, 0), -1)

            cv2.putText(self.img_res, f"Y:{face['yaw']:6.1f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(self.img_res, f"P:{face['pitch']:6.1f}",
                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(self.img_res, f"R:{face['roll']:6.1f}",
                        (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return self.img_res
