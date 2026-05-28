import cv2
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python.vision import RunningMode
import os


LANDMARK_INDICES = [1, 33, 263, 61, 291, 199]

DEFAULT_MODEL = "models/face_landmarker.task"

def _euler_to_rvec(roll, pitch, yaw):
    r, p, y = np.radians(roll), np.radians(pitch), np.radians(yaw)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    rvec, _ = cv2.Rodrigues(R)
    return rvec

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


class FacePoseDetector:
    def __init__(self, dis_mode=1, max_faces=1,
                 min_detection_confidence=0.5,
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
            running_mode=RunningMode.IMAGE,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(opts)

        self.img_raw = None
        self.img_res = None
        self.res = None
        self.camera_matrix = None
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        # --- 新增：标准 3D 人脸关键点模型 (X右, Y下, Z前) 单位: mm ---
        # 这里的点位顺序必须与 LANDMARK_INDICES = [1, 33, 263, 61, 291, 199] 一一对应
        self.model_points = np.array([
            [0.0, 0.0, 0.0],             # 1: 鼻尖 (将坐标原点牢牢钉在鼻尖)
            [-30.0, -30.0, -20.0],       # 33: 左眼角 (画面左侧)
            [30.0, -30.0, -20.0],        # 263: 右眼角 (画面右侧)
            [-15.0, 30.0, -10.0],        # 61: 左嘴角
            [15.0, 30.0, -10.0],         # 291: 右嘴角
            [0.0, 60.0, -15.0]           # 199: 下巴
        ], dtype=np.float64)

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
        result = self.landmarker.detect(mp_image)

        self.res = []

        if result is None or not result.face_landmarks:
            return self.res

        transforms = (result.facial_transformation_matrixes
                      if result.facial_transformation_matrixes else [])

        for i, face_landmarks in enumerate(result.face_landmarks):
            if i >= len(transforms):
                continue

            T = np.array(transforms[i], dtype=np.float64)

            # 1. 提取 MediaPipe 原生的旋转矩阵和平移向量
            R_mp = T[:3, :3]
            t_mp = T[:3, 3]

            # 2. 坐标系转换 (MediaPipe OpenGL -> OpenCV 标准)
            # S 矩阵用于反转 Y 轴和 Z 轴
            S = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
            # R_cv = S @ R_mp @ S 既转换了相机坐标系，也翻转了物体局部坐标系
            # 保证绘制出的 Y 轴指向下巴，Z 轴指向脑后
            R_cv = S @ R_mp @ S
            rvec, _ = cv2.Rodrigues(R_cv)
            roll, pitch, yaw = _mat_to_euler(R_cv)

            # 3. 平移解耦：利用 2D 鼻尖像素坐标反投影，将原点绝对锚定在鼻尖
            # 鼻尖在 MediaPipe 中的索引为 1
            nose_x = face_landmarks[1].x * w
            nose_y = face_landmarks[1].y * h
            
            # MediaPipe 原生 t_mp[2] 是头部中心的负深度，取反得到正深度
            # 鼻尖比头部中心更靠近相机，减去 20.0mm 作为深度补偿
            depth = -t_mp[2] - 20.0 
            
            # 针孔相机反投影公式，利用你自定义的 camera_matrix
            cx, cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]
            fx, fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            tx = (nose_x - cx) * depth / fx
            ty = (nose_y - cy) * depth / fy
            tz = depth
            
            tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)

            # 记录用于后续绘制的 2D 关键点
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
                "tx": float(tx),
                "ty": float(ty),
                "tz": float(tz),
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
