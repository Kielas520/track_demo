import cv2
import numpy as np
from ultralytics import YOLO


SKELETON_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
]

KEYPOINT_COLORS = [
    (0, 100, 255), (0, 100, 255), (0, 100, 255), (0, 100, 255), (0, 100, 255),
    (255, 0, 0), (0, 0, 255), (255, 0, 0), (0, 0, 255),
    (255, 0, 0), (0, 0, 255),
    (0, 255, 0), (255, 0, 255), (0, 255, 0), (255, 0, 255),
    (0, 255, 0), (255, 0, 255),
]

EDGE_COLORS = [
    (0, 255, 0), (0, 255, 0), (255, 128, 0), (255, 0, 0), (255, 128, 0), (255, 0, 0),
    (0, 255, 255), (0, 255, 255),
    (0, 255, 128), (0, 255, 128), (255, 0, 255), (255, 0, 255),
    (128, 128, 255), (128, 128, 255), (128, 128, 255), (128, 128, 255),
]


class Detector3D:
    def __init__(self, model_path="yolo11n-pose.pt", device="mps", dis_mode=1, conf_thres=0.5):
        """
        初始化YOLO Pose检测器
        :param model_path: YOLO Pose模型路径
        :param device: 推理设备 (mps, cpu, cuda)
        :param dis_mode: 显示模式 (0->不显示, 1->显示)
        :param conf_thres: 置信度阈值
        """
        self.model = YOLO(model_path)
        self.device = device
        self.dis_mode = dis_mode
        self.conf_thres = conf_thres

        self.img_raw = None
        self.img_res = None
        self.res = None

    def detect(self, frame):
        """
        姿态检测：输入frame，返回每个人的关键点和bbox
        :param frame: 输入视频帧
        :return: list[dict], 每项包含 box, conf, keypoints, kpt_confs
        """
        if frame is None:
            return None

        self.img_raw = frame.copy()

        results = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,
            stream=False,
            conf=self.conf_thres,
        )

        detections = []
        r = results[0]

        if r.keypoints is not None and r.keypoints.data.shape[0] > 0:
            kpts_data = r.keypoints.data.cpu().numpy()
            boxes_data = r.boxes.xyxy.cpu().numpy() if r.boxes is not None else None
            confs_data = r.boxes.conf.cpu().numpy() if r.boxes is not None else None

            for i in range(len(kpts_data)):
                kpt = kpts_data[i]
                box = boxes_data[i].tolist() if boxes_data is not None else None
                conf = confs_data[i].item() if confs_data is not None else None

                keypoints = []
                for j in range(len(kpt)):
                    kx, ky, kc = kpt[j]
                    keypoints.append({
                        "x": float(kx),
                        "y": float(ky),
                        "conf": float(kc),
                    })

                detections.append({
                    "box": box,
                    "conf": float(conf) if conf is not None else None,
                    "keypoints": keypoints,
                })

        self.res = detections
        return self.res

    def draw(self, res=None):
        """
        绘制骨骼关键点
        :param res: 检测结果
        :return: 绘制后的图像 或 None
        """
        if res is None:
            res = self.res

        if res is None or self.img_raw is None:
            return None

        if self.dis_mode == 0:
            return None

        self.img_res = self.img_raw.copy()

        for person in res:
            kpts = person["keypoints"]
            box = person.get("box")
            conf = person.get("conf")

            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(self.img_res, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if conf is not None:
                    cv2.putText(self.img_res, f"{conf:.2f}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for a, b in SKELETON_EDGES:
                if a >= len(kpts) or b >= len(kpts):
                    continue
                ka, kb = kpts[a], kpts[b]
                if ka["conf"] < 0.5 or kb["conf"] < 0.5:
                    continue
                x1, y1 = int(ka["x"]), int(ka["y"])
                x2, y2 = int(kb["x"]), int(kb["y"])
                edge_idx = SKELETON_EDGES.index((a, b))
                color = EDGE_COLORS[edge_idx]
                cv2.line(self.img_res, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

            for idx, kp in enumerate(kpts):
                if kp["conf"] < 0.5:
                    continue
                x, y = int(kp["x"]), int(kp["y"])
                color = KEYPOINT_COLORS[idx]
                cv2.circle(self.img_res, (x, y), 4, color, -1, cv2.LINE_AA)

        return self.img_res
