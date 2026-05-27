import copy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np

class Detector2D:
    def __init__(self, cls_id, model_path="yolo26n.pt", device="mps", dis_mode=1):
        """
        初始化检测器
        :param model_path: YOLO模型路径
        :param device: 推理设备 (mps, cpu, cuda)
        :param dis_mode: 显示模式 (0->不显示结果, 1->显示结果)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.dis_mode = dis_mode

        self.id = cls_id

        self.img_raw = None
        self.img_res = None
        self.res = None

    def detect(self, frame):
        """
        检测函数：输入frame，进行推理，保存原始图像和结果
        :param frame: 输入的视频帧 (numpy array)
        :return: res (检测到的对象列表信息)
        """
        if frame is None:
            return None

        self.img_raw = frame.copy()

        results = self.model.predict(
            source=frame,
            classes=[self.id],
            device=self.device,
            verbose=False,
            stream=False
        )

        detected_objects = []
        r = results[0]

        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes:
                conf = box.conf[0].item()
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                detected_objects.append({
                    "box": [x1, y1, x2, y2],
                    "conf": conf,
                    "cls_id": cls_id,
                    "cls_name": self.model.names[cls_id]
                })

        self.res = detected_objects
        return self.res

    def draw(self, res=None):
        """
        绘制函数：根据 dis_mode 决定是否绘制，并将结果绘制到 img_res
        :param res: 检测结果列表 (如果为None，则使用 self.res)
        :return: img_res (绘制后的图像) 或 None
        """
        if res is None:
            res = self.res

        if res is None or self.img_raw is None:
            return None

        if self.dis_mode == 0:
            return None

        self.img_res = self.img_raw.copy()

        annotator = Annotator(self.img_res, line_width=2, font_size=10)

        for obj in res:
            box = obj["box"]
            conf = obj["conf"]
            name = obj["cls_name"]

            label = f"{name} {conf:.2f}"

            annotator.box_label(box, label, color=(255, 0, 0))

        return self.img_res
