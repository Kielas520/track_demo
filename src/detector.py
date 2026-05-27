import copy
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
import cv2
import numpy as np

class Detector:
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
        
        # COCO数据集中 'bottle' 的类别 ID 为 39
        self.id = cls_id
        
        # 属性初始化
        self.img_raw = None   # 原始输入帧
        self.img_res = None   # 绘制后的结果帧
        self.res = None       # 检测结果 (boxes, confs, etc.)

    def detect(self, frame):
        """
        检测函数：输入frame，进行推理，保存原始图像和结果
        :param frame: 输入的视频帧 (numpy array)
        :return: res (检测到的对象列表信息)
        """
        if frame is None:
            return None

        # 1. 把frame复制到属性 img_raw
        # 使用 copy.deepcopy 或 .copy() 确保后续操作不影响原始引用，视具体需求而定，这里浅拷贝通常足够用于展示
        self.img_raw = frame.copy()

        # 2. 对frame进行推理
        # classes=[self.id] 确保只检测水瓶
        results = self.model.predict(
            source=frame,
            classes=[self.id],
            device=self.device,
            verbose=False, # 关闭控制台详细日志，保持整洁
            stream=False   # 单帧处理不需要stream
        )

        # 3. 解析结果并存到 res 属性
        # 我们提取有用的信息：box坐标, 置信度, 类别ID
        detected_objects = []
        r = results[0] # 获取第一个结果对象
        
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
        # 使用传入的 res 或者内部存储的 self.res
        if res is None:
            res = self.res
            
        # 如果没有检测结果或没有原始图像，返回 None
        if res is None or self.img_raw is None:
            return None

        # 判断 dis_mode: 0->不显示结果(返回None), 1->显示结果
        if self.dis_mode == 0:
            return None

        # 从属性获得 img_raw 并复制到 img_res
        self.img_res = self.img_raw.copy()

        # 将结果绘制到 img_res
        # 使用 Ultralytics 自带的 Annotator 进行绘制，比手动画矩形更规范
        annotator = Annotator(self.img_res, line_width=2, font_size=10)
        
        for obj in res:
            box = obj["box"]
            conf = obj["conf"]
            name = obj["cls_name"]
            
            # 格式化标签: "bottle 0.85"
            label = f"{name} {conf:.2f}"
            
            # 绘制边框和标签
            annotator.box_label(box, label, color=(255, 0, 0)) # 蓝色框

        return self.img_res

# 使用示例
if __name__ == "__main__":
    # 1. 初始化检测器
    detector = Detector(cls_id=39, dis_mode=1)
    
    # 模拟打开摄像头
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    print("启动推理流... (按 'q' 退出)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 2. 执行检测
        res = detector.detect(frame)
        
        # 3. 执行绘制
        output_frame = detector.draw(res)
        
        # 4. 显示结果
        if output_frame is not None:
            cv2.imshow("Bottle Detection", output_frame)
            
            # 打印控制台信息 (可选)
            if res:
                for obj in res:
                    print(f"目标: {obj['cls_name']} | 置信度: {obj['conf']:.2f} | 坐标: {obj['box']}")

        # 按 'q' 键退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()