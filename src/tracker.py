import cv2

class TrackerManager:
    """封装 OpenCV 追踪算法"""
    def __init__(self):
        self.tracker = None

    def start(self, frame, bbox, mode="CSRT"):
        """初始化追踪器"""
        if mode == "CSRT":
            # 兼容不同版本 OpenCV 的 API
            if hasattr(cv2, 'TrackerCSRT_create'):
                self.tracker = cv2.TrackerCSRT_create()
            else:
                params = cv2.TrackerCSRT_Params()
                self.tracker = cv2.TrackerCSRT.create(params)
        else:
            if hasattr(cv2, 'TrackerKCF_create'):
                self.tracker = cv2.TrackerKCF_create()
            else:
                self.tracker = cv2.TrackerKCF.create()
        
        self.tracker.init(frame, bbox)

    def update(self, frame):
        """更新追踪结果"""
        if self.tracker is None:
            return False, None
        return self.tracker.update(frame)

    def reset(self):
        """清理追踪器实例"""
        self.tracker = None