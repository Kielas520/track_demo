import cv2

class TrackerManager:
    """
    OpenCV 跟踪器管理器
    ==================
    封装 OpenCV 的 CSRT 和 KCF 跟踪器，
    统一接口，并兼容不同 OpenCV 版本的 API 差异。

    支持的模式:
        - "CSRT": 精度高但慢，适合小目标或需要精细跟踪的场景
        - "KCF":  速度快但精度略低，适合实时性要求高的场景
    """

    def __init__(self):
        """初始化，跟踪器实例为空"""
        self.tracker = None   # OpenCV Tracker 实例，None 表示未初始化

    def start(self, frame, bbox, mode="CSRT"):
        """
        初始化并启动跟踪器

        根据 mode 创建对应的 OpenCV Tracker 实例，
        用给定帧和包围框初始化。兼容新旧两套 OpenCV API。

        Args:
            frame: 当前视频帧 (numpy array, BGR 格式)
            bbox:  目标的初始包围框 (x, y, w, h)
            mode:  跟踪算法，"CSRT" 或 "KCF"
        """
        if mode == "CSRT":
            # OpenCV 4.x 使用 create() 静态方法
            # OpenCV 3.x 使用 TrackerCSRT_create() 函数
            if hasattr(cv2, 'TrackerCSRT_create'):
                self.tracker = cv2.TrackerCSRT_create()
            else:
                params = cv2.TrackerCSRT_Params()
                self.tracker = cv2.TrackerCSRT.create(params)
        else:
            # KCF 模式 (Kernelized Correlation Filter)
            if hasattr(cv2, 'TrackerKCF_create'):
                self.tracker = cv2.TrackerKCF_create()
            else:
                self.tracker = cv2.TrackerKCF.create()

        # 用当前帧和目标包围框初始化跟踪器
        self.tracker.init(frame, bbox)

    def update(self, frame):
        """
        在新一帧上更新跟踪结果

        调用底层 OpenCV Tracker 的 update 方法，
        返回是否成功以及新的包围框。

        Args:
            frame: 当前视频帧 (numpy array, BGR 格式)

        Returns:
            (success, bbox): success 为 bool，bbox 为 (x, y, w, h) 或 None
        """
        if self.tracker is None:
            return False, None
        return self.tracker.update(frame)

    def reset(self):
        """
        重置跟踪器

        释放当前跟踪器实例，回到未初始化状态。
        下次需要重新调用 start()。
        """
        self.tracker = None
