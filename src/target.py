class Target:
    """
    目标状态管理器
    ==============
    维护当前被跟踪目标的所有状态信息，
    包括跟踪状态、包围框、跟踪器模式和帧计数。
    本身不涉及跟踪算法逻辑，仅作为数据容器。
    """

    def __init__(self):
        """初始化目标为未跟踪状态"""
        self.is_tracking = False   # 是否正在跟踪
        self.bbox = None           # 当前包围框 (x, y, w, h)，None 表示无目标
        self.mode = "CSRT"         # 跟踪器模式："CSRT" 或 "KCF"
        self.frame_count = 0       # 跟踪持续的帧数

    def start(self, bbox, mode="CSRT"):
        """
        启动或重启跟踪

        当用户选定目标并启动跟踪器时调用，
        将状态置为跟踪中，记录初始包围框和模式。

        Args:
            bbox: 包围框元组 (x, y, w, h)
            mode: 跟踪器类型，"CSRT" 或 "KCF"
        """
        self.is_tracking = True
        self.bbox = bbox
        self.mode = mode
        self.frame_count = 0       # 新目标，帧计数归零

    def update_state(self, success, bbox):
        """
        每帧更新目标状态

        由主循环在调用 tracker.update() 之后调用。
        跟踪成功时更新 bbox 并递增计数器；
        跟踪失败时保持 is_tracking=True，
        以便主循环通过 mask 重检测触发重连。

        Args:
            success: 跟踪器本帧是否成功
            bbox:   跟踪器返回的包围框 (x, y, w, h)，失败时为 None
        """
        if success:
            self.bbox = bbox
            self.frame_count += 1
        # 注意：即使失败也不设 is_tracking=False，
        # 这样主循环中的候选目标检测可以自动重连

    def reset(self):
        """
        重置目标状态

        清除所有跟踪信息，回到初始未跟踪状态。
        通常在按 'r' 键时调用。
        """
        self.is_tracking = False
        self.bbox = None
        self.frame_count = 0
