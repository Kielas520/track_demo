class Target:
    """管理目标状态的数据类"""
    def __init__(self):
        self.is_tracking = False
        self.bbox = None
        self.mode = "CSRT"
        self.frame_count = 0

    def start(self, bbox, mode="CSRT"):
        """启动/重启追踪时的状态更新"""
        self.is_tracking = True
        self.bbox = bbox
        self.mode = mode
        self.frame_count = 0

    def update_state(self, success, bbox):
        """每帧更新状态"""
        if success:
            self.bbox = bbox
            self.frame_count += 1
        # 注意：原逻辑中，即便失败也保持 is_tracking=True 以便触发重连机制

    def reset(self):
        """重置状态"""
        self.is_tracking = False
        self.bbox = None
        self.frame_count = 0