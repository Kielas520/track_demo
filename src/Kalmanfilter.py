import time
import cv2
import numpy as np

class KalmanFilter3D:
    """
    三维卡尔曼滤波器 (CV 匀速模型)
    ==============================

    状态向量 (6x1):
        [x, y, z, vx, vy, vz]
        - x, y:   图像平面位置 (像素)
        - z:      目标面积 (像素²)，如 HSV 掩膜白点计数
        - vx, vy: x、y 方向速度 (像素/秒)
        - vz:     面积变化率 (像素²/秒)

    观测向量 (3x1):
        [x, y, z]
        - 来自跟踪器 bbox 中心点 + ROI掩膜面积

    运动模型: 匀速直线运动 (Constant Velocity)
        x_{k+1} = x_k + vx * dt
        y_{k+1} = y_k + vy * dt
        z_{k+1} = z_k + vz * dt

    主要功能:
        - 滤波平滑跟踪位置，去除观测噪声
        - 前馈预测：补偿系统延迟，提前预估目标位置
        - 实时可调 Q/R 噪声参数
    """

    def __init__(self, predict_mode="hybrid", predict_time=0.1,
                 q_x=1e-2, q_y=1e-2, q_z=1e-2,
                 q_vx=1e-2, q_vy=1e-2, q_vz=1e-2,
                 r_x=1e-1, r_y=1e-1, r_z=1e-1):
        """
        初始化卡尔曼滤波器

        Args:
            predict_mode: 预测模式 "manual"|"auto"|"hybrid"
                - "manual": 使用固定 predict_time
                - "auto":   使用系统实际循环延时 dt
                - "hybrid": 两者相加 (manual + dt)
            predict_time: 手动预测时间 (秒)，仅在 manual/hybrid 模式下使用

            q_x, q_y, q_z:  位置/面积过程噪声 (过程噪声对角阵前3项)
                             值越大 → 越信任观测，滤波平滑程度越低
            q_vx, q_vy, q_vz:速度/面积变化率过程噪声 (对角阵后3项)
                             值越大 → 速度估计越灵活，响应越快
            r_x, r_y, r_z:  观测噪声 (测量噪声对角阵)
                             值越大 → 越信任模型预测，平滑程度越高
        """
        # 创建 6 状态、3 观测的 KalmanFilter
        self.kf = cv2.KalmanFilter(6, 3)

        # 初始化转移矩阵、测量矩阵、误差协方差
        self.set_params()

        # ---- 预测模式配置 ----
        self.predict_mode = predict_mode
        self.predict_time = predict_time

        # ---- 系统延迟计时 ----
        self.system_delay = 0.0    # 当前帧与上一帧的时间差 (秒)
        self.last_tick = time.time()

        # ---- 噪声参数 ----
        self.q_x, self.q_y, self.q_z = q_x, q_y, q_z
        self.q_vx, self.q_vy, self.q_vz = q_vx, q_vy, q_vz
        self.r_x, self.r_y, self.r_z = r_x, r_y, r_z
        self._apply_qr()           # 将参数写入矩阵

        # ---- 存储最近一次原始观测值 ----
        self.last_raw_x = None
        self.last_raw_y = None
        self.last_raw_z = None

    # ===================== 矩阵初始化 =====================

    def set_params(self):
        """
        设置卡尔曼滤波器的固定矩阵参数

        包括:
            - 状态转移矩阵 F (6x6): 匀速模型，dt 在 predict() 中动态更新
            - 测量矩阵 H (3x6):     直接观测位置/面积，不观测速度
            - 误差协方差矩阵 P (6x6): 初始化为单位阵
        """
        # F = [[I(3x3), dt*I(3x3)],
        #      [0(3x3),   I(3x3)  ]]
        # 其中 dt 在每次 predict() 调用时动态填入
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)

        # H = [I(3x3), 0(3x3)]
        # 只测量位置和面积，不直接测量速度
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)

        # P = I * 1.0
        # 初始状态不确定性（大值意味着初始估计不可靠，滤波器会快速收敛）
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

    def _apply_qr(self):
        """
        将当前噪声参数写入过程噪声矩阵 Q 和测量噪声矩阵 R

        Q (6x6 对角阵):
            diag(q_x, q_y, q_z, q_vx, q_vy, q_vz)
            表示状态预测的不确定性（模型误差）

        R (3x3 对角阵):
            diag(r_x, r_y, r_z)
            表示观测值的不确定性（传感器误差）

        调参原则:
            - Q 大, R 小 → 更信任观测，响应快、平滑少
            - Q 小, R 大 → 更信任模型，平滑强、滞后大
        """
        self.kf.processNoiseCov = np.array([
            [self.q_x, 0, 0, 0, 0, 0],
            [0, self.q_y, 0, 0, 0, 0],
            [0, 0, self.q_z, 0, 0, 0],
            [0, 0, 0, self.q_vx, 0, 0],
            [0, 0, 0, 0, self.q_vy, 0],
            [0, 0, 0, 0, 0, self.q_vz]
        ], np.float32)

        self.kf.measurementNoiseCov = np.array([
            [self.r_x, 0, 0],
            [0, self.r_y, 0],
            [0, 0, self.r_z]
        ], np.float32)

    def set_qr(self, q_x=None, q_y=None, q_z=None,
               q_vx=None, q_vy=None, q_vz=None,
               r_x=None, r_y=None, r_z=None):
        """
        运行时动态调整 Q/R 噪声参数

        只需传入要修改的参数，其余保持不变。
        通常由主循环中的 trackbar 回调触发。

        Args:
            q_x, q_y, q_z:  位置/面积过程噪声 (None 表示不修改)
            q_vx, q_vy, q_vz: 速度过程噪声
            r_x, r_y, r_z:  观测噪声
        """
        if q_x is not None: self.q_x = q_x
        if q_y is not None: self.q_y = q_y
        if q_z is not None: self.q_z = q_z
        if q_vx is not None: self.q_vx = q_vx
        if q_vy is not None: self.q_vy = q_vy
        if q_vz is not None: self.q_vz = q_vz
        if r_x is not None: self.r_x = r_x
        if r_y is not None: self.r_y = r_y
        if r_z is not None: self.r_z = r_z
        self._apply_qr()

    # ===================== 状态初始化 =====================

    def init(self, x, y, z):
        """
        初始化卡尔曼滤波器状态

        通常在首次检测到新目标时调用。
        设置初始位置和面积，速度分量初始化为 0。

        Args:
            x: 目标中心 x 坐标 (像素)
            y: 目标中心 y 坐标 (像素)
            z: 目标掩膜面积 (像素²)
        """
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)],
                                      [0.0], [0.0], [0.0]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()

    # ===================== 滤波核心 =====================

    def predict(self, dt):
        """
        时间更新（预测步）

        根据时间间隔 dt 推进状态，补偿匀速运动。
        每次主循环中在获取新观测前调用。

        Args:
            dt: 距离上一帧的时间间隔 (秒)
        """
        # 动态填入 dt，实现真正的 CV 模型
        self.kf.transitionMatrix[0, 3] = np.float32(dt)
        self.kf.transitionMatrix[1, 4] = np.float32(dt)
        self.kf.transitionMatrix[2, 5] = np.float32(dt)

        # OpenCV 内部执行: x_pred = F * x, P_pred = F * P * F^T + Q
        self.kf.predict()

    def update(self, x, y, z):
        """
        测量更新（校正步）

        将新的观测值送入滤波器，融合预测和观测，
        更新后验状态 statePost。

        Args:
            x: 观测到的目标中心 x 坐标 (像素)
            y: 观测到的目标中心 y 坐标 (像素)
            z: 观测到的掩膜面积 (像素²)
        """
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])

        # K = P_pred * H^T * (H * P_pred * H^T + R)^{-1}
        # x_post = x_pred + K * (z - H * x_pred)
        self.kf.correct(measured)

        # 保存原始观测，用于 predict_future(0) 直接返回原始值
        self.last_raw_x = x
        self.last_raw_y = y
        self.last_raw_z = z

    # ===================== 系统延迟计时 =====================

    def tick(self):
        """
        记录并返回系统循环时间

        应在每帧开始时调用，测量实际帧间隔。

        Returns:
            dt: 当前帧与上一帧的时间差 (秒)
        """
        current_time = time.time()
        self.system_delay = current_time - self.last_tick
        self.last_tick = current_time
        return self.system_delay

    def get_predict_dt(self):
        """
        根据预测模式计算前馈预测的时间量

        Returns:
            预测时间 (秒)
            - "manual": 返回手动设定的 predict_time
            - "auto":   返回系统实际循环延时
            - "hybrid": 返回两者之和
        """
        if self.predict_mode == "manual":
            return self.predict_time
        elif self.predict_mode == "auto":
            return self.system_delay
        elif self.predict_mode == "hybrid":
            return self.predict_time + self.system_delay
        else:
            return 0.0

    # ===================== 位置查询 & 预测 =====================

    def get_filtered_pos(self):
        """
        获取卡尔曼滤波后的当前位置

        返回 statePost 中的 (x, y, z)，
        即融合了当前观测的平滑估计。

        Returns:
            (x, y, z): 滤波后的像素位置和面积，均为 int
        """
        state = self.kf.statePost
        return int(state[0, 0]), int(state[1, 0]), int(state[2, 0])

    def predict_future(self, future_time=None):
        """
        前馈预测未来位置

        基于当前后验状态和速度，向前预测 future_time 秒后的位置。
        用于补偿系统延迟（采集+处理+显示的总延迟）。

        Args:
            future_time: 预测时间 (秒)，None 则自动计算

        Returns:
            (x, y, z): 预测的未来位置和面积，均为 int

        特殊处理:
            当 future_time == 0 时，直接返回原始观测值 (last_raw_*)，
            而非滤波后的平滑值。这确保手动模式 predict_time=0
            时预测标记与原始观测完全重合。
        """
        if future_time is None:
            future_time = self.get_predict_dt()

        # future_time == 0 表示不做预测，返回原始观测
        if future_time == 0 and self.last_raw_x is not None:
            return int(self.last_raw_x), int(self.last_raw_y), int(self.last_raw_z)

        # 匀速直线运动前馈: pos_future = pos_now + vel * dt
        state = self.kf.statePost
        x, y, z = state[0, 0], state[1, 0], state[2, 0]
        vx, vy, vz = state[3, 0], state[4, 0], state[5, 0]

        pred_x = x + vx * future_time
        pred_y = y + vy * future_time
        pred_z = z + vz * future_time

        return int(pred_x), int(pred_y), int(pred_z)

    # ===================== 配置接口 =====================

    def set_mode(self, mode):
        """
        设置预测模式

        Args:
            mode: "manual" | "auto" | "hybrid"
        """
        self.predict_mode = mode

    def set_predict_time(self, t):
        """
        设置手动预测时间

        Args:
            t: 预测时长 (秒)
        """
        self.predict_time = t
