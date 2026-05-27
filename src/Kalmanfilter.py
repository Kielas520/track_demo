import time
import numpy as np

class KalmanFilter3D:
    """
    三维卡尔曼滤波器 (CV 匀速模型) - 纯 NumPy 无依赖版本
    ===================================================

    状态向量 (6x1):
        [x, y, z, vx, vy, vz]
        - x, y:   图像平面位置 (像素)
        - z:      目标面积 (像素²)，如 HSV 掩膜白点计数
        - vx, vy: x、y 方向速度 (像素/秒)
        - vz:     面积变化率 (像素²/秒)

    观测向量 (3x1):
        [x, y, z]
        - 来自跟踪器 bbox 中心点 + ROI掩膜面积

    主要功能:
        - 纯矩阵运算，零 OpenCV 依赖
        - 状态机机制：引入 DETECTING 预热期，防止初始速度未收敛导致预测乱飞
        - 动态过程噪声：根据真实 dt 动态计算 Q 矩阵
        - 前馈预测：补偿系统延迟，提前预估目标位置
    """

    def __init__(self, predict_mode="hybrid", predict_time=0.1,
                 q_x=10.0, q_y=10.0, q_z=10.0,
                 q_vx=50.0, q_vy=50.0, q_vz=50.0,
                 r_x=1.0, r_y=1.0, r_z=1.0):
        """
        初始化卡尔曼滤波器

        Args:
            predict_mode: 预测模式 "manual"|"auto"|"hybrid"
            predict_time: 手动预测时间 (秒)
            q_x, q_y, q_z: 位置/面积的基础过程噪声系数
            q_vx, q_vy, q_vz: 速度/面积变化率的基础过程噪声系数
            r_x, r_y, r_z: 观测噪声
        """
        # ---- 预测与系统延迟 ----
        self.predict_mode = predict_mode
        self.predict_time = predict_time
        self.system_delay = 0.0    
        self.last_tick = time.time()

        # ---- 状态机配置 ----
        self.frame_count = 0 
        self.DETECTING_THRESHOLD = 10  # 前 10 帧拦截前馈预测

        # ---- 卡尔曼状态与矩阵 ----
        # x(6): [x, y, z, vx, vy, vz]
        self.x = np.zeros(6, dtype=np.float64)
        # 初始协方差 P (位置不确定性较小，速度不确定性较大)
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)
        
        # 测量矩阵 H (仅观测位置和面积)
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], dtype=np.float64)
        self.I = np.eye(6, dtype=np.float64)

        # ---- 噪声参数 ----
        self.q_x, self.q_y, self.q_z = q_x, q_y, q_z
        self.q_vx, self.q_vy, self.q_vz = q_vx, q_vy, q_vz
        self.r_x, self.r_y, self.r_z = r_x, r_y, r_z
        self._apply_r()

        # 存储最近一次原始观测值
        self.last_raw_x = None
        self.last_raw_y = None
        self.last_raw_z = None

    def _apply_r(self):
        """更新测量噪声矩阵 R (将传入的标准差转换为方差)"""
        self.R = np.diag([self.r_x**2, self.r_y**2, self.r_z**2]).astype(np.float64)

    def set_qr(self, q_x=None, q_y=None, q_z=None,
               q_vx=None, q_vy=None, q_vz=None,
               r_x=None, r_y=None, r_z=None):
        """运行时动态调整 Q/R 噪声参数"""
        if q_x is not None: self.q_x = q_x
        if q_y is not None: self.q_y = q_y
        if q_z is not None: self.q_z = q_z
        if q_vx is not None: self.q_vx = q_vx
        if q_vy is not None: self.q_vy = q_vy
        if q_vz is not None: self.q_vz = q_vz
        if r_x is not None: self.r_x = r_x
        if r_y is not None: self.r_y = r_y
        if r_z is not None: self.r_z = r_z
        self._apply_r()

    # ===================== 状态初始化 =====================

    def init(self, x, y, z):
        """
        初始化卡尔曼滤波器状态，并重置状态机
        """
        self.x = np.array([x, y, z, 0.0, 0.0, 0.0], dtype=np.float64)
        # 重置协方差
        self.P = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0]).astype(np.float64)
        
        self.frame_count = 1
        self.last_raw_x = x
        self.last_raw_y = y
        self.last_raw_z = z

    # ===================== 滤波核心 =====================

    def predict(self, dt):
        """
        时间更新（预测步）- 包含动态过程噪声 Q 的计算
        """
        # 动态构建状态转移矩阵 F
        F = np.array([
            [1, 0, 0, dt, 0,  0],
            [0, 1, 0, 0,  dt, 0],
            [0, 0, 1, 0,  0,  dt],
            [0, 0, 0, 1,  0,  0],
            [0, 0, 0, 0,  1,  0],
            [0, 0, 0, 0,  0,  1]
        ], dtype=np.float64)

        # 动态计算过程噪声 Q，随 dt 缩放
        qx = (self.q_x * dt) ** 2
        qy = (self.q_y * dt) ** 2
        qz = (self.q_z * dt) ** 2
        qvx = (self.q_vx * dt) ** 2
        qvy = (self.q_vy * dt) ** 2
        qvz = (self.q_vz * dt) ** 2
        Q = np.diag([qx, qy, qz, qvx, qvy, qvz])

        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, x, y, z):
        """
        测量更新（校正步）
        """
        z_meas = np.array([x, y, z], dtype=np.float64)
        self.last_raw_x, self.last_raw_y, self.last_raw_z = x, y, z
        
        # 推动状态机
        self.frame_count += 1

        y_err = z_meas - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y_err
        self.P = (self.I - K @ self.H) @ self.P

    # ===================== 系统延迟计时 =====================

    def tick(self):
        """记录并返回系统循环时间 dt"""
        current_time = time.time()
        self.system_delay = current_time - self.last_tick
        self.last_tick = current_time
        return self.system_delay

    def get_predict_dt(self):
        """根据预测模式计算前馈预测的时间量"""
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
        """获取当前滤波平滑位置"""
        return int(self.x[0]), int(self.x[1]), int(self.x[2])

    def predict_future(self, future_time=None):
        """
        前馈预测未来位置 (自带状态机拦截)
        """
        if future_time is None:
            future_time = self.get_predict_dt()

        # 1. 状态机拦截：DETECTING 阶段拦截前馈预测，透传当前滤波位置
        if self.frame_count < self.DETECTING_THRESHOLD:
            return int(self.x[0]), int(self.x[1]), int(self.x[2])

        # 2. future_time == 0 表示不做预测，直接返回滤波位置
        if future_time == 0.0:
            return int(self.x[0]), int(self.x[1]), int(self.x[2])

        # 3. 匀速直线运动前馈
        pred_x = self.x[0] + self.x[3] * future_time
        pred_y = self.x[1] + self.x[4] * future_time
        pred_z = self.x[2] + self.x[5] * future_time

        return int(pred_x), int(pred_y), int(pred_z)

    # ===================== 配置接口 =====================

    def set_mode(self, mode):
        self.predict_mode = mode

    def set_predict_time(self, t):
        self.predict_time = t