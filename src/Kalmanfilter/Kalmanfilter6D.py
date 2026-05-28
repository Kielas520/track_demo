import time
import numpy as np
from numba import njit


# ============================================================
# Numba JIT 编译的快速 EKF 预测与更新函数 (核心数学计算)
# ============================================================

@njit(fastmath=True, nogil=True)
def _fast_ekf_predict(X, P, F, Q):
    """EKF 预测步骤（原地操作，零拷贝）"""
    X[:] = F @ X                  
    P[:] = F @ P @ F.T + Q       


@njit(fastmath=True, nogil=True)
def _fast_ekf_update_inplace(X, P, H, R, Y, I, mahalanobis_threshold=200.0):
    """EKF 更新步骤（含马氏距离门控，原地操作，零拷贝）"""
    S = H @ P @ H.T + R
    S_inv_Y = np.linalg.solve(S, Y)
    mahalanobis_sq = np.dot(Y, S_inv_Y)

    if mahalanobis_sq > mahalanobis_threshold:
        return False

    K = np.linalg.solve(S.T, (P @ H.T).T).T 
    X[:] = X + K @ Y       

    I_KH = I - K @ H                                    
    P[:] = I_KH @ P @ I_KH.T + K @ R @ K.T            
    return True


class KalmanFilter6D:
    """
    ============================================================
    6DOF 卡尔曼滤波器 (CV 匀速模型) - 对齐工程管线版本
    ============================================================

    状态向量 X (12x1):
        [x, vx, y, vy, z, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
        - 平移与旋转混合状态
    
    观测向量 Z (6x1):
        [x_obs, y_obs, z_obs, roll_obs, pitch_obs, yaw_obs]

    工程特性对齐 (基于 3D 版本):
        - 状态机机制：引入 DETECTING 预热期，防止初始速度未收敛导致预测乱飞
        - 系统时间流控制：tick() 与 get_predict_dt() 补偿系统延迟
        - 降级前馈预测：仅对平移 (x,y,z) 进行前馈预测，姿态保持当前滤波值输出以确保稳定
    """

    def __init__(self, predict_mode="hybrid", predict_time=0.1):
        # ---- 预测与系统延迟 ----
        self.predict_mode = predict_mode
        self.predict_time = predict_time
        self.system_delay = 0.0    
        self.last_tick = time.time()

        # ---- 状态机配置 ----
        self.frame_count = 0 
        self.DETECTING_THRESHOLD = 10  # 前 10 帧拦截前馈预测

        # ---- 状态向量与协方差初始化 ----
        self.X = np.zeros(12, dtype=np.float64)
        self.P = np.eye(12, dtype=np.float64)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0      # 位置/角度 → 低不确定性
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0     # 速度 → 高不确定性

        # ---- 过程与观测噪声参数 ----
        self.s2q_pos = 10.0
        self.s2q_rot = 100.0
        self.mahalanobis_threshold = 200.0
        self.r_pos_factor = 0.01
        self.r_rot_factor = 0.5

        # ---- 预分配矩阵内存 ----
        self.F = np.eye(12, dtype=np.float64)     
        self.Q = np.zeros((12, 12), dtype=np.float64)  
        self.H = np.zeros((6, 12), dtype=np.float64)   
        self.R = np.zeros((6, 6), dtype=np.float64)    
        self.I = np.eye(12, dtype=np.float64)          

        # 初始化观测矩阵 H
        for i in range(6):
            self.H[i, i * 2] = 1.0  

        # ---- Numba JIT 预热 ----
        self._warmup_jit()

    def _warmup_jit(self):
        """内部方法：触发 Numba 编译"""
        dummy_X = np.zeros(12, dtype=np.float64)
        dummy_P = np.eye(12, dtype=np.float64)
        dummy_F = np.eye(12, dtype=np.float64)
        dummy_Q = np.zeros((12, 12), dtype=np.float64)
        dummy_H = np.zeros((6, 12), dtype=np.float64)
        dummy_R = np.eye(6, dtype=np.float64)
        dummy_Y = np.zeros(6, dtype=np.float64)
        _fast_ekf_predict(dummy_X, dummy_P, dummy_F, dummy_Q)
        _fast_ekf_update_inplace(dummy_X, dummy_P, dummy_H, dummy_R, dummy_Y, self.I, 200.0)

    # ===================== 配置接口 =====================

    def set_noise(self, q_pos=0.001, q_rot=0.1, r_pos_factor=0.01, r_rot_factor=0.5,
                  mahalanobis_threshold=200.0):
        """动态调整卡尔曼滤波器的噪声参数"""
        self.s2q_pos = q_pos
        self.s2q_rot = q_rot
        self.r_pos_factor = r_pos_factor
        self.r_rot_factor = r_rot_factor
        self.mahalanobis_threshold = mahalanobis_threshold

    def set_mode(self, mode):
        """设置预测模式: 'manual' | 'auto' | 'hybrid'"""
        self.predict_mode = mode

    def set_predict_time(self, t):
        """设置手动预测时间提前量 (秒)"""
        self.predict_time = t

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

    # ===================== 状态初始化 =====================

    def init_state(self, x, y, z, roll, pitch, yaw):
        """
        初始化卡尔曼滤波器状态，并重置状态机与计时器
        """
        self.X[:] = [x, 0.0, y, 0.0, z, 0.0, roll, 0.0, pitch, 0.0, yaw, 0.0]

        # 协方差矩阵重置
        self.P.fill(0.0)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0      
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0     
            
        # 重置状态机与计时
        self.frame_count = 1
        self.last_tick = time.time()

    # ===================== 滤波核心 =====================

    def predict(self, dt):
        """时间更新（预测步）"""
        self.F.fill(0)
        for i in range(6):
            base = i * 2
            self.F[base, base] = 1.0           
            self.F[base, base + 1] = dt        
            self.F[base + 1, base + 1] = 1.0   

        self.Q.fill(0)
        t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4

        # 平移过程噪声
        for i in range(3):
            idx = i * 2
            self.Q[idx, idx]         = t4 / 4 * self.s2q_pos
            self.Q[idx + 1, idx]     = t3 / 2 * self.s2q_pos
            self.Q[idx, idx + 1]     = t3 / 2 * self.s2q_pos
            self.Q[idx + 1, idx + 1] = t2     * self.s2q_pos

        # 旋转过程噪声
        for i in range(3, 6):
            idx = i * 2
            self.Q[idx, idx]         = t4 / 4 * self.s2q_rot
            self.Q[idx + 1, idx]     = t3 / 2 * self.s2q_rot
            self.Q[idx, idx + 1]     = t3 / 2 * self.s2q_rot
            self.Q[idx + 1, idx + 1] = t2     * self.s2q_rot

        _fast_ekf_predict(self.X, self.P, self.F, self.Q)
        return self.X

    def update(self, measurement):
        """测量更新（校正步）"""
        Z = np.array(measurement, dtype=np.float64)
        
        # 推动状态机
        self.frame_count += 1

        self.R.fill(0)
        dist = abs(Z[2]) + 1e-6

        # 平移与旋转观测噪声（距离自适应）
        base_pos = self.r_pos_factor * dist + 0.01
        self.R[0, 0] = self.R[1, 1] = self.R[2, 2] = base_pos

        base_rot = self.r_rot_factor
        rot_noise = base_rot * max(1.0, abs(dist) / 50.0)
        self.R[3, 3] = self.R[4, 4] = self.R[5, 5] = rot_noise

        Z_pred = self.H @ self.X
        Y = Z - Z_pred

        return _fast_ekf_update_inplace(self.X, self.P, self.H, self.R, Y, self.I,
                                          self.mahalanobis_threshold)

    # ===================== 结果查询 & 预测 =====================

    def get_filtered_pos(self):
        """获取滤波后的平移量"""
        return float(self.X[0]), float(self.X[2]), float(self.X[4])

    def get_filtered_rot(self):
        """获取滤波后的旋转姿态"""
        return float(self.X[6]), float(self.X[8]), float(self.X[10])

    def predict_future(self, future_time=None):
        """
        前馈预测未来位姿 (带状态机拦截与旋转保护)
        
        策略：
        - 仅对 x, y, z 进行匀速前馈预测补偿延迟
        - roll, pitch, yaw 保持当前滤波值，防止欧拉角预测突变与抖动
        
        返回:
            tuple: ((pred_x, pred_y, pred_z), (cur_roll, cur_pitch, cur_yaw))
        """
        if future_time is None:
            future_time = self.get_predict_dt()

        # 获取当前最新滤波状态
        cur_x, cur_y, cur_z = self.get_filtered_pos()
        cur_rot = self.get_filtered_rot()  # (roll, pitch, yaw)

        # 1. 状态机拦截：DETECTING 阶段拦截平移前馈预测，直接透传当前位置
        if self.frame_count < self.DETECTING_THRESHOLD:
            return (cur_x, cur_y, cur_z), cur_rot

        # 2. 无预测时间，直接返回当前状态
        if future_time <= 0.0:
            return (cur_x, cur_y, cur_z), cur_rot

        # 3. 仅对平移做匀速直线运动前馈补偿
        # X[1] 为 vx, X[3] 为 vy, X[5] 为 vz
        pred_x = cur_x + self.X[1] * future_time
        pred_y = cur_y + self.X[3] * future_time
        pred_z = cur_z + self.X[5] * future_time

        return (pred_x, pred_y, pred_z), cur_rot

    def smooth_reset_covariance(self):
        """协方差平滑重置，帮助滤波器从漂移中恢复"""
        for i in range(0, 12, 2):
            self.P[i, i] += 0.05