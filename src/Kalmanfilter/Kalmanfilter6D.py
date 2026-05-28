import numpy as np
from numba import njit


# ============================================================
# Numba JIT 编译的快速 EKF 预测函数
# ============================================================
# 使用 @njit 装饰器将 Python 代码编译为机器码，大幅提升执行速度。
# - fastmath=True: 启用快速数学优化（允许不严格的浮点运算）
# - nogil=True:   释放 GIL 全局解释器锁，支持多线程调用
# 函数操作的都是外部传入的 ndarray 引用，直接原地修改，避免内存分配开销。
# ============================================================

@njit(fastmath=True, nogil=True)
def _fast_ekf_predict(X, P, F, Q):
    """
    EKF 预测步骤（原地操作，零拷贝）

    数学公式：
        状态预测:  x̂_{k|k-1} = F · x̂_{k-1|k-1}
        协方差预测: P_{k|k-1} = F · P_{k-1|k-1} · F^T + Q

    参数:
        X : np.ndarray 形状 (12,)  —— 状态向量，被原地更新
        P : np.ndarray 形状 (12,12) —— 协方差矩阵，被原地更新
        F : np.ndarray 形状 (12,12) —— 状态转移矩阵（含 dt）
        Q : np.ndarray 形状 (12,12) —— 过程噪声协方差矩阵（含 dt^2/3/4）
    """
    X[:] = F @ X                  # 状态预测：匀速模型外推
    P[:] = F @ P @ F.T + Q       # 协方差更新：传播不确定性 + 模型噪声


@njit(fastmath=True, nogil=True)
def _fast_ekf_update_inplace(X, P, H, R, Y, I):
    """
    EKF 更新步骤（原地操作，零拷贝）

    数学公式：
        新息协方差:   S = H·P·H^T + R
        卡尔曼增益:   K = P·H^T·S^{-1}
        状态更新:     x̂ = x̂ + K·Y          (Y = z - H·x̂, 新息)
        协方差更新:   P = (I - K·H)·P·(I - K·H)^T + K·R·K^T  (Joseph 稳定形式)

    Joseph 形式 vs 简化形式 P = (I-KH)P：
    Joseph 形式保证协方差矩阵的正定性，数值更稳定，适合 Numba 编译环境。

    参数:
        X : np.ndarray 形状 (12,)  —— 状态向量，被原地更新
        P : np.ndarray 形状 (12,12) —— 协方差矩阵，被原地更新
        H : np.ndarray 形状 (6,12)  —— 观测矩阵 [I|0]，位置分量直接观测，速度分量间接
        R : np.ndarray 形状 (6,6)   —— 观测噪声协方差矩阵（距离自适应）
        Y : np.ndarray 形状 (6,)    —— 新息向量 (innovation) = Z - H·X
        I : np.ndarray 形状 (12,12) —— 单位矩阵（缓存，避免重复创建）

    返回:
        bool: True 表示正常更新，False 表示被马氏距离门控拒绝（野值/异常点）
    """
    # --------- 计算新息协方差 S ---------
    S = H @ P @ H.T + R
    # 6x6 对称正定矩阵，编码了状态不确定性和测量噪声两方面

    # --------- 计算马氏距离的平方 ---------
    # 马氏距离: d² = Y^T · S^{-1} · Y
    # 用 np.linalg.solve 代替求逆，数值更稳定，速度也更快
    S_inv_Y = np.linalg.solve(S, Y)
    mahalanobis_sq = np.dot(Y, S_inv_Y)

    # --------- 野值检测 (Outlier Rejection / Gating) ---------
    # 马氏距离服从 χ²(6) 卡方分布，阈值 200 对应极低假阳性率
    # 若当前观测与预测值偏差过大（被遮挡、误匹配等），直接丢弃此次更新
    if mahalanobis_sq > 200.0:
        return False

    # --------- 计算卡尔曼增益 K ---------
    # K = P·H^T·S^{-1}，使用 solve 避免显式求逆
    # 等价于: K = (P @ H.T) @ np.linalg.inv(S)
    K = np.linalg.solve(S.T, (P @ H.T).T).T  # 12x6 矩阵，加权因子

    # --------- 状态更新 ---------
    X[:] = X + K @ Y       # 用新息修正预测值

    # --------- 协方差更新 (Joseph 稳定形式) ---------
    I_KH = I - K @ H                                    # 12x12
    P[:] = I_KH @ P @ I_KH.T + K @ R @ K.T            # 保证 P 正定

    return True


class KalmanFilter6D:
    """
    ============================================================
    6DOF 卡尔曼滤波器 (CV 匀速模型 / Constant Velocity Model)
    ============================================================

    设计目标：
        对 SolvePnP 输出的 6DOF 目标姿态（平移 + 旋转）进行在线滤波与平滑。
        6 个自由度各自独立建模为匀速运动（位置 + 速度），共 12 维状态。

    状态向量 X (12x1):
        [x, vx, y, vy, z, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
        ┌───────────┬───────────┬──────────────┐
        │  平移部分  │  速度部分  │   语义说明    │
        ├───────────┼───────────┼──────────────┤
        │    x      │    vx     │ X 轴平移 + X 速度 │
        │    y      │    vy     │ Y 轴平移 + Y 速度 │
        │    z      │    vz     │ Z 轴平移 + Z 速度 │
        │   roll    │  vroll    │ 绕 X 轴旋转 + 角速度 │
        │  pitch    │  vpitch   │ 绕 Y 轴旋转 + 角速度 │
        │   yaw     │  vyaw     │ 绕 Z 轴旋转 + 角速度 │
        └───────────┴───────────┴──────────────┘
        - x, y, z: 平移量，来自 SolvePnP 的 tvec（相机坐标系）
        - roll, pitch, yaw: 旋转欧拉角（度），来自 SolvePnP 的 rvec → 欧拉角
        - v*: 各自由度的变化率（速度 / 角速度）

    观测向量 Z (6x1):
        [x_obs, y_obs, z_obs, roll_obs, pitch_obs, yaw_obs]
        - 来自 PoseDetector 的 SolvePnP 输出（每次检测直接可观测）

    状态转移矩阵 F (12x12):
        每个自由度独立建模为匀速运动，F 为块对角形式：
            F_block = [[1, dt],
                       [0,  1]]
        即：位置 = 上一位置 + 速度·dt，速度保持不变（匀速假设）
        完整 F 由 6 个这样的 2x2 块沿对角线排列构成。

    过程噪声 Q (12x12):
        对匀速模型，速度的随机扰动（加速度积分）引入过程噪声。
        Q_block = σ²_q · [[dt⁴/4, dt³/2],
                          [dt³/2, dt² ]]
        其中 σ²_q 是可调的过程噪声谱密度 (s2q_...)。

    观测矩阵 H (6x12):
        H = [I₆ | 0₆]，即只直接观测位置/角度分量，速度分量不可直接观测。
        H[i, i*2] = 1.0  —— 第 i 个自由度的位置 → 第 i 个观测量

    观测噪声 R (6x6):
        对角线矩阵，噪声随目标距离自适应缩放：
        - 平移观测噪声 ∝ 距离 (z深度)，因为远距离的 SolvePnP 平移精度下降
        - 旋转观测噪声 ∝ max(1, 距离/50)，远距离旋转也更不可靠
    """

    # ============================================================
    # 构造器
    # ============================================================
    def __init__(self):
        # ---- 状态向量 X: 12x1，全零初始化 ----
        # 初始状态待 init_state() 赋予第一个有效观测值
        self.X = np.zeros(12, dtype=np.float64)

        # ---- 协方差矩阵 P: 12x12，对观测值高置信，速度分量高不确定 ----
        # 初始协方差表示我们对初始估计的信任程度
        #   - 位置/角度的方差 = 1.0  （对第一个观测值比较信任）
        #   - 速度的方差     = 50.0  （初始不知道速度，非常不确定）
        self.P = np.eye(12, dtype=np.float64)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0      # 偶数索引：位置/角度 → 低不确定性
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0     # 奇数索引：速度 → 高不确定性

        # ---- 过程噪声谱密度（可调参数） ----
        # s2q_pos: 平移加速度的谱密度，决定了预测时平移位置的不确定性增长速度
        # s2q_rot: 旋转角加速度的谱密度，决定了预测时旋转角度的不确定性增长速度
        # 旋转的 s2q 默认比平移大 10 倍，因为欧拉角抖动更剧烈
        self.s2q_pos = 10.0
        self.s2q_rot = 100.0

        # ---- 观测噪声系数（可调参数） ----
        # r_pos_factor: 平移观测噪声因子，乘以距离得到对角线噪声
        # r_rot_factor: 旋转观测噪声基准值，再乘以距离因子
        self.r_pos_factor = 0.01
        self.r_rot_factor = 0.5

        # ---- 预分配矩阵内存，避免每次 predict/update 重复分配 ----
        self.F = np.eye(12, dtype=np.float64)     # 状态转移矩阵
        self.Q = np.zeros((12, 12), dtype=np.float64)  # 过程噪声协方差
        self.H = np.zeros((6, 12), dtype=np.float64)   # 观测矩阵 H[i, i*2]=1
        self.R = np.zeros((6, 6), dtype=np.float64)    # 观测噪声协方差
        self.I = np.eye(12, dtype=np.float64)          # 单位矩阵（缓存复用）

        # ---- 初始化观测矩阵 H：只观测位置/角度，不观测速度 ----
        for i in range(6):
            self.H[i, i * 2] = 1.0  # H[i, 2i] = 1，其余为 0

        # ---- Numba JIT 预热：避免首次调用时的编译延迟 ----
        # 用 dummy 数据调用一次 predict 和 update，触发 Numba 编译
        # 实际运行时的第一次调用就能享受 JIT 加速
        dummy_X = np.zeros(12, dtype=np.float64)
        dummy_P = np.eye(12, dtype=np.float64)
        dummy_F = np.eye(12, dtype=np.float64)
        dummy_Q = np.zeros((12, 12), dtype=np.float64)
        dummy_H = np.zeros((6, 12), dtype=np.float64)
        dummy_R = np.eye(6, dtype=np.float64)
        dummy_Y = np.zeros(6, dtype=np.float64)
        dummy_I = np.eye(12, dtype=np.float64)

        _fast_ekf_predict(dummy_X, dummy_P, dummy_F, dummy_Q)
        _fast_ekf_update_inplace(dummy_X, dummy_P, dummy_H, dummy_R, dummy_Y, dummy_I)

    # ============================================================
    # 设置噪声参数
    # ============================================================
    def set_noise(self, q_pos=0.001, q_rot=0.1, r_pos_factor=0.01, r_rot_factor=0.5):
        """
        动态调整卡尔曼滤波器的噪声参数，用于在线调参。

        参数:
            q_pos (float): 平移过程噪声谱密度 σ²_q_pos
                越大 → 滤波器更信任观测值，平滑度降低
                越小 → 滤波器更信任匀速模型，平滑度提高但响应变慢
            q_rot (float): 旋转过程噪声谱密度 σ²_q_rot
                含义同上，作用于 roll/pitch/yaw
            r_pos_factor (float): 平移观测噪声因子
                越大 → 观测噪声越大 → 滤波器更信任预测，平滑度提高
                越小 → 观测噪声越小 → 滤波器更信任原始观测，响应更快
            r_rot_factor (float): 旋转观测噪声基准值
                含义同上，作用于 roll/pitch/yaw 观测
        """
        self.s2q_pos = q_pos
        self.s2q_rot = q_rot
        self.r_pos_factor = r_pos_factor
        self.r_rot_factor = r_rot_factor

    # ============================================================
    # 状态初始化
    # ============================================================
    def init_state(self, x, y, z, roll, pitch, yaw):
        """
        用首次检测结果初始化滤波器状态。

        将第一次 SolvePnP 的观测值作为初始位置/角度，
        速度初始化为 0（在不知道初速度时，这是最佳无偏估计）。
        协方差矩阵重置，对位置高信任、对速度不确定。

        参数:
            x, y, z:     平移量（tvec）
            roll, pitch, yaw: 旋转欧拉角（度）
        """
        # 位置取自观测，速度全部设为 0
        self.X[:] = [x, 0.0, y, 0.0, z, 0.0, roll, 0.0, pitch, 0.0, yaw, 0.0]

        # 协方差矩阵重置为初始值
        self.P.fill(0.0)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0      # 位置/角度 → 低不确定性
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0     # 速度 → 高不确定性

    # ============================================================
    # 预测步骤
    # ============================================================
    def predict(self, dt):
        """
        卡尔曼滤波预测步骤：根据时间间隔 dt 将状态外推到当前时刻。

        1. 构建状态转移矩阵 F（匀速运动模型）
           对于每个自由度 (i = x, y, z, roll, pitch, yaw)：
               position_{k} = position_{k-1} + velocity_{k-1} · dt
               velocity_{k} = velocity_{k-1}                （不变）

        2. 构建过程噪声协方差 Q（积分白噪声模型）
           假设速度变化由零均值高斯白噪声加速度驱动，
           Q 中的子块为：
               [σ²·dt⁴/4  σ²·dt³/2]
               [σ²·dt³/2  σ²·dt²  ]
           解释：位置的不确定性 ∝ dt⁴（两个积分），速度的不确定性 ∝ dt²（一个积分），
           交叉项 ∝ dt³。

        3. 调用 Numba 编译的预测函数，原地更新 X 和 P。

        参数:
            dt (float): 自上次 predict/update 以来的时间间隔（秒）
                        dt 越小 → 预测越准，不确���性增长越少
                        帧率越高 dt 越小，滤波效果越好

        返回:
            np.ndarray: 预测后的状态向量 X (12,)
        """
        # ---- 构建状态转移矩阵 F ----
        # F 是一个 12x12 的块对角矩阵，每个 2x2 块为 [[1, dt], [0, 1]]
        self.F.fill(0)
        for i in range(6):
            base = i * 2
            self.F[base, base] = 1.0           # 位置 = 1·位置_旧
            self.F[base, base + 1] = dt        #        + dt·速度_旧
            self.F[base + 1, base + 1] = 1.0   # 速度 = 1·速度_旧

        # ---- 构建过程噪声协方差 Q ----
        # Q 同样为块对角结构，6 个独立的 2x2 子块
        self.Q.fill(0)
        t2 = dt ** 2   # dt²
        t3 = dt ** 3   # dt³
        t4 = dt ** 4   # dt⁴

        # 前 3 个自由度 (x, y, z): 使用平移过程噪声谱密度 s2q_pos
        for i in range(3):
            idx = i * 2                                      # 子块在矩阵中的起始索引
            self.Q[idx, idx]         = t4 / 4 * self.s2q_pos  # 位置方差 = σ²·dt⁴/4
            self.Q[idx + 1, idx]     = t3 / 2 * self.s2q_pos  # 速度-位置协方差 = σ²·dt³/2
            self.Q[idx, idx + 1]     = t3 / 2 * self.s2q_pos  # 位置-速度协方差 = σ²·dt³/2
            self.Q[idx + 1, idx + 1] = t2     * self.s2q_pos  # 速度方差 = σ²·dt²

        # 后 3 个自由度 (roll, pitch, yaw): 使用旋转过程噪声谱密度 s2q_rot
        for i in range(3, 6):
            idx = i * 2
            self.Q[idx, idx]         = t4 / 4 * self.s2q_rot
            self.Q[idx + 1, idx]     = t3 / 2 * self.s2q_rot
            self.Q[idx, idx + 1]     = t3 / 2 * self.s2q_rot
            self.Q[idx + 1, idx + 1] = t2     * self.s2q_rot

        # ---- 执行预测 ----
        _fast_ekf_predict(self.X, self.P, self.F, self.Q)
        return self.X

    # ============================================================
    # 更新步骤
    # ============================================================
    def update(self, measurement):
        """
        卡尔曼滤波更新步骤：用新观测值修正预测状态。

        1. 构建距离自适应的观测噪声协方差 R
           - 平移噪声随深度 z 线性增大（远距离 SolvePnP 精度差）
           - 旋转噪声也随距离增大（但增速较缓，50mm 起算）
           这确保了在目标靠近时更信任观测，远离时更信任模型预测。

        2. 计算新息 Y = Z - H·X_pred
           新息反映了观测与预测的偏差程度。

        3. 调用 Numba 编译的更新函数，内含马氏距离门控 (gating)：
           如果马氏距离 > sqrt(200) ≈ 14.14，则视为野值/误匹配，
           直接丢弃此次更新（返回但状态不变）。

        参数:
            measurement (list/tuple/array): 6 元素观测值
                [x_obs, y_obs, z_obs, roll_obs, pitch_obs, yaw_obs]

        返回:
            np.ndarray: 更新后的状态向量 X (12,)
        """
        Z = np.array(measurement, dtype=np.float64)

        # ---- 构建观测噪声协方差 R（距离自适应） ----
        self.R.fill(0)

        # 用 z 深度作为距离的近似（实际可用 sqrt(x²+y²+z²)，但 z 占主导时近似即可）
        obs_x, obs_y, obs_z = Z[0], Z[1], Z[2]
        dist = abs(obs_z) + 1e-6       # 避免除零

        # 平移观测噪声：基准噪声 + 随距离线性增长项
        # base_pos = r_pos_factor * dist + 0.01
        #   - r_pos_factor * dist: 距离越远，平移不确定度越大
        #   - +0.01: 最小噪声基底，防止观测噪声过小导致滤波器不稳定（过信任有噪观测）
        base_pos = self.r_pos_factor * dist + 0.01
        self.R[0, 0] = base_pos
        self.R[1, 1] = base_pos
        self.R[2, 2] = base_pos

        # 旋转观测噪声：也随距离增大
        # 旋转噪声 = base_rot * max(1.0, dist/50.0)
        #   - 50mm 以内时，旋转噪声 = base_rot（常数）
        #   - 超过 50mm 后，噪声线性增长
        # 这是因为 SolvePnP 的旋转估计对距离不太敏感，但远距离仍然会恶化
        base_rot = self.r_rot_factor
        self.R[3, 3] = base_rot * max(1.0, abs(dist) / 50.0)
        self.R[4, 4] = base_rot * max(1.0, abs(dist) / 50.0)
        self.R[5, 5] = base_rot * max(1.0, abs(dist) / 50.0)

        # ---- 计算新息 (Innovation) Y = Z - H·X ----
        Z_pred = self.H @ self.X       # 从状态空间映射到观测空间
        Y = Z - Z_pred                 # 新息：观测值与预测值的差异

        # ---- 执行更新（含马氏距离门控） ----
        _fast_ekf_update_inplace(self.X, self.P, self.H, self.R, Y, self.I)
        return self.X

    # ============================================================
    # 结果查询接口
    # ============================================================

    def get_filtered_pos(self):
        """
        获取滤波后的平移量 (x, y, z)。

        返回:
            tuple[float, float, float]: 滤波后 X[0]=x, X[2]=y, X[4]=z
            （注意状态向量中位置在偶数索引 0,2,4，速度在奇数索引 1,3,5）
        """
        return float(self.X[0]), float(self.X[2]), float(self.X[4])

    def get_filtered_rot(self):
        """
        获取滤波后的旋转欧拉角 (roll, pitch, yaw)。

        返回:
            tuple[float, float, float]: 滤波后 X[6]=roll, X[8]=pitch, X[10]=yaw
        """
        return float(self.X[6]), float(self.X[8]), float(self.X[10])

    def get_filtered_vel(self):
        """
        获取滤波后的平移速度 (vx, vy, vz)。

        返回:
            tuple[float, float, float]: 滤波后 X[1]=vx, X[3]=vy, X[5]=vz
            （注意此处只返回平移速度，未返回角速度）
        """
        return float(self.X[1]), float(self.X[3]), float(self.X[5])

    def smooth_reset_covariance(self):
        """
        协方差平滑重置：对位置/角度分量的小幅膨胀。

        当滤波器状态可能偏离真实值（如长时间未更新、遮挡恢复后），
        调用此函数增加状态不确定性，让后续观测有更大的修正权重，
        帮助滤波器从可能的状态漂移中快速恢复。

        机制：
            P[i,i] += 0.05  对每个位置/角度分量 (i = 0,2,4,6,8,10)
            小幅增加对角协方差，相当于"降低对当前状态估计的信心"。
        """
        for i in range(0, 12, 2):
            self.P[i, i] += 0.05