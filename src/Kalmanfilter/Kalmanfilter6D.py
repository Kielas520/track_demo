import numpy as np
from numba import njit

@njit(fastmath=True, nogil=True)
def _fast_ekf_predict(X, P, F, Q):
    X[:] = F @ X
    P[:] = F @ P @ F.T + Q

@njit(fastmath=True, nogil=True)
def _fast_ekf_update_inplace(X, P, H, R, Y, I):
    S = H @ P @ H.T + R
    S_inv_Y = np.linalg.solve(S, Y)
    mahalanobis_sq = np.dot(Y, S_inv_Y)
    if mahalanobis_sq > 200.0:
        return False
    K = np.linalg.solve(S.T, (P @ H.T).T).T
    X[:] = X + K @ Y
    I_KH = I - K @ H
    P[:] = I_KH @ P @ I_KH.T + K @ R @ K.T
    return True


class KalmanFilter6D:
    """
    6DOF 卡尔曼滤波器 (CV 匀速模型)

    状态向量 (12x1):
        [x, vx, y, vy, z, vz, roll, vroll, pitch, vpitch, yaw, vyaw]
        - x, y, z:       平移 (SolvePnP tvec)
        - roll, pitch, yaw: 旋转角 (度)
        - v*: 对应速度

    观测向量 (6x1):
        [x, y, z, roll, pitch, yaw]
        - 来自 PoseDetector 的 SolvePnP 输出
    """

    def __init__(self):
        self.X = np.zeros(12, dtype=np.float64)

        self.P = np.eye(12, dtype=np.float64)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0

        self.s2q_pos = 10.0
        self.s2q_rot = 100.0
        self.r_pos_factor = 0.01
        self.r_rot_factor = 0.5

        self.F = np.eye(12, dtype=np.float64)
        self.Q = np.zeros((12, 12), dtype=np.float64)
        self.H = np.zeros((6, 12), dtype=np.float64)
        self.R = np.zeros((6, 6), dtype=np.float64)
        self.I = np.eye(12, dtype=np.float64)

        for i in range(6):
            self.H[i, i * 2] = 1.0

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

    def set_noise(self, q_pos=0.001, q_rot=0.1, r_pos_factor=0.01, r_rot_factor=0.5):
        self.s2q_pos = q_pos
        self.s2q_rot = q_rot
        self.r_pos_factor = r_pos_factor
        self.r_rot_factor = r_rot_factor

    def init_state(self, x, y, z, roll, pitch, yaw):
        self.X[:] = [x, 0.0, y, 0.0, z, 0.0, roll, 0.0, pitch, 0.0, yaw, 0.0]

        self.P.fill(0.0)
        for i in range(0, 12, 2):
            self.P[i, i] = 1.0
        for i in range(1, 12, 2):
            self.P[i, i] = 50.0

    def predict(self, dt):
        self.F.fill(0)
        for i in range(6):
            self.F[i * 2, i * 2] = 1.0
            self.F[i * 2, i * 2 + 1] = dt
            self.F[i * 2 + 1, i * 2 + 1] = 1.0

        self.Q.fill(0)
        t2 = dt ** 2
        t3 = dt ** 3
        t4 = dt ** 4

        for i in range(3):
            idx = i * 2
            self.Q[idx, idx] = t4 / 4 * self.s2q_pos
            self.Q[idx + 1, idx] = t3 / 2 * self.s2q_pos
            self.Q[idx, idx + 1] = t3 / 2 * self.s2q_pos
            self.Q[idx + 1, idx + 1] = t2 * self.s2q_pos

        for i in range(3, 6):
            idx = i * 2
            self.Q[idx, idx] = t4 / 4 * self.s2q_rot
            self.Q[idx + 1, idx] = t3 / 2 * self.s2q_rot
            self.Q[idx, idx + 1] = t3 / 2 * self.s2q_rot
            self.Q[idx + 1, idx + 1] = t2 * self.s2q_rot

        _fast_ekf_predict(self.X, self.P, self.F, self.Q)
        return self.X

    def update(self, measurement):
        Z = np.array(measurement, dtype=np.float64)

        self.R.fill(0)
        obs_x, obs_y, obs_z = Z[0], Z[1], Z[2]
        dist = abs(obs_z) + 1e-6
        base_pos = self.r_pos_factor * dist + 0.01
        self.R[0, 0] = base_pos
        self.R[1, 1] = base_pos
        self.R[2, 2] = base_pos

        base_rot = self.r_rot_factor
        # 距离越远，旋转观测越不可靠
        self.R[3, 3] = base_rot * max(1.0, abs(dist) / 50.0)
        self.R[4, 4] = base_rot * max(1.0, abs(dist) / 50.0)
        self.R[5, 5] = base_rot * max(1.0, abs(dist) / 50.0)

        Z_pred = self.H @ self.X
        Y = Z - Z_pred

        _fast_ekf_update_inplace(self.X, self.P, self.H, self.R, Y, self.I)
        return self.X

    def get_filtered_pos(self):
        return float(self.X[0]), float(self.X[2]), float(self.X[4])

    def get_filtered_rot(self):
        return float(self.X[6]), float(self.X[8]), float(self.X[10])

    def get_filtered_vel(self):
        return float(self.X[1]), float(self.X[3]), float(self.X[5])

    def smooth_reset_covariance(self):
        for i in range(0, 12, 2):
            self.P[i, i] += 0.05