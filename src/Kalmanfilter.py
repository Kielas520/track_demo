import time
import cv2
import numpy as np

class KalmanFilter3D:
    def __init__(self, predict_mode="hybrid", predict_time=0.1,
                 q_x=1e-2, q_y=1e-2, q_z=1e-2, 
                 q_vx=1e-2, q_vy=1e-2, q_vz=1e-2,
                 r_x=1e-1, r_y=1e-1, r_z=1e-1):
        # 6个状态变量 (x, y, z, vx, vy, vz)，3个测量变量 (x, y, z)
        self.kf = cv2.KalmanFilter(6, 3)
        self.set_params()
        self.predict_mode = predict_mode
        self.predict_time = predict_time
        self.system_delay = 0.0
        self.last_tick = time.time()
        
        # 噪声参数
        self.q_x, self.q_y, self.q_z = q_x, q_y, q_z
        self.q_vx, self.q_vy, self.q_vz = q_vx, q_vy, q_vz
        self.r_x, self.r_y, self.r_z = r_x, r_y, r_z
        self._apply_qr()
        
        self.last_raw_x = None
        self.last_raw_y = None
        self.last_raw_z = None

    def set_params(self):
        # 状态转移矩阵 F (6x6)
        self.kf.transitionMatrix = np.array([
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1]
        ], np.float32)
        
        # 测量矩阵 H (3x6)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0]
        ], np.float32)
        
        self.kf.errorCovPost = np.eye(6, dtype=np.float32) * 1.0

    def _apply_qr(self):
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

    def init(self, x, y, z):
        self.kf.statePost = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)], 
                                      [0.0], [0.0], [0.0]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()

    def predict(self, dt):
        """推进状态，更新动态 dt"""
        self.kf.transitionMatrix[0, 3] = np.float32(dt)
        self.kf.transitionMatrix[1, 4] = np.float32(dt)
        self.kf.transitionMatrix[2, 5] = np.float32(dt)
        self.kf.predict()

    def update(self, x, y, z):
        measured = np.array([[np.float32(x)], [np.float32(y)], [np.float32(z)]])
        self.kf.correct(measured)
        self.last_raw_x = x
        self.last_raw_y = y
        self.last_raw_z = z

    def tick(self):
        current_time = time.time()
        self.system_delay = current_time - self.last_tick
        self.last_tick = current_time
        return self.system_delay

    def get_predict_dt(self):
        if self.predict_mode == "manual":
            return self.predict_time
        elif self.predict_mode == "auto":
            return self.system_delay
        elif self.predict_mode == "hybrid":
            return self.predict_time + self.system_delay
        else:
            return 0.0

    def get_filtered_pos(self):
        state = self.kf.statePost
        return int(state[0, 0]), int(state[1, 0]), int(state[2, 0])

    def predict_future(self, future_time=None):
        if future_time is None:
            future_time = self.get_predict_dt()

        if future_time == 0 and self.last_raw_x is not None:
            return int(self.last_raw_x), int(self.last_raw_y), int(self.last_raw_z)

        state = self.kf.statePost
        x, y, z = state[0, 0], state[1, 0], state[2, 0]
        vx, vy, vz = state[3, 0], state[4, 0], state[5, 0]

        pred_x = x + vx * future_time
        pred_y = y + vy * future_time
        pred_z = z + vz * future_time

        return int(pred_x), int(pred_y), int(pred_z)

    def set_mode(self, mode):
        self.predict_mode = mode

    def set_predict_time(self, t):
        self.predict_time = t