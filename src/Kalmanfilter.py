import time
import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self, predict_mode="hybrid", predict_time=0.1,
                 q_x=1e-2, q_y=1e-2, q_vx=1e-2, q_vy=1e-2,
                 r_x=1e-1, r_y=1e-1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.set_params()
        self.predict_mode = predict_mode
        self.predict_time = predict_time
        self.system_delay = 0.0
        self.last_tick = time.time()
        self.q_x = q_x
        self.q_y = q_y
        self.q_vx = q_vx
        self.q_vy = q_vy
        self.r_x = r_x
        self.r_y = r_y
        self._apply_qr()
        self.last_raw_x = None
        self.last_raw_y = None

    def set_params(self):
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], np.float32)
        
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], np.float32)
        
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

    def _apply_qr(self):
        self.kf.processNoiseCov = np.array([
            [self.q_x, 0, 0, 0],
            [0, self.q_y, 0, 0],
            [0, 0, self.q_vx, 0],
            [0, 0, 0, self.q_vy]
        ], np.float32)
        self.kf.measurementNoiseCov = np.array([
            [self.r_x, 0],
            [0, self.r_y]
        ], np.float32)

    def set_qr(self, q_x=None, q_y=None, q_vx=None, q_vy=None, r_x=None, r_y=None):
        if q_x is not None: self.q_x = q_x
        if q_y is not None: self.q_y = q_y
        if q_vx is not None: self.q_vx = q_vx
        if q_vy is not None: self.q_vy = q_vy
        if r_x is not None: self.r_x = r_x
        if r_y is not None: self.r_y = r_y
        self._apply_qr()

    def init(self, x, y):
        self.kf.statePost = np.array([[np.float32(x)], 
                                      [np.float32(y)], 
                                      [0.0], 
                                      [0.0]], np.float32)
        self.kf.statePre = self.kf.statePost.copy()

    def predict(self, dt):
        """传入每帧真实的时间差，推进状态并保证速度单位是 像素/秒"""
        self.kf.transitionMatrix[0, 2] = np.float32(dt)
        self.kf.transitionMatrix[1, 3] = np.float32(dt)
        self.kf.predict()

    def update(self, x, y):
        measured = np.array([[np.float32(x)], [np.float32(y)]])
        self.kf.correct(measured)
        self.last_raw_x = x
        self.last_raw_y = y

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
        return int(state[0, 0]), int(state[1, 0])

    def predict_future(self, future_time=None):
        if future_time is None:
            future_time = self.get_predict_dt()

        if future_time == 0 and self.last_raw_x is not None:
            return int(self.last_raw_x), int(self.last_raw_y)

        state = self.kf.statePost
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]

        pred_x = x + vx * future_time
        pred_y = y + vy * future_time

        return int(pred_x), int(pred_y)

    def set_mode(self, mode):
        self.predict_mode = mode

    def set_predict_time(self, t):
        self.predict_time = t