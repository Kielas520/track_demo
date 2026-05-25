import time
import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self, predict_mode="hybrid", predict_time=0.1):
        self.kf = cv2.KalmanFilter(4, 2)
        self.set_params()
        self.predict_mode = predict_mode
        self.predict_time = predict_time
        self.system_delay = 0.0
        self.last_tick = time.time()

    def set_params(self):
        # 初始 dt 占位，每次 predict 时会动态更新
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
        
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 1e-2
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1e-1
        self.kf.errorCovPost = np.eye(4, dtype=np.float32) * 1.0

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

    def predict_future(self, future_time=None):
        state = self.kf.statePost
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]

        if future_time is None:
            future_time = self.get_predict_dt()

        pred_x = x + vx * future_time
        pred_y = y + vy * future_time

        return int(pred_x), int(pred_y)

    def set_mode(self, mode):
        self.predict_mode = mode

    def set_predict_time(self, t):
        self.predict_time = t