import cv2
import numpy as np

class KalmanFilter2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.set_params()

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

    def predict_future(self, future_time):
        """
        基于当前后验状态，前馈预测未来的位置
        future_time: 想要预测的未来总时间（秒）
        """
        state = self.kf.statePost
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        
        # 匀速直线运动模型
        pred_x = x + vx * future_time
        pred_y = y + vy * future_time
        
        return int(pred_x), int(pred_y)