import time

class Predictor:
    def __init__(self, mode="hybrid", predict_time=0.1):
        """
        :param mode: "manual" (手动), "auto" (自动), "hybrid" (混合)
        :param predict_time: 手动预测时间 (单位: 秒)
        """
        self.mode = mode
        self.predict_time = predict_time
        self.system_delay = 0.0
        self.last_tick = time.time()

    def tick(self):
        """放在主循环开始处，计算每帧的系统延时"""
        current_time = time.time()
        self.system_delay = current_time - self.last_tick
        self.last_tick = current_time
        return self.system_delay

    def get_predict_dt(self):
        """根据当前模式获取预测的总时间差"""
        if self.mode == "manual":
            return self.predict_time
        elif self.mode == "auto":
            return self.system_delay
        elif self.mode == "hybrid":
            return self.predict_time + self.system_delay
        else:
            return 0.0

    def predict(self, kf):
        """
        基于卡尔曼滤波器的后验状态进行预测
        状态向量为 [x, y, vx, vy]，单位需保证为 像素 和 像素/秒
        """
        state = kf.kf.statePost
        x, y, vx, vy = state[0, 0], state[1, 0], state[2, 0], state[3, 0]
        
        dt = self.get_predict_dt()
        
        # 匀速直线运动预测：预测位置 = 当前位置 + 速度 * 预测时间
        pred_x = x + vx * dt
        pred_y = y + vy * dt
        
        return int(pred_x), int(pred_y)

    def set_mode(self, mode):
        self.mode = mode

    def set_predict_time(self, t):
        self.predict_time = t