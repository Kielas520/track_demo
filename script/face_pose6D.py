import cv2
import numpy as np
from src.detector.face_pose import FacePoseDetector, _euler_to_rvec
from src.Kalmanfilter.Kalmanfilter6D import KalmanFilter6D

# ============================================================
# KF 噪声参数控制面板 (保留原有的 6D 专属控制面板)
# ============================================================
CONTROL_WIN = "KF Noise Control"
PARAM_NAMES = ["q_pos", "q_rot", "r_pos_factor", "r_rot_factor", "mahalanobis_threshold"]
DEFAULT_VALUES = [10.0, 100.0, 0.01, 0.5, 200.0]
PANEL_W, PANEL_H = 480, 420


class NoisePanel:
    """用 cv2 窗口实现的可交互数值输入面板，用于动态调整卡尔曼噪声参数。"""
    def __init__(self):
        self.values = list(DEFAULT_VALUES)
        self.edit_bufs = [str(v) for v in DEFAULT_VALUES]
        self.selected = 0
        self.applied_values = list(DEFAULT_VALUES)
        self.pending_apply = False
        self._row_rects = []

    def get_values(self):
        return self.values

    def draw(self):
        img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)

        n = len(PARAM_NAMES)
        row_h = (PANEL_H - 30) // n
        for i, name in enumerate(PARAM_NAMES):
            y = 20 + (i + 1) * row_h
            if i == self.selected:
                cv2.rectangle(img, (5, y - row_h + 6), (PANEL_W - 5, y - 4), (60, 60, 60), -1)

            cv2.putText(img, name, (15, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
            label = f"applied: {self.applied_values[i]:.5g}"
            cv2.putText(img, label, (15, y - 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1)

            cursor = "|" if i == self.selected else ""
            buf_text = f"[ {self.edit_bufs[i]}{cursor} ]"
            color = (0, 220, 255) if i == self.selected else (200, 200, 200)
            cv2.putText(img, buf_text, (200, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        hint_y = PANEL_H - 8
        if self.pending_apply:
            cv2.putText(img, "ENTER: Apply & Restart KF | Click: select field",
                        (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        else:
            cv2.putText(img, "Click to select | Type value | ENTER to apply",
                        (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        self._row_rects = [(20 + i * row_h, 20 + (i + 1) * row_h) for i in range(n)]
        return img

    def handle_click(self, x, y):
        for i, (y0, y1) in enumerate(self._row_rects):
            if y0 <= y < y1:
                self.selected = i
                return

    def handle_key(self, key_char):
        if key_char == '\r':
            if self.pending_apply:
                self._apply()
                return True
            return False
        elif key_char == '\x1b':
            self.edit_bufs[self.selected] = str(self.applied_values[self.selected])
            self._check_pending()
            return False
        elif key_char == '\t':
            self.selected = (self.selected + 1) % len(PARAM_NAMES)
            return False
        elif key_char == '\x7f':
            if len(self.edit_bufs[self.selected]) > 0:
                self.edit_bufs[self.selected] = self.edit_bufs[self.selected][:-1]
            self._check_pending()
            return False
        elif key_char in '0123456789.-+eE':
            self.edit_bufs[self.selected] += key_char
            self._check_pending()
            return False
        return False

    def parse_and_validate(self):
        result = []
        for buf in self.edit_bufs:
            try:
                val = float(buf)
            except ValueError:
                return None
            result.append(val)
        return result

    def _check_pending(self):
        parsed = self.parse_and_validate()
        if parsed is None:
            self.pending_apply = False
            return
        self.values = parsed
        self.pending_apply = (parsed != self.applied_values)

    def _apply(self):
        self.applied_values = list(self.values)
        self.edit_bufs = [str(v) for v in self.values]
        self.pending_apply = False


def on_trackbar(val):
    pass


def _make_mouse_cb(panel):
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            panel.handle_click(x, y)
    return cb


# ============================================================
# 辅助投影函数：将 3D 坐标映射到 2D 像素平面用于画圈/连线
# ============================================================
def project_3d_to_2d(tx, ty, tz, cam_matrix, dist_coeffs):
    pts3d = np.array([[[tx, ty, tz]]], dtype=np.float64)
    # 假设位姿是在相机坐标系下，所以 rvec 和 tvec 均为 0
    rvec_zero = np.zeros((3, 1), dtype=np.float64)
    tvec_zero = np.zeros((3, 1), dtype=np.float64)
    pts2d, _ = cv2.projectPoints(pts3d, rvec_zero, tvec_zero, cam_matrix, dist_coeffs)
    return int(pts2d[0][0][0]), int(pts2d[0][0][1])


# ============================================================
# 主循环
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    window_name = "Face 6DOF Pose + KF Pipeline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # UI: 对齐 yolo2D 的预测参数滑动条
    cv2.createTrackbar("Font Scale", window_name, 6, 20, on_trackbar)
    cv2.createTrackbar('Mode(0:m 1:a 2:h)', window_name, 0, 2, on_trackbar)
    cv2.createTrackbar('PredictTime', window_name, 0, 500, on_trackbar)

    panel = NoisePanel()
    cv2.namedWindow(CONTROL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CONTROL_WIN, _make_mouse_cb(panel))
    cv2.imshow(CONTROL_WIN, panel.draw())

    detector = FacePoseDetector(dis_mode=1, max_faces=1)
    kf = KalmanFilter6D()
    kf_initialized = False

    while True:
        # ---- 1. 系统延迟计时 ----
        dt = kf.tick()
        if dt == 0 or dt > 0.5:
            dt = 0.033

        # ---- 2. 读取参数 ----
        mode_idx = cv2.getTrackbarPos('Mode(0:m 1:a 2:h)', window_name)
        mode_map = {0: "manual", 1: "auto", 2: "hybrid"}
        kf.set_mode(mode_map[mode_idx])
        kf.set_predict_time(cv2.getTrackbarPos('PredictTime', window_name) / 100.0)
        font_scale = max(0.3, cv2.getTrackbarPos("Font Scale", window_name) / 10.0)

        ret, frame = cap.read()
        if not ret:
            break

        # ---- 3. 人脸姿态检测 ----
        detections = detector.detect(frame)
        output = detector.draw()
        if output is None:
            output = frame.copy()

        cam_mat = detector.camera_matrix
        dist_coef = detector.dist_coeffs

        # ---- 4. 追踪与滤波逻辑 ----
        if detections and len(detections) > 0:
            face = detections[0]
            tx, ty, tz = face["tx"], face["ty"], face["tz"]
            roll, pitch, yaw = face["roll"], face["pitch"], face["yaw"]

            if not kf_initialized:
                kf.init_state(tx, ty, tz, roll, pitch, yaw)
                kf_initialized = True
            else:
                kf.predict(dt)
                success = kf.update([tx, ty, tz, roll, pitch, yaw])
                if not success:
                    # 被马氏距离拒绝，平滑增加协方差防止锁死
                    kf.smooth_reset_covariance()

            # 获取状态
            fil_x, fil_y, fil_z = kf.get_filtered_pos()
            fil_roll, fil_pitch, fil_yaw = kf.get_filtered_rot()

            (pred_x, pred_y, pred_z), (pred_roll, pred_pitch, pred_yaw) = kf.predict_future()

            # ---- 5. 投影与可视化 (按要求定制) ----
            # 投影 3D 坐标到 2D 像素
            raw_pt = project_3d_to_2d(tx, ty, tz, cam_mat, dist_coef)
            fil_pt = project_3d_to_2d(fil_x, fil_y, fil_z, cam_mat, dist_coef)
            pred_pt = project_3d_to_2d(pred_x, pred_y, pred_z, cam_mat, dist_coef)

            # 连线 (观测 -> 滤波 -> 预测)
            cv2.line(output, raw_pt, fil_pt, (200, 200, 200), 1, cv2.LINE_AA)
            cv2.line(output, fil_pt, pred_pt, (0, 255, 255), 2, cv2.LINE_AA)

            # A. 原始检测 (Raw) - 十字准星 + 坐标轴
            cv2.drawMarker(output, raw_pt, (255, 255, 0), cv2.MARKER_CROSS, 10, 2)
            raw_rvec = _euler_to_rvec(roll, pitch, yaw)
            raw_tvec = np.array([[tx], [ty], [tz]], dtype=np.float64)
            cv2.drawFrameAxes(output, cam_mat, dist_coef, raw_rvec, raw_tvec, length=1.0, thickness=2)

            # B. 滤波结果 (Filtered) - 仅白色实心圆圈 (不画坐标轴)
            cv2.circle(output, fil_pt, 5, (255, 255, 255), -1, cv2.LINE_AA)

            # C. 预测结果 (Predicted) - 黄色空心圆圈 + 斜十字 + 坐标轴
            cv2.circle(output, pred_pt, 7, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.drawMarker(output, pred_pt, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 7, 2)
            pred_rvec = _euler_to_rvec(pred_roll, pred_pitch, pred_yaw) # 预测的姿态(透传当前滤波值)
            pred_tvec = np.array([[pred_x], [pred_y], [pred_z]], dtype=np.float64)
            cv2.drawFrameAxes(output, cam_mat, dist_coef, pred_rvec, pred_tvec, length=1.3, thickness=2)

            # 状态文本
            status_text = f"Mode: {kf.predict_mode} | Delay: {dt*1000:.1f}ms | Frame: {kf.frame_count}"
            cv2.putText(output, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), max(1, int(font_scale*2)))

        else:
            # ---- 目标丢失处理 ----
            cv2.putText(output, "Lost! Searching...", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), max(1, int(font_scale*2)))
            kf_initialized = False  # 对齐 yolo2D 的逻辑，丢失后重置初始化标记


        cv2.imshow(window_name, output)
        cv2.imshow(CONTROL_WIN, panel.draw())

        # ---- 6. 按键与控制面板事件处理 ----
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        if key == 13:
            if panel.handle_key('\r'):
                q_pos, q_rot, r_pos, r_rot, maha = panel.applied_values
                kf.set_noise(q_pos=q_pos, q_rot=q_rot,
                             r_pos_factor=r_pos, r_rot_factor=r_rot,
                             mahalanobis_threshold=maha)
        elif key == 27:
            panel.handle_key('\x1b')
        elif key == 9:
            panel.handle_key('\t')
        elif key == 8 or key == 127:
            panel.handle_key('\x7f')
        elif key >= 32 and key < 127:
            panel.handle_key(chr(key))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()