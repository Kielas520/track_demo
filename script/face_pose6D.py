import cv2
import time
import numpy as np
from src.detector.face_pose import FacePoseDetector, _euler_to_rvec
from src.Kalmanfilter.Kalmanfilter6D import KalmanFilter6D


# ============================================================
# KF 噪声参数控制面板
# ============================================================
CONTROL_WIN = "KF Noise Control"
PARAM_NAMES = ["q_pos", "q_rot", "r_pos_factor", "r_rot_factor"]
DEFAULT_VALUES = [10.0, 100.0, 0.01, 0.5]
PANEL_W, PANEL_H = 480, 360


class NoisePanel:
    """用 cv2 窗口实现的可交互数值输入面板，用于动态调整卡尔曼噪声参数。"""

    def __init__(self):
        self.values = list(DEFAULT_VALUES)      # 当前生效的值
        self.edit_bufs = [str(v) for v in DEFAULT_VALUES]  # 编辑缓冲区（字符串）
        self.selected = 0                        # 当前选中行索引
        self.applied_values = list(DEFAULT_VALUES)
        self.pending_apply = False               # 是否有待应用的变更
        self._row_rects = []                     # 每行的点击区域 (y_start, y_end)

    def get_values(self):
        return self.values

    def draw(self):
        """绘制控制面板，返回 BGR 图像。"""
        img = np.zeros((PANEL_H, PANEL_W, 3), dtype=np.uint8)
        img[:] = (40, 40, 40)

        row_h = PANEL_H // 6
        for i, name in enumerate(PARAM_NAMES):
            y = (i + 1) * row_h
            # 高亮选中行
            if i == self.selected:
                cv2.rectangle(img, (5, y - row_h + 6), (PANEL_W - 5, y - 4), (60, 60, 60), -1)

            # 参数名
            cv2.putText(img, name, (15, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

            # 当前生效值
            label = f"applied: {self.applied_values[i]:.5g}"
            cv2.putText(img, label, (15, y - 36), cv2.FONT_HERSHEY_SIMPLEX,
                        0.45, (120, 120, 120), 1)

            # 编辑缓冲区（显示输入中的值）
            cursor = "|" if i == self.selected else ""
            buf_text = f"[ {self.edit_bufs[i]}{cursor} ]"
            color = (0, 220, 255) if i == self.selected else (200, 200, 200)
            cv2.putText(img, buf_text, (200, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

        # 底部提示
        hint_y = PANEL_H - 12
        if self.pending_apply:
            cv2.putText(img, "ENTER: Apply & Restart KF | Click: select field",
                        (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)
        else:
            cv2.putText(img, "Click to select | Type value | ENTER to apply",
                        (10, hint_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (140, 140, 140), 1)

        # 保存行区域供点击检测
        self._row_rects = [(i * row_h, (i + 1) * row_h) for i in range(len(PARAM_NAMES))]
        return img

    def handle_click(self, x, y):
        """处理鼠标点击，选中对应行。"""
        for i, (y0, y1) in enumerate(self._row_rects):
            if y0 <= y < y1:
                self.selected = i
                return

    def handle_key(self, key_char):
        """处理键盘输入，返回 True 表示触发了 apply + restart。"""
        if key_char == '\r':  # Enter
            if self.pending_apply:
                self._apply()
                return True
            return False
        elif key_char == '\x1b':  # Escape
            self.edit_bufs[self.selected] = str(self.applied_values[self.selected])
            self._check_pending()
            return False
        elif key_char == '\t':  # Tab
            self.selected = (self.selected + 1) % len(PARAM_NAMES)
            return False
        elif key_char == '\x7f':  # Backspace
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
        """尝试解析编辑缓冲区为浮点数，返回 list[float] 或 None。"""
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


# ============================================================
# 绘制函数
# ============================================================
def _draw_kf_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=80):
    axis_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ])
    pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = pts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])

    cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 3)
    cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 3)
    cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 3)
    cv2.circle(img, origin, 5, (255, 255, 255), -1)


def on_trackbar(val):
    pass


# ============================================================
# 鼠标回调：将点击转发给 NoisePanel
# ============================================================
def _make_mouse_cb(panel):
    def cb(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            panel.handle_click(x, y)
    return cb


# ============================================================
# 主循环
# ============================================================
def main():
    cap = cv2.VideoCapture(0)
    window_name = "Face 6DOF Pose + KF (State Machine)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("Font Scale", window_name, 6, 20, on_trackbar)

    # ---- 控制面板窗口 ----
    panel = NoisePanel()
    cv2.namedWindow(CONTROL_WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(CONTROL_WIN, _make_mouse_cb(panel))
    panel_img = panel.draw()
    cv2.imshow(CONTROL_WIN, panel_img)

    detector = FacePoseDetector(dis_mode=1, max_faces=1)
    kf = KalmanFilter6D()
    kf_initialized = False

    lost_timeout_sec = 0.5
    max_jump_dist = 100.0
    last_obs_time = 0.0
    reject_count = 0

    fx, fy, fz = 0.0, 0.0, 0.0
    froll, fpitch, fyaw = 0.0, 0.0, 0.0

    prev_tick = cv2.getTickCount()
    tick_freq = cv2.getTickFrequency()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cur_tick = cv2.getTickCount()
        dt = (cur_tick - prev_tick) / tick_freq
        prev_tick = cur_tick
        dt = max(dt, 0.001)
        now = time.time()

        detections = detector.detect(frame)
        output = detector.draw()
        if output is None:
            output = frame.copy()

        font_scale = max(0.3, cv2.getTrackbarPos("Font Scale", window_name) / 10.0)
        thickness = max(1, int(font_scale * 2))
        line_height = int(30 * font_scale)
        current_state = "UNKNOWN"
        raw_pose = None

        if detections and len(detections) > 0:
            face = detections[0]
            tx, ty, tz = face["tx"], face["ty"], face["tz"]
            roll, pitch, yaw = face["roll"], face["pitch"], face["yaw"]
            z_obs = np.array([tx, ty, tz])
            raw_pose = (tx, ty, tz, roll, pitch, yaw)

            if not kf_initialized:
                kf.init_state(tx, ty, tz, roll, pitch, yaw)
                kf_initialized = True
                last_obs_time = now
                reject_count = 0
                current_state = "INIT"
            else:
                fx, fy, fz = kf.get_filtered_pos()
                dist = np.linalg.norm(z_obs - np.array([fx, fy, fz]))

                if dist > max_jump_dist:
                    kf.init_state(tx, ty, tz, roll, pitch, yaw)
                    reject_count = 0
                    current_state = f"JUMP RESET ({dist:.1f}mm)"
                else:
                    kf.predict(dt)
                    is_updated = True
                    if hasattr(kf, 'update_with_status'):
                        is_updated = kf.update_with_status([tx, ty, tz, roll, pitch, yaw])
                    else:
                        kf.update([tx, ty, tz, roll, pitch, yaw])

                    if not is_updated:
                        reject_count += 1
                        current_state = f"REJECTED ({reject_count})"
                        if reject_count > 5:
                            kf.init_state(tx, ty, tz, roll, pitch, yaw)
                            reject_count = 0
                            current_state = "FORCED RESET"
                    else:
                        reject_count = 0
                        current_state = "TRACKING"

            last_obs_time = now
        else:
            if kf_initialized:
                time_since_obs = now - last_obs_time
                if time_since_obs > lost_timeout_sec:
                    kf_initialized = False
                    current_state = "TIMEOUT (LOST)"
                else:
                    kf.predict(dt)
                    current_state = "BLIND PREDICT"

        # ---- 渲染主窗口 ----
        y_offset = int(30 * font_scale)

        color_state = (0, 255, 0) if current_state == "TRACKING" else (0, 165, 255)
        cv2.putText(output, f"State: {current_state}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_state, thickness)
        y_offset += line_height

        cv2.putText(output, f"dt: {dt*1000:.1f}ms", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)
        y_offset += line_height

        if raw_pose:
            tx, ty, tz, r, p, y = raw_pose
            cv2.putText(output, f"Raw T: x={tx:.1f} y={ty:.1f} z={tz:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 200), thickness)
            y_offset += line_height
            cv2.putText(output, f"Raw R: r={r:.1f} p={p:.1f} y={y:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 200), thickness)
            y_offset += line_height

        if kf_initialized:
            fx, fy, fz = kf.get_filtered_pos()
            froll, fpitch, fyaw = kf.get_filtered_rot()

            cv2.putText(output, f"KF  T: x={fx:.1f} y={fy:.1f} z={fz:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 255, 200), thickness)
            y_offset += line_height
            cv2.putText(output, f"KF  R: r={froll:.1f} p={fpitch:.1f} y={fyaw:.1f}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 255, 200), thickness)

            rvec_kf = _euler_to_rvec(froll, fpitch, fyaw)
            tvec_kf = np.array([[fx], [fy], [fz]], dtype=np.float64)
            _draw_kf_axis(output, detector.camera_matrix,
                          detector.dist_coeffs, rvec_kf, tvec_kf, length=80)

        cv2.imshow(window_name, output)

        # ---- 渲染控制面板 ----
        panel_img = panel.draw()
        cv2.imshow(CONTROL_WIN, panel_img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

        # ---- 键盘事件分发到控制面板 ----
        if key == 13:  # Enter
            if panel.handle_key('\r'):
                # 应用新参数并重启卡尔曼
                q_pos, q_rot, r_pos, r_rot = panel.applied_values
                kf.set_noise(q_pos=q_pos, q_rot=q_rot,
                             r_pos_factor=r_pos, r_rot_factor=r_rot)
                # 如果有当前观测值，用它重新初始化
                if raw_pose:
                    kf.init_state(*raw_pose)
                print(f"[NoisePanel] Applied: q_pos={q_pos}, q_rot={q_rot}, "
                      f"r_pos={r_pos}, r_rot={r_rot}  (KF restarted)")
        elif key == 27:  # Escape
            panel.handle_key('\x1b')
        elif key == 9:   # Tab
            panel.handle_key('\t')
        elif key == 8 or key == 127:  # Backspace
            panel.handle_key('\x7f')
        elif key >= 32 and key < 127:
            panel.handle_key(chr(key))

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
