import cv2
import time
import numpy as np
from src.detector.face_pose import FacePoseDetector, _euler_to_rvec
from src.Kalmanfilter.Kalmanfilter6D import KalmanFilter6D

def _draw_kf_axis(img, camera_matrix, dist_coeffs, rvec, tvec, length=80):
    """
    绘制 3D 坐标轴
    红色 (0, 0, 255): X轴 (通常指向面部右侧)
    绿色 (0, 255, 0): Y轴 (通常指向面部下方)
    蓝色 (255, 0, 0): Z轴 (通常指向面部正前方)
    """
    axis_3d = np.float32([
        [0, 0, 0],
        [length, 0, 0],
        [0, length, 0],
        [0, 0, length],
    ])
    pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, camera_matrix, dist_coeffs)
    pts = pts.reshape(-1, 2).astype(int)
    origin = tuple(pts[0])
    
    # OpenCV 颜色格式为 BGR
    cv2.line(img, origin, tuple(pts[1]), (0, 0, 255), 3)   # X - Red
    cv2.line(img, origin, tuple(pts[2]), (0, 255, 0), 3)   # Y - Green
    cv2.line(img, origin, tuple(pts[3]), (255, 0, 0), 3)   # Z - Blue
    cv2.circle(img, origin, 5, (255, 255, 255), -1)

def on_trackbar(val):
    pass

def main():
    cap = cv2.VideoCapture(0)
    window_name = "Face 6DOF Pose + KF (State Machine)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # 创建字体缩放滑动条，初始值 6 代表 font_scale = 0.6
    cv2.createTrackbar("Font Scale", window_name, 6, 20, on_trackbar)

    detector = FacePoseDetector(dis_mode=1, max_faces=1)
    kf = KalmanFilter6D()
    kf_initialized = False

    # ==========================================
    # 状态机 & 异常处理参数配置
    # ==========================================
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

        # 读取滑动条的值并限制最小缩放比例
        font_scale = max(0.3, cv2.getTrackbarPos("Font Scale", window_name) / 10.0)
        thickness = max(1, int(font_scale * 2))
        line_height = int(30 * font_scale)
        current_state = "UNKNOWN"

        # 用于记录当前帧的原始观测值，方便显示
        raw_pose = None 

        # ==========================================
        # 状态机流转逻辑
        # ==========================================
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

        # ==========================================
        # 渲染输出与调试信息
        # ==========================================
        y_offset = int(30 * font_scale)

        # 打印状态机信息
        color_state = (0, 255, 0) if current_state == "TRACKING" else (0, 165, 255)
        cv2.putText(output, f"State: {current_state}", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_state, thickness)
        y_offset += line_height

        # 打印耗时
        cv2.putText(output, f"dt: {dt*1000:.1f}ms", (10, y_offset), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (200, 200, 200), thickness)
        y_offset += line_height

        # 如果有原始观测，打印出来做对比
        if raw_pose:
            tx, ty, tz, r, p, y = raw_pose
            cv2.putText(output, f"Raw T: x={tx:.1f} y={ty:.1f} z={tz:.1f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 200), thickness)
            y_offset += line_height
            cv2.putText(output, f"Raw R: r={r:.1f} p={p:.1f} y={y:.1f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 200, 200), thickness)
            y_offset += line_height

        # 打印卡尔曼滤波输出，并绘制坐标轴
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
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()