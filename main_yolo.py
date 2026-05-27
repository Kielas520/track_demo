import cv2
import numpy as np
from src.target import Target
from src.tracker import TrackerManager
from src.Kalmanfilter import KalmanFilter3D
from src.detector import Detector


def nothing(x):
    """OpenCV trackbar 空回调，trackbar 必须关联一个回调函数"""
    pass


def main():
    # ======================== 1. 摄像头 & 窗口初始化 ========================

    cap = cv2.VideoCapture(0)             
    
    # 统一的合并窗口
    window_name = "Unified Tracking System"
    # 使用 WINDOW_NORMAL 允许调整窗口大小，防止双画面并排导致窗口过宽超出屏幕
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # ==================== 2. YOLO 参数滑动条 (带 [YOLO] 前缀) =================
    
    cv2.createTrackbar('[YOLO] Conf(%)', window_name, 50, 100, nothing)  
    cv2.createTrackbar('[YOLO] Font Scale', window_name, 13, 20, nothing)       

    # ==================== 3. 卡尔曼滤波器参数滑动条 (带 [MAIN] 前缀) ===========

    # --- 过程噪声 Q (位置部分) ---
    cv2.createTrackbar('[MAIN] Q_x', window_name, 600, 1000, nothing)   
    cv2.createTrackbar('[MAIN] Q_y', window_name, 600, 1000, nothing)   
    cv2.createTrackbar('[MAIN] Q_z', window_name, 600, 1000, nothing)   

    # --- 过程噪声 Q (速度部分) ---
    cv2.createTrackbar('[MAIN] Q_vx', window_name, 320, 1000, nothing)  
    cv2.createTrackbar('[MAIN] Q_vy', window_name, 320, 1000, nothing)  
    cv2.createTrackbar('[MAIN] Q_vz', window_name, 320, 1000, nothing)  

    # --- 测量噪声 R ---
    cv2.createTrackbar('[MAIN] R_x', window_name, 500, 1000, nothing)   
    cv2.createTrackbar('[MAIN] R_y', window_name, 500, 1000, nothing)   
    cv2.createTrackbar('[MAIN] R_z', window_name, 500, 1000, nothing)   

    # --- 预测配置 ---
    cv2.createTrackbar('[MAIN] Mode(0:m 1:a 2:h)', window_name, 0, 2, nothing)  
    cv2.createTrackbar('[MAIN] PredictTime', window_name, 0, 500, nothing)      

    # ======================== 4. 组件初始化 ===============================

    detector = Detector(cls_id=39, device="mps", dis_mode=1)  
    target = Target()                                                      
    tracker = TrackerManager()                                             

    kf = KalmanFilter3D(predict_mode="manual", predict_time=0.0,
                 q_x=0.05, q_y=0.05, q_z=0.05,
                 q_vx=0.1, q_vy=0.1, q_vz=0.1,
                 r_x=5.0,  r_y=5.0,  r_z=5.0
                )

    # ======================== 5. 主循环 ===================================

    while True:
        # ---- 5.1 系统延迟计时 ----
        dt = kf.tick()                
        if dt == 0 or dt > 0.5:       
            dt = 0.033

        # ---- 5.2 读取摄像头帧 ----
        ret, frame = cap.read()
        if not ret:
            break

        # 保留用于 YOLO 视图的独立副本，避免在 YOLO 视图上画上跟踪器的内容
        frame_for_yolo = frame.copy()

        # ---- 5.3 读取所有滑动条值 ----

        font_scale = cv2.getTrackbarPos('[YOLO] Font Scale', window_name) / 10.0
        conf_thresh = cv2.getTrackbarPos('[YOLO] Conf(%)', window_name) / 100.0

        q_x = cv2.getTrackbarPos('[MAIN] Q_x', window_name) / 100.0
        q_y = cv2.getTrackbarPos('[MAIN] Q_y', window_name) / 100.0
        q_z = cv2.getTrackbarPos('[MAIN] Q_z', window_name) / 100.0
        q_vx = cv2.getTrackbarPos('[MAIN] Q_vx', window_name) / 100.0
        q_vy = cv2.getTrackbarPos('[MAIN] Q_vy', window_name) / 100.0
        q_vz = cv2.getTrackbarPos('[MAIN] Q_vz', window_name) / 100.0
        r_x = cv2.getTrackbarPos('[MAIN] R_x', window_name) / 100.0
        r_y = cv2.getTrackbarPos('[MAIN] R_y', window_name) / 100.0
        r_z = cv2.getTrackbarPos('[MAIN] R_z', window_name) / 100.0
        
        kf.set_qr(q_x=q_x, q_y=q_y, q_z=q_z,
                  q_vx=q_vx, q_vy=q_vy, q_vz=q_vz,
                  r_x=r_x, r_y=r_y, r_z=r_z)

        mode_idx = cv2.getTrackbarPos('[MAIN] Mode(0:m 1:a 2:h)', window_name)
        mode_map = {0: "manual", 1: "auto", 2: "hybrid"}
        kf.set_mode(mode_map[mode_idx])
        kf.set_predict_time(cv2.getTrackbarPos('[MAIN] PredictTime', window_name) / 100.0)

        # ---- 5.4 YOLO 目标检测 ----
        # 注意：需要确保 detector 内部是在未被污染的帧上绘制
        detections = detector.detect(frame_for_yolo)  

        candidate_bbox = None
        if detections:
            valid = [d for d in detections if d["conf"] >= conf_thresh]
            if valid:
                best = max(valid, key=lambda d: d["conf"])
                x1, y1, x2, y2 = best["box"]
                candidate_bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))  

        # ---- 5.5 键盘输入 ----
        key = cv2.waitKey(1) & 0xFF
        target_mode = None
        if key == ord('c'):
            target_mode = "CSRT"     
        if key == ord('k'):
            target_mode = "KCF"      

        # ---- 5.6 跟踪逻辑 ----

        if target.is_tracking:
            kf.predict(dt)
            success, bbox = tracker.update(frame)
            target.update_state(success, bbox)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2        
                z = max(0, w * h)  

                kf.update(cx, cy, z)

                filtered_x, filtered_y, filtered_z = kf.get_filtered_pos()
                future_x, future_y, future_z = kf.predict_future()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (filtered_x, filtered_y), 8, (255, 0, 0), -1)
                cv2.circle(frame, (future_x, future_y), 12, (0, 0, 255), 3)

                cv2.putText(frame,
                            f"Mode: {kf.predict_mode} | "
                            f"Delay: {dt*1000:.1f}ms | "
                            f"z:{filtered_z}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost! Searching...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 255), 2)

                if candidate_bbox:
                    tracker.start(frame, candidate_bbox, mode=target.mode)
                    target.start(candidate_bbox, mode=target.mode)

                    x_c, y_c, w_c, h_c = candidate_bbox
                    init_z = max(0, w_c * h_c)
                    kf.init(x_c + w_c // 2, y_c + h_c // 2, init_z)
        else:
            if candidate_bbox:
                x, y, w, h = candidate_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Found! 'c' for CSRT / 'k' for KCF",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 255, 255), 2)

            if target_mode and candidate_bbox:
                tracker.start(frame, candidate_bbox, mode=target_mode)
                target.start(candidate_bbox, mode=target_mode)

                x_c, y_c, w_c, h_c = candidate_bbox
                init_z = max(0, w_c * h_c)
                kf.init(x_c + w_c // 2, y_c + h_c // 2, init_z)

        # ---- 5.7 全局控制 ----
        if key == ord('r'):
            target.reset()
            tracker.reset()
        if key == ord('q'):
            break

        # ---- 5.8 画面合并与显示 ----
        
        yolo_view = detector.draw()  # 获取 YOLO 检测可视化结果
        
        # 若检测器 draw 方法未返回图像，默认使用当前帧备份
        if yolo_view is None:
            yolo_view = frame_for_yolo 
            
        # 确保两张图的高宽一致以进行横向拼接
        if yolo_view.shape != frame.shape:
            yolo_view = cv2.resize(yolo_view, (frame.shape[1], frame.shape[0]))

        # 左侧放 yolo_view，右侧放主视图 frame
        combined_frame = np.hstack((yolo_view, frame))
        
        # 在中心位置添加一条白线分隔左右视图
        h_comb, w_comb = combined_frame.shape[:2]
        mid_x = w_comb // 2
        cv2.line(combined_frame, (mid_x, 0), (mid_x, h_comb), (255, 255, 255), 2)
        
        # 显示合并后的图像
        cv2.imshow(window_name, combined_frame)         

    # ======================== 6. 清理 =====================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()