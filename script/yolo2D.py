import cv2
import numpy as np
from src.target import Target
from src.tracker import TrackerManager
from src.Kalmanfilter import KalmanFilter3D
from src.detector.detector2d import Detector2D


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
    
    # --- 过程噪声 Q (位置部分，标准差：通常很小，仅代表非速度因素导致的位移) ---
    # 除以 100.0，范围 [0.0, 5.0]
    cv2.createTrackbar('[MAIN] Q_x', window_name, 50, 500, nothing)   # 默认 0.1
    cv2.createTrackbar('[MAIN] Q_y', window_name, 50, 500, nothing)   # 默认 0.1
    cv2.createTrackbar('[MAIN] Q_z', window_name, 50, 500, nothing)   # 默认 0.5

    # --- 过程噪声 Q (速度部分，标准差：代表物体运动加速度产生的不确定性) ---
    # 除以 1.0，范围 [0, 1000] 像素/秒
    cv2.createTrackbar('[MAIN] Q_vx', window_name, 250, 1000, nothing) # 默认 100
    cv2.createTrackbar('[MAIN] Q_vy', window_name, 250, 1000, nothing) # 默认 100
    cv2.createTrackbar('[MAIN] Q_vz', window_name, 500, 5000, nothing) # 默认 500 (面积变化率波动大)

    # --- 测量噪声 R (观测部分，标准差：代表 YOLO 检测框的抖动像素) ---
    # 除以 1.0，范围 x/y: [0, 50] 像素, z: [0, 5000] 面积像素
    cv2.createTrackbar('[MAIN] R_x', window_name, 2, 50, nothing)      # 默认 5 像素抖动
    cv2.createTrackbar('[MAIN] R_y', window_name, 2, 50, nothing)      # 默认 5 像素抖动
    cv2.createTrackbar('[MAIN] R_z', window_name, 500, 5000, nothing)  # 默认 500 面积抖动   

    # --- 预测配置 ---
    cv2.createTrackbar('[MAIN] Mode(0:m 1:a 2:h)', window_name, 0, 2, nothing)  
    cv2.createTrackbar('[MAIN] PredictTime', window_name, 0, 500, nothing)      

    # ======================== 4. 组件初始化 ===============================

    detector = Detector2D(cls_id=39, device="mps", dis_mode=1)  
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

        # Q位置 (缩放 100)
        q_x = cv2.getTrackbarPos('[MAIN] Q_x', window_name) / 100.0
        q_y = cv2.getTrackbarPos('[MAIN] Q_y', window_name) / 100.0
        q_z = cv2.getTrackbarPos('[MAIN] Q_z', window_name) / 100.0
        
        # Q速度 (无需缩放，代表 像素/秒)
        q_vx = float(cv2.getTrackbarPos('[MAIN] Q_vx', window_name))
        q_vy = float(cv2.getTrackbarPos('[MAIN] Q_vy', window_name))
        q_vz = float(cv2.getTrackbarPos('[MAIN] Q_vz', window_name))
        
        # R观测 (无需缩放，直接作为标准差)
        r_x = float(cv2.getTrackbarPos('[MAIN] R_x', window_name))
        r_y = float(cv2.getTrackbarPos('[MAIN] R_y', window_name))
        r_z = float(cv2.getTrackbarPos('[MAIN] R_z', window_name))
        
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
                # ---------- 跟踪成功 ----------
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2        # 观测值：检测到的目标中心点
                z = max(0, w * h)  

                # 【校正步】将观测值送入卡尔曼滤波器
                kf.update(cx, cy, z)

                filtered_x, filtered_y, filtered_z = kf.get_filtered_pos()
                future_x, future_y, future_z = kf.predict_future()

                # ====== 可视化 ======
                
                # 1. 计算动态缩放比例 dyn_scale (以 w+h=300 为基准 1.0)
                # 设置最小缩放比例为 0.3，避免目标过小时标尺完全看不见
                dyn_scale = max(0.3, (w + h) / 300.0)
                
                # 2. 根据 dyn_scale 动态计算所有的几何尺寸与粗细
                line_thick = max(1, int(2 * dyn_scale))
                cross_size = max(5, int(20 * dyn_scale))
                fil_radius = max(2, int(6 * dyn_scale))
                pred_radius = max(3, int(12 * dyn_scale))
                
                # 3. 颜色配置 (针对红色水瓶目标，使用高对比度的冷色或明亮色)
                color_bbox = (0, 255, 0)       # 绿色 (边界框)
                color_obs = (255, 255, 0)      # 青色 (观测点中心)
                color_fil = (255, 255, 255)    # 白色 (滤波点中心)
                color_pred = (0, 255, 255)     # 黄色 (预测点中心)

                # --- 绘制开始 ---
                
                # A. 边界框
                cv2.rectangle(frame, (x, y), (x + w, y + h), color_bbox, line_thick)

                # B. 状态趋势连接线 (观测 -> 滤波 -> 预测)
                # 使用细线连接，直观展示当前物体的运动趋势与滤波器的工作状态
                cv2.line(frame, (cx, cy), (filtered_x, filtered_y), (200, 200, 200), max(1, line_thick - 1), cv2.LINE_AA)
                cv2.line(frame, (filtered_x, filtered_y), (future_x, future_y), color_pred, line_thick, cv2.LINE_AA)

                # C. 观测值 (使用十字准星 cv2.MARKER_CROSS 替代方形)
                cv2.drawMarker(frame, (cx, cy), color_obs, markerType=cv2.MARKER_CROSS, markerSize=cross_size, thickness=line_thick)

                # D. 滤波值 (白色实心圆)
                cv2.circle(frame, (filtered_x, filtered_y), fil_radius, color_fil, -1, cv2.LINE_AA)

                # E. 预测值 (黄色空心圆 + 内部斜十字)
                cv2.circle(frame, (future_x, future_y), pred_radius, color_pred, line_thick, cv2.LINE_AA)
                cv2.drawMarker(frame, (future_x, future_y), color_pred, markerType=cv2.MARKER_TILTED_CROSS, markerSize=pred_radius, thickness=line_thick)

                # 状态文本输出 (文本粗细也跟随界面字体 scale 缩放)
                text_thick = max(1, int(2 * font_scale))
                cv2.putText(frame,
                            f"Mode: {kf.predict_mode} | Delay: {dt*1000:.1f}ms | z:{int(filtered_z)}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), text_thick)
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