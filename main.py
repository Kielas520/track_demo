import cv2
import numpy as np
from src.target import Target
from src.tracker import TrackerManager
from src.Kalmanfilter import KalmanFilter2D


def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    main_window_name = "Tracker Main"
    mask_window_name = "Mask & Controls"
    cv2.namedWindow(main_window_name)
    cv2.namedWindow(mask_window_name)

    # 阈值滑动条...
    cv2.createTrackbar('H Min', mask_window_name, 160, 179, nothing)
    cv2.createTrackbar('H Max', mask_window_name, 10, 179, nothing)
    cv2.createTrackbar('S Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('S Max', mask_window_name, 255, 255, nothing)
    cv2.createTrackbar('V Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('V Max', mask_window_name, 255, 255, nothing)
    cv2.createTrackbar('Min Area', mask_window_name, 500, 20000, nothing)
    cv2.createTrackbar('Max Area', mask_window_name, 50000, 300000, nothing)
    cv2.createTrackbar('Font Scale', mask_window_name, 7, 20, nothing)

    target = Target()
    tracker = TrackerManager()
    kf = KalmanFilter2D()

    while True:
        dt = kf.tick()
        if dt == 0 or dt > 0.5:
            dt = 0.033

        ret, frame = cap.read()
        if not ret: break

        font_scale = cv2.getTrackbarPos('Font Scale', mask_window_name) / 10.0
        h_min, h_max = cv2.getTrackbarPos('H Min', mask_window_name), cv2.getTrackbarPos('H Max', mask_window_name)
        s_min, s_max = cv2.getTrackbarPos('S Min', mask_window_name), cv2.getTrackbarPos('S Max', mask_window_name)
        v_min, v_max = cv2.getTrackbarPos('V Min', mask_window_name), cv2.getTrackbarPos('V Max', mask_window_name)
        min_area, max_area = cv2.getTrackbarPos('Min Area', mask_window_name), cv2.getTrackbarPos('Max Area', mask_window_name)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if h_min <= h_max:
            mask = cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
        else:
            mask = cv2.bitwise_or(cv2.inRange(hsv, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max])),
                                  cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max])))
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel), cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if min_area < cv2.contourArea(c) < max_area]
        
        candidate_bbox = None
        if valid_contours:
            candidate_bbox = cv2.boundingRect(max(valid_contours, key=cv2.contourArea))

        key = cv2.waitKey(1) & 0xFF
        target_mode = None
        if key == ord('c'): target_mode = "CSRT"
        if key == ord('k'): target_mode = "KCF"

        if target.is_tracking:
            # 传入 dt 推进状态，保证滤波器内部的 vx, vy 准确
            kf.predict(dt)

            success, bbox = tracker.update(frame)
            target.update_state(success, bbox)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2
                
                # 传入观测值修正 KF 状态
                kf.update(cx, cy)
                
                future_x, future_y = kf.predict_future()

                # 绘制当前实际位置（绿色）
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)
                
                # 绘制未来预测位置（红色十字）
                cv2.drawMarker(frame, (future_x, future_y), (0, 0, 255), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                
                cv2.putText(frame, f"Mode: {kf.predict_mode} | Delay: {dt*1000:.1f}ms", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost! Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                if candidate_bbox:
                    tracker.start(frame, candidate_bbox, mode=target.mode)
                    target.start(candidate_bbox, mode=target.mode)
                    x, y, w, h = candidate_bbox
                    kf.init(x + w // 2, y + h // 2)
        else:
            if candidate_bbox:
                x, y, w, h = candidate_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Found! 'c' for CSRT / 'k' for KCF", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            
            if target_mode and candidate_bbox:
                tracker.start(frame, candidate_bbox, mode=target_mode)
                target.start(candidate_bbox, mode=target_mode)
                x, y, w, h = candidate_bbox
                kf.init(x + w // 2, y + h // 2)

        if key == ord('r'): 
            target.reset()
            tracker.reset()
        if key == ord('q'): break

        cv2.imshow(mask_window_name, mask)
        cv2.imshow(main_window_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()