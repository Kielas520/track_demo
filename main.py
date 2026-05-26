import cv2
import numpy as np
from src.target import Target
from src.tracker import TrackerManager
from src.Kalmanfilter import KalmanFilter3D


def nothing(x):
    pass

def main():
    cap = cv2.VideoCapture(0)
    main_window_name = "Tracker Main"
    mask_window_name = "Mask & Controls"
    cv2.namedWindow(main_window_name)
    cv2.createTrackbar('Q_x', main_window_name, 600, 1000, nothing)
    cv2.createTrackbar('Q_y', main_window_name, 600, 1000, nothing)
    cv2.createTrackbar('Q_vx', main_window_name, 320, 1000, nothing)
    cv2.createTrackbar('Q_vy', main_window_name, 320, 1000, nothing)
    cv2.createTrackbar('R_x', main_window_name, 500, 1000, nothing)
    cv2.createTrackbar('R_y', main_window_name, 500, 1000, nothing)
    cv2.createTrackbar('Q_z', main_window_name, 600, 1000, nothing)
    cv2.createTrackbar('Q_vz', main_window_name, 320, 1000, nothing)
    cv2.createTrackbar('R_z', main_window_name, 500, 1000, nothing)
    cv2.createTrackbar('Mode(0:man 1:auto 2:hyb)', main_window_name, 0, 2, nothing)
    cv2.createTrackbar('PredictTime', main_window_name, 0, 500, nothing)
    cv2.namedWindow(mask_window_name)

    # 阈值滑动条...
    cv2.createTrackbar('H Min', mask_window_name, 160, 179, nothing)
    cv2.createTrackbar('H Max', mask_window_name, 179, 179, nothing)
    cv2.createTrackbar('S Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('S Max', mask_window_name, 255, 255, nothing)
    cv2.createTrackbar('V Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('V Max', mask_window_name, 255, 255, nothing)
    cv2.createTrackbar('Min Area', mask_window_name, 25422, 200000, nothing)
    cv2.createTrackbar('Max Area', mask_window_name, 300000, 300000, nothing)
    cv2.createTrackbar('Font Scale', mask_window_name, 13, 20, nothing)

    target = Target()
    tracker = TrackerManager()
    kf = KalmanFilter3D(predict_mode="manual", predict_time=0.0,
                 q_x=0.05, q_y=0.05, q_z=0.05,
                 q_vx=0.1, q_vy=0.1, q_vz=0.1,
                 r_x=5.0,  r_y=5.0,  r_z=5.0
                )

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

        q_x = cv2.getTrackbarPos('Q_x', main_window_name) / 100.0
        q_y = cv2.getTrackbarPos('Q_y', main_window_name) / 100.0
        q_z = cv2.getTrackbarPos('Q_z', main_window_name) / 100.0
        q_vx = cv2.getTrackbarPos('Q_vx', main_window_name) / 100.0
        q_vy = cv2.getTrackbarPos('Q_vy', main_window_name) / 100.0
        q_vz = cv2.getTrackbarPos('Q_vz', main_window_name) / 100.0
        r_x = cv2.getTrackbarPos('R_x', main_window_name) / 100.0
        r_y = cv2.getTrackbarPos('R_y', main_window_name) / 100.0
        r_z = cv2.getTrackbarPos('R_z', main_window_name) / 100.0
        kf.set_qr(q_x=q_x, q_y=q_y, q_z=q_z,
                  q_vx=q_vx, q_vy=q_vy, q_vz=q_vz,
                  r_x=r_x, r_y=r_y, r_z=r_z)

        mode_idx = cv2.getTrackbarPos('Mode(0:man 1:auto 2:hyb)', main_window_name)
        mode_map = {0: "manual", 1: "auto", 2: "hybrid"}
        kf.set_mode(mode_map[mode_idx])
        kf.set_predict_time(cv2.getTrackbarPos('PredictTime', main_window_name) / 100.0)

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

                z = 0
                if w > 0 and h > 0:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    roi = frame[y1:y2, x1:x2]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    if h_min <= h_max:
                        roi_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                    else:
                        roi_mask = cv2.bitwise_or(
                            cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max])),
                            cv2.inRange(hsv_roi, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max])))
                    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                    z = cv2.countNonZero(roi_mask)
                    frame[y1:y2, x1:x2][roi_mask > 0] = (0, 255, 255)
                    cv2.putText(frame, f"Area:{z}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                kf.update(cx, cy, z)

                filtered_x, filtered_y, filtered_z = kf.get_filtered_pos()
                future_x, future_y, future_z = kf.predict_future()

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (filtered_x, filtered_y), 8, (255, 0, 0), -1)
                cv2.circle(frame, (future_x, future_y), 12, (0, 0, 255), 3)

                cv2.putText(frame, f"Mode: {kf.predict_mode} | Delay: {dt*1000:.1f}ms | z:{filtered_z}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Lost! Searching...", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                if candidate_bbox:
                    tracker.start(frame, candidate_bbox, mode=target.mode)
                    target.start(candidate_bbox, mode=target.mode)
                    x, y, w, h = candidate_bbox
                    init_z = 0
                    if w > 0 and h > 0:
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        roi = frame[y1:y2, x1:x2]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        if h_min <= h_max:
                            roi_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                        else:
                            roi_mask = cv2.bitwise_or(
                                cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max])),
                                cv2.inRange(hsv_roi, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max])))
                        init_z = cv2.countNonZero(roi_mask)
                    kf.init(x + w // 2, y + h // 2, init_z)
        else:
            if candidate_bbox:
                x, y, w, h = candidate_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Found! 'c' for CSRT / 'k' for KCF", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            
            if target_mode and candidate_bbox:
                tracker.start(frame, candidate_bbox, mode=target_mode)
                target.start(candidate_bbox, mode=target_mode)
                x, y, w, h = candidate_bbox
                init_z = 0
                if w > 0 and h > 0:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    roi = frame[y1:y2, x1:x2]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    if h_min <= h_max:
                        roi_mask = cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([h_max, s_max, v_max]))
                    else:
                        roi_mask = cv2.bitwise_or(
                            cv2.inRange(hsv_roi, np.array([h_min, s_min, v_min]), np.array([179, s_max, v_max])),
                            cv2.inRange(hsv_roi, np.array([0, s_min, v_min]), np.array([h_max, s_max, v_max])))
                    init_z = cv2.countNonZero(roi_mask)
                kf.init(x + w // 2, y + h // 2, init_z)

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