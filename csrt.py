import cv2
import numpy as np

def nothing(x):
    pass

class Target:
    def __init__(self):
        self.is_tracking = False
        self.bbox = None
        self.tracker = None
        self.mode = "CSRT" # 新增：记录当前追踪类型

    def start_tracking(self, frame, bbox, mode="CSRT"):
        self.mode = mode
        if mode == "CSRT":
            params = cv2.TrackerCSRT_Params()
            self.tracker = cv2.TrackerCSRT.create(params)
        else: # 使用 KCF
            self.tracker = cv2.TrackerKCF.create()
        
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.is_tracking = True
        self.frame_count = 0

    def update(self, frame):
        if not self.is_tracking:
            return False, None
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = bbox
            self.frame_count += 1
        return success, bbox
        
    def reset(self):
        self.is_tracking = False
        self.bbox = None
        self.tracker = None
        self.frame_count = 0

def main():
    cap = cv2.VideoCapture(0)
    main_window_name = "CSRT Tracker"
    mask_window_name = "Mask & Controls"
    cv2.namedWindow(main_window_name)
    cv2.namedWindow(mask_window_name)

    # 滑动条设置
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

    while True:
        ret, frame = cap.read()
        if not ret: break

        font_scale = cv2.getTrackbarPos('Font Scale', mask_window_name) / 10.0
        h_min, h_max = cv2.getTrackbarPos('H Min', mask_window_name), cv2.getTrackbarPos('H Max', mask_window_name)
        s_min, s_max = cv2.getTrackbarPos('S Min', mask_window_name), cv2.getTrackbarPos('S Max', mask_window_name)
        v_min, v_max = cv2.getTrackbarPos('V Min', mask_window_name), cv2.getTrackbarPos('V Max', mask_window_name)
        min_area, max_area = cv2.getTrackbarPos('Min Area', mask_window_name), cv2.getTrackbarPos('Max Area', mask_window_name)

        # 1. 始终运行检测逻辑
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

        # 2. 状态机逻辑
        key = cv2.waitKey(1) & 0xFF
        
        # 键盘检测：按 'c' 用 CSRT，按 'k' 用 KCF
        target_mode = None
        if key == ord('c'): target_mode = "CSRT"
        if key == ord('k'): target_mode = "KCF"

        if target.is_tracking:
            success, bbox = target.update(frame)
            if success:
                x, y, w, h = [int(v) for v in bbox]
                # 在画面上显示当前使用的算法
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{target.mode} | Frames: {target.frame_count}", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "STATUS: Lost! Searching...", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 255), 2)
                # 丢失后，如果检测器再次看到目标，自动重连
                if candidate_bbox:
                    target.start_tracking(frame, candidate_bbox)
        else:
            # 检测模式
            if candidate_bbox:
                x, y, w, h = candidate_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Found! Press 'c' for CSRT / 'k' for KCF", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 2)
            
            # 检测到候选框时，根据按键启动
            if target_mode and candidate_bbox:
                target.start_tracking(frame, candidate_bbox, mode=target_mode)

        if key == ord('r'): target.reset()
        if key == ord('q'): break

        cv2.imshow(mask_window_name, mask)
        cv2.imshow(main_window_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()