import cv2
import numpy as np

def nothing(x):
    pass

class Target:
    """管理目标跟踪状态与追踪器的类"""
    def __init__(self):
        self.is_tracking = False
        self.bbox = None
        self.tracker = None

    def start_tracking(self, frame, bbox):
        """初始化 CSRT 跟踪器并切换为跟踪状态"""
        # 创建参数对象
        params = cv2.TrackerCSRT_Params()

        # 关闭尺度估计（默认为 True）
        params.use_scale_estimation = False

        # 使用自定义参数创建跟踪器
        self.tracker = cv2.TrackerCSRT_create(params)
        self.tracker.init(frame, bbox)
        self.bbox = bbox
        self.is_tracking = True

    def update(self, frame):
        """更新跟踪器状态"""
        if not self.is_tracking:
            return False, None
        
        success, bbox = self.tracker.update(frame)
        if success:
            self.bbox = bbox
        return success, bbox
        
    def reset(self):
        """重置跟踪状态，返回检测模式"""
        self.is_tracking = False
        self.bbox = None
        self.tracker = None


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("错误：无法打开摄像头。")
        return

    # 创建主窗口与控制/掩码窗口
    main_window_name = "CSRT Tracker"
    mask_window_name = "Mask & Controls"
    cv2.namedWindow(main_window_name)
    cv2.namedWindow(mask_window_name)

    # 创建 HSV 滑动条
    # OpenCV 中 H 的范围是 0-179，S 和 V 的范围是 0-255
    # 红色初始建议值：H 包含 160~179 及 0~10 的范围
    cv2.createTrackbar('H Min', mask_window_name, 160, 179, nothing)
    cv2.createTrackbar('H Max', mask_window_name, 10, 179, nothing)
    cv2.createTrackbar('S Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('S Max', mask_window_name, 255, 255, nothing)
    cv2.createTrackbar('V Min', mask_window_name, 100, 255, nothing)
    cv2.createTrackbar('V Max', mask_window_name, 255, 255, nothing)

    cv2.createTrackbar('Min Area', mask_window_name, 500, 20000, nothing)
    cv2.createTrackbar('Max Area', mask_window_name, 50000, 300000, nothing)

    target = Target()

    print("操作说明:")
    print(" - 调整 HSV 滑动条使目标在 Mask 窗口中清晰显示为白色。")
    print("   (注：当寻找红色时，H Min 可能大于 H Max，程序会自动处理跨越 0 度的区间。)")
    print(" - 按下 'SPACE' (空格键) 锁定面积最大的有效目标并开始跟踪。")
    print(" - 按下 'r' 键取消跟踪，重新回到检测模式。")
    print(" - 按下 'q' 键退出程序。")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 获取滑动条当前值
        h_min = cv2.getTrackbarPos('H Min', mask_window_name)
        h_max = cv2.getTrackbarPos('H Max', mask_window_name)
        s_min = cv2.getTrackbarPos('S Min', mask_window_name)
        s_max = cv2.getTrackbarPos('S Max', mask_window_name)
        v_min = cv2.getTrackbarPos('V Min', mask_window_name)
        v_max = cv2.getTrackbarPos('V Max', mask_window_name)
        
        min_area = cv2.getTrackbarPos('Min Area', mask_window_name)
        max_area = cv2.getTrackbarPos('Max Area', mask_window_name)

        if not target.is_tracking:
            # ---------------------------------------------------
            # 阶段 1: 检测模式 (寻找有效目标)
            # ---------------------------------------------------
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # 处理 H 通道跨界情况 (红色区域在 OpenCV 中通常为 160-179 和 0-10)
            if h_min <= h_max:
                lower = np.array([h_min, s_min, v_min])
                upper = np.array([h_max, s_max, v_max])
                mask = cv2.inRange(hsv, lower, upper)
            else:
                # 当 H_Min > H_Max 时，分为两段提取并合并
                lower1 = np.array([h_min, s_min, v_min])
                upper1 = np.array([179, s_max, v_max])
                mask1 = cv2.inRange(hsv, lower1, upper1)
                
                lower2 = np.array([0, s_min, v_min])
                upper2 = np.array([h_max, s_max, v_max])
                mask2 = cv2.inRange(hsv, lower2, upper2)
                
                mask = cv2.bitwise_or(mask1, mask2)

            # 形态学操作：消除小噪点并填补目标内部空洞
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # 寻找轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_contours = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if min_area < area < max_area:
                    valid_contours.append(cnt)

            candidate_bbox = None
            if valid_contours:
                # 选取符合面积要求的最大轮廓作为候选目标
                largest_contour = max(valid_contours, key=cv2.contourArea)
                candidate_bbox = cv2.boundingRect(largest_contour)
                
                # 在主画面上绘制黄色虚线框作为候选预览
                x, y, w, h = candidate_bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Target Candidate", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.putText(frame, "Detecting Mode - Press SPACE to lock", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # 显示处理后的 HSV 掩码
            cv2.imshow(mask_window_name, mask)

            # 按键处理
            key = cv2.waitKey(30) & 0xFF
            if key == ord(' ') and candidate_bbox is not None:
                target.start_tracking(frame, candidate_bbox)
                print(f"目标已锁定，开始跟踪。BBox: {candidate_bbox}")
            elif key == ord('q'):
                break

        else:
            # ---------------------------------------------------
            # 阶段 2: 跟踪模式 (CSRT 运行中)
            # ---------------------------------------------------
            success, bbox = target.update(frame)

            if success:
                x, y, w, h = [int(v) for v in bbox]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, "CSRT Tracking", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Tracking Lost", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "Tracking Mode - Press 'r' to reset", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 跟踪模式下保持窗口活跃
            blank_mask = np.zeros_like(frame[:, :, 0])
            cv2.putText(blank_mask, "Tracking Active", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            cv2.imshow(mask_window_name, blank_mask)

            # 按键处理
            key = cv2.waitKey(30) & 0xFF
            if key == ord('r'):
                target.reset()
                print("跟踪已重置，返回检测模式。")
            elif key == ord('q'):
                break

        # 刷新主窗口
        cv2.imshow(main_window_name, frame)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()