"""
目标跟踪演示系统 - 主程序
==========================

基于 OpenCV 的实时目标跟踪演示程序。

功能流程:
    1. 摄像头采集 → HSV 颜色空间目标检测
    2. 用户选定目标 → CSRT/KCF 跟踪器接管
    3. 跟踪框内 HSV 掩膜 → 计算目标面积
    4. 3D 卡尔曼滤波 (x, y, z=面积) → 平滑 + 预测
    5. 可视化: 原始观测 / 滤波位置 / 预测位置

按键说明:
    c  - 启动 CSRT 跟踪器 (精度高)
    k  - 启动 KCF 跟踪器 (速度快)
    r  - 重置跟踪
    q  - 退出

依赖:
    opencv-contrib-python >= 4.13.0.92
"""

import cv2
import numpy as np
from src.target import Target
from src.tracker import TrackerManager
from src.Kalmanfilter import KalmanFilter3D


def nothing(x):
    """OpenCV trackbar 空回调，trackbar 必须关联一个回调函数"""
    pass


def main():
    # ======================== 1. 摄像头 & 窗口初始化 ========================

    cap = cv2.VideoCapture(0)             # 打开默认摄像头 (索引0)
    main_window_name = "Tracker Main"      # 主窗口：显示跟踪结果
    mask_window_name = "Mask & Controls"   # 副窗口：显示 HSV 掩膜 + 调参滑动条

    cv2.namedWindow(main_window_name)
    cv2.namedWindow(mask_window_name)

    # ==================== 2. 卡尔曼滤波器参数滑动条 ========================
    # 显示在 "Tracker Main" 窗口上方
    # trackbar 范围 0-1000，实际值 = trackbar值 / 100.0

    # --- 过程噪声 Q (位置部分) ---
    cv2.createTrackbar('Q_x', main_window_name, 600, 1000, nothing)   # x 轴位置噪声
    cv2.createTrackbar('Q_y', main_window_name, 600, 1000, nothing)   # y 轴位置噪声
    cv2.createTrackbar('Q_z', main_window_name, 600, 1000, nothing)   # 面积噪声

    # --- 过程噪声 Q (速度部分) ---
    cv2.createTrackbar('Q_vx', main_window_name, 320, 1000, nothing)  # x 方向速度噪声
    cv2.createTrackbar('Q_vy', main_window_name, 320, 1000, nothing)  # y 方向速度噪声
    cv2.createTrackbar('Q_vz', main_window_name, 320, 1000, nothing)  # 面积变化率噪声

    # --- 测量噪声 R ---
    cv2.createTrackbar('R_x', main_window_name, 500, 1000, nothing)   # x 测量噪声
    cv2.createTrackbar('R_y', main_window_name, 500, 1000, nothing)   # y 测量噪声
    cv2.createTrackbar('R_z', main_window_name, 500, 1000, nothing)   # 面积测量噪声

    # --- 预测配置 ---
    cv2.createTrackbar('Mode(0:man 1:auto 2:hyb)', main_window_name, 0, 2, nothing)  # 预测模式
    cv2.createTrackbar('PredictTime', main_window_name, 0, 500, nothing)              # 手动预测时间 (×0.01秒)

    # =================== 3. HSV 颜色阈值滑动条 ============================
    # 显示在 "Mask & Controls" 窗口
    # 用于调整目标颜色检测范围（红色瓶盖典型值：H 160-10）

    cv2.createTrackbar('H Min', mask_window_name, 160, 179, nothing)     # 色相下限
    cv2.createTrackbar('H Max', mask_window_name, 179, 179, nothing)     # 色相上限
    cv2.createTrackbar('S Min', mask_window_name, 100, 255, nothing)     # 饱和度下限
    cv2.createTrackbar('S Max', mask_window_name, 255, 255, nothing)     # 饱和度上限
    cv2.createTrackbar('V Min', mask_window_name, 100, 255, nothing)     # 明度下限
    cv2.createTrackbar('V Max', mask_window_name, 255, 255, nothing)     # 明度上限
    cv2.createTrackbar('Min Area', mask_window_name, 25422, 200000, nothing)   # 最小轮廓面积
    cv2.createTrackbar('Max Area', mask_window_name, 300000, 300000, nothing)   # 最大轮廓面积
    cv2.createTrackbar('Font Scale', mask_window_name, 13, 20, nothing)         # 文字大小

    # ======================== 4. 组件初始化 ===============================

    target = Target()                                                      # 目标状态管理器
    tracker = TrackerManager()                                             # OpenCV 跟踪器封装

    # 初始为 manual 模式且 predict_time=0，即不做前馈预测
    kf = KalmanFilter3D(predict_mode="manual", predict_time=0.0,
                 q_x=0.05, q_y=0.05, q_z=0.05,
                 q_vx=0.1, q_vy=0.1, q_vz=0.1,
                 r_x=5.0,  r_y=5.0,  r_z=5.0
                )

    # ======================== 5. 主循环 ===================================

    while True:
        # ---- 5.1 系统延迟计时 ----
        dt = kf.tick()                # 测量帧间隔
        if dt == 0 or dt > 0.5:       # 首帧或间隔异常 → 假设 30fps
            dt = 0.033

        # ---- 5.2 读取摄像头帧 ----
        ret, frame = cap.read()
        if not ret:
            break

        # ---- 5.3 读取所有滑动条值 ----

        font_scale = cv2.getTrackbarPos('Font Scale', mask_window_name) / 10.0

        # HSV 阈值
        h_min = cv2.getTrackbarPos('H Min', mask_window_name)
        h_max = cv2.getTrackbarPos('H Max', mask_window_name)
        s_min = cv2.getTrackbarPos('S Min', mask_window_name)
        s_max = cv2.getTrackbarPos('S Max', mask_window_name)
        v_min = cv2.getTrackbarPos('V Min', mask_window_name)
        v_max = cv2.getTrackbarPos('V Max', mask_window_name)

        # 轮廓面积过滤
        min_area = cv2.getTrackbarPos('Min Area', mask_window_name)
        max_area = cv2.getTrackbarPos('Max Area', mask_window_name)

        # 卡尔曼 Q/R 参数 (trackbar ÷ 100 = 实际值)
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

        # 预测模式和时长
        mode_idx = cv2.getTrackbarPos('Mode(0:man 1:auto 2:hyb)', main_window_name)
        mode_map = {0: "manual", 1: "auto", 2: "hybrid"}
        kf.set_mode(mode_map[mode_idx])
        kf.set_predict_time(cv2.getTrackbarPos('PredictTime', main_window_name) / 100.0)

        # ---- 5.4 HSV 颜色空间目标检测 ----

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 红色瓶盖的 H 值跨越 0 度，需要分两段处理
        # h_min > h_max 时，例如 H ∈ [160, 179] ∪ [0, 10]
        if h_min <= h_max:
            mask = cv2.inRange(hsv,
                               np.array([h_min, s_min, v_min]),
                               np.array([h_max, s_max, v_max]))
        else:
            # 两段取并集，覆盖 H 环绕 0 度的情况
            mask = cv2.bitwise_or(
                cv2.inRange(hsv,
                            np.array([h_min, s_min, v_min]),
                            np.array([179, s_max, v_max])),
                cv2.inRange(hsv,
                            np.array([0, s_min, v_min]),
                            np.array([h_max, s_max, v_max])))

        # 形态学处理：开运算去噪，闭运算填洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel),
                                cv2.MORPH_CLOSE, kernel)

        # 提取轮廓，过滤面积不在范围内的
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours
                          if min_area < cv2.contourArea(c) < max_area]

        # 选择面积最大的作为候选目标
        candidate_bbox = None
        if valid_contours:
            candidate_bbox = cv2.boundingRect(
                max(valid_contours, key=cv2.contourArea))

        # ---- 5.5 键盘输入 ----
        key = cv2.waitKey(1) & 0xFF
        target_mode = None
        if key == ord('c'):
            target_mode = "CSRT"     # 启动 CSRT 跟踪
        if key == ord('k'):
            target_mode = "KCF"      # 启动 KCF 跟踪

        # ---- 5.6 跟踪逻辑 ----

        if target.is_tracking:
            # 【预测步】传入真实 dt，推进卡尔曼状态
            kf.predict(dt)

            # 【跟踪步】在新帧中定位目标
            success, bbox = tracker.update(frame)
            target.update_state(success, bbox)

            if success:
                # ---------- 跟踪成功 ----------
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2        # 包围框中心点

                # —— ROI 内 HSV 掩膜面积计算 ——
                # 在跟踪框范围内重新做 HSV 阈值，得到目标像素计数
                z = 0   # 面积默认值，防止 ROI 无效时出错
                if w > 0 and h > 0:
                    # 边界裁剪，防止 bbox 超出图像
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)

                    roi = frame[y1:y2, x1:x2]           # 截取 ROI
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

                    if h_min <= h_max:
                        roi_mask = cv2.inRange(hsv_roi,
                                               np.array([h_min, s_min, v_min]),
                                               np.array([h_max, s_max, v_max]))
                    else:
                        roi_mask = cv2.bitwise_or(
                            cv2.inRange(hsv_roi,
                                        np.array([h_min, s_min, v_min]),
                                        np.array([179, s_max, v_max])),
                            cv2.inRange(hsv_roi,
                                        np.array([0, s_min, v_min]),
                                        np.array([h_max, s_max, v_max])))

                    # 闭运算连接相邻区域
                    roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)
                    z = cv2.countNonZero(roi_mask)       # 面积 = 白点计数

                    # 在原图上高亮匹配区域 (黄色)
                    frame[y1:y2, x1:x2][roi_mask > 0] = (0, 255, 255)
                    cv2.putText(frame, f"Area:{z}", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 【校正步】将 (x, y, 面积) 作为观测送入卡尔曼滤波器
                kf.update(cx, cy, z)

                # 获取滤波后位置 (蓝色实心圆)
                filtered_x, filtered_y, filtered_z = kf.get_filtered_pos()

                # 获取前馈预测位置 (红色空心圆)
                future_x, future_y, future_z = kf.predict_future()

                # ====== 可视化 ======

                # 绿色空心矩形: 跟踪器原始 bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # 蓝色实心圆: 卡尔曼滤波位置
                cv2.circle(frame, (filtered_x, filtered_y), 8, (255, 0, 0), -1)

                # 红色空心圆: 前馈预测位置
                cv2.circle(frame, (future_x, future_y), 12, (0, 0, 255), 3)

                # 状态信息: 模式 | 系统延迟 | 滤波面积
                cv2.putText(frame,
                            f"Mode: {kf.predict_mode} | "
                            f"Delay: {dt*1000:.1f}ms | "
                            f"z:{filtered_z}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), 2)

            else:
                # ---------- 跟踪丢失 ----------
                cv2.putText(frame, "Lost! Searching...",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 0, 255), 2)

                # 尝试用 HSV 检测的候选目标重连
                if candidate_bbox:
                    tracker.start(frame, candidate_bbox, mode=target.mode)
                    target.start(candidate_bbox, mode=target.mode)

                    # 计算初始化面积作为 z 轴初始值
                    x, y, w, h = candidate_bbox
                    init_z = 0
                    if w > 0 and h > 0:
                        x1, y1 = max(0, x), max(0, y)
                        x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                        roi = frame[y1:y2, x1:x2]
                        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                        if h_min <= h_max:
                            roi_mask = cv2.inRange(hsv_roi,
                                                   np.array([h_min, s_min, v_min]),
                                                   np.array([h_max, s_max, v_max]))
                        else:
                            roi_mask = cv2.bitwise_or(
                                cv2.inRange(hsv_roi,
                                            np.array([h_min, s_min, v_min]),
                                            np.array([179, s_max, v_max])),
                                cv2.inRange(hsv_roi,
                                            np.array([0, s_min, v_min]),
                                            np.array([h_max, s_max, v_max])))
                        init_z = cv2.countNonZero(roi_mask)

                    kf.init(x + w // 2, y + h // 2, init_z)

        else:
            # ---------- 未在跟踪状态 (等待用户选择目标) ----------
            if candidate_bbox:
                x, y, w, h = candidate_bbox

                # 黄色矩形: 检测到的候选目标
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, "Found! 'c' for CSRT / 'k' for KCF",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale,
                            (0, 255, 255), 2)

            # 用户按下 c 或 k 启动跟踪
            if target_mode and candidate_bbox:
                tracker.start(frame, candidate_bbox, mode=target_mode)
                target.start(candidate_bbox, mode=target_mode)

                # 计算初始化面积
                x, y, w, h = candidate_bbox
                init_z = 0
                if w > 0 and h > 0:
                    x1, y1 = max(0, x), max(0, y)
                    x2, y2 = min(frame.shape[1], x + w), min(frame.shape[0], y + h)
                    roi = frame[y1:y2, x1:x2]
                    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                    if h_min <= h_max:
                        roi_mask = cv2.inRange(hsv_roi,
                                               np.array([h_min, s_min, v_min]),
                                               np.array([h_max, s_max, v_max]))
                    else:
                        roi_mask = cv2.bitwise_or(
                            cv2.inRange(hsv_roi,
                                        np.array([h_min, s_min, v_min]),
                                        np.array([179, s_max, v_max])),
                            cv2.inRange(hsv_roi,
                                        np.array([0, s_min, v_min]),
                                        np.array([h_max, s_max, v_max])))
                    init_z = cv2.countNonZero(roi_mask)

                kf.init(x + w // 2, y + h // 2, init_z)

        # ---- 5.7 全局控制 ----
        if key == ord('r'):
            target.reset()
            tracker.reset()
        if key == ord('q'):
            break

        # ---- 5.8 显示 ----
        cv2.imshow(mask_window_name, mask)          # HSV 掩膜窗口
        cv2.imshow(main_window_name, frame)         # 跟踪结果窗口

    # ======================== 6. 清理 =====================================

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
