"""
目标跟踪演示项目
=================

基于 OpenCV 的实时目标跟踪系统，集成：
- HSV 颜色空间目标检测
- CSRT / KCF 跟踪器
- 3D 卡尔曼滤波器 (x, y, 面积)
- 前馈预测（补偿系统延迟）

模块索引:
- Detector: 基于 YOLO 的目标检测器
- TrackerManager: OpenCV 跟踪器管理器 (CSRT/KCF)
- KalmanFilter3D: 3D 卡尔曼滤波器
- Target: 目标状态管理器
"""

# 导入项目中的各个模块，以便于外部使用
from .detector import Detector2D, FacePoseDetector
from .tracker import TrackerManager
from .Kalmanfilter import KalmanFilter3D, KalmanFilter6D
from .target import Target

__all__ = [
    'Detector2D',
    'FacePoseDetector',
    'TrackerManager',
    'KalmanFilter3D',
    'KalmanFilter6D',
    'Target',
]