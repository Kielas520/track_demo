# Track Demo

实时目标跟踪与姿态估计演示系统，集成 YOLO 目标检测、MediaPipe 人脸姿态估计、CSRT/KCF 跟踪器以及 3D/6D 卡尔曼滤波。

## 功能

- **YOLO 目标检测**：基于 Ultralytics YOLO 的 2D 目标检测与置信度过滤
- **MediaPipe 人脸姿态估计**：6DOF 人脸姿态检测（yaw/pitch/roll + 3D 位置）
- **CSRT / KCF 跟踪器**：一键切换，丢失自动重连
- **3D 卡尔曼滤波**：对 (x, y, 面积) 平滑去噪 + 前馈预测补偿延迟
- **6D 卡尔曼滤波**：对 (x, y, z, yaw, pitch, roll) 进行 EKF 滤波，含 Numba JIT 加速与马氏距离门控
- **实时调参**：滑动条调整 Q/R 噪声参数、预测模式、预测时间、检测置信度

## 安装

```bash
uv sync
```

## 运行

```bash
# YOLO 2D 目标跟踪
python script/yolo2D.py

# 人脸 6DOF 姿态跟踪
python script/face_pose6D.py
```

## 操作

### yolo2D.py

| 按键 | 功能 |
|------|------|
| `c` | 启动 CSRT 跟踪器（精度高） |
| `k` | 启动 KCF 跟踪器（速度快） |
| `r` | 重置跟踪 |
| `q` | 退出 |

### face_pose6D.py

| 按键 | 功能 |
|------|------|
| `q` | 退出 |
| `Enter` | 应用 KF 噪声参数 |
| `Tab` | 切换面板选项 |
| 点击 | 选择面板字段 |

## 窗口界面

### Unified Tracking System（yolo2D.py）
左侧为 YOLO 检测视图，右侧为跟踪视图，中间白色分隔线。

### KF Noise Control（face_pose6D.py）
独立的 6D 卡尔曼噪声参数控制面板，支持精确数值输入。

## 可视化说明

| 标记 | 颜色 | 含义 |
|------|------|------|
| 十字准星 | 青色（2D）/ 黄色（6D） | 检测/观测原始位置 |
| 实心圆 | 白色 | 卡尔曼滤波平滑位置 |
| 空心圆 + 斜十字 | 黄色 | 前馈预测位置 |
| 连线 | 灰色 → 黄色 | 观测→滤波→预测趋势 |
| 坐标轴 | 彩色 | 姿态方向（6D 模式） |

## 滑动条

### yolo2D.py — Unified Tracking System 窗口

| 滑动条 | 范围 | 说明 |
|--------|------|------|
| [YOLO] Conf(%) | 0 - 100 | 检测置信度阈值 |
| [YOLO] Font Scale | 0.1 - 2.0 | 文字大小 |
| [MAIN] Q_x, Q_y, Q_z | 0.00 - 5.00 | 位置/面积过程噪声标准差 |
| [MAIN] Q_vx, Q_vy, Q_vz | 0 - 5000 | 速度过程噪声标准差（像素²/秒） |
| [MAIN] R_x, R_y, R_z | 0 - 5000 | 测量噪声标准差 |
| [MAIN] Mode | 0-2 | 预测模式：手动/自动/混合 |
| [MAIN] PredictTime | 0.0 - 5.0s | 手动预测时长 |

### face_pose6D.py — KF Noise Control 窗口

| 参数 | 说明 |
|------|------|
| q_pos | 平移过程噪声谱密度 |
| q_rot | 旋转过程噪声谱密度 |
| r_pos_factor | 平移观测噪声因子（距离自适应） |
| r_rot_factor | 旋转观测噪声因子 |
| mahalanobis_threshold | 马氏距离门控阈值 |

## 项目依赖

| 包 | 用途 |
|---|------|
| opencv-contrib-python | 视频采集、CSRT/KCF 跟踪、可视化 |
| ultralytics | YOLO 目标检测 |
| mediapipe | 人脸关键点与姿态估计 |
| numpy | 矩阵运算 |
| numba | 6D EKF JIT 编译加速 |

## 项目结构

```
track_demo/
├── script/
│   ├── yolo2D.py              # YOLO 2D 目标跟踪演示
│   └── face_pose6D.py         # 人脸 6DOF 姿态跟踪演示
├── src/
│   ├── __init__.py
│   ├── target.py              # 目标状态管理
│   ├── tracker.py             # CSRT/KCF 跟踪器封装
│   ├── detector/
│   │   ├── detector2d.py      # YOLO 检测器封装
│   │   └── face_pose.py       # MediaPipe 人脸姿态检测器
│   └── Kalmanfilter/
│       ├── Kalmanfilter3D.py  # 3D 卡尔曼滤波器 (CV 模型)
│       └── Kalmanfilter6D.py  # 6D 卡尔曼滤波器 (EKF + Numba)
├── models/
│   └── face_landmarker.task   # MediaPipe 人脸关键点模型
├── yolo26n.pt                 # YOLO 模型权重
├── pyproject.toml
└── README.md
```
