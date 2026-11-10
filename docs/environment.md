# 服务器环境记录

记录日期：2026-11-10。通过包元数据读取，未升级或安装服务器依赖。

- Python：`3.8.10`
- 平台：`Linux-5.10.104-tegra-aarch64-with-glibc2.29`
- 源目录：`/home/unitree/unitree_sdk2_python`
- 上游提交：`794fb2b`（上传仓库保留原 Git 历史）

| Python 分发包 | 服务器版本 |
|---|---|
| `cyclonedds` | `0.10.2` |
| `numpy` | `1.24.4` |
| `opencv-python` | `4.13.0.92` |
| `ultralytics` | `8.3.0` |
| `torch` | `2.4.1` |
| `torchvision` | `0.19.1` |
| `scipy` | `1.10.1` |
| `pexpect` | `4.6.0` |
| `pyrealsense2` | `2.55.1.6486` |
| `clip` | `1.0` |

这些版本描述现有机器，不是适用于所有平台的锁定安装方案。Jetson 的 PyTorch、TorchVision、RealSense 和 ROS 依赖需与目标机器的系统环境匹配。`clip` 在此表中是安装后的分发包名；此记录不推断其安装源。

## 主流程依赖

- SDK：`cyclonedds==0.10.2`、NumPy、OpenCV；安装说明见 [上游文档](UNITREE_SDK_README.md)。
- YOLO-World 检测：Ultralytics、PyTorch 及其文本编码依赖；仓库根目录包含现有 `yolov8s-world.pt`。
- 图像源：外部 RGB-D 服务提供 `http://127.0.0.1:18082/rgbd.npz`。`start_detection_services.py` 只管理检测器。
- 使用 ROS 图像服务器时，还需要目标机器上的 `rospy`、`sensor_msgs` 和相机话题；相应脚本为 `example/g1/vision/g1_ros_rgbd_http_server.py`。
- 旧版 SSH / RealSense 抓帧路径会使用 Pexpect、RealSense 等依赖。

## 验证范围

本次上传核对了源文件哈希、Python 语法、文档主流程脚本及仓库提交内容。未执行机器人动作、网络配置脚本或模型启动命令，也未在新机器上做端到端运行验证。
