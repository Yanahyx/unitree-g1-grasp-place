# Unitree G1 视觉抓取与放置

基于 Unitree SDK2 Python 的 G1 右臂抓取与放置项目，使用 YOLO-World 视觉检测和 RH56E2 灵巧手，包含力/电流反馈闭合控制、瓶子抓取、纸箱放置及相关调试脚本。

本仓库整理自机器人上的 `/home/unitree/unitree_sdk2_python`，保留 Unitree SDK 的原始 Git 历史。完整操作入口为 **[机器人抓取放置全流程操作指令](docs/机器人抓取放置全流程操作指令.md)**。

## 文件入口

| 功能 | 文件 |
|---|---|
| 启动或复用检测服务 | `example/g1/grasp/start_detection_services.py` |
| 瓶子检测（8765） | `example/g1/grasp/g1_yolo_world_detection_server.py` |
| 箱体检测（8766） | `example/g1/grasp/g1_yolo_world_place_detection_server.py` |
| 右臂力控抓取 | `example/g1/real_arm_grasp/g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py` |
| 右臂放置 | `example/g1/real_arm_grasp/g1_right_arm_lateral_place_sequence.py` |
| 灵巧手临时网络配置 | `example/g1/real_arm_grasp/setup_hand_network.sh` |
| 视觉和坐标处理 | `example/g1/vision/` |
| 手部控制 | `example/g1/hands/`、`example/g1/high_level/rh56e2_*` |
| Unitree SDK | `unitree_sdk2py/` |
| 检测权重 | `yolov8s-world.pt` |

`grasp/` 和其他变体保留服务器现有版本。实际操作请从完整操作文档所列脚本开始；部分实验变体还依赖项目目录之外的环境或服务。

## 获取与环境

在已配置相应机器人依赖的新目录中克隆：

```bash
git clone git@github.com:Yanahyx/unitree-g1-grasp-place.git
cd unitree-g1-grasp-place
python3 -m pip install -e .
```

SDK 安装仅覆盖 `setup.py` 中的基础依赖。视觉模型、ROS / 相机服务、Jetson PyTorch 等还需按目标机器配置；现有服务器版本见 **[环境记录](docs/environment.md)**。上游的安装与 CycloneDDS 故障排查说明见 [Unitree SDK 原始 README](docs/UNITREE_SDK_README.md)。

不要用上述克隆步骤覆盖现有机器人工作目录。原服务器仍可使用原操作文档中的路径运行。

## 操作流程

1. 按操作文档完成机器人上电、支撑与模式切换。
2. 连接机器人，进入项目根目录，配置灵巧手网络。
3. 确认 RGB-D 图像服务在 `127.0.0.1:18082` 提供数据。
4. 启动检测服务，然后分别执行文档中的抓取、放置命令。

```bash
# 配置临时手部网络；需要时由 sudo 提示输入密码
bash example/g1/real_arm_grasp/setup_hand_network.sh

# 只管理检测器；不会启动相机服务
python3 example/g1/grasp/start_detection_services.py all
```

抓取与放置命令包含 `--execute --yes`，会实际驱动机器人；按完整操作文档逐步执行，不要并行启动两个动作脚本。服务就绪不等于相机、目标位置或抓取条件已经确认。

## 配置与上传范围

- 已移除 Wi-Fi、SSH、sudo 明文密码。需要 SSH 抓帧的脚本从 `ROBOT_SSH_PASSWORD` 环境变量读取默认密码，原命令行参数仍可覆盖。
- 服务器上的网卡、相机序列号、网络地址及动作参数保留原样；迁移前按实际设备核对。
- 包含项目源码、SDK 历史、操作说明及现有检测权重。
- 排除运行日志、临时图像、缓存、安装元数据、空占位文件和 `.bak` / `pre_safety` / `pre_retry` 备份。
- 本次发布验证了文件完整性和 Python 语法，未执行机器人动作或新环境端到端测试。

## 上游与许可证

上游项目：[unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)。保留原 [BSD-3-Clause LICENSE](LICENSE) 和作者信息；第三方模型及依赖适用其各自许可证。
