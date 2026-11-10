# Unitree G1 视觉抓取与放置

基于 Unitree SDK2 Python 的 G1 右臂抓取与放置项目，使用 RealSense RGB-D、YOLO-World 和 RH56E2 灵巧手。本页直接收录单次抓取放置全流程，以及带图像、坐标、耗时和人工反馈记录的 20 次连续抓取测试教程。

## 目录

- [获取与环境](#获取与环境)
- [机器人抓取放置全流程](#机器人抓取放置全流程)
- [20 次连续抓取测试教程](#20-次连续抓取测试教程)
- [结果文件与统计口径](#结果文件与统计口径)
- [常见问题与逻辑检查](#常见问题与逻辑检查)
- [文件入口](#文件入口)
- [配置与上传范围](#配置与上传范围)

## 获取与环境

在已配置相应机器人依赖的新目录中克隆：

```bash
git clone git@github.com:Yanahyx/unitree-g1-grasp-place.git
cd unitree-g1-grasp-place
python3 -m pip install -e .
```

SDK 安装仅覆盖 `setup.py` 中的基础依赖。视觉模型、ROS / 相机服务、Jetson PyTorch 等还需按目标机器配置；现有服务器版本见 **[环境记录](docs/environment.md)**。上游的安装与 CycloneDDS 故障排查说明见 [Unitree SDK 原始 README](docs/UNITREE_SDK_README.md)。

不要用上述克隆步骤覆盖现有机器人工作目录。原服务器仍可使用原操作文档中的路径运行。

## 机器人抓取放置全流程

以下保留原全流程的单次动作参数。连续测试采用后文单独列出的提速参数。示例 IP 以机器人当前地址为准。

### 一、机器人开机与模式切换

机器人上电后，按提供的图片依次操作遥控器：

| 顺序 | 模式 | 遥控器按键 |
|---|---|---|
| 1 | 零力矩 | L2 + Y |
| 2 | 阻尼 | L2 + B |
| 3 | 预备 | L2 + 方向键上 |
| 4 | 走跑 | R2 + A |

注意：以上为上电后的模式切换顺序，不包含电源按钮操作。进入零力矩模式前需做好支撑或吊装，避免机器人失去支撑倾倒。

### 二、连接网络与机器人服务器

#### 2.1 连接机器人 Wi-Fi

- 网络名称：`Unitree_G1`
- 网络密码：由设备管理员线下提供，不写入仓库。

#### 2.2 SSH 登录机器人

电脑连接上述网络后，在终端执行：

```bash
ssh unitree@10.113.9.174
```

登录密码由设备管理员线下提供。按提示输入，输入时终端不显示密码字符属于正常现象。

#### 2.3 进入工作目录

连接成功后执行，后续命令均在该目录下运行：

```bash
cd /home/unitree/unitree_sdk2_python
```

### 三、开机或重启后执行一次

机器人重启后需要启动检测模型。下面的新启动命令可重复执行：服务已就绪则复用，未运行才启动，正在初始化则等待；遇到异常端口占用会报错，不强行关闭已有进程。

启动器只管理检测模型，不启动摄像头服务。已有 RGB-D 图像服务须提供 `http://127.0.0.1:18082/rgbd.npz`，模型就绪不代表已识别到目标。

#### 3.1 配置灵巧手 IP

```bash
bash example/g1/real_arm_grasp/setup_hand_network.sh
```

脚本需要提权时会由 `sudo` 提示输入密码。请勿把密码写入代码或命令历史。

#### 3.2 启动抓取和放置检测模型（推荐）

```bash
python3 example/g1/grasp/start_detection_services.py all
```

此命令依次确保瓶子检测（8765）和箱体检测（8766）就绪。检测程序在后台持续运行，无需再加 `nohup` 或 `&`，也不需要再执行旧的模型启动命令。

#### 3.3 仅启动其中一种模型（可选）

只用抓取时：

```bash
python3 example/g1/grasp/start_detection_services.py grasp
```

只用放置时：

```bash
python3 example/g1/grasp/start_detection_services.py place
```

### 四、执行抓取（力控版）

以下抓取、放置命令会实际驱动机器人。请分步执行，保持运动范围无障碍，不要同时运行两个动作脚本。

```bash
python3 example/g1/real_arm_grasp/g1_right_arm_lateral_grasp_sequence_mjpg_force_thumb.py eth1 \
  --use-vision \
  --detector-server-url http://127.0.0.1:8765/detect \
  --object bottle \
  --vision-xy-offset -0.06 0.0 \
  --side-standoff-y 0.0 \
  --hand-speed 300 \
  --hand-force 800 \
  --hand-hold 3.0 \
  --thumb-rotate-speed 500 \
  --thumb-rotate-force 2000 \
  --thumb-rotate-step 40 \
  --thumb-rotate-hold 0.03 \
  --thumb-close-tolerance 120 \
  --feedback-force-delta 40 \
  --feedback-current-delta 35 \
  --feedback-min-phase 0.85 \
  --close-at-approach-end \
  --no-announce \
  --execute --yes
```

### 五、执行放置

确认瓶子已可靠抓持、纸箱位置合适后执行：

```bash
python3 example/g1/real_arm_grasp/g1_right_arm_lateral_place_sequence.py \
  --box-vision \
  --box-vision-source auto \
  --box-detector-server-url http://127.0.0.1:8766/detect \
  --box-object "cardboard box" \
  --box-camera-pitch 0.8 \
  --box-placement-region center \
  --box-place-offset -0.27 -0.04 0.52 \
  --box-approach-offset 0.0 -0.10 0.0 \
  --pre-duration 3.0 \
  --place-duration 2.0 \
  --return-duration 2.0 \
  --side-lift-return-duration 2.0 \
  --max-joint-speed 0.35 \
  --return-max-joint-speed 0.45 \
  --hold-after-side-lift 0 \
  --hold-before-release 0 \
  --hold-after-release 0 \
  --hold-after-hand-close 0 \
  --announce-on-arm-return \
  --announce-text "放置完成" \
  --weight 0.75 \
  --execute \
  --yes
```

### 六、迁移与配置说明

- 本文命令适用于已完成环境配置的机器人；全新机器请先阅读仓库首页和 `docs/environment.md`。
- Wi-Fi、SSH 与 sudo 密码不随项目分发。仅在使用需要 SSH 抓帧的旧版脚本时，在当前终端设置 `ROBOT_SSH_PASSWORD`；也可使用原有 `--password` / `--vision-password` 参数。主流程的本机检测服务不需要 SSH 密码。
- 仓库保留服务器代码中的网卡、相机序列号、坐标偏移及旧版网络地址。迁移时应核对配置；它们不会自动跟随上面的 SSH 登录地址变化。
- 启动检测模型不会启动 RGB-D 服务，也不会证明相机或目标检测已经正常。不要把服务健康检查当作可执行抓取的确认。

需要临时设置 SSH 密码时，可避免在命令行中写出明文：

```bash
read -rsp "Robot SSH password: " ROBOT_SSH_PASSWORD; echo
export ROBOT_SSH_PASSWORD
# 在同一终端运行确实需要 SSH 抓帧的脚本
unset ROBOT_SSH_PASSWORD
```


## 20 次连续抓取测试教程

### 1. 准备条件

入口脚本：[test_grasp_20.py](example/g1/real_arm_grasp/test_grasp_20.py)。在机器人 Linux 的交互终端运行，默认连续启动 20 次抓取尝试，逐次记录实际检测画面、瓶子位置、耗时和人工结果。

- 先按上文完成模式切换、灵巧手网络和相机准备，单次抓取应已调试可用。
- 项目根目录包含 `yolov8s-world.pt`，视觉与控制环境配置见 [环境记录](docs/environment.md)。
- RGB-D 服务须提供 `http://127.0.0.1:18082/rgbd.npz`。
- 本程序每轮只加载一次专用检测模型，复用仓库 `DetectorState` 实现，不依赖或修改现有 8765 服务。检测器绑定本机动态端口，不独占新的相机设备。
- 不用 `nohup`、`&` 或标准输入重定向运行，需要现场输入结果。
- 每次只执行抓取，不自动放置。抓取结束后由操作人员取下瓶子并准备下一次。

### 2. 先检查图像与检测，不运动

在包含本教程脚本的仓库根目录运行：

```bash
python3 example/g1/real_arm_grasp/test_grasp_20.py \
  --sdk-dir "$PWD" \
  --check
```

正常时输出 `CHECK_OK`，保存检测原图、标注图和坐标，不启动抓取动作，也不占正式测试次数。检查结果在 `/home/unitree/grasp_trials/formal/check_日期_时间/`。

未检测到瓶子时会报错，并保留已采集图像及错误文件。相机无法采图时不会生成虚构图像。程序不回退到其他相机来源，以保证图像与定位来自同一次检测。

### 3. 启动连续 20 次测试

仍在仓库根目录运行：

```bash
python3 example/g1/real_arm_grasp/test_grasp_20.py \
  --sdk-dir "$PWD" \
  --trials 20
```

程序先加载模型，然后等待第一次摆放确认，**按回车后才开始第一次真实抓取动作**。

已有服务器若使用项目外部署的副本，也可运行：

```bash
cd /home/unitree/unitree_sdk2_python
python3 /home/unitree/grasp_trials/test_grasp_20.py --sdk-dir "$PWD" --trials 20
```

项目外副本需与仓库同步。首次使用本仓库，优先采用相对路径入口，并通过 `--sdk-dir "$PWD"` 指向当前根目录，避免运行到旧脚本或旧工作目录。

### 4. 每次输入结果后启动下一次

1. 放好第一次的瓶子，按回车开始。
2. 动作脚本正常退出后，程序询问 `1=成功，0=失败`。正常退出不自动等于抓取成功。
3. **先取下瓶子、摆好下一次位置，再输入本次的 `1` 或 `0` 并回车。输入后立即开始下一次。**
4. 输入其他内容会重新询问，不开始动作。
5. 漏检、目标越界、逆运动学未通过等导致动作脚本非零退出时，程序自动记为 `0`；准备好下一次后按回车继续，无需再输入成功或失败。
6. 报错也占一次，最多启动 20 次，不自动补做失败试次。最后一次反馈后输出统计。

测试前固定成功判据，例如“瓶子抓起后保持稳定至收手结束，无掉落、无人工协助”。同一轮使用同一判据和动作参数。

### 5. 连续测试的动作参数

这组参数与上文原始单次抓取命令不同，配置在 `test_grasp_20.py` 的 `command()` 中。**连续测试关闭所有动作语音播报。**

| 参数 | 设置 | 说明 |
|---|---:|---|
| `--pre-duration` | 2.0 s | 抬手请求时长，现有速度限时可能自动延长 |
| `--lateral-duration` | 默认 3.0 s | 靠近及退回预抓取点使用该段时间 |
| `--return-duration` | 3.0 s | 最后收手 |
| `--hand-hold` | 1.5 s | 每次手部动作后的等待 |
| `--hold-before-open` | 0 s | 张手前停顿 |
| `--hold-after-open` | 0.2 s | 张手后停顿 |
| `--hold-before-grasp` | 0 s | 抓取前阶段停顿 |
| `--hold-after-grasp` | 0.3 s | 抓取后停顿 |
| `--thumb-inward-delay` | 0.3 s | 拇指内旋后等待 |
| `--hand-speed` / `--hand-force` | 300 / 800 | 手速与手部力设置 |
| `--vision-xy-offset` | -0.06 / 0.0 m | 视觉定位 XY 偏移 |
| `--no-announce` | 启用 | 关闭开始与完成播报 |

已有调试出现过等待清零后，手尚未张开就开始移动的现象，不要直接将手部等待清零。若需调参，先单次验证，再另开正式测试轮次，不把不同配置混作同一组性能数据。

### 6. 自定义次数、目录与超时

```bash
python3 example/g1/real_arm_grasp/test_grasp_20.py \
  --sdk-dir "$PWD" \
  --trials 20 \
  --output-root /home/unitree/grasp_trials/formal \
  --rgbd-url http://127.0.0.1:18082/rgbd.npz \
  --timeout 180
```

`--timeout` 单位为秒，仅针对单次动作子进程。超时后发送中断，等待原动作脚本释放手臂控制，不强制杀死仍在执行的动作进程。若程序仍未返回，先在现场核对状态，不另开第二份程序。

### 7. 中止与重新开始

`Ctrl+C` 中止整轮；动作运行中会将中断发送给子进程，并等待动作脚本清理。已有记录保留，不自动重试。

- 动作被中断的试次带有 `error: operator_interrupted`，通常自动标签为 `0`，统计时应与正常运行失败区分。
- 动作正常结束、尚未输入 `1/0` 时退出，记录保留为 `awaiting_feedback`，不擅自记为失败。
- 每次重新运行创建新的 `session_...`，不覆盖以前结果，也不自动续接中断会话。
- 程序使用文件锁并检测其他动作脚本，防止并行启动测试。相机检查也会被正在运行的动作阻止。

## 结果文件与统计口径

### 文件目录

每轮目录使用北京时间命名：

```text
/home/unitree/grasp_trials/formal/session_日期_时间/
  session.json                  本轮命令、动作脚本哈希与计时定义
  results.csv                   表格汇总，供后续绘图
  results.json                  完整结果汇总
  trial_001/
    detection_01_rgb.png         实际用于本次检测的原始图像
    detection_01_annotated.png   同一帧检测框与置信度
    detection_01_rgbd.npz        同一帧彩色、深度、内参及时间戳
    detection_01.json            检测框、像素中心和相机系三维位置
    result.json                 标签、耗时、骨盆系位置和抓取目标
    run.log                     完整动作输出及报错
  trial_002/
  ...
```

检测文件编号是整轮请求的递增编号，后续文件夹可能使用 `detection_02`、`detection_03`，并非都叫 `detection_01`。漏检时保留已经采到的图像和 `detection_*_error.json`，位置留空；相机本身无法采图时不能生成该帧。程序在保存检测数据后才向动作脚本返回定位结果。

### 主要字段

| 字段 | 含义 |
|---|---|
| `success` | 1 成功、0 失败，待反馈时可能不存在 |
| `label_source` | `user` 人工输入，`automatic_error` 程序自动标签 |
| `status` | `labelled` 已反馈、`awaiting_feedback` 待反馈、`error` 错误 |
| `exit_code` | 动作退出码，0 不能代替人工成功判定 |
| `error` | 中断、超时、启动异常等原因，可能不存在 |
| `total_seconds` | 动作子进程从启动到退出的运行时间 |
| `detection.center_xy` | 瓶子中心 `[u, v]`，单位像素，在检测 JSON 中 |
| `point_camera_m` | 相机系三维位置，单位米，在检测 JSON 中 |
| `positions.vision_point_pelvis_m` | 骨盆系瓶子位置，单位米，在结果 JSON 中 |
| `positions.final_grasp_target` | 动作规划目标，单位米，在结果 JSON 中 |

骨盆系数据来自动作日志，精度受日志打印小数位限制。`results.csv` 汇总了结果、标签来源、耗时、像素中心以及相机系和骨盆系位置。

### 耗时与成功率

总耗时包含动作脚本初始化、检测、规划、手臂与灵巧手动作、手部等待；不包含整轮模型加载、人工反馈、摆放瓶子和下载记录。语音已禁用。提前报错的短耗时也保留，但不能当作完整抓取耗时。计算完成流程均值时，应说明使用 `exit_code == 0` 的试次，以及其中成功和失败各多少次。

- **完整流程成功率**：成功数 / 全部已完成、具有结果的有效尝试数，保留检测和规划失败。
- **已检出条件下的抓取成功率**：成功数 / 有目标检测位置的有效尝试数，明确标注条件，不能代替完整流程成功率。
- **动作正常完成后的条件成功率**：成功数 / 动作正常退出且已有人工反馈的尝试数。
- 待反馈记录不进入 0/1 统计。人为中断单独列出，是否排除需说明，不能与其他自动报错混在一起删除。

不要用后续成功记录替换先前失败来提高总体成功率。展示案例可以筛选，统计范围应保持明确、可追溯。

### 位置图与热力图

用 `detection.center_xy` 将各次位置叠加到参考相机画面。漏检没有可靠坐标，单列次数，不补造位置。

固定相机、机器人和桌面有利于比较。若测试间发生移动，应分批展示或先配准，直接像素叠加不等于标定后的桌面位置。颜色分区若由邻近样本平滑得到，应注明“局部成功比例的平滑估计”，保留原始采样点，只覆盖有样本支持的区域。

## 常见问题与逻辑检查

| 现象 | 处理方式 |
|---|---|
| `18082` 连接拒绝或没有帧 | 检查已有 RGB-D 服务，检测启动器不启动相机 |
| `no detection for object='bottle'` | 查看原图、视野与光照，本次保留为自动失败 |
| `IK did not meet tolerance` 或越界 | 查看检测位置和日志，重新摆到已验证范围，不直接放宽容差 |
| 手指没有及时到位 | 查看角度反馈，先单次调试，不在正式测试中临时清零等待 |
| 检测到其他动作程序 | 等已有动作退出，不并行运行 |
| 结果不足 20 次 | 检查人工中止、终端断开或启动失败，已有记录保留 |
| 新克隆目录找不到模型或代码 | 核对根目录，并传入 `--sdk-dir "$PWD"` |

在仓库根目录运行不驱动机器人的逻辑检查：

```bash
python3 -m unittest discover \
  -s example/g1/real_arm_grasp \
  -p test_grasp_20_checks.py \
  -v
```

检查涵盖 `1/0` 交互、报错自动计 0、待反馈保留、禁用播报和模拟 20 次循环。完整循环测试依赖 Linux 文件锁，Windows 跳过这一项。模拟检查不替代真机验证。


## 文件入口

| 功能 | 文件 |
|---|---|
| 20 次连续抓取测试 | `example/g1/real_arm_grasp/test_grasp_20.py` |
| 连续测试的逻辑检查 | `example/g1/real_arm_grasp/test_grasp_20_checks.py` |
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

## 配置与上传范围

- 已移除 Wi-Fi、SSH、sudo 明文密码。需要 SSH 抓帧的脚本从 `ROBOT_SSH_PASSWORD` 环境变量读取默认密码，原命令行参数仍可覆盖。
- 服务器上的网卡、相机序列号、网络地址及动作参数保留原样；迁移前按实际设备核对。
- 包含项目源码、SDK 历史、操作说明及现有检测权重。
- 排除运行日志、临时图像、缓存、安装元数据、空占位文件和 `.bak` / `pre_safety` / `pre_retry` 备份。
- 原始仓库发布验证了文件完整性和 Python 语法。连续测试脚本另提供模拟逻辑检查；使用前仍需核对目标机器人环境并执行 `--check`。

## 上游与许可证

上游项目：[unitreerobotics/unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python)。保留原 [BSD-3-Clause LICENSE](LICENSE) 和作者信息；第三方模型及依赖适用其各自许可证。
