[README.md](https://github.com/user-attachments/files/33591968/README.md)
# YOLO26n 装甲板模型 · 视频推理（infer.py）

用参考材料中的 **YOLO26n 装甲板模型 `best.pt`**（Ultralytics Pose 模型，6 类：
`B1 B2 B3 R1 R2 R3`，每个目标 4 个关键点）对自己拍摄的视频做推理，
把**检测框 + 类别 + 置信度 + 4 个关键点**画在视频上，并保存一个与原视频逐帧对应、
可以正常播放的结果视频。

## 1. 目录结构

```
infer.py/
├── infer.py              # 推理脚本（作业主体，PyCharm 里直接运行这个文件）
├── best.pt               # YOLO26n 装甲板关键点模型权重
├── videos/
│   └── eb3f6c6e19e16dca3b5c634497e6af2c.mp4   # 自己拍摄的视频
├── outputs/              # 结果视频输出目录（运行后自动生成）
├── requirements.txt
└── README.md
```

> 推理用的视频是 `videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4`。
> 想换别的视频：同名覆盖它，或用 `--source` 指定路径，或改 `infer.py` 顶部的 `DEFAULT_SOURCE`。

## 2. 环境依赖

```bash
pip install -r requirements.txt
```

已安装的机器上（本机 `infer-py` 环境）无需重复安装。CPU 也能跑，有 GPU 会快很多。

## 3. 运行方式

### 3.1 在 PyCharm 里运行

1. 用 PyCharm 打开本项目，`File → Settings → Project: infer.py → Python Interpreter`
   选择已经装好 `ultralytics`/`opencv-python`/`torch` 的解释器；
2. 打开 `infer.py`，按 **Shift+F10**（或点绿色三角）运行 —— 会使用脚本顶部的默认配置。

默认配置在 `infer.py` 开头，一眼就能找到、直接改：

```python
PROJECT_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL = str(PROJECT_DIR / "best.pt")                         # 模型权重
DEFAULT_SOURCE = str(PROJECT_DIR / "videos" / "eb3f6c6e19e16dca3b5c634497e6af2c.mp4")  # 自己拍摄的视频
DEFAULT_OUTPUT = str(PROJECT_DIR / "outputs" / "eb3f6c6e19e16dca3b5c634497e6af2c.mp4")  # 结果视频
```

路径都是相对脚本所在目录自动拼出来的，整个文件夹拷到别的机器也不用改。

### 3.2 命令行运行（路径通过参数修改）

```bash
# 用默认配置
python infer.py

# 指定模型 / 视频 / 输出位置
python infer.py --model best.pt --source videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4 --output outputs/result.mp4

# 只用 CPU、降低阈值、弹窗预览
python infer.py --device cpu --conf 0.3 --show

# 用摄像头（编号 0）
python infer.py --source 0 --output outputs/camera.mp4
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | `best.pt` | 模型权重路径 |
| `--source` | `videos/eb3f6c6e19e16dca3b5c634497e6af2c.mp4` | 输入视频路径（写 `0`/`1` 表示摄像头） |
| `--output` | `outputs/eb3f6c6e19e16dca3b5c634497e6af2c.mp4` | 结果视频路径，目录不存在会自动创建 |
| `--conf` | `0.25` | 置信度阈值 |
| `--iou` | `0.70` | NMS 的 IoU 阈值 |
| `--imgsz` | `640` | 推理输入尺寸 |
| `--device` | `auto` | `auto` / `cpu` / `0` / `1` |
| `--max-det` | `50` | 单帧最大检测数 |
| `--kpt-conf` | `0.25` | 关键点置信度阈值（模型输出带关键点置信度时生效） |
| `--codec` | `auto` | 输出编码器：`auto`/`mp4v`/`avc1`/`XVID`/`MJPG` |
| `--show` | 关 | 推理时弹窗实时预览，按 `q` 提前退出 |
| `--save-txt` | 关 | 额外保存 `结果.txt`（帧号、类别、置信度、框、4 个关键点坐标） |
| `--max-frames` | `0` | 只处理前 N 帧，便于快速调试 |

运行日志会打印：类别与关键点信息、视频分辨率/帧率/帧数、推理设备、
每 2 秒一次的进度、类别统计与平均置信度，以及最后对输出视频的校验结果。

## 4. 实现要点（对应作业要求）

1. **加载权重 + 读取视频**：`YOLO(best.pt)` 载入权重（自动识别为 `pose` 任务、
   4 个关键点），`cv2.VideoCapture` 逐帧读取自己拍的视频，不一次性把视频读进内存。
2. **逐帧推理**：每帧调用 `model.predict(frame, conf=…, iou=…, imgsz=…, device=…)`。
3. **画检测框 / 类别 / 置信度 / 4 个关键点**：
   - 检测框用类别颜色绘制，框上方标签形如 `T1 R2 0.93`（目标编号 + 类别 + 置信度）；
   - **4 个关键点直接取自模型输出 `results[0].keypoints`**，不是用检测框的四个角代替；
     代码里有关键点来源的注释，坐标 `(N,4,2)` 原样使用；
   - 每个关键点画成"实心圆 + 白色外环 + 黑色细边"，旁边标注序号 1~4，
     4 个点再按顺序连成四边形 —— 同一目标使用同一颜色，
     多个目标同时出现时靠颜色和 `T1/T2` 编号区分归属，任何背景下都清晰可见；
   - 模型若输出关键点置信度（`kpt_shape=[4,3]`），低于阈值的点画成灰色空心圈。
4. **保存可播放的结果视频**：
   - 用原视频的 **帧率、分辨率** 建立 `cv2.VideoWriter`（`mp4v`，必要时自动回退
     `avc1`/`MJPG`），逐帧写入，保证与原视频**帧帧对应、时长一致**；
   - 写完会重新用 `cv2.VideoCapture` 打开结果视频，校验帧数/帧率/分辨率并打印结论，
     确认文件能正常解码播放；
   - 宽高不是偶数时自动补 1 像素（H.264/MPEG-4 的要求），避免花屏。

## 5. 常见问题

* **`ModuleNotFoundError: No module named 'ultralytics'`**
  当前 PyCharm 解释器没装依赖。切到装好的解释器，或 `pip install -r requirements.txt`。
* **找不到视频**
  报错信息会给出应有的路径；用 `--source` 指定自己的视频即可。
* **输出视频播放不了 / 花屏**
  换编码器重跑：`--output outputs/result.avi --codec MJPG`（兼容性最好）。
* **运行慢**
  允许用 CPU，不要求实时；有 NVIDIA 显卡时 `--device 0`（默认 `auto` 会自动用 GPU）。
* **手机竖拍视频方向不对**
  脚本按原始像素保存（不改画面内容）。如果原视频带旋转元数据，
  建议先用播放器/剪辑软件把视频转正再推理，日志里也会给出提示。
* **想检查每一帧的坐标**
  加 `--save-txt`，输出 `outputs/xxx.txt`（帧号、类别、置信度、xyxy 框、4 个关键点坐标）。
