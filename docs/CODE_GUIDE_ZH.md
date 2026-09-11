# E-DynVLA 中文代码导读

这份文档面向第一次阅读本仓库代码的人。它不逐行翻译代码，而是回答四个问题：

1. 数据是怎样生成并保存的？
2. RGB/HDR 是怎样变成 Event 的？
3. Event 怎样做动静与光照分离？
4. RGB、Event、语言和机器人状态怎样进入 VLA 并产生动作？

## 1. 先看当前状态

仓库目前的数据生成、事件分离和训练接口已经统一。

### 1.1 当前批量数据生成链路

```text
DOM / Isaac Lab
  -> 三相机 RGB + 状态 + 动作 + v4-hybrid Event
  -> 每个 sample 保存：
       data/*.parquet
       rgb/{wrist,opst,side}.mp4
       events/{wrist,opst,side}.aedat4
       reproduction.json
```

这里保存的是 **raw Event**。AEDAT4 只保存 `x/y/t/p`，不保存
`q_static/q_dynamic/q_illumination`。这是为了让数据更接近真实 Event Camera：
传感器先给原始事件，动静分离由后续算法完成。

### 1.2 当前训练链路

```text
Parquet + 两路 RGB MP4 + wrist AEDAT4 + wrist support HDF5
  -> 首次预处理：动静/光照分离并保存带版本指纹的 q 缓存
  -> 训练读取：按 observation 截取 80ms Event
  -> static/dynamic voxel
  -> Event Token
  -> DynamicVLA + RGB/Event WAM
```

正式训练使用 `EDVSupportDataset`。Raw AEDAT4 仍是可复现的源数据；缓存保存
`q_static/q_dynamic/q_illumination`，分离算法或输入文件发生变化时会自动失效。
建议在启动多卡训练前先运行 `scripts/precompute_edv_event_cache.py`，不要让多个
DataLoader worker 在首个 epoch 同时生成大缓存。

当前训练只使用 wrist Event/support；三路 support 都保留在数据集中，便于后续扩展。
真实部署仍需由在线传感器模块产生同样的 static/dynamic voxel。

## 2. 仓库结构

```text
E-DynVLA 仓库根目录
├── E-DynVLA/                 VLA、DOM 仿真、数据打包和训练
│   ├── configs/              训练配置
│   ├── core/                 train/test 循环
│   ├── policies/
│   │   ├── dynamicvla/       保留并适配的 DynamicVLA backbone
│   │   └── edynvla/          Event 数据、Tokenizer、Event-WAM
│   ├── scripts/              数据生成、打包、校验和状态工具
│   ├── simulations/          DOM / Isaac Lab 仿真
│   └── run.py                训练入口
├── V2E-VLA/                  Event 生成与动静分离
│   ├── dvs_gen/
│   │   ├── sensors/          Isaac Camera 接口
│   │   ├── warp/             帧间 warp 与有效区域
│   │   ├── dvs/              事件触发和 HDF5 recorder
│   │   └── io/               视频等 I/O
│   ├── scripts/              分离、可视化和评价
│   └── tests/
├── benchmark/                v2/v3/v4 和动静分离对照实验
└── docs/                     项目文档
```

代码来源可以分成三类：

- 自研部分：`policies/edynvla/`、EDV 数据脚本、v3/v4、动静分离和 benchmark；
- 适配部分：`simulations/simulate.py`、DynamicVLA 的 event 接口；
- 必需上游核心：`policies/dynamicvla/` 和 `V2E-VLA/dvs_gen/` 的基础实现。

## 3. 一条样本是怎样生成的

### 3.1 批量入口

文件：[`E-DynVLA/scripts/generate_edv_dataset_to_size.sh`](../E-DynVLA/scripts/generate_edv_dataset_to_size.sh)

它负责调度，不负责算法本身：

1. 计算 `success/` 和 `failure/` 已占容量；
2. 使用文件锁分配不会重复的 sample index；
3. 在 `cuda:2` 和 `cuda:3` 上各启动一个 worker；
4. 每个 worker 循环调用单样本脚本；
5. 达到目标容量或剩余空间不足 100 GiB 时停止；
6. 单个样本失败不会终止整个批次。

服务器根盘很小，因此脚本把 `TMPDIR/TMP/TEMP` 固定到 VEPFS 大盘。

### 3.2 单样本入口

文件：[`E-DynVLA/scripts/generate_edv_samples.sh`](../E-DynVLA/scripts/generate_edv_samples.sh)

它完成一次事务式生成：

```text
创建 .edv_staging 临时目录
  -> 调用仿真脚本
  -> 调用打包脚本
  -> 校验成功后原子移动到 success/ 或 failure/
  -> 删除 staging
```

如果过程中崩溃，最终 sample 目录不会被半成品覆盖。

正式采样器是 `dom_stratified`：

- 16 类 DOM 目标物体轮换覆盖；
- 物体 USD variant 确定性轮换；
- 速度分为 `0.15–0.30`、`0.30–0.45`、`0.45–0.60`、`0.60–0.75 m/s`；
- `seed = EDV_SEED_BASE + sample_index`；
- 采样参数写入 `reproduction.json`。

### 3.3 仿真包装器

文件：[`E-DynVLA/scripts/run_pick_csv_event_demo.py`](../E-DynVLA/scripts/run_pick_csv_event_demo.py)

这个文件位于批量脚本和 DOM simulator 之间，主要做四件事：

1. 根据 sample index 选择物体类别、USD variant、速度层和 seed；
2. 启动 Isaac Lab `AppLauncher`；
3. 对 DOM 的物体状态采样函数做受控覆盖；
4. 调用 `simulations/simulate.py` 并保存生成 manifest。

它仍保留 CSV 模式和固定物体模式，正式批量生成使用的是
`--random-dom-init`，不依赖 CSV 初始条件。

### 3.4 DOM 主仿真

文件：[`E-DynVLA/simulations/simulate.py`](../E-DynVLA/simulations/simulate.py)

建议先看这些函数：

- `_get_object_states()`：生成场景中物体的状态；
- `_get_dynamic_object_state()`：生成运动物体的位置、方向和速度；
- `_set_up_scene_cameras()`：建立 wrist/opst/side 三相机；
- `get_camera_views()`：读取 RGB、深度、motion vector 等观测；
- `simulate()`：环境循环、状态机、相机与 Event 总调度；
- `is_object_stopped()`：判断运动物体是否停止；
- `get_frames()`：整理保存帧。

抓取动作来自 `simulations/state_machines/pick_sm.py`。数据是否成功由最终打包器
根据“夹爪闭合且物体抬升至少 0.10m”重新标注，而不是简单相信仿真进程正常退出。

### 3.5 最终打包

文件：[`E-DynVLA/scripts/package_edv_lerobot_sample.py`](../E-DynVLA/scripts/package_edv_lerobot_sample.py)

它把仿真的 staging HDF5 转成最终样本：

```text
sample_000000/
├── data/episode_000000.parquet
├── rgb/
│   ├── wrist_cam.mp4
│   ├── opst_cam.mp4
│   └── side_cam.mp4
├── events/
│   ├── wrist_cam.aedat4
│   ├── opst_cam.aedat4
│   └── side_cam.aedat4
└── reproduction.json
```

Parquet 每一行对应一个 25 Hz observation，字段为：

- `action`：末端动作，四元数已转 Euler；
- `observation.state`：末端位置与 Euler 姿态；
- `observation.environment_state`：物体位置、姿态和速度；
- `timestamp`：以第一帧为 0 的 observation 时间；
- `frame_index/episode_index/index/task_index`：索引信息。

AEDAT4 每个事件保存：

- `x, y`：像素坐标；
- `t`：相对共同时间原点的微秒时间戳；
- `p`：OFF/ON 极性。

`reproduction.json` 保存 seed、采样器、物体、速度层、仿真配置、Event 参数、软件
版本、代码哈希、文件哈希和结果标签。

## 4. V2E-VLA：RGB/HDR 怎样变成 Event

完整调用顺序如下：

```text
simulate.py
  -> DVSCamera.from_scene()
  -> DVSCamera.snapshot()
  -> DVSCamera.warp_and_process()
       -> adaptive_warp_steps()
       -> bidir_warp_gap()
       -> BatchedMultiCamProcessor()
  -> GeneralDVSRecorder.record()
  -> DVSCamera.flush()
```

### 4.1 Isaac 相机适配

文件：[`V2E-VLA/dvs_gen/sensors/dvs_camera.py`](../V2E-VLA/dvs_gen/sensors/dvs_camera.py)

- `DVSCameraCfg` 在普通 CameraCfg 上增加 HDR、motion vector、depth 和事件参数；
- `DVSCamera.from_scene()` 为每个相机建立 processor 与 recorder；
- `snapshot()` 读取当前关键帧；
- `warp_and_process()` 在相邻关键帧之间生成高频中间亮度；
- `_feed()` 把中间亮度、有效 mask、warp confidence 送入事件触发器；
- `flush()` 把 episode 的事件写盘。

### 4.2 帧间 warp

文件：[`V2E-VLA/dvs_gen/warp/interpolation.py`](../V2E-VLA/dvs_gen/warp/interpolation.py)

核心是 `bidir_warp_gap(A, B, mvA, mvB, K, ...)`：

1. 使用 A/B 两端 motion vector 双向 warp；
2. 使用深度与前后运动一致性估计 valid mask；
3. 对 motion-vector 边界膨胀，避免遮挡边缘误插值；
4. 对共同可见区域进行时间连续的对数亮度融合；
5. 空洞和不可靠区域保持参考亮度，不让它们凭空触发事件；
6. 输出中间帧、置信度和有效区域。

`adaptive_warp_steps()` 根据局部亮度变化决定需要多少时间节点，避免所有区间都
使用同样的时间采样密度。

### 4.3 Event 触发器

文件：[`V2E-VLA/dvs_gen/dvs/processor.py`](../V2E-VLA/dvs_gen/dvs/processor.py)

`BatchedMultiCamProcessor` 为每个像素维护上一次事件参考对数亮度。当亮度变化跨过
阈值时产生 ON/OFF event，并支持一次跨越多个阈值时生成多个事件。

v4-hybrid 的重点不是简单删事件，而是：

- 保留真实渲染关键帧和高置信度 warp 产生的阈值跨越；
- 对低置信度 warp 中的弱事件做门控；
- 使用邻域同极性强事件为真实运动边缘提供支撑；
- 被拒绝的弱事件仍更新光感受器参考，避免下一关键帧集中爆发。

### 4.4 Event recorder

文件：[`V2E-VLA/dvs_gen/dvs/recorder.py`](../V2E-VLA/dvs_gen/dvs/recorder.py)

仿真阶段先保存 HDF5，因为批量数组写入方便。最终打包器再把每个相机的
`x/y/t/p` 转成独立 AEDAT4，并执行事件数量、分辨率和时间范围 round-trip 校验。

## 5. 动静与光照分离

文件：[`V2E-VLA/scripts/separate_dynamic_static_events.py`](../V2E-VLA/scripts/separate_dynamic_static_events.py)

算法目标是把 raw event 分成软置信度，而不是硬二值标签：

```text
q_static       与相机 ego-motion 一致的静态世界事件
q_dynamic      不能被 ego-motion 解释的独立运动事件
q_illumination 更可能由曝光、阴影或高光变化造成的事件
q_unknown      几何或光度证据不足的事件
```

### 5.1 几何部分

`ego_geometry()` 或 `StreamingMotionSeparator._ego_geometry()` 使用：

- 当前深度；
- 前后相机位姿；
- 相机内参；

把当前像素反投影到三维，再投影到下一帧，得到静态世界应有的 `ego flow`。
随后比较：

```text
flow residual = observed flow - predicted ego flow
```

残差越大，独立运动置信度越高。深度前后不一致、越界和遮挡边缘会降低有效性。

### 5.2 光照抑制部分

算法把下一帧按 ego flow 对齐回来，然后计算：

- exposure-compensated log intensity residual；
- RGB chromaticity residual；
- depth residual；
- 运动残差的空间与时间持续性。

如果变化主要是亮度变化，却没有深度、色度或持续运动证据，就提高
`q_illumination` 并压低 `q_dynamic`。因此机械臂运动引起的阴影或反光不应直接被
当作动态物体。

### 5.3 离线和在线实现

- `robust_motion_calibration()` + `classify_maps()`：离线标定/评价路径，会尝试
  offset、方向和 scale，质量更稳但较慢；
- `StreamingMotionSeparator.step()`：在线路径，使用固定标定参数、预分配数组和
  相邻帧计算；
- `ReusableSeparatedEventVoxelizer`：重复使用 GPU/CPU buffer，减少数组分配；
- `SeparatedEventVoxelRing`：维护 observation 所需的 Event 历史；
- `AsyncSeparatedEventVoxelizer`：允许分离与主 observation 流水并行。

## 6. 当前训练代码怎样读取数据

### 6.1 训练入口

文件：[`E-DynVLA/run.py`](../E-DynVLA/run.py)

它只负责：

1. 读取 YAML；
2. 应用命令行覆盖；
3. 初始化 CUDA/DDP；
4. 调用 `core.train()` 或 `core.test()`。

### 6.2 训练循环

文件：[`E-DynVLA/core/train.py`](../E-DynVLA/core/train.py)

主要流程：

```text
get_dataset()
  -> DataLoader
  -> get_policy()
  -> policy.forward(batch)
  -> loss.backward()
  -> AdamW + cosine scheduler
  -> 每个 epoch 测试并保存 checkpoint
```

当前 GPU 模式默认使用 DDP。配置中的 `GRAD_ACCUM_STEPS=2` 表示累积两个 batch
再更新一次参数。

### 6.3 数据适配器

主文件：

- [`E-DynVLA/policies/edynvla/edv_support.py`](../E-DynVLA/policies/edynvla/edv_support.py)：读取 EDV sample、生成/验证缓存、读取 RGB/Parquet；
- [`E-DynVLA/policies/edynvla/data.py`](../E-DynVLA/policies/edynvla/data.py)：Event 时间窗口和 voxel 化；
- [`E-DynVLA/policies/edynvla/motion_separation.py`](../E-DynVLA/policies/edynvla/motion_separation.py)：调用 V2E-VLA 分离器。

`EDVSupportDataset` 直接读取当前的 `Parquet + MP4 + AEDAT4 + support HDF5`。
首次预处理把每个 raw event 的三种软置信度写入派生 HDF5。缓存包含输入文件信息、
算法代码哈希和 schema 版本；稳定训练阶段不会再次运行几何分离。

`SeparatedEventWindowReader.frame(frame_index)` 的对齐方式是：

```text
observation_time = event_time_origin + frame_index / 25
history_start = observation_time - 8 * 10ms
```

它用 `searchsorted` 从有序事件时间戳中截取 80ms 历史，并生成：

```text
observation.events.static   [T=8, 2, 96, 128]
observation.events.dynamic  [T=8, 2, 96, 128]
```

其中：

```text
static weight  = q_static  * (1 - q_illumination)
dynamic weight = q_dynamic * (1 - q_illumination)
```

计数经过 clip 和 `log1p` 归一化。它还生成未来 100ms 的 patch activity，作为
Event-WAM 标签：

```text
observation.events.future_activity [10, 4, 12, 16]
```

四个通道依次是 static-OFF、static-ON、dynamic-OFF、dynamic-ON。

旧的 `DOMEventDataset` 仅保留给十个 demo 的 HDF5 对照实验使用。

## 7. Event 怎样变成 Token

文件：[`E-DynVLA/policies/edynvla/event_tokenizer.py`](../E-DynVLA/policies/edynvla/event_tokenizer.py)

`SparseEventTokenizer` 不做目标检测，也不需要类别标签。输入是：

```text
static_voxels  [B, 8, 2, 96, 128]
dynamic_voxels [B, 8, 2, 96, 128]
```

处理步骤：

1. 以 `16×16` patch 做共享 Conv2d embedding；
2. 计算每个 patch 的事件密度；
3. 每个时间 bin、每种类型各选密度最高的 8 个 patch；
4. 加上时间、空间坐标和 static/dynamic 类型 embedding；
5. 两层 Transformer 在 Event token 内部做时空聚合；
6. 投影到 VLM 的 768 维 token 空间。

默认最大 token 数：

```text
2 个 summary token + 2 种事件 × 8 bins × 8 patches = 130 tokens
```

没有事件的 patch 会通过 mask 屏蔽，而不是用伪事件填充。

## 8. Event Token 怎样进入 VLA

核心文件：[`E-DynVLA/policies/dynamicvla/modeling_dynamicvla.py`](../E-DynVLA/policies/dynamicvla/modeling_dynamicvla.py)

需要优先看两个类：

- `DynamicVLAPolicy`：数据归一化、模态预处理、训练/推理外层接口；
- `VLAFlowMatching`：VLM、Event token、state token 和 action expert 的主体。

`DynamicVLAPolicy.forward()` 的顺序是：

```text
保留 Event 字段（不做 state/action normalizer）
  -> 归一化 RGB/state/action
  -> prepare_images / prepare_language / prepare_state
  -> prepare_events
  -> SparseEventTokenizer
  -> VLAFlowMatching.forward
```

`VLAFlowMatching._embed_prefix()` 把各模态拼成 prefix：

```text
[RGB tokens, Event tokens, Language tokens, State token]
```

动作 chunk 加噪后作为 suffix。模型学习 flow matching 速度：

```text
x_t = t * noise + (1 - t) * action
target velocity u_t = noise - action
action loss = MSE(predicted velocity, u_t)
```

推理时从噪声动作开始，多步调用 `denoise_step()` 得到动作 chunk。

## 9. Event-WAM 是什么

文件：[`E-DynVLA/policies/edynvla/event_wam.py`](../E-DynVLA/policies/edynvla/event_wam.py)

当前 WAM 不是全分辨率视频生成器。它是训练期的短时世界模型辅助头：

```text
RGB/Event/语言/状态共享上下文 + action context
  -> Transformer Decoder
  -> 未来 100ms 的 12×16 RGB
  -> 未来 10×10ms 的四通道 Event patch activity
```

总损失为：

```text
total loss = action flow-matching loss
           + WAM_LOSS_WEIGHT * (RGB Smooth-L1 + Event BCE)
```

默认权重是 `0.1`，正样本权重是 `4.0`，用于缓解未来事件图稀疏造成的正负不平衡。
关闭 `WAM_ENABLED` 即可做消融实验。训练时 WAM 使用真实 action chunk；推理时可以
用预测 action chunk 调用 `predict_action_chunk_with_world()` 查看未来世界预测。

## 10. 训练配置怎么读

文件：[`E-DynVLA/configs/edynvla.yaml`](../E-DynVLA/configs/edynvla.yaml)

最重要的配置如下：

| 配置 | 当前值 | 含义 |
|---|---:|---|
| `N_OBS_STEPS` | 2 | RGB observation 历史长度 |
| `CHUNK_SIZE` | 20 | 一次预测的动作长度 |
| `VLM_MODEL_NAME` | SmolLM2-360M | 语言/VLM 主干 |
| `TEMPORAL_FUSION` | attn | 多帧 RGB 融合方式 |
| `USE_EVENT_TOKENS` | true | 开启 Event 分支 |
| `EVENT_HISTORY_BINS` | 8 | Event 历史 bin 数 |
| `EVENT_BIN_MS` | 10ms | 每个 bin 的时间长度 |
| `EVENT_OUTPUT_SIZE` | 96×128 | voxel 空间分辨率 |
| `EVENT_PATCH_SIZE` | 16 | Event patch 大小 |
| `EVENT_MAX_PATCHES_PER_BIN` | 8 | 每个 bin/类型保留 patch 数 |
| `WAM_ENABLED` | true | 开启未来 RGB/Event 辅助任务 |
| `WAM_LOSS_WEIGHT` | 0.1 | WAM loss 权重 |
| `BATCH_SIZE` | 16 | 每张卡的 batch 大小 |
| `GRAD_ACCUM_STEPS` | 2 | 梯度累积步数 |

视觉、connector 和文本模型当前被冻结，主要训练 action expert、state projection、
Event Tokenizer 和 Event-WAM。正式训练时仍需根据显存核对真实可训练参数和 batch。

## 11. 推荐阅读顺序

如果只想用一两个小时理解整体，按下面顺序看：

1. 本文第 1、3、4、6、7、8 节；
2. `configs/edynvla.yaml`；
3. `generate_edv_samples.sh`；
4. `run_pick_csv_event_demo.py`；
5. `dvs_camera.py` 的 `warp_and_process()`；
6. `interpolation.py` 的 `bidir_warp_gap()`；
7. `processor.py` 的 `__call__()`；
8. `separate_dynamic_static_events.py` 的 `StreamingMotionSeparator.step()`；
9. `policies/edynvla/data.py`；
10. `event_tokenizer.py`；
11. `DynamicVLAPolicy.forward()` 和 `VLAFlowMatching._embed_prefix()`；
12. `event_wam.py`；
13. 最后再看 `core/train.py`。

`modeling_fastvlm.py` 和 `modeling_vlm_with_expert.py` 很长，第一次阅读不必逐行看。
它们主要是保留的 backbone 实现，先理解输入输出接口即可。

## 12. 想修改某部分时应该改哪里

| 想改的内容 | 首要文件 |
|---|---|
| DOM 物体、速度、seed 采样 | `run_pick_csv_event_demo.py` |
| 抓取控制逻辑 | `simulations/state_machines/pick_sm.py` |
| v4 光晕与 warp | `V2E-VLA/dvs_gen/warp/interpolation.py` |
| Event 阈值和 hybrid gate | `V2E-VLA/dvs_gen/dvs/processor.py` |
| 相机和 Event 总调度 | `V2E-VLA/dvs_gen/sensors/dvs_camera.py` |
| 动静/光照分离 | `V2E-VLA/scripts/separate_dynamic_static_events.py` |
| AEDAT4/Parquet/MP4 打包 | `package_edv_lerobot_sample.py` |
| observation 时间窗口 | `policies/edynvla/data.py` |
| Event token 数量和结构 | `policies/edynvla/event_tokenizer.py` |
| Event-WAM | `policies/edynvla/event_wam.py` |
| Event 与 VLM 融合位置 | `modeling_dynamicvla.py` 的 `_embed_prefix()` |
| loss、优化器、checkpoint | `core/train.py` 与 `configs/edynvla.yaml` |

## 13. 正式训练前的代码清单

训练前执行：

1. 设置 `EDV_SUPPORT_ROOT`、`V2E_VLA_ROOT` 和可写的 `EDV_CACHE_ROOT`；
2. 单进程运行 `scripts/precompute_edv_event_cache.py`；
3. 运行 `pytest -q tests` 和至少一个 DataLoader → model forward smoke test；
4. 用 `--ckpt` 指定 DynamicVLA 预训练权重，冻结主干时禁止随机初始化；
5. 核对两张训练卡、batch size、缓存盘容量和 train/test 数量；
6. 先训练一个短 epoch，检查 action、WAM RGB 和 WAM Event loss 都能正常下降。

推理输入必须包含当前时刻的 `observation.events.static` 和
`observation.events.dynamic`。真实相机或仿真服务需要在发送 observation 前完成在线
窗口聚合；推理脚本不会凭空从 RGB 恢复 Event。

建议保留以下消融：RGB only、RGB+raw Event、RGB+static、RGB+dynamic、
RGB+static+dynamic、RGB+static+dynamic+Event-WAM。
