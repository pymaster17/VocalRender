# VocalRender 代码结构

## 1. 目录结构

```text
VocalRender/
├── conf/                              # 训练 / 推理 / 预处理配置
│   ├── svs_train.yaml
│   ├── svs_infer.yaml
│   └── svs_preprocess.yaml
├── scripts/                           # 命令行入口
│   ├── train_vocalrender_svs.py       # SVS 训练 wrapper（解析配置后调用 runner）
│   ├── infer_vocalrender_svs.py       # 批量推理与离线评估
│   ├── infer_vocalrender_svs_single.py # 单样本推理
│   ├── preprocess_svs_data.py         # SVS 预处理入口
│   └── setup_svs_tokenizer.py         # SVS token 扩展工具
│   └── sync_space.py                  # 将 demo/ + src/vocalrender/ 发布到 HF Space
├── src/
│   └── vocalrender/
│       ├── model/                     # 主模型与 SVS token 工具
│       ├── modules/                   # 神经网络子模块
│       ├── training/                  # 训练基础设施与 runner
│       ├── preprocessing/             # SVS 预处理库
│       ├── evaluation/                # 推理与评估
│       ├── inference/                 # 批推理脚本共享的 backend 抽象
│       └── utils/                     # score rendering、ABC/MusicXML 导入等工具
├── demo/                              # 网页 Demo（Gradio 钢琴卷帘 UI），同步到 HF Space
│   ├── app.py
│   ├── assets/piano_roll/             # 钢琴卷帘前端（score_model.js / piano_roll.js）
│   ├── assets/*.wav                   # 参考音色（Git LFS）
│   └── tests/                         # 前端单元测试、浏览器与 GPU 冒烟测试
├── nanovllm-voxcpm/                   # nano-vllm 推理 backend（git submodule）
├── docs/
└── pyproject.toml
```

## 2. 分层说明

### 2.1 模型层 (`src/vocalrender/model/`)

- `voxcpm.py` / `VoxCPMModel`
  - V1 / V1.5 主模型。
  - 负责 TSLM、RALM、LocEnc、LocDiT、AudioVAE、FSQ 的初始化与串接。
  - 对外提供 `forward()`（训练）与 `generate_batch()`（推理）。
- `voxcpm2.py` / `VoxCPM2Model`
  - V2 主模型，对接 VoxCPM2 预训练权重与 V2 声码器 / DiT 结构。
  - 保持与 V1 基本一致的训练 / 推理接口。
- `svs_utils.py`
  - 管理 SVS token 定义、embedding 初始化与时长估计工具。
- `utils.py` / `model_factory` 辅助（输出采样率等）。

### 2.2 子模块层 (`src/vocalrender/modules/`)

- `minicpm4/` — TSLM / RALM 的共享骨干（KV cache、注意力、gradient
  checkpointing）。
- `layers/` — FSQ 与 LoRA 等基础层实现。
- `locenc/` — Prompt audio / 历史音频的局部编码器。
- `locdit/` — 连续声学特征生成；`unified_cfm.py` 承担 flow matching 与
  DiTAR 温度采样。
- `audiovae/` — `audio_vae.py`（V1 / V1.5）与 `audio_vae_v2.py`（V2）。

### 2.3 训练层 (`src/vocalrender/training/`)

#### 配置层

- `config.py` — `SVSTrainConfig`（typed dataclass 分组配置），散平 YAML
  normalize 兼容层，脚本统一入口 `parse_script_config()`。

#### 公共基础设施

- `accelerator.py` — DDP / FSDP / ZeRO-2 / ZeRO-3 封装。
- `tracker.py` — 训练日志与 TensorBoard 记录。
- `runtime.py` — 保存目录、writer / tracker、optimizer / scheduler、
  训练精度解析、LR 调度器工厂、SIGTERM/SIGINT 处理。
- `loop_schedule.py` — 验证、音频评估与 checkpoint 触发条件。
- `checkpoint.py` — FSDP / DDP 感知的 checkpoint 读写。
- `resume.py` — DataLoader 位置与 RNG 状态恢复。
- `diagnostics.py` — 显存归因与 per-rank 耗时诊断。
- `model_factory.py` — 读取 `config.json` 的 `architecture` 字段，
  选择 V1 / V2 模型类。

#### Runner 编排层

- `runners/svs.py` — SVS 训练主编排入口，拆分为 `build_runtime`、
  `build_model_and_tokenizer`、`build_datasets_and_loaders`、
  `build_eval_context`、`train_one_step`、`run_validation_and_checkpoint`。

#### 数据层

- `svs_loading.py` — Arrow 数据加载与 train / val 切分。
- `svs_data.py` — Dataset wrapper 与 dataloader：多策略 score masking、
  prompt audio 拼接、动态 batch。
- `svs_raw_data.py` — 原始文件夹管线与注解转换。
- `dataset_ops.py` — 数据集过滤、时长裁剪、上采样、歌曲索引构建。
- `dynamic_batch.py` — `DynamicBatchSampler`。
- `packers.py` — `AudioFeatureProcessingPacker`：将文本 token 与音频波形
  打包为统一多模态序列（SVS 使用 aggregated prompt 布局，melisma 行复用
  上一音节的文本 token），实时 AudioVAE 编码，输出 text_mask /
  audio_mask / loss_mask / labels。
- `data.py` — 多 dataset stream 拼接与按比例采样。

#### 验证与评估装配

- `svs_eval_setup.py` — 评估器与 AudioVAE loader 工厂。
- `validation.py` — loss eval 与音频 eval 的总调度。
- `val_audio.py` — 多 GPU 验证音频生成与 TensorBoard 记录。
- `svs_tokenizer.py` — 运行时向 tokenizer 注入 SVS token 并收集
  mask 索引（与 `model/svs_utils.py` 分工：后者定义 token 映射表与
  embedding 初始化逻辑）。
- `vae_loader.py` — AudioVAE 懒加载（验证时按需上 GPU）。

`training/__init__.py` 通过 PEP 562 `__getattr__` 惰性解析 `Accelerator`、
`SVSTrainConfig` 等再导出名：推理路径（预处理里的 `svs_data` 音节工具、网页
Demo）导入子模块时不会连带加载 argbind / 分布式训练栈。

### 2.4 预处理层 (`src/vocalrender/preprocessing/`)

- `svs_preprocessor.py` — `SVSPreprocessor`：VAE 编码与 token 序列构建。
- `svs_prompt.py` — 从元数据重建 SVS prompt。
- `data_loaders.py` — 多格式数据加载（json_file / folder_based）。
- `text_tensor.py` — 文本张量构建与音符时长估计。
- `arrow_writer.py` — Arrow 数据集写入（多 GPU 动态分发）。

### 2.5 推理与评估层 (`src/vocalrender/evaluation/`)

- `inference.py` — 单样本 / 批量 / 多卡推理。
- `multi_gpu.py` — 多 GPU worker 池（训练验证 `val_audio.py` 直接使用；
  外部批推理脚本走 `inference/backends/` 抽象层，见 §2.6）。
- `svs_metrics.py` — `SVSEvaluator` 统一评估入口（SingMOS / AES +
  `register_metric_backend` 自定义指标 seam）。
- `metrics.py` — SingMOS / AES 指标实现。
- `visualization.py` — 乐谱条件可视化（TensorBoard 图）。
- `audio_utils.py` — 音频归一化、参考音频解码与批量 latent 解码原语。

与训练层相同，`evaluation/__init__.py` 惰性导出：只用 `audio_utils` 的调用方
（如网页 Demo）不会触发 s3prl / matplotlib 等指标依赖的导入。

### 2.6 推理 backend 层 (`src/vocalrender/inference/`)

批推理脚本共享的**模型生成 backend 抽象层**。

- `backends/base.py` — `TTSInferenceBackend` ABC + `TTSRequest` /
  `TTSResult` 数据契约。
- `backends/multi_gpu.py` — `MultiGPUBackend`：process-per-GPU 生成
  backend，每张卡一个持久 worker process，支持 V1/V1.5/V2 与
  `prompt_audio_feats`。
- `backends/nano_vllm.py` — `NanoVLLMBackend`：包 `nano-vllm-voxcpm` 的
  `AsyncVoxCPM{,2}ServerPool`，continuous batching。启动时对 SVS finetune
  ckpt 做 vocab-size side-car 补丁（产物写入 `<ckpt>.nanovllm_patched/`）。
- `backends/factory.py` — `build_tts_backend(cfg, pretrained_path)`，按
  YAML `inference_backend.type` 选择 backend。

详见 `docs/inference_backends.md`。

### 2.7 工具层 (`src/vocalrender/utils/`)

- `score_rendering.py` — 乐谱渲染工具（音符序列 → 五线谱 PNG，依赖
  可选 extra `[viz]`：music21 + lilypond）。
- `score_import.py` — ABC / MusicXML（`.musicxml` / `.xml` / `.mxl`）乐谱导入：
  解析 → 选择声部与小节区间 → 转成 `word / pitch / duration_index` 事件行
  （连音线合并为 melisma，休止符记作 `SP`）。依赖 extra `[demo]`（music21 +
  defusedxml）。测试见 `tests/test_score_import.py`。

### 2.8 网页 Demo 层 (`demo/`)

Gradio 应用，也是 HF Space `pymaster/VocalRender-demo` 的唯一代码来源。

- `app.py` — 页面布局与事件绑定；`_ModelCache` 单驻留模型缓存（切换
  checkpoint 时先释放旧模型再加载）；`@spaces.GPU` 在 ZeroGPU 上调度生成，
  自部署时 `spaces` 缺失则退化为空装饰器；`VOCALRENDER_UI_ONLY=1` 跳过模型
  加载以便无 GPU 开发。乐谱的单一数据源是
  `{"rows": [...], "bpm": int, "beats_per_bar": int}`，`rows` 与预设 / 导入器
  产出的 `word / word_index / uid / pitch / duration_index` 行完全一致。
- `assets/piano_roll/` — 钢琴卷帘前端：`score_model.js`（纯乐谱逻辑，无 DOM，
  `node --test` 可测）、`piano_roll.js`（交互、撤销重做、WebAudio 试听）、
  `piano_roll.html` / `.css`（`gr.HTML` 模板）。
- `assets/*.wav`、`assets/fonts/Bravura.woff2` — 参考音色与乐谱字体（Git LFS）。
- `requirements.txt` — Space 侧安装清单；仓库内安装用 `pip install -e ".[demo]"`。
- `tests/` — 值转换单测、Playwright 浏览器驱动、`gpu_smoke.py` + Slurm 脚本。

发布链：`scripts/sync_space.py` 把 `demo/` 与 `src/vocalrender/` 拼成 Space 目录
并 `upload_folder` 镜像推送；`.github/workflows/sync-space.yml` 在 `main` 的相关
路径变更时自动执行（需要仓库 secret `HF_TOKEN`）。

## 3. 数据流

### 3.1 训练数据流

```text
原始音频 + 标注
       │
       ▼
  preprocessing/
  ├─ data_loaders.py   多格式加载
  ├─ text_tensor.py    文本张量 + 时长估计
  └─ svs_preprocessor.py  VAE 编码 + token 序列构建
       │
       ▼
  Arrow 文件 (磁盘持久化)
       │
       ▼
  training/
  ├─ svs_loading.py    Arrow → train/val 切分
  ├─ svs_data.py       Dataset wrapper + masking + prompt 拼接
  ├─ dynamic_batch.py  DynamicBatchSampler
  ├─ dataset_ops.py    过滤 / 裁剪 / 上采样
  └─ packers.py        文本+音频 → 统一多模态序列
       │
       ▼
  runners/svs.py → model forward
```

### 3.2 推理数据流

```text
SVS 输入 (乐谱标注 + 必需的 prompt 音频)
       │
       ▼
  evaluation/inference.py  (或 inference/backends/*)
  ├─ 构建 prompt token 序列
  ├─ 模型自回归生成
  └─ AudioVAE 解码 → 波形
       │
       ▼
  evaluation/svs_metrics.py  →  SingMOS / AES 指标
```

网页 Demo 的路径更短，不经过 backend 抽象与指标：

```text
钢琴卷帘 rows + bpm (或 ABC / MusicXML → utils/score_import.py)
       │
       ▼
  demo/app.py
  ├─ preprocessing.rebuild_svs_prompt → prompt token 序列
  ├─ 模型自回归生成（单驻留 checkpoint）
  └─ AudioVAE 解码 → 48 kHz 波形 → 浏览器播放
```
