# VocalRender

[English](README.md) | **简体中文**

[![arXiv](https://img.shields.io/badge/arXiv-2607.27768-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2607.27768)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-VocalRender-FFD21E?logo=huggingface)](https://huggingface.co/pymaster/VocalRender)
[![Hugging Face Dataset](https://img.shields.io/badge/Hugging%20Face-CrawlSinger--OS-FFD21E?logo=huggingface)](https://huggingface.co/datasets/pymaster/CrawlSinger-OS)
[![在线推理 Demo](https://img.shields.io/badge/Hugging%20Face-%E5%9C%A8%E7%BA%BF%E6%8E%A8%E7%90%86%20Demo-FFD21E?logo=huggingface)](https://huggingface.co/spaces/pymaster/VocalRender-demo)
[![音频演示](https://img.shields.io/badge/Audio-Demo-183d32.svg)](https://pymaster17.github.io/VocalRender/)

无需本地部署，即可在浏览器中通过 **[在线推理 Demo](https://huggingface.co/spaces/pymaster/VocalRender-demo)** 体验两个已发布的 checkpoint。

**VocalRender** 是一个面向真实作曲流程的乐谱原生歌声合成(SVS)系统。它通过
原创的歌词—音符交错表示、连续声学 latent 与自回归扩散建模,直接将作曲者使用的
符号乐谱渲染为歌声音频。模型采用**「字 / 音高 / 音符」交错的乐谱 prompt**:

```
<BPM_90> 感<P_62><NOTE_8> 受<P_62><NOTE_DOT_16><P_60><NOTE_16> 停<P_59><NOTE_8> ... <audio_start> [audio latents...]
```

每个歌词字后面跟一个或多个 `(音高, 音符时值)` token 对,再加一个全局 BPM token。
模型同时接收必需的 prompt 歌声片段作为音色条件,并将乐谱渲染为 48 kHz 歌声。

本仓库是最小化开源版本,包含:数据预处理、训练与推理。

## 原理简介

不同于需要为每个字或音素指定精确时长的 SVS 系统,或依赖时序对齐参考音频、
F0 曲线的系统,VocalRender 直接接收作曲者使用的符号信息:歌词、MIDI 音高、
音符时值和全局速度。模型在遵循乐谱的同时自主生成自然时序与富有表现力的
演唱细节。

![基于时长、基于参考以及本文提出的乐谱原生歌声合成输入方式对比](assets/intro.png)

VocalRender 包含三个关键设计:

1. **乐谱原生的交错表示。** 乐谱 tokenizer 先编码 BPM,再依次排列每个歌词
   音节及其对应的全部 `(音高, 音符时值)` 对,显式保留歌词与音符的对应关系,
   并自然支持一个音节横跨多个音符的转音(melisma)。
2. **连续声学表示。** Audio VAE 将歌声音频编码为紧凑的连续 latent,避免离散
   声学量化的信息瓶颈,保留细粒度的音高、音色与咬字信息。
3. **自回归扩散生成。** AR Transformer 生成韵律草图并逐个 patch 预测序列长度,
   LocDiT 进一步还原高保真的局部声学 latent,最后由 VAE decoder 解码为波形。
   因而模型无需显式时长预测器或时序对齐的声学引导。

![VocalRender 整体架构与乐谱 tokenization 流程](assets/structure.png)

## Demo 推理

这是最短的端到端复现路径:安装依赖、下载已发布 checkpoint,然后运行仓库内置的
乐谱与 prompt 音频。生成需要支持 CUDA 的 GPU,但**无需修改配置文件、准备 JSON
或自行提供 prompt 音频**。

```bash
git clone --recurse-submodules https://github.com/pymaster17/VocalRender.git
cd VocalRender

python -m venv .venv && source .venv/bin/activate
pip install -e .

hf download pymaster/VocalRender \
    --include "VocalRender/*" \
    --local-dir pretrained_models

python scripts/infer_vocalrender_svs_single.py \
    --ckpt_dir pretrained_models/VocalRender \
    --json_file examples/opencpop_demo.json \
    --item_name 2003000087 \
    --prompt_audio examples/prompt_audio/2003000081.wav \
    --output outputs/demo_2003000087.wav
```

命令会直接生成 48 kHz 音频 `outputs/demo_2003000087.wav`。上述模型、乐谱和
prompt 配对已通过从 Hugging Face 全新下载 checkpoint 的完整复现测试。仓库共
提供三组可直接运行的样例:

| 乐谱 `item_name` | 内置 prompt 音频 | Prompt 时长 |
|---|---|---:|
| `2003000087` | `examples/prompt_audio/2003000081.wav` | 6.17 秒 |
| `2017000646` | `examples/prompt_audio/2017000644.wav` | 4.19 秒 |
| `2044001652` | `examples/prompt_audio/2044001666.wav` | 5.33 秒 |

运行其他样例时只需按照表格替换 `--item_name`、`--prompt_audio` 和 `--output`。
Demo JSON 仅保留推理字段;`word_dur` 和 `pitch_dur` 是用于可视化和评测的可选
元数据。内置片段选自 [OpenCpop](https://wenet.org.cn/opencpop/),仍遵循其原始
使用条款。

Prompt 音频是必需输入:已发布 checkpoint 的所有训练样本均使用 prompt 音频
(`prompt_audio_prob: 1.0`)。使用自定义乐谱时,请提供 2-8 秒的干净歌声片段,
它也用于指定目标音色。

**VocalRender** 与 **VocalRender-Pro** 的架构、参数量及语音预训练基础模型初始化
完全相同,两者仅采用不同的训练策略(训练数据与训练 schedule):VocalRender 在
CrawlSinger-OS 上先进行合成数据预训练、再使用真实数据 finetune;
VocalRender-Pro 则在规模更大的真实歌声 CrawlSinger 上进行更长时间训练。完整
设置详见论文。Pro checkpoint 使用相同的推理命令:

```bash
hf download pymaster/VocalRender \
    --include "VocalRender-Pro/*" \
    --local-dir pretrained_models

# 然后将 --ckpt_dir 替换为 pretrained_models/VocalRender-Pro。
```

每个 checkpoint 约需下载 9.5 GB。

## 网页 Demo(钢琴卷帘)

[在线推理 Demo](https://huggingface.co/spaces/pymaster/VocalRender-demo) 的代码维护在
[`demo/`](demo/) 目录:一个带钢琴卷帘乐谱编辑器、ABC / MusicXML 导入、参考音色选择和
checkpoint 切换的 Gradio 应用。Space 由本仓库发布(`scripts/sync_space.py`),同一个应用也可以
部署在自己的 GPU 服务器上:

```bash
pip install -e ".[demo]"          # 需要 Python >= 3.11;请先按显卡驱动安装 torch
python demo/app.py                # http://127.0.0.1:7860
```

Docker 镜像、环境变量和测试见 [demo/README.md](demo/README.md)。

## 批量推理

批量推理在预处理好的验证集上运行,输出生成的 WAV、可选乐谱 PNG 以及
`metrics_summary.json`:

```bash
python scripts/infer_vocalrender_svs.py --config_path conf/svs_infer.yaml
```

在 [conf/svs_infer.yaml](conf/svs_infer.yaml) 中配置数据集和 checkpoint 路径。
提供两种后端(详见 [docs/inference_backends.md](docs/inference_backends.md)):
`multi_gpu` 是默认的进程内后端,支持 prompt 音频与乐谱渲染;
`nano_vllm` 通过连续批处理加速仅计算指标的任务。

仅在需要对应功能时安装可选组件:

```bash
# 五线谱乐谱渲染(save_score: true)
pip install -e ".[viz]"

# nano-vllm 推理后端
pip install -e ./nanovllm-voxcpm
```

## 训练

### 数据预处理

训练数据已发布在
[CrawlSinger-OS](https://huggingface.co/datasets/pymaster/CrawlSinger-OS),
其中包含用于训练的 Muse、Muchin、SongFormDB、OpenSinger、M4Singer 和
GTSinger,以及作为独立验证集的 Opencpop。
下载独立 TAR 分片,并恢复
[conf/svs_preprocess.yaml](conf/svs_preprocess.yaml) 默认使用的目录结构:

```bash
hf download pymaster/CrawlSinger-OS \
    --repo-type dataset \
    --local-dir data/CrawlSinger-OS-release

mkdir -p data/CrawlSinger-OS
find data/CrawlSinger-OS-release -type f -name '*.tar' -print0 |
    while IFS= read -r -d '' shard; do
        tar -xf "$shard" -C data/CrawlSinger-OS
    done

for dataset in opensinger m4singer gtsinger opencpop; do
    cp "data/CrawlSinger-OS-release/${dataset}/annotations.json" \
       "data/CrawlSinger-OS/${dataset}/annotations.json"
done
```

归档中直接包含三个 `folder_based` 数据集;四个 `json_file` 数据集的音频
位于 `<dataset>/audio/`。复制标注文件后即可使用默认预处理配置。请为下载的
归档和解包后的数据同时预留足够磁盘空间。

为每个音频片段标注 字 / 音高 / 音符 字段(schema 说明见
[conf/svs_preprocess.yaml](conf/svs_preprocess.yaml)):

```jsonc
[
  {
    "item_name": "Alto-1#newboy#0000",
    "wav_fn":    "Alto-1#newboy/0000.wav",
    "word":       ["AP", "感", "受", "SP"],
    "word_dur":   [0.14, 0.31, 0.42, 0.20],
    "pitch":      [0, 62, 62, 0],
    "note":       ["<NOTE_8>", "<NOTE_8>", "<NOTE_DOT_16>", "<NOTE_8>"],
    "pitch_dur":  [0.14, 0.31, 0.42, 0.20],
    "pitch2word": [0, 1, 2, 3],
    "bpm":        90
  }
]
```

`word_dur` 和 `pitch_dur` 是仅用于可视化和评测的可选输入字段,不参与乐谱
prompt 构建或模型训练。必需的乐谱字段为 `word`、`pitch`、`note`、
`pitch2word` 和 `bpm`。

然后将音频编码为 AudioVAE-V2 latent(Arrow 分片):

```bash
python scripts/preprocess_svs_data.py conf/svs_preprocess.yaml
```

### 准备基础模型

以下步骤仅用于训练,Demo 推理无需执行:

1. 将 **VoxCPM2** 预训练 checkpoint 下载到 `pretrained_models/VoxCPM2`
   (需包含 `config.json`、模型权重以及 tokenizer 文件)。
2. 为 tokenizer 扩展 128 个音高、12 个音符时值和 256 个 BPM token:

```bash
python scripts/setup_svs_tokenizer.py \
    --tokenizer_path pretrained_models/VoxCPM2 \
    --save_path pretrained_models/VoxCPM2
```

模型 embedding 会在训练开始时自动 resize。

### 开始训练

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 \
    scripts/train_vocalrender_svs.py --config_path conf/svs_train.yaml
```

任意配置项都可以用点分路径覆盖,例如
`--set train.batch_size=32 --set runtime.save_path=checkpoints/run2`。
训练会自动从 `save_path` 下最新的 checkpoint 恢复。

验证阶段会将 loss 及音质指标(SingMOS、Audiobox Aesthetics)和采样音频记录到
TensorBoard。

## 评测指标

- **SingMOS** —— 歌声 MOS 预测器(通过 `torch.hub` 加载,需要 `s3prl`)。
- **AES** —— Audiobox Aesthetics 维度(CE = 内容愉悦度,PQ = 制作质量)。
- `vocalrender.evaluation.svs_metrics` 中提供可插拔的 `register_metric_backend`
  接口,无需修改评测器即可添加自定义指标。

## 仓库结构

```
conf/               训练 / 推理 / 预处理的 YAML 配置
scripts/            入口脚本(预处理、训练、推理)
src/vocalrender/    核心包(模型、训练、推理、评测)
demo/               网页 Demo(Gradio 钢琴卷帘界面),发布到 HF Space
nanovllm-voxcpm/    可选的 nano-vllm 推理后端(git 子模块)
docs/               架构与使用文档
```

完整目录树见 [docs/structure.md](docs/structure.md)。

## 致谢

- [VoxCPM](https://github.com/OpenBMB/VoxCPM)(OpenBMB)—— 本工作所基于的
  TTS 基础模型;模型架构(TSLM / LocEnc / LocDiT / AudioVAE)与预训练权重
  均来自 VoxCPM 项目。
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) —— 轻量级 vLLM
  实现,`nano_vllm` 推理后端在此基础上改编。
- [SingMOS](https://github.com/South-Twilight/SingMOS) 与
  [audiobox-aesthetics](https://github.com/facebookresearch/audiobox-aesthetics)
  —— 评测模型。

## 许可证

Apache-2.0(与上游 VoxCPM 一致)。见 [LICENSE](LICENSE)。
