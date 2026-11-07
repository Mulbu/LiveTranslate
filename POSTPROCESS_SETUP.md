# 标点还原 + 说话人识别（kotoba-whisper-v2.2 同款组件）

本功能让 LiveTranslate 在使用 **CT2 版 kotoba-whisper-v2.2**（`models/` 下的本地 faster-whisper 模型）时，
额外获得 kotoba-whisper-v2.2 的 **标点还原** 与 **说话人区分** 能力。

> 说明：这两个能力**不在 CT2 权重里**。官方 v2.2 是把它们做在一层 transformers `pipeline` 里：
> 标点用 `punctuators` 库，说话人用 `pyannote.audio`。本项目复用**同样的组件**，但把说话人做成了
> 适配实时流的**在线声纹聚类**（官方的整段离线 diarization 在 1–8 秒的 VAD 片段上无法保持一致的说话人标签）。

## 1. 安装依赖

```bash
pip install punctuators==0.0.5 onnxruntime
pip install pyannote.audio
# torch / torchaudio 按 CUDA 版本单独装（见 requirements.txt 末尾注释）
```

## 2. Hugging Face 登录 + 接受 gated 模型

说话人的声纹嵌入模型是 gated 的，需要：

1. 登录并接受条款：<https://huggingface.co/pyannote/embedding>
   （若改用别的声纹模型，也要接受对应模型条款）
2. 登录：

   ```bash
   huggingface-cli login
   # 或设置环境变量 HF_TOKEN=hf_xxx
   ```

标点模型 `1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase` 不需要 gated 授权，
首次运行会自动下载（ONNX，CPU 即可）。

## 3. CT2 模型放置

把转换好的 CT2 模型（`config.json` / `model.bin` / `preprocessor_config.json` /
`tokenizer.json` / `vocabulary.json`）整个目录放到：

```
LiveTranslate/models/kotoba-whisper-v2.2-faster/
```

重启控制面板 → 「Whisper 模型」下拉会出现「本地: kotoba-whisper-v2.2-faster」，选中即可。

## 4. 开启功能

控制面板 → VAD/ASR 标签页 → 勾选：

- **标点符号还原**（Punctuation restoration）
- **说话人识别**（Speaker identification）

或直接改 `config.yaml` 的 `asr:` 段：

```yaml
asr:
  add_punctuation: true
  punctuation_model: "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
  diarization: true
  diarization_model: "pyannote/embedding"   # 也可换 pyannote/wespeaker-voxceleb-resnet34-LM
  diarization_threshold: 0.72                # 余弦相似度阈值：越高越"严格"，越不容易合并不同的人
  diarization_max_speakers: 8
  diarization_min_segment: 0.4               # 少于该时长的片段不判定说话人（秒）
```

字幕会以 `[说话人1] 原文` 的形式显示在原文行，翻译行保持干净文本；转写文件同样带说话人标签。

## 5. 调参建议

- 把不同人错误合并 → 调高 `diarization_threshold`（如 0.80）。
- 同一个人被拆成多个 → 调低阈值（如 0.65）。
- 人声太短判不出来 → 调低 `diarization_min_segment`，但过短会不准。

## 6. 已知限制

- 说话人是**在线聚类**（增量、按会话），不是官方整段离线 diarization；直播中途出现新说话人会新分配 `SPEAKER_xx`。
- 增量 ASR（`incremental_asr`）开启时，被 incremental 提前提交的句子走的是 partial 路径、不做说话人判定；
  默认（增量关闭）的整段/收尾路径会正常标注。
- 任一组件缺失 / 模型未授权时，功能会自动降级为“不标注”，不影响正常识别与翻译（日志有 warning）。
- 本仓库的服务器（Python 3.8 / 无 GPU / 无依赖）无法运行验证，请在 Windows 的 conda 环境 `livetranslate`（Python 3.12 + CUDA）上验收。
