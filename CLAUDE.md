# CLAUDE.md

## 项目简介

本仓库是 ICLR 2026 Oral 论文 *Latent Particle World Models (LPWM)* 及 DLPv3 的官方 PyTorch 实现（fork 自 taldatech/lpwm）。
LPWM 是自监督、以物体为中心的随机世界模型：从视频中无监督地发现关键点、包围盒和物体掩码（"particles"），
用 latent action 模块建模粒子动力学，可按动作 / 语言 / 目标图像进行条件生成。

**我们在做的研究**（LPWM Acceleration Research，导师 Tal Daniel）：加速 DLP / LPWM 的训练与推理，
并研究模型改动（时间位置编码 `temporal_pos`、`temporal_rope`、滑动 KV cache、记忆）对质量的影响。
基准：OBJ3D 64×64，验证集 LPIPS / PSNR / SSIM。具体实验状态以 Notion 为准（见下方「工作流程」）。

## 目录结构

| 路径 | 内容 |
|---|---|
| `models.py` | DLPv3 与 LPWM 模型实现（含 `temporal_pos`、KV cache 等 opt-in 选项） |
| `modules/modules.py` | 基础网络模块 |
| `train_lpwm.py` / `train_dlp.py` | 单 GPU 训练（支持 `resume_dir` 抢占后续训） |
| `train_lpwm_accelerate.py` / `train_dlp_accelerate.py` | 多 GPU 训练（HuggingFace Accelerate，配置 `accel_conf.yml`） |
| `generate_lpwm_video_prediction.py` | 用预训练模型生成视频预测 |
| `eval/` | `eval_gen_metrics.py`（LPIPS/PSNR/SSIM）、`eval_fvd.py`（FVD）、`eval_model.py` |
| `configs/*.json` | 各数据集超参数配置 |
| `datasets/` | 数据集加载代码（是代码，不是数据） |
| `lpwm_stn/` | 隔离的 Spatial Transformer Network 参考实现与 Triton 后端 |
| `benchmarks/stn/` | 端到端基准、profiling、质量对比与集群运行脚本（`run_*.sh`），见其 README |
| `tests/` | 单元测试（`temporal_pos/`、`composite/`、`stn/`，含 golden fixtures） |
| `checkpoints/` | 放置模型权重（已被 git 忽略） |
| `docs/` | 安装说明、超参数说明、示例用法 |

## 环境配置

- conda：`conda env create -f environment_linux.yml`（Windows 用 `environment_win.yml`），或 `pip install -r requirements.txt`
- 依赖：Python ≥ 3.9，torch ≥ 2.6，accelerate，piqa，einops 等（见 README）
- 本机是 macOS（无 CUDA），只适合写代码和跑 CPU 测试；GPU 训练与评估在集群上通过 sbatch 运行（A6000）

## 常用命令

```bash
# CPU 测试
python -m pytest tests/temporal_pos -q
python -m pytest tests -q

# 单 GPU 训练
python train_lpwm.py --dataset obj3d128

# 多 GPU 训练
accelerate launch --config_file ./accel_conf.yml train_lpwm_accelerate.py --dataset obj3d128

# 评估
python eval/eval_gen_metrics.py --help
python eval/eval_fvd.py --help
python generate_lpwm_video_prediction.py --help
```

## 关键约定

- **新功能一律 opt-in**：通过配置键加入（如 `temporal_pos`、`temporal_rope`、`kv_cache`），默认值必须与原代码 bitwise 一致，并配测试证明这一点。
- 对比实验的预设规则（指标、判定阈值）要在看到结果**之前**写进 Notion；事后形成的判断要标注出来。
- 不要提交数据集、模型权重、日志、`.env` 或任何密钥（见 `.gitignore`）。已跟踪的测试 fixtures（`tests/**/fixtures/*.pt`）和 `eval/lpips/vgg.pth` 除外。
- commit message 风格：`文件或模块: 改了什么`（参考 `git log`）。

## 工作流程（多账号协作）

本目录由两个 Claude 账号（`claude-lab` / `claude-me`）轮流使用。**项目进度以 Notion 为准，不依赖对话历史。**

Notion 页面（通过项目级 Notion MCP 访问）：
- 项目根页面：[LPWM Acceleration Research](https://app.notion.com/p/407360a2828a81a39089ed34cefab74b)
- 当前实验文档：[KV Cache](https://app.notion.com/p/407360a2828a812fa7adf6a8b41b1325)（当前在做的实验都记在这里）
- 每日进度：[Research Log](https://app.notion.com/p/407360a2828a81e7884acd3949efb7c1)
- 其他：Master Plan (Tree)、LPWM Experiment Tracker 数据库（同一套内容的不同记录格式）

a. **每次会话开始**：先用 Notion MCP 读取 KV Cache 页面和 Research Log 的最新条目，了解最新实验状态和待办，再开始工作。
b. **完成阶段性工作或会话结束前**：把进度、结果、下一步更新回 Notion —— 当天进度写进 Research Log，实验设计 / 结果 / 结论写进 KV Cache 实验文档。
c. **切换账号前先 git commit**，保证工作区干净（`git status` 无改动）。
d. **同一时间只有一个账号在修改这个目录。**

## 额度说明

实验脚本（训练、评估、测试）的运行不消耗 Claude 额度；只有让 Claude 协助写代码、分析、调试时才消耗。
长时间的训练交给集群或终端自己跑，不需要开着 Claude 会话等待。
