# Model Checkpoints

Download the required model checkpoints into this directory.

## LingBot-Map

Download from Hugging Face Hub:

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='robbyant/lingbot-map', filename='lingbot_map_stream.pt', local_dir='.')
"
```

Or visit: https://huggingface.co/robbyant/lingbot-map

Expected file: `checkpoints/lingbot-map.pt` (~70 MB)

## SAM3.1

从魔搭社区 (ModelScope) 下载，使用 git clone + git-lfs：

```bash
# 确保 git-lfs 已安装并初始化
git lfs install

# 从魔搭社区克隆 SAM3.1 仓库
git clone https://www.modelscope.cn/facebook/sam3.1.git
```

如果网络不稳定，可以先安装 ModelScope CLI：

```bash
pip install modelscope
```

然后可以使用 modelscope 命令行工具下载。

Expected directory: `checkpoints/sam3.1/` containing:
- `sam3.1_multiplex.pt` (~2 GB)
- `config.json`, `tokenizer.json`, etc.
