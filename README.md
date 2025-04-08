# Resource-Efficient Foundation Model Programming

A modular framework for **cost-efficient inference** in complex multi-modal tasks using _Foundation Model Programs_. This system enables dynamic routing of sub-tasks to the most appropriate foundation model backends, optimizing the trade-off between accuracy and computational cost.

---

## 🌟 Overview  

This repository implements a neurosymbolic execution system where user queries are compiled into Python-like **Foundation Model Programs**. Each program is composed of generic neural functions (e.g., object detection, VQA) and dynamically chooses among multiple backend models during inference.

Key Features:
- **Streaming VQA support** for binary and open-form tasks  
- **Online policy learning** for dynamic backend selection  
- **Structured REINFORCE + Thompson Sampling** for cost-aware inference  
- **Composable program interface** for multi-modal reasoning

Use Cases:
- Visual Question Answering (VQA)  
- Multi-modal scene understanding  
- Cost-constrained decision-making for FM workflows  

---

## 🗂️ Project Structure

```
📁FMP
├── 📁config                  # YAML configuration files for model setup and routing parameters
│   ├── binary_vqa.yaml       # FM backend and hyperparameter configurations for Streaming Binary VQA experiments
│   ├── binary_vqa_supp.yaml  # FM backend and hyperparameter configurations for Streaming Binary VQA experiments (supplementary)
│   └── open_form_vqa.yaml    # FM backend and hyperparameter configurations for Streaming Open-Form VQA experiments
├── 📁data                    # Input data and saved retrieval images
│   ├── binary_vqa.json       # Dataset annotations and queries for the Streaming Binary VQA task
│   ├── 📁open_form_vqa       # Dataset annotations and queries for the Streaming Open-Form VQA task
├── 📁execution               # Core execution modules
│   ├── backend.py            # Routing backend & program execution engine
│   ├── image_patch.py        # ImagePatch class & spatial reasoning
│   ├── models.py             # ObjectDetection, VQA, LLM wrappers
│   ├── modules.py            # Initialization & model pooling
│   └── utils.py              # Shared utilities (e.g., transforms, loaders)
├── 📁routing                 # Routing algorithms & networks
│   ├── __init__.py
│   ├── algorithms.py         # Reinforcement learning routing logic
│   └── networks.py           # Neural network models for routing decisions
├── api.key                   # API key for remote models (ignored in .gitignore)
├── environment.yml           # Conda environment specification
├── main_verification.py      # Entry point for running Streaming Binary VQA tasks
├── main_vqa.py               # Entry point for running Streaming Open-Form VQA tasks
├── README.md                 # Project documentation
├── utils_verification.py     # Task-specific utilities
├── utils_vqa.py              # Task-specific utilities
└── utils.py                  # Shared utility functions
```


---

## 🛠️ Installation

```bash
git clone https://github.com/yourusername/FMP.git
cd FMP
conda env create -f environment.yml
conda activate FMP
```

Download COCO:
```bash
mkdir -p data/coco && cd data/coco
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip && unzip val2017.zip
```

---

## 📄 Benchmarks

Download binary VQA and open-form VQA benchmarks:
```bash
cd data
wget https://utexas.box.com/shared/static/rdcykkjg41i2rfo7itna4tvrygbmzkct.json -O binary_vqa.json
wget https://utexas.box.com/shared/static/p62xf6oqrp92zeboj8kyne21mbeej6pv.zip -O open_form_vqa.zip && unzip open_form_vqa.zip
```

---

## 🚀 Usage

### Run Streaming Binary VQA
```bash
python main_verification.py \
  --cost_weighting 0.001 0.003 0.005 \
  --config config/binary_vqa.yaml \
  --data data/binary_vqa.json \
  --log logs/
```

| Flag             | Description                                         |
|------------------|-----------------------------------------------------|
| `--cost_weighting` | Trade-off between accuracy and cost.               |
| `--config`         | Program + backend routing YAML.                    |
| `--data`           | JSON file with query/image annotations.            |
| `--log`            | Logging directory.                                 |

### Run Streaming Open-form VQA
```bash
python main_vqa.py \
  --cost_weighting 0.001 0.003 0.005 \
  --config config/open_form_vqa.yaml \
  --data data/open_form_vqa \
  --log logs/
```

| Flag             | Description                                         |
|------------------|-----------------------------------------------------|
| `--cost_weighting` | Trade-off between accuracy and cost.               |
| `--config`         | Program + backend routing YAML.                    |
| `--data`           | Root directory of open-form VQA data               |
| `--log`            | Logging directory.                                 |
| `--type`           | Query categories to evaluate. Default to all.      |


---

### ⚙️ Configuration

- Backend model selection, routing policy, and program control flow are defined in `config/`.
- Supports:
  - MMDetection (object detection)
  - HuggingFace / vLLM (LLMs and VLMs)
  - AgentLego (tool-based execution)
  - OpenAI/Remote APIs

---

## 🙏 Acknowledgments
- Based on ideas from [ViperGPT](https://github.com/cvlab-columbia/viperGPT), [VisProg](https://github.com/allenai/visprog), and neurosymbolic LLM systems.
- Built using PyTorch, MMDetection, HuggingFace Transformers, and AgentLego.
- Benchmarks adapted from GQA, COCO, and A-OKVQA datasets.
