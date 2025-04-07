# Resource-Efficient Foundation Model Programming

A modular framework for **cost-efficient inference** in complex multi-modal tasks using _Foundation Model Programs_. This system enables dynamic routing of sub-tasks to the most appropriate foundation model backends, optimizing the trade-off between accuracy and computational cost.

---

## 📚 Table of Contents  
- [Overview](#overview)  
- [Project Structure](#project-structure)  
- [Installation](#installation)  
- [Usage](#usage)  
  - [Run Streaming Verification](#run-streaming-verification)  
  - [Configuration](#configuration)  
- [Data Format](#data-format)  
- [Benchmarks](#benchmarks)  
- [Acknowledgments](#acknowledgments)

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
│   ├── code_generation.ipynb
│   ├── dsl.prompt            # DSL prompt template for generating user programs
│   ├── openai.key            # OpenAI API key (don't commit this!)
│   ├── verification.json     # Dataset annotations and queries
│   └── 📁saved_retrieval_imgs # Cached positive/negative image tensors
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

## 🚀 Usage

### Run Streaming Verification
```bash
python main_verification.py \
  --cost_weighting 0.1 0.5 1.0 \
  --config config/verification.yaml \
  --data data/verification.json \
  --log logs/
```

| Flag             | Description                                         |
|------------------|-----------------------------------------------------|
| `--cost_weighting` | Trade-off between accuracy and cost.               |
| `--config`         | Program + backend routing YAML.                    |
| `--data`           | JSON file with query/image annotations.            |
| `--log`            | Logging directory.                                 |

---

### Configuration

- Backend model selection, routing policy, and program control flow are defined in `config/`.
- Supports:
  - MMDetection (object detection)
  - HuggingFace / vLLM (LLMs and VLMs)
  - AgentLego (tool-based execution)
  - OpenAI/Remote APIs (via `openai.key` or `api.key`)

---

## 📄 Data Format

### `verification.json`
```json
{
  "query": "Are there at least four horses on a beach?",
  "code": "def execute_command(image): ...",
  "positive_images": [101, 205],
  "negative_images": [109, 320]
}
```

- DSL-style or auto-generated Python code represents reasoning structure.
- Each query maps to a sequence of images.

---

## 🙏 Acknowledgments
- Based on ideas from [ViperGPT](https://github.com/cvlab-columbia/viperGPT), [VisProg](https://github.com/allenai/visprog), and neurosymbolic LLM systems.
- Built using PyTorch, MMDetection, HuggingFace Transformers, and AgentLego.
- Benchmarks adapted from GQA, COCO, and A-OKVQA datasets.
