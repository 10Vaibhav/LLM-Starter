# LLMs Environment Setup

This repository contains the code and environment setup for the Large Language Models (LLMs).


## Quick Start


### 1. Create and Activate Conda Environment
```bash
# Create virtual environment with Python 3.11
conda create --name llms_env python=3.11

# Activate the environment
conda activate llms_env
```

### 2. Install Required Packages
```bash
# Install core packages
pip install anthropic config==0.5.1 langchain==0.0.297 pydantic==1.10.9 tiktoken==0.5.1 faiss-cpu==1.7.4 transformers==4.47.1 torch==2.5.1 datasets==3.2.0 evaluate==0.4.3 accelerate==1.2.1 ipywidgets==8.1.5 matplotlib==3.10.0 seaborn==0.13.2 clean-text==0.6.0 scikit-learn==1.6.0 sentencepiece==0.2.0 pandas==2.0.0

# Install Jupyter components
pip install ipykernel jupyterlab notebook

# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=llms_course_env
```

### 3. Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```

**Important:** Make sure to select the `llms_env` kernel when working with notebooks.

## Installed Packages

| Package | Version | Purpose |
|---------|---------|---------|
| `anthropic` | Anthropic API client |
| `config` | 0.5.1 | Configuration management |
| `langchain` | 0.0.297 | LLM application framework |
| `pydantic` | 1.10.9 | Data validation |
| `tiktoken` | 0.5.1 | Token counting for OpenAI models |
| `faiss-cpu` | 1.7.4 | Similarity search and clustering |
| `transformers` | 4.47.1 | Hugging Face transformers |
| `torch` | 2.5.1 | PyTorch deep learning framework |
| `datasets` | 3.2.0 | Hugging Face datasets |
| `evaluate` | 0.4.3 | Model evaluation metrics |
| `accelerate` | 1.2.1 | Distributed training utilities |
| `ipywidgets` | 8.1.5 | Interactive Jupyter widgets |
| `matplotlib` | 3.10.0 | Plotting library |
| `seaborn` | 0.13.2 | Statistical data visualization |
| `clean-text` | 0.6.0 | Text preprocessing |
| `scikit-learn` | 1.6.0 | Machine learning library |
| `sentencepiece` | 0.2.0 | Text tokenization |
| `pandas` | 2.0.0 | Data manipulation and analysis |



## Environment Management

### Activating the Environment
```bash
conda activate llms_env
```

### Deactivating the Environment
```bash
conda deactivate
```

### Removing the Environment (if needed)
```bash
conda remove --name llms_env --all
```

### Listing Available Kernels
```bash
jupyter kernelspec list
```
