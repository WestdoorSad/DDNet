# DDNet

**A Dual-Driven Meta-Learning Framework for Few-Shot Modulation Recognition under Varying SNR Conditions**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)

---

## News

- **2026-05-11** — We **open-sourced part of the code** to facilitate reproduction and discussion. This repository currently includes the core training scripts, models, and data-processing code. Full experiment configurations, pretrained weights, and additional comparison methods may be added gradually as needed. **Issues and pull requests are welcome.**

---

## Overview

DDNet targets **few-shot automatic modulation recognition (AMR)** under **varying SNR**. This repository provides a episodic meta-learning on the RadioML **RML201610A** dataset.

> **Note:** This is a **partial** release. If something you need is missing, please open an issue.

---

## Requirements

- Python 3.9+ (recommended)
- NVIDIA GPU with CUDA (training script uses `.cuda()`)
- Dependencies: see [`requirements.txt`](requirements.txt)

The pinned `torch` build in `requirements.txt` targets **CUDA 12.8** (`+cu128`). For CPU-only or other CUDA versions, edit the PyTorch lines per [official install instructions](https://pytorch.org/get-started/locally/).

### Install

```bash
conda create -n ddnet python=3.10 -y
conda activate ddnet
pip install -r requirements.txt
```

---

## Dataset

This code expects the **RML2016.10a** dictionary pickle used by `data/RML201610A.py`.

1. Obtain **RML2016.10a** (e.g. from the [RadioML / DeepSig resources](https://www.deepsig.ai/datasets)) and prepare the `.pkl` in the format your loader expects.
2. Set the dataset path in `data/RML201610A.py` (default in the code may point to a local Windows path — **change it to your machine**).

---

## Citation

If this work helps your research, please cite our paper **(bibtex to be added when available)**.

---

## License

Specify your license here (e.g. MIT / Apache-2.0). *Add a `LICENSE` file in the repo root if you choose an open license.*

---

## Acknowledgements

- Thanks to the authors of open-source projects and datasets that make reproducible AMR research possible.
- Implementation references common practices from PyTorch-based few-shot learning and self-supervised learning codebases on GitHub.

---

## Contact

For questions or collaboration: open an [issue](https://github.com/WestdoorSad/DDNet/issues) or reach out via your preferred channel (add email / homepage if you like).
