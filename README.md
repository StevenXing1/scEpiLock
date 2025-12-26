# scEpiLock

A deep learning framework for single-cell epigenomic analysis with interpretability features.

## Overview

scEpiLock is a comprehensive deep learning framework designed for analyzing single-cell epigenomic data. The project consists of three main modules:

1. **Deep Learning Module**: Train and evaluate deep learning models on single-cell chromatin accessibility data
2. **Grad-CAM Module**: Generate interpretable visualizations of model predictions using Gradient-weighted Class Activation Mapping
3. **Variant Impact Module**: Assess the functional impact of genetic variants on chromatin accessibility

## Features

- 🧬 Single-cell chromatin accessibility prediction
- 🔍 Model interpretability with Grad-CAM visualization
- 🧪 Genetic variant impact assessment
- 📊 Support for multiple cell types and tissues
- ⚡ GPU-accelerated training and inference
- 🚀 **Hyena operators** for efficient long-range sequence modeling
- 🎯 Multiple model architectures (CNN, Hyena, Hybrid)

## Installation

### Requirements

- Python 3.7+
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/StevenXing1/scEpiLock.git
cd scEpiLock

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Deep Learning Module

Train a model on single-cell chromatin accessibility data:

**Standard CNN model:**
```bash
cd deep_learning_module
python main.py --config config/config.json
```

**Hyena model (faster, better long-range modeling):**
```bash
cd deep_learning_module
python main.py --config config/config_hyena.json
```

**Configuration**: Edit `config/config.json` or `config/config_hyena.json` to customize:
- Model architecture (`scEpiLock`, `scEpiLock_Hyena`, or `scEpiLock_HyenaDeep`)
- Training hyperparameters
- Data paths and preprocessing options
- Output directories

### 2. Grad-CAM Module

Generate interpretability visualizations for trained models:

```bash
cd grad_cam_module
python main.py --config config/config_galaxy_brain.json
```

This module produces Grad-CAM heatmaps showing which genomic regions the model focuses on when making predictions.

### 3. Variant Impact Module

Evaluate the effect of genetic variants on chromatin accessibility:

```bash
cd variant_impact_module
python main.py --config config/config_snp_paper_brain_all.json
```

This module:
- Takes reference and alternate sequences
- Predicts chromatin accessibility for both
- Computes the differential impact score

## Project Structure

```
scEpiLock/
├── deep_learning_module/     # Main training pipeline
│   ├── main.py               # Entry point for training
│   ├── config/               # Configuration files
│   │   ├── config.json       # Standard CNN config
│   │   └── config_hyena.json # Hyena model config
│   ├── data/                 # Data loading and preprocessing
│   ├── model/                # Model architectures
│   │   ├── scEpiLock.py      # Standard CNN model
│   │   └── scEpiLock_Hyena.py # Hyena models
│   ├── train/                # Training logic
│   └── assess/               # Model evaluation
├── grad_cam_module/          # Interpretability analysis
│   ├── main.py               # Grad-CAM entry point
│   ├── cam/                  # CAM implementations
│   └── utils/                # Visualization utilities
└── variant_impact_module/    # Variant effect prediction
    ├── main.py               # Variant analysis entry point
    ├── data/                 # Sequence generation
    └── evaluate/             # Impact scoring
```

## Configuration

Each module uses JSON configuration files. Key parameters include:

- `model_path`: Path to trained model weights
- `data_path`: Input data location
- `output_path`: Where to save results
- `batch_size`: Batch size for processing
- `device`: CPU or CUDA device

## Data Format

The framework expects:
- **Input**: Genomic sequences in FASTA format or preprocessed numpy arrays
- **Labels**: Cell type accessibility labels
- **Variants**: VCF format or custom TSV with genomic coordinates

## Model Architectures

scEpiLock supports multiple model architectures optimized for genomic sequence analysis:

### 1. scEpiLock (Default)
The baseline model using residual convolutional blocks:
- Convolutional layers for local sequence feature extraction
- Residual blocks (ResBlocks) for deep architecture stability
- Batch normalization and dropout for regularization
- Efficient for standard single-cell chromatin accessibility tasks

**Usage:**
```json
{
  "model_name": "scEpiLock",
  "n_class": 7
}
```

### 2. scEpiLock_Hyena ⚡ NEW
Modern architecture using Hyena operators for efficient long-range modeling:
- **Hyena Operators**: Subquadratic alternative to attention mechanisms
- **Long Convolutions**: Implicit long convolutions via FFT for efficient computation
- **Data-Controlled Gating**: Dynamic feature routing based on input content
- **Scalability**: Linear complexity in sequence length, ideal for long genomic sequences

**Key advantages:**
- Faster training and inference compared to attention-based models
- Better long-range dependency modeling (up to full sequence length)
- Memory efficient for processing thousands of sequences
- State-of-the-art performance on long sequence tasks

**Usage:**
```bash
cd deep_learning_module
python main.py --config config/config_hyena.json
```

**Configuration:**
```json
{
  "model_name": "scEpiLock_Hyena",
  "seq_len": 1000,
  "d_model": 320,
  "n_layers": 4,
  "hyena_order": 2,
  "n_class": 7
}
```

**Parameters:**
- `d_model`: Hidden dimension size (default: 320)
- `n_layers`: Number of Hyena blocks (default: 4)
- `hyena_order`: Order of Hyena operator (default: 2, higher = more complex interactions)
- `seq_len`: Input sequence length (default: 1000)

### 3. scEpiLock_HyenaDeep
Hybrid architecture combining convolutions and Hyena:
- Initial convolutional layers for local pattern recognition
- Hyena blocks for long-range chromatin interactions
- Hierarchical feature learning with progressive downsampling
- Best for complex multi-cell-type prediction tasks

**Usage:**
```json
{
  "model_name": "scEpiLock_HyenaDeep",
  "n_class": 7
}
```

### Architecture Comparison

| Model | Parameters | Speed | Long-Range | Best For |
|-------|-----------|-------|------------|----------|
| scEpiLock | ~10M | Fast | Limited | Standard tasks, baseline |
| scEpiLock_Hyena | ~8M | Very Fast | Excellent | Long sequences, efficiency |
| scEpiLock_HyenaDeep | ~15M | Medium | Excellent | Complex patterns, accuracy |

### When to Use Hyena Models

**Choose Hyena models when:**
- Working with long genomic sequences (>1kb)
- Need to capture long-range chromatin interactions
- Want faster training times
- Working with limited GPU memory
- Require state-of-the-art performance on sequence modeling

**Stick with scEpiLock when:**
- Using shorter sequences (<1kb)
- Need quick prototyping
- Want established baseline results

## Output

- **Training**: Model checkpoints, loss curves, evaluation metrics
- **Grad-CAM**: Heatmap visualizations as PNG/PDF files
- **Variant Impact**: TSV files with variant scores and predictions

## Citation

If you use scEpiLock in your research, please cite:

```
[Citation information to be added]
```

## License

[License information to be added]

## Contact

For questions or issues, please contact:
- Email: haiyix2@uci.edu
- GitHub: [@StevenXing1](https://github.com/StevenXing1)

## Acknowledgments

This project was developed for single-cell epigenomics research.
