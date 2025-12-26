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

```bash
cd deep_learning_module
python main.py --config config/config.json
```

**Configuration**: Edit `config/config.json` to customize:
- Model architecture parameters
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
│   ├── data/                 # Data loading and preprocessing
│   ├── model/                # Model architectures
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

The default model (scEpiLock) uses:
- Convolutional layers for sequence feature extraction
- Dilated convolutions for multi-scale context
- Attention mechanisms for interpretability

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
