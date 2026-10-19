# CROPS — Deep Learning for Agricultural Crop Classification

**Reproduction and Improvement of MCTNet Using Multi-Source Sentinel-2 Time Series**

This project investigates pixel-based agricultural crop classification using satellite time-series data and deep learning. It reproduces the MCTNet architecture and explores how environmental covariates and an improved temporal modeling architecture can enhance crop classification performance.

The experiments focus on two study areas in the United States: **Arkansas and California**.

## Project Overview

Accurate crop classification is essential for agricultural monitoring, food security assessment, and environmental management. Sentinel-2 satellite imagery provides temporal observations of vegetation throughout the growing season, enabling the identification of crop types from their phenological patterns.

This project is organized into three main parts:

- **Part 1 — Baseline Reproduction:** Reimplementation and evaluation of MCTNet using Sentinel-2 time-series data.
- **Part 2 — Environmental Covariates:** Investigation of climate, soil, and topographic variables through controlled ablation experiments.
- **Part 3 — Model Improvement:** Design and evaluation of MS-SE-BiGRU-Attn, an improved architecture for temporal feature extraction and multi-source crop classification.

## Objectives

- Reproduce the methodology of the paper *A Lightweight CNN-Transformer Network for Pixel-Based Crop Mapping Using Time-Series Sentinel-2 Imagery*.
- Explore temporal satellite observations and investigate data coverage and missing values.
- Evaluate the contribution of climate, soil, and topographic covariates.
- Design an improved deep learning architecture for satellite time-series classification.
- Compare experimental configurations using standard classification metrics.

## Methodology

### Part 1 — MCTNet Reproduction

The baseline model uses Sentinel-2 time-series observations to classify agricultural pixels.

The main steps include:

1. Dataset preparation and alignment with crop labels.
2. Temporal preprocessing and handling of missing observations.
3. Exploratory data analysis and inspection of temporal patterns.
4. Implementation and training of the MCTNet architecture.
5. Evaluation on the Arkansas and California datasets.

The input data consist of **36 temporal observations and 10 Sentinel-2 spectral bands**, resulting in a time-series representation of 360 features per pixel.

The model combines convolutional and Transformer-based components to learn temporal representations for crop classification.

### Part 2 — Integration of Environmental Covariates

This part investigates whether environmental information improves classification beyond Sentinel-2 imagery alone.

The following configurations are evaluated:

| Configuration | Input data |
|---|---|
| S2 Only | Sentinel-2 |
| S2 + Climate | Sentinel-2 and climate variables |
| S2 + Soil | Sentinel-2 and soil properties |
| S2 + Topography | Sentinel-2 and topographic variables |
| S2 + All | Sentinel-2 and all environmental covariates |

An ablation study is conducted to compare the configurations and assess the contribution of each data source.

### Part 3 — Proposed Architecture: MS-SE-BiGRU-Attn

An improved architecture, **MS-SE-BiGRU-Attn**, is designed to strengthen temporal representation learning.

The proposed model combines:

- **MS:** Multi-scale feature extraction.
- **SE:** Squeeze-and-Excitation mechanisms for adaptive feature recalibration.
- **BiGRU:** Bidirectional Gated Recurrent Units for modeling temporal dependencies.
- **Attention:** Attention mechanisms for emphasizing informative temporal features.

The model is trained and evaluated on both study areas. Its performance is compared with the baseline and the covariate-enhanced configurations.

## Experimental Results

### Baseline MCTNet

The reproduced baseline achieves the following test results:

| Study area | Accuracy (OA) | Kappa | Macro F1 |
|---|---:|---:|---:|
| Arkansas | 85.69% | 0.7929 | 79.75% |
| California | 81.61% | 0.7605 | 74.04% |

### Proposed Model

The proposed MS-SE-BiGRU-Attn architecture achieves the following reported accuracy values:

| Study area | Accuracy |
|---|---:|
| Arkansas | 89.65% |
| California | 83.13% |

These results demonstrate the potential of the proposed architecture to improve crop classification performance, particularly in Arkansas.

**Note:** The reported accuracy values should be interpreted alongside the evaluation protocol and dataset configuration used in each experiment. A complete comparison should use consistent test splits and evaluation metrics.

## Technologies

- **Programming language:** Python
- **Deep learning:** PyTorch
- **Data processing:** NumPy, Pandas
- **Visualization and evaluation:** Matplotlib, Seaborn, Scikit-learn
- **Remote sensing:** Sentinel-2 time-series data
- **Development environment:** Jupyter Notebook / Google Colab
- **Version control:** Git and GitHub

## Repository Structure

```text
CROPS/
├── models/
│   ├── mctnet_arkansas_final.pth
│   ├── mctnet_california_final.pth
│   ├── part2_AR_S2_only.pth
│   ├── part2_AR_S2_climate.pth
│   ├── part2_AR_S2_soil.pth
│   ├── part2_AR_S2_topo.pth
│   ├── part2_AR_S2_all.pth
│   ├── part2_CA_S2_only.pth
│   ├── part2_CA_S2_climate.pth
│   ├── part2_CA_S2_soil.pth
│   ├── part2_CA_S2_topo.pth
│   └── part2_CA_S2_all.pth
├── ARKANSAS_10000x360.csv
├── CALIFORNIA_10000x360.csv
├── partie1_enrichi.ipynb
├── partie2_visualisations_1.ipynb
├── partie3_amelioration.ipynb
├── sentinel.md
└── README.md
```

*The structure above reflects the main files and checkpoints used in the project. Update it if additional scripts, datasets, or proposed-model checkpoints are added.*

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/lynaar/crops.git
cd crops
```

### 2. Create a Python environment

```bash
python -m venv .venv
```

Activate the environment:

**Windows — PowerShell**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Linux / macOS**

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the required libraries:

```bash
pip install torch numpy pandas matplotlib seaborn scikit-learn jupyter
```

Depending on the notebook, additional geospatial libraries or other dependencies may be required. For GPU training, install the PyTorch build compatible with your CUDA environment using the official PyTorch installation instructions.

### 4. Run the notebooks

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Execute the notebooks in the following order:

1. `partie1_enrichi.ipynb` — baseline reproduction and data preparation.
2. `partie2_visualisations_1.ipynb` — covariate experiments and visualizations.
3. `partie3_amelioration.ipynb` — proposed model and performance evaluation.

**Important:** The notebooks may depend on the CSV datasets, model checkpoints, and paths configured in the original development environment. Verify these paths before running the experiments.

## Dataset

The project uses agricultural data from two study areas:

- **Arkansas, USA**
- **California, USA**

The data combine Sentinel-2 spectral time series with crop labels and, in the second part, environmental covariates.

The provided CSV files contain 10,000 samples per study area, with 360 Sentinel-2 features per sample.

The original satellite imagery and environmental datasets may require separate acquisition or preprocessing. Refer to `sentinel.md` and the notebooks for additional information about data preparation and assumptions.

## Evaluation Metrics

Model performance is assessed using:

- **Overall Accuracy (OA):** Proportion of correctly classified samples.
- **Cohen's Kappa:** Agreement between predicted and reference labels, adjusted for chance.
- **Macro F1-score:** Average F1-score across classes, giving equal weight to each class.

These metrics provide complementary views of classification performance, particularly when crop classes are imbalanced.

## Project Deliverables

- Reproduced MCTNet baseline.
- Environmental covariate integration and ablation experiments.
- Proposed MS-SE-BiGRU-Attn architecture.
- Trained model checkpoints.
- Experimental visualizations and performance comparisons.
- Technical report documenting the methodology, experiments, results, and discussion.

## Future Improvements

- Evaluate generalization across additional geographical regions and growing seasons.
- Investigate class imbalance and improve minority-class recognition.
- Perform additional ablation studies on the proposed architecture.
- Improve computational efficiency and reproducibility.
- Explore uncertainty estimation and spatial validation strategies.

## Academic Context

**University:** University of Science and Technology Houari Boumediene (USTHB)  
**Faculty:** Faculty of Computer Science  
**Program:** Master 1 — Intelligent Computer Systems (M1 SII)  
**Academic Year:** 2025–2026

**Project title:** *Deep Learning for Crop Classification Using Multi-Source Satellite Data*

This project was carried out as an academic research project focused on deep learning, time-series modeling, and remote sensing.

## Acknowledgments

This work builds on the research presented in *A Lightweight CNN-Transformer Network for Pixel-Based Crop Mapping Using Time-Series Sentinel-2 Imagery*. The original authors' methodology provides the baseline for the reproduction and subsequent experiments.

## License

No license has been specified yet. Add a `LICENSE` file if you intend to distribute the code under an open-source license.
