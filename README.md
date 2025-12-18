# Keypoint Detection with Uncertainty Estimation

This project implements keypoint detection for pose estimation using two different approaches: **KP-RCNN** (Detectron2-based) and **YOLO-NAS** (SuperGradients-based), both enhanced with Monte Carlo Dropout for uncertainty quantification.

## Project Overview

The project focuses on detecting keypoints for agricultural applications, with uncertainty estimation to assess prediction reliability. This is particularly useful for robotic harvesting systems where prediction confidence is crucial for safe operation.

### Key Features
- **Dual Architecture Support**: KP-RCNN and YOLO-NAS implementations
- **Monte Carlo Dropout**: Uncertainty quantification for reliable predictions
- **Safety Assessment**: Automated safety evaluation for robotic cutting applications
- **Comprehensive Analysis**: Statistical analysis tools for uncertainty evaluation
- **Visualization Tools**: Rich visualization of predictions with uncertainty indicators

## Project Structure

```
├── KP_RCNN/                    # Detectron2-based keypoint detection
│   ├── my_detectron2/          # Modified Detectron2 source
│   ├── train.py               # Training script
│   ├── test_process2.py       # Testing and evaluation
│   ├── uncertainty_analyzer.py # Uncertainty analysis tools
│   └── ThresholdAnalyzer.py   # Threshold analysis utilities
├── YOLONAS/                   # SuperGradients-based keypoint detection
│   ├── super-gradients-master/ # SuperGradients source
│   ├── train.py              # Training script
│   ├── validation.py         # Validation script
│   ├── test.py              # Testing with MC Dropout
│   └── Uncertainty_Analyzer.py # Uncertainty analysis tools
├── requirement.txt           # Python dependencies
└── README.md                # This file
```

## Quick Start

### 1. Environment Setup

Create and activate a virtual environment:

```bash
conda create -n detection python=3.10.19
conda activate detection
```

### 2. Install Dependencies

```bash
cd Uncertainty_Estimation
pip install -r requirements.txt
```

### 3. IDE Configuration

#### For PyCharm/IntelliJ:
- **KP-RCNN**: Mark `KP_RCNN/my_detectron2` as source root
- **YOLO-NAS**: Mark `YOLONAS/super-gradients-master/src` as source root

#### For VS Code:
Add to your `settings.json`:
```json
{
    "python.analysis.extraPaths": [
        "./KP_RCNN/my_detectron2",
        "./YOLONAS/super-gradients-master/src"
    ]
}
```

## Usage

### KP-RCNN Workflow

1. **Training**
   ```bash
   cd KP_RCNN
   python train.py
   ```

2. **Testing & Evaluation**
   ```bash
   python test_process2.py
   ```

3. **Uncertainty Analysis**
   ```bash
   python uncertainty_analyzer.py
   ```

### YOLO-NAS Workflow

1. **Training**
   ```bash
   cd YOLONAS
   python train.py
   ```

2. **Validation**
   ```bash
   python validation.py
   ```

3. **Testing with MC Dropout**
   ```bash
   python test.py
   ```

4. **Uncertainty Analysis**
   ```bash
   python Uncertainty_Analyzer.py
   ```

## Monte Carlo Dropout Implementation

Both implementations use Monte Carlo Dropout for uncertainty estimation:

### Key Features:
- **Epistemic Uncertainty**: Model uncertainty due to limited training data
- **Aleatoric Uncertainty**: Data uncertainty inherent in the observations
- **Safety Assessment**: Automated evaluation for robotic applications
- **Visualization**: Uncertainty-aware prediction visualization

### Safety Criteria:
The system evaluates safety for robotic cutting based on keypoint uncertainty:
- **Critical Keypoints**: First 3 keypoints must have uncertainty < 50%
- **Color Coding**: Green (safe), Yellow (borderline), Red (unsafe)
- **Automated Decision**: Binary safe/unsafe classification

## Analysis Tools

### Uncertainty Analyzer Features:
- **Statistical Analysis**: Correlation analysis between uncertainty and accuracy
- **Threshold Analysis**: Optimal uncertainty thresholds for different applications
- **Visualization**: Comprehensive plots and charts
- **Performance Metrics**: OKS (Object Keypoint Similarity) evaluation
- **Comparative Analysis**: Statistical tests between different uncertainty levels

### Output Metrics:
- **Global Uncertainty**: Combined bbox and keypoint uncertainty
- **Per-Keypoint Uncertainty**: Individual keypoint reliability scores
- **Spatial Uncertainty**: Position variation across MC dropout passes
- **Confidence Scores**: Model confidence in predictions

## Visualization

The system provides rich visualizations including:
- **Uncertainty Heatmaps**: Visual representation of prediction uncertainty
- **Safety Indicators**: Color-coded safety assessment
- **Keypoint Confidence**: Per-keypoint uncertainty visualization
- **Bounding Box Uncertainty**: Spatial uncertainty visualization

## Performance Monitoring

### Key Metrics Tracked:
- **Prediction Accuracy**: Standard keypoint detection metrics
- **Uncertainty Calibration**: How well uncertainty correlates with errors
- **Safety Rate**: Percentage of images deemed safe for robotic operation
- **Processing Time**: Inference speed with uncertainty estimation

## Technical Details

### Monte Carlo Dropout Configuration:
- **Dropout Rate**: Configurable (typically 0.02-0.10)
- **MC Passes**: 10 passes per image for uncertainty estimation
- **Dropout Placement**: Strategic placement in neck and head layers

### Model Architectures:
- **KP-RCNN**: Detectron2-based with custom keypoint head
- **YOLO-NAS**: SuperGradients YOLO-NAS-Pose with MC Dropout modifications

##  Troubleshooting

### Common Issues:

1. **CUDA Memory Issues**
   - Reduce batch size in training scripts
   - Clear CUDA cache between predictions

2. **Import Errors**
   - Ensure source roots are properly configured
   - Check virtual environment activation

3. **Model Loading Issues**
   - Verify checkpoint paths in configuration files
   - Ensure model weights are compatible with code version

### Performance Optimization:
- Safe features out from backbone for multiple run, which can reduce 0.1s per image

---

**Note**: This project is designed for research and educational purposes in agricultural robotics and computer vision with uncertainty quantification.


