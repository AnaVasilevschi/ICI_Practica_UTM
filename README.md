# Parkinson's Disease Detection using LSTM Networks

This project implements a deep learning approach for detecting Parkinson's disease from gait data, based on the research paper: "Automatic and non-invasive Parkinson's disease diagnosis and severity rating using LSTM network" ([Zhao et al., 2021](https://www.sciencedirect.com/science/article/abs/pii/S1568494621003860)).

## Overview

Parkinson's disease (PD) is a progressive neurological disorder that affects movement, causing tremors, stiffness, and difficulty with walking, balance, and coordination. Early detection of PD is crucial for effective treatment and management of the disease.

This project implements an LSTM (Long Short-Term Memory) neural network to detect Parkinson's disease based on gait patterns. The model analyzes time series data from force sensors placed under the feet, identifying the subtle differences in gait patterns between PD patients and healthy individuals.

## Dataset

The model is designed to work with the [Gait in Parkinson's Disease dataset](https://physionet.org/content/gaitpdb/1.0.0/) from PhysioNet, which contains:

- Gait measurements from 93 patients with Parkinson's Disease and 73 healthy controls
- Data collected using force sensors beneath the feet as subjects walked at their self-selected pace
- Each foot had 8 sensors measuring vertical ground reaction force (in Newtons) as a function of time
- Data digitized at 100 samples per second
- Additional demographic information and disease severity metrics (HoehnYahr scale and UPDRS scores)

For demonstration purposes, this implementation uses a synthetic dataset that mimics the structure and characteristics of the original PhysioNet dataset.

## Model Architecture

The implemented LSTM model has the following architecture:

- Input layer: Time series data from 16 sensors (8 per foot)
- First LSTM layer: 64 units with L2 regularization and dropout (0.2)
- Second LSTM layer: 32 units with L2 regularization and dropout (0.2)
- Output layer: Single neuron with sigmoid activation for binary classification

The model is trained using:
- Binary cross-entropy loss function
- Adam optimizer
- Early stopping to prevent overfitting

## Features

- Data preprocessing pipeline for gait time series data
- LSTM-based neural network for binary classification
- Comprehensive evaluation metrics (accuracy, F1-score, MAE, MSE)
- Visualization tools for model performance analysis
- Support for both real and synthetic datasets

## Requirements

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/parkinsons-detection.git
cd parkinsons-detection
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Data Preparation

To generate synthetic data for demonstration:

```bash
python src/generate_synthetic_data.py
```

To use the real PhysioNet dataset (requires internet connection):

```bash
python src/download_data_fixed.py
python src/preprocess_data.py
```

### Model Training and Evaluation

To train and evaluate the model:

```bash
python src/train_and_evaluate.py
```

This will:
1. Load the preprocessed data
2. Train the LSTM model
3. Evaluate the model on the test set
4. Generate performance visualizations
5. Save the trained model and results

## Project Structure

```
parkinsons_detection/
├── data/
│   ├── raw/             # Raw data files
│   └── processed/       # Preprocessed data ready for training
├── models/              # Saved models and evaluation results
├── src/
│   ├── __init__.py
│   ├── download_data.py          # Script to download the dataset
│   ├── preprocess_data.py        # Data preprocessing pipeline
│   ├── generate_synthetic_data.py # Generate synthetic data
│   ├── model.py                  # LSTM model implementation
│   ├── train_and_evaluate.py     # Training and evaluation script
│   └── utils.py                  # Utility functions
└── README.md
```

## Results

The model was trained on a synthetic dataset mimicking the characteristics of the PhysioNet Gait in Parkinson's Disease dataset. The performance metrics on the test set are:

- Accuracy: 63.19%
- F1 Score: 63.95%
- Mean Absolute Error: 0.4726
- Mean Squared Error: 0.2387

These results are for demonstration purposes using synthetic data. Performance on the actual PhysioNet dataset would likely differ. The original research paper reported an accuracy of 98.6% for binary classification.

### Visualizations

The training process and model performance are visualized through:

- Training history plot (accuracy and loss)
- Confusion matrix
- ROC curve
- Precision-recall curve

These visualizations are saved in the `models/` directory after training.

## Limitations and Future Work

- This implementation uses synthetic data for demonstration. For real-world applications, the actual PhysioNet dataset should be used.
- The model could be extended to predict disease severity using regression techniques.
- Additional features from the gait data could be extracted to improve model performance.
- Transfer learning approaches could be explored to leverage pre-trained models.

## References

1. Zhao, A., Qi, L., Li, J., Dong, J., & Yu, H. (2021). Automatic and non-invasive Parkinson's disease diagnosis and severity rating using LSTM network. Applied Soft Computing, 104, 107210.

2. Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R., ... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation, 101(23), e215-e220.

3. PhysioNet Gait in Parkinson's Disease Dataset: https://physionet.org/content/gaitpdb/1.0.0/

## License

This project is licensed under the MIT License - see the LICENSE file for details.
