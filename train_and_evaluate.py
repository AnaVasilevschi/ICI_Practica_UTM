"""
Script to train and evaluate the LSTM model for Parkinson's disease detection.

This script loads the preprocessed synthetic data and trains the LSTM model
for binary classification of Parkinson's disease.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Add the project root directory to the Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.model import create_lstm_model, train_model, evaluate_model, plot_training_history, plot_confusion_matrix, save_predictions
from src.utils import plot_roc_curve, plot_precision_recall_curve, analyze_demographics

def main():
    """Main function to train and evaluate the LSTM model."""
    # Create directories for saving models and results
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Load preprocessed data
    data_dir = 'data/processed'
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    print(f"Loaded data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.summary()
    
    # Train model
    print("\nTraining model...")
    model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=32,
        epochs=50,  # Reduced for demonstration
        patience=10,
        model_path='models/lstm_model.h5'
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    results = evaluate_model(model, X_test, y_test)
    
    # Save evaluation metrics
    metrics = {
        'accuracy': results['accuracy'],
        'f1_score': results['f1_score'],
        'mae': results['mae'],
        'mse': results['mse']
    }
    
    with open('results/evaluation_metrics.txt', 'w') as f:
        f.write('Evaluation Metrics:\n')
        for metric, value in metrics.items():
            f.write(f'{metric}: {value:.4f}\n')
        f.write('\nClassification Report:\n')
        f.write(classification_report(y_test, results['y_pred']))
    
    # Generate and save plots
    plot_training_history(history, 'models/training_history.png')
    plot_confusion_matrix(results['confusion_matrix'], 'models/confusion_matrix.png')
    plot_roc_curve(y_test, results['y_pred_prob'], 'models/roc_curve.png')
    plot_precision_recall_curve(y_test, results['y_pred_prob'], 'models/precision_recall_curve.png')
    
    # Save predictions
    save_predictions(y_test, results['y_pred'], 'models/predictions.csv')
    
    # Analyze demographics
    demographics_path = os.path.join(data_dir, 'demographics.csv')
    analyze_demographics(demographics_path, 'visualizations')
    
    print("Training and evaluation completed successfully!")
    print(f"Model saved to: models/lstm_model.h5")
    print(f"Results saved to: results/evaluation_metrics.txt")
    print(f"Plots saved to: models/ directory")
    print(f"Demographics analysis saved to: visualizations/ directory")

if __name__ == "__main__":
    main()
