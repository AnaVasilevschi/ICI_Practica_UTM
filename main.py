"""
Main script to run the Parkinson's disease detection pipeline.

This script orchestrates the entire process from data preprocessing to model training and evaluation.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_preprocessing import download_dataset, load_demographics, load_gait_data, extract_features, preprocess_data, save_processed_data
from model import create_lstm_model, train_model, evaluate_model, plot_training_history, plot_confusion_matrix, save_predictions

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Parkinson\'s Disease Detection using LSTM')
    parser.add_argument('--data_dir', type=str, default='../data/raw',
                        help='Directory to store raw data')
    parser.add_argument('--processed_dir', type=str, default='../data/processed',
                        help='Directory to store processed data')
    parser.add_argument('--model_dir', type=str, default='../models',
                        help='Directory to store model and results')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=10,
                        help='Patience for early stopping')
    parser.add_argument('--skip_preprocessing', action='store_true',
                        help='Skip data preprocessing if already done')
    return parser.parse_args()

def main():
    """Main function to run the pipeline."""
    # Parse arguments
    args = parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.processed_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Define paths
    base_url = "https://physionet.org/files/gaitpdb/1.0.0"
    model_path = os.path.join(args.model_dir, 'lstm_model.h5')
    
    # Step 1: Data Preprocessing
    if not args.skip_preprocessing:
        print("\n=== Step 1: Data Preprocessing ===")
        
        # Download the dataset
        download_dataset(base_url, args.data_dir)
        
        # Load demographics data
        demographics_path = os.path.join(args.data_dir, "demographics.html")
        demographics = load_demographics(demographics_path)
        if demographics is not None:
            print(f"Loaded demographics data with {len(demographics)} subjects")
            # Save demographics to CSV for easier access
            demographics.to_csv(os.path.join(args.processed_dir, 'demographics.csv'), index=False)
        
        # Load gait data
        gait_data = load_gait_data(args.data_dir)
        print(f"Loaded gait data for {len(gait_data)} subjects")
        
        # Extract features
        X, y = extract_features(gait_data)
        print(f"Extracted features: X shape = {X.shape}, y shape = {y.shape}")
        
        # Preprocess data
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)
        print(f"Preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
        
        # Save processed data
        save_processed_data(args.processed_dir, X_train, X_val, X_test, y_train, y_val, y_test)
        
        # Save scaler for future use
        import joblib
        joblib.dump(scaler, os.path.join(args.processed_dir, 'scaler.pkl'))
        
        print("Data preprocessing completed successfully.")
    else:
        print("\n=== Step 1: Skipping Data Preprocessing ===")
    
    # Step 2: Model Training and Evaluation
    print("\n=== Step 2: Model Training and Evaluation ===")
    
    # Load preprocessed data
    X_train = np.load(os.path.join(args.processed_dir, 'X_train.npy'))
    X_val = np.load(os.path.join(args.processed_dir, 'X_val.npy'))
    X_test = np.load(os.path.join(args.processed_dir, 'X_test.npy'))
    y_train = np.load(os.path.join(args.processed_dir, 'y_train.npy'))
    y_val = np.load(os.path.join(args.processed_dir, 'y_val.npy'))
    y_test = np.load(os.path.join(args.processed_dir, 'y_test.npy'))
    
    print(f"Loaded data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # Create model
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = create_lstm_model(input_shape)
    model.summary()
    
    # Train model
    model, history = train_model(
        model, X_train, y_train, X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        model_path=model_path
    )
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    
    # Save results to file
    with open(os.path.join(args.model_dir, 'evaluation_results.txt'), 'w') as f:
        f.write("Model Evaluation Results:\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"F1 Score: {results['f1_score']:.4f}\n")
        f.write(f"Mean Absolute Error: {results['mae']:.4f}\n")
        f.write(f"Mean Squared Error: {results['mse']:.4f}\n\n")
        f.write("Classification Report:\n")
        from sklearn.metrics import classification_report
        f.write(classification_report(y_test, results['y_pred']))
    
    # Plot training history
    history_path = os.path.join(args.model_dir, 'training_history.png')
    plot_training_history(history, output_path=history_path)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.model_dir, 'confusion_matrix.png')
    plot_confusion_matrix(results['confusion_matrix'], output_path=cm_path)
    
    # Save predictions
    pred_path = os.path.join(args.model_dir, 'predictions.csv')
    save_predictions(y_test, results['y_pred'], output_path=pred_path)
    
    print("Model training and evaluation completed successfully.")
    print(f"Model saved to: {model_path}")
    print(f"Training history plot saved to: {history_path}")
    print(f"Confusion matrix plot saved to: {cm_path}")
    print(f"Predictions saved to: {pred_path}")

if __name__ == "__main__":
    main()
