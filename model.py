"""
LSTM model implementation for Parkinson's disease detection using gait data.

This module implements the LSTM network architecture described in the paper:
"Automatic and non-invasive Parkinson's disease diagnosis and severity rating using LSTM network"
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score

def create_lstm_model(input_shape, dropout_rate=0.2, l2_reg=0.001):
    """
    Create an LSTM model for Parkinson's disease detection.
    
    Args:
        input_shape (tuple): Shape of input data (timesteps, features)
        dropout_rate (float): Dropout rate for regularization
        l2_reg (float): L2 regularization parameter
        
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential()
    
    # First LSTM layer with L2 regularization and dropout
    model.add(LSTM(
        units=64,
        input_shape=input_shape,
        return_sequences=True,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    ))
    model.add(Dropout(dropout_rate))
    
    # Second LSTM layer
    model.add(LSTM(
        units=32,
        return_sequences=False,
        kernel_regularizer=l2(l2_reg),
        recurrent_regularizer=l2(l2_reg)
    ))
    model.add(Dropout(dropout_rate))
    
    # Output layer for binary classification
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model with Adam optimizer
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=100, patience=10, model_path='../models/lstm_model.h5'):
    """
    Train the LSTM model.
    
    Args:
        model (tf.keras.Model): LSTM model to train
        X_train (np.array): Training features
        y_train (np.array): Training labels
        X_val (np.array): Validation features
        y_val (np.array): Validation labels
        batch_size (int): Batch size for training
        epochs (int): Maximum number of epochs
        patience (int): Patience for early stopping
        model_path (str): Path to save the best model
        
    Returns:
        tuple: (trained model, training history)
    """
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    model_checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the trained model on the test set.
    
    Args:
        model (tf.keras.Model): Trained LSTM model
        X_test (np.array): Test features
        y_test (np.array): Test labels
        
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Get model predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred_prob)
    mse = mean_squared_error(y_test, y_pred_prob)
    
    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Compile results
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'mae': mae,
        'mse': mse,
        'classification_report': report,
        'confusion_matrix': cm,
        'y_pred': y_pred,
        'y_pred_prob': y_pred_prob
    }
    
    return results

def plot_training_history(history, output_path='../models/training_history.png'):
    """
    Plot the training history.
    
    Args:
        history: Training history from model.fit()
        output_path (str): Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_confusion_matrix(cm, output_path='../models/confusion_matrix.png'):
    """
    Plot the confusion matrix.
    
    Args:
        cm (np.array): Confusion matrix
        output_path (str): Path to save the plot
        
    Returns:
        None
    """
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Control', 'Parkinson\'s']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(output_path)
    plt.close()

def save_predictions(y_test, y_pred, output_path='../models/predictions.csv'):
    """
    Save the test set predictions to a CSV file.
    
    Args:
        y_test (np.array): True labels
        y_pred (np.array): Predicted labels
        output_path (str): Path to save the CSV file
        
    Returns:
        None
    """
    import pandas as pd
    
    # Create DataFrame with true and predicted labels
    df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred
    })
    
    # Save to CSV
    df.to_csv(output_path, index=False)

def main():
    """Main function to create, train, and evaluate the LSTM model."""
    import os
    
    # Load preprocessed data
    data_dir = '../data/processed'
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
    model, history = train_model(model, X_train, y_train, X_val, y_val)
    
    # Evaluate model
    results = evaluate_model(model, X_test, y_test)
    
    # Print evaluation results
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1 Score: {results['f1_score']:.4f}")
    print(f"Mean Absolute Error: {results['mae']:.4f}")
    print(f"Mean Squared Error: {results['mse']:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, results['y_pred']))
    
    # Plot training history
    plot_training_history(history)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])
    
    # Save predictions
    save_predictions(y_test, results['y_pred'])
    
    print("Model training and evaluation completed successfully.")

if __name__ == "__main__":
    main()
