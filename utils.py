"""
Utility functions for the Parkinson's disease detection project.

This module contains helper functions for data visualization, model interpretation,
and other utilities used across the project.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

def plot_roc_curve(y_true, y_pred_prob, output_path='../models/roc_curve.png'):
    """
    Plot the ROC curve for the model predictions.
    
    Args:
        y_true (np.array): True labels
        y_pred_prob (np.array): Predicted probabilities
        output_path (str): Path to save the plot
        
    Returns:
        float: Area under the ROC curve (AUC)
    """
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_pred_prob)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred_prob, output_path='../models/precision_recall_curve.png'):
    """
    Plot the Precision-Recall curve for the model predictions.
    
    Args:
        y_true (np.array): True labels
        y_pred_prob (np.array): Predicted probabilities
        output_path (str): Path to save the plot
        
    Returns:
        float: Average precision score
    """
    # Calculate precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)
    avg_precision = average_precision_score(y_true, y_pred_prob)
    
    # Plot precision-recall curve
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.3)
    plt.savefig(output_path)
    plt.close()
    
    return avg_precision

def visualize_sensor_data(data, subject_id, output_dir='../visualizations'):
    """
    Visualize the sensor data for a specific subject.
    
    Args:
        data (pd.DataFrame): Sensor data
        subject_id (str): Subject ID
        output_dir (str): Directory to save the visualizations
        
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot time series for each foot
    plt.figure(figsize=(15, 10))
    
    # Left foot sensors
    plt.subplot(2, 1, 1)
    for i in range(1, 9):
        plt.plot(data['Time'], data[f'L{i}'], label=f'L{i}')
    plt.title(f'Left Foot Sensors - Subject {subject_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Right foot sensors
    plt.subplot(2, 1, 2)
    for i in range(1, 9):
        plt.plot(data['Time'], data[f'R{i}'], label=f'R{i}')
    plt.title(f'Right Foot Sensors - Subject {subject_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{subject_id}_sensor_data.png'))
    plt.close()
    
    # Plot total forces
    plt.figure(figsize=(12, 6))
    plt.plot(data['Time'], data['L_Total'], label='Left Foot Total')
    plt.plot(data['Time'], data['R_Total'], label='Right Foot Total')
    plt.title(f'Total Force - Subject {subject_id}')
    plt.xlabel('Time (s)')
    plt.ylabel('Force (N)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, f'{subject_id}_total_force.png'))
    plt.close()

def analyze_demographics(demographics_path, output_dir='../visualizations'):
    """
    Analyze and visualize the demographics data.
    
    Args:
        demographics_path (str): Path to the demographics CSV file
        output_dir (str): Directory to save the visualizations
        
    Returns:
        pd.DataFrame: Summary statistics of the demographics data
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load demographics data
    demographics = pd.read_csv(demographics_path)
    
    # Create summary statistics
    summary = demographics.groupby('Group').agg({
        'Age': ['count', 'mean', 'std', 'min', 'max'],
        'Gender': lambda x: (x == 'male').mean(),  # Proportion of males
        'HoehnYahr': ['mean', 'std', 'min', 'max'],
        'UPDRS': ['mean', 'std', 'min', 'max']
    }).reset_index()
    
    # Rename columns for better readability
    summary.columns = ['Group', 'Count', 'Age_Mean', 'Age_Std', 'Age_Min', 'Age_Max', 
                      'Male_Proportion', 'HY_Mean', 'HY_Std', 'HY_Min', 'HY_Max',
                      'UPDRS_Mean', 'UPDRS_Std', 'UPDRS_Min', 'UPDRS_Max']
    
    # Save summary to CSV
    summary.to_csv(os.path.join(output_dir, 'demographics_summary.csv'), index=False)
    
    # Visualize age distribution by group
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Group', y='Age', data=demographics)
    plt.title('Age Distribution by Group')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'age_distribution.png'))
    plt.close()
    
    # Visualize gender distribution by group
    gender_counts = demographics.groupby(['Group', 'Gender']).size().unstack()
    gender_counts.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title('Gender Distribution by Group')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'gender_distribution.png'))
    plt.close()
    
    # For PD patients only, visualize disease severity
    pd_patients = demographics[demographics['Group'] == 'PD']
    
    # Hoehn & Yahr distribution
    plt.figure(figsize=(10, 6))
    sns.countplot(x='HoehnYahr', data=pd_patients)
    plt.title('Hoehn & Yahr Stage Distribution')
    plt.xlabel('Hoehn & Yahr Stage')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'hoehn_yahr_distribution.png'))
    plt.close()
    
    # UPDRS distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(pd_patients['UPDRS'], bins=15, kde=True)
    plt.title('UPDRS Score Distribution')
    plt.xlabel('UPDRS Score')
    plt.ylabel('Count')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'updrs_distribution.png'))
    plt.close()
    
    return summary

def feature_importance_analysis(model, X_test, feature_names=None, output_path='../models/feature_importance.png'):
    """
    Analyze feature importance using permutation importance.
    
    Args:
        model: Trained model
        X_test (np.array): Test features
        feature_names (list): Names of features
        output_path (str): Path to save the plot
        
    Returns:
        pd.DataFrame: Feature importance scores
    """
    from sklearn.inspection import permutation_importance
    
    # Reshape X_test if needed (for LSTM input)
    if len(X_test.shape) == 3:
        n_samples, n_timesteps, n_features = X_test.shape
        X_test_reshaped = X_test.reshape(n_samples, n_timesteps * n_features)
    else:
        X_test_reshaped = X_test
    
    # Calculate permutation importance
    result = permutation_importance(model, X_test_reshaped, y_test, n_repeats=10, random_state=42)
    
    # Create feature names if not provided
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(X_test_reshaped.shape[1])]
    
    # Create DataFrame with importance scores
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': result.importances_mean,
        'Std': result.importances_std
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=importance_df.head(20))
    plt.title('Feature Importance (Top 20)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    return importance_df
