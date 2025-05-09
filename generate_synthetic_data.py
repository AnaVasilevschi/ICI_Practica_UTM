"""
Script to generate synthetic gait data for Parkinson's disease detection.

This script creates synthetic data that mimics the structure of the PhysioNet
Gait in Parkinson's Disease dataset for demonstration purposes.
"""

import os
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

def generate_subject_data(subject_id, is_pd, num_samples=1000):
    """
    Generate synthetic gait data for a single subject.
    
    Args:
        subject_id (str): Subject identifier
        is_pd (bool): Whether the subject has Parkinson's disease
        num_samples (int): Number of time samples to generate
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic gait data
    """
    # Create time vector (100 Hz sampling rate)
    time = np.linspace(0, num_samples/100, num_samples)
    
    # Base frequency for the gait cycle (slightly different for PD vs control)
    if is_pd:
        # PD patients typically have more irregular gait patterns
        base_freq = 1.0 + 0.2 * np.random.randn()  # Around 1 Hz but more variable
        noise_level = 0.3  # More noise/variability
        asymmetry = 0.3 * np.random.rand()  # More asymmetry between left and right
    else:
        # Controls have more regular gait patterns
        base_freq = 1.2 + 0.1 * np.random.randn()  # Around 1.2 Hz with less variability
        noise_level = 0.15  # Less noise/variability
        asymmetry = 0.1 * np.random.rand()  # Less asymmetry
    
    # Generate synthetic sensor data
    data = {}
    data['Time'] = time
    
    # Generate left foot sensor data (8 sensors)
    for i in range(1, 9):
        # Each sensor has a different phase and amplitude
        phase = 2 * np.pi * np.random.rand()
        amp = 50 + 30 * np.random.rand()
        
        # Base signal with gait frequency
        signal = amp * np.sin(2 * np.pi * base_freq * time + phase)
        
        # Add some higher frequency components
        signal += 0.3 * amp * np.sin(4 * np.pi * base_freq * time + phase)
        
        # Add noise
        signal += noise_level * amp * np.random.randn(num_samples)
        
        # Ensure non-negative values and add offset
        signal = np.maximum(signal, 0) + 10 * np.random.rand()
        
        # For PD patients, add some random fluctuations to simulate gait variability
        if is_pd:
            # Add random dips to simulate freezing of gait
            freeze_indices = np.random.choice(num_samples, size=int(0.05*num_samples), replace=False)
            signal[freeze_indices] *= 0.5
            
            # Add some tremor-like high-frequency oscillations
            tremor_freq = 4.0 + 1.0 * np.random.rand()  # 4-5 Hz tremor
            signal += 0.2 * amp * np.sin(2 * np.pi * tremor_freq * time)
        
        data[f'L{i}'] = signal
    
    # Generate right foot sensor data (8 sensors)
    for i in range(1, 9):
        # Each sensor has a different phase and amplitude
        phase = 2 * np.pi * np.random.rand()
        amp = 50 + 30 * np.random.rand()
        
        # Base signal with gait frequency, phase shifted from left foot
        signal = amp * np.sin(2 * np.pi * base_freq * time + phase + np.pi)  # Phase shift of π (half cycle)
        
        # Add some higher frequency components
        signal += 0.3 * amp * np.sin(4 * np.pi * base_freq * time + phase + np.pi)
        
        # Add noise
        signal += noise_level * amp * np.random.randn(num_samples)
        
        # Ensure non-negative values and add offset
        signal = np.maximum(signal, 0) + 10 * np.random.rand()
        
        # Add asymmetry between left and right foot for PD patients
        if is_pd:
            signal *= (1 - asymmetry)
            
            # Add random dips to simulate freezing of gait
            freeze_indices = np.random.choice(num_samples, size=int(0.05*num_samples), replace=False)
            signal[freeze_indices] *= 0.5
            
            # Add some tremor-like high-frequency oscillations
            tremor_freq = 4.0 + 1.0 * np.random.rand()  # 4-5 Hz tremor
            signal += 0.2 * amp * np.sin(2 * np.pi * tremor_freq * time)
        
        data[f'R{i}'] = signal
    
    # Calculate total forces for each foot
    data['L_Total'] = sum(data[f'L{i}'] for i in range(1, 9))
    data['R_Total'] = sum(data[f'R{i}'] for i in range(1, 9))
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    return df

def generate_demographics(num_pd_subjects=20, num_control_subjects=20):
    """
    Generate synthetic demographics data.
    
    Args:
        num_pd_subjects (int): Number of PD subjects to generate
        num_control_subjects (int): Number of control subjects to generate
        
    Returns:
        pd.DataFrame: DataFrame containing synthetic demographics data
    """
    # Lists to store data
    subject_ids = []
    groups = []
    genders = []
    ages = []
    heights = []
    weights = []
    hoehn_yahr = []
    updrs = []
    
    # Generate PD subjects
    for i in range(1, num_pd_subjects + 1):
        subject_id = f"S{i:02d}"
        subject_ids.append(subject_id)
        groups.append("PD")
        genders.append(random.choice(["male", "female"]))
        ages.append(random.randint(55, 85))
        heights.append(round(random.uniform(1.5, 1.9), 2))
        weights.append(random.randint(50, 100))
        hoehn_yahr.append(random.choice([1, 1.5, 2, 2.5, 3, 4]))
        updrs.append(random.randint(10, 50))
    
    # Generate control subjects
    for i in range(1, num_control_subjects + 1):
        subject_id = f"C{i:02d}"
        subject_ids.append(subject_id)
        groups.append("Control")
        genders.append(random.choice(["male", "female"]))
        ages.append(random.randint(55, 85))
        heights.append(round(random.uniform(1.5, 1.9), 2))
        weights.append(random.randint(50, 100))
        hoehn_yahr.append(0)  # Controls have 0 Hoehn & Yahr score
        updrs.append(0)  # Controls have 0 UPDRS score
    
    # Create DataFrame
    demographics = pd.DataFrame({
        'Subject': subject_ids,
        'Group': groups,
        'Gender': genders,
        'Age': ages,
        'Height': heights,
        'Weight': weights,
        'HoehnYahr': hoehn_yahr,
        'UPDRS': updrs
    })
    
    return demographics

def extract_features(gait_data, window_size=100, stride=50):
    """
    Extract features from the gait data for use in the LSTM model.
    
    Args:
        gait_data (dict): Dictionary containing gait data for each subject
        window_size (int): Size of the sliding window (in samples)
        stride (int): Stride of the sliding window (in samples)
        
    Returns:
        tuple: (X, y, subject_ids) where X is the feature matrix, 
               y is the target vector, and subject_ids are the corresponding subject IDs
    """
    features = []
    labels = []
    subject_ids = []
    
    for subject_id, data in gait_data.items():
        # Determine if subject has PD
        is_pd = subject_id.startswith('S')  # S for PD subjects, C for controls
        
        # Extract time series data using sliding window
        for i in range(0, len(data) - window_size, stride):
            window = data.iloc[i:i+window_size]
            
            # Extract features from the window (all sensor data excluding time and totals)
            sensor_data = window.iloc[:, 1:17].values  # All sensors excluding time and totals
            
            features.append(sensor_data)
            labels.append(1 if is_pd else 0)  # 1 for PD, 0 for control
            subject_ids.append(subject_id)
    
    return np.array(features), np.array(labels), np.array(subject_ids)

def preprocess_data(X, y, subject_ids, test_size=0.2, val_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into train/val/test sets and scaling.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector
        subject_ids (np.array): Subject IDs for each sample
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # Get unique subject IDs
    unique_subjects = np.unique(subject_ids)
    
    # Split subjects into train+val and test
    train_val_subjects, test_subjects = train_test_split(
        unique_subjects, test_size=test_size, random_state=random_state,
        stratify=[1 if s.startswith('S') else 0 for s in unique_subjects]  # Stratify by PD/control
    )
    
    # Split train+val subjects into train and val
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
        random_state=random_state,
        stratify=[1 if s.startswith('S') else 0 for s in train_val_subjects]  # Stratify by PD/control
    )
    
    # Create masks for each set
    train_mask = np.isin(subject_ids, train_subjects)
    val_mask = np.isin(subject_ids, val_subjects)
    test_mask = np.isin(subject_ids, test_subjects)
    
    # Split data using masks
    X_train = X[train_mask]
    y_train = y[train_mask]
    X_val = X[val_mask]
    y_val = y[val_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]
    
    print(f"Train set: {len(train_subjects)} subjects, {X_train.shape[0]} samples")
    print(f"Validation set: {len(val_subjects)} subjects, {X_val.shape[0]} samples")
    print(f"Test set: {len(test_subjects)} subjects, {X_test.shape[0]} samples")
    
    # Reshape for scaling (combine time and features dimensions)
    n_samples_train, n_timesteps, n_features = X_train.shape
    n_samples_val = X_val.shape[0]
    n_samples_test = X_test.shape[0]
    
    X_train_reshaped = X_train.reshape(n_samples_train, -1)
    X_val_reshaped = X_val.reshape(n_samples_val, -1)
    X_test_reshaped = X_test.reshape(n_samples_test, -1)
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_val_scaled = scaler.transform(X_val_reshaped)
    X_test_scaled = scaler.transform(X_test_reshaped)
    
    # Reshape back to 3D
    X_train_scaled = X_train_scaled.reshape(n_samples_train, n_timesteps, n_features)
    X_val_scaled = X_val_scaled.reshape(n_samples_val, n_timesteps, n_features)
    X_test_scaled = X_test_scaled.reshape(n_samples_test, n_timesteps, n_features)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, demographics=None, scaler=None):
    """
    Save the processed data to disk.
    
    Args:
        output_dir (str): Directory to save the processed data
        X_train, X_val, X_test (np.array): Feature matrices
        y_train, y_val, y_test (np.array): Target vectors
        demographics (pd.DataFrame, optional): Demographics data
        scaler (object, optional): Fitted scaler object
        
    Returns:
        bool: True if save was successful, False otherwise
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
        np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
        np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
        
        if demographics is not None:
            demographics.to_csv(os.path.join(output_dir, 'demographics.csv'), index=False)
        
        if scaler is not None:
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        print(f"Processed data saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

def main():
    """Main function to generate and preprocess synthetic data."""
    # Define paths
    output_dir = "data/processed"
    raw_dir = "data/raw"
    
    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)
    
    print("Generating synthetic data...")
    
    # Generate demographics data
    num_pd_subjects = 20
    num_control_subjects = 20
    demographics = generate_demographics(num_pd_subjects, num_control_subjects)
    print(f"Generated demographics data for {len(demographics)} subjects")
    
    # Save demographics to CSV
    demographics.to_csv(os.path.join(raw_dir, 'demographics.csv'), index=False)
    
    # Generate gait data for each subject
    gait_data = {}
    for _, row in demographics.iterrows():
        subject_id = row['Subject']
        is_pd = row['Group'] == 'PD'
        
        # Generate data with random length
        num_samples = random.randint(800, 1200)
        data = generate_subject_data(subject_id, is_pd, num_samples)
        
        # Save to dictionary
        gait_data[subject_id] = data
        
        # Save to CSV
        data.to_csv(os.path.join(raw_dir, f'{subject_id}.csv'), index=False)
    
    print(f"Generated gait data for {len(gait_data)} subjects")
    
    # Extract features
    X, y, subject_ids = extract_features(gait_data)
    print(f"Extracted features: X shape = {X.shape}, y shape = {y.shape}")
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y, subject_ids)
    print(f"Preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # Save processed data
    save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, demographics, scaler)
    
    print("Synthetic data generation and preprocessing completed successfully.")

if __name__ == "__main__":
    main()
