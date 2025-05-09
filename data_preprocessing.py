"""
Data preprocessing module for Parkinson's disease detection using gait data.

This module handles downloading, extracting, and preprocessing the gait data
from the PhysioNet Gait in Parkinson's Disease dataset.
"""

import os
import numpy as np
import pandas as pd
import requests
import zipfile
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def download_dataset(base_url, output_dir):
    """
    Download the Gait in Parkinson's Disease dataset from PhysioNet.
    
    Args:
        base_url (str): Base URL for the dataset
        output_dir (str): Directory to save the downloaded data
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    print("Downloading dataset...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Download the dataset files
    try:
        # Download demographics file
        demographics_url = f"{base_url}/demographics.html"
        response = requests.get(demographics_url)
        with open(os.path.join(output_dir, "demographics.html"), "wb") as f:
            f.write(response.content)
        
        # Download format description
        format_url = f"{base_url}/format.txt"
        response = requests.get(format_url)
        with open(os.path.join(output_dir, "format.txt"), "wb") as f:
            f.write(response.content)
        
        # Download data files (this would need to be expanded to get all files)
        # For demonstration, we'll download a few example files
        
        # Get list of files from the dataset page
        response = requests.get(base_url)
        # This is a simplified approach - in a real implementation, 
        # we would parse the HTML to get all data files
        
        # For now, we'll manually specify some example files to download
        example_files = [
            "GaCo01_01.txt", "GaCo02_01.txt", "GaCo03_01.txt",
            "GaPt01_01.txt", "GaPt02_01.txt", "GaPt03_01.txt",
            "JuCo01_01.txt", "JuCo02_01.txt", 
            "JuPt01_01.txt", "JuPt02_01.txt",
            "SiCo01_01.txt", "SiCo02_01.txt",
            "SiPt01_01.txt", "SiPt02_01.txt"
        ]
        
        for filename in example_files:
            file_url = f"{base_url}/{filename}"
            try:
                response = requests.get(file_url)
                response.raise_for_status()  # Raise an exception for HTTP errors
                with open(os.path.join(output_dir, filename), "wb") as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            except requests.exceptions.RequestException as e:
                print(f"Error downloading {filename}: {e}")
                
        print("Dataset download completed.")
        return True
    
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

def load_demographics(demographics_file):
    """
    Load and parse the demographics data.
    
    Args:
        demographics_file (str): Path to the demographics HTML file
        
    Returns:
        pd.DataFrame: DataFrame containing demographics data
    """
    try:
        # Read the HTML file
        demographics = pd.read_html(demographics_file)[0]
        return demographics
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return None

def load_gait_data(data_dir, file_pattern=None):
    """
    Load gait data files from the specified directory.
    
    Args:
        data_dir (str): Directory containing the gait data files
        file_pattern (str, optional): Pattern to match specific files
        
    Returns:
        dict: Dictionary with subject IDs as keys and data as values
    """
    gait_data = {}
    
    # Get list of all text files in the directory
    files = [f for f in os.listdir(data_dir) if f.endswith('.txt') and not f == 'format.txt']
    
    if file_pattern:
        files = [f for f in files if file_pattern in f]
    
    for filename in files:
        try:
            # Parse the filename to get subject info
            parts = filename.split('_')[0]
            study = parts[:2]  # Ga, Ju, or Si
            group = parts[2:4]  # Co or Pt
            subject_num = parts[4:] if len(parts) > 4 else ""
            subject_id = f"{study}{group}{subject_num}"
            
            # Load the data file
            file_path = os.path.join(data_dir, filename)
            data = pd.read_csv(file_path, delimiter='\s+', header=None)
            
            # Assign column names based on the format description
            column_names = ['Time']
            column_names.extend([f'L{i+1}' for i in range(8)])  # Left foot sensors
            column_names.extend([f'R{i+1}' for i in range(8)])  # Right foot sensors
            column_names.extend(['L_Total', 'R_Total'])  # Total forces
            
            data.columns = column_names
            
            # Store in dictionary
            if subject_id not in gait_data:
                gait_data[subject_id] = []
            
            gait_data[subject_id].append({
                'filename': filename,
                'data': data
            })
            
        except Exception as e:
            print(f"Error loading {filename}: {e}")
    
    return gait_data

def extract_features(gait_data):
    """
    Extract features from the gait data for use in the LSTM model.
    
    Args:
        gait_data (dict): Dictionary containing gait data for each subject
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target vector
    """
    features = []
    labels = []
    
    for subject_id, subject_data in gait_data.items():
        # Determine if subject is PD patient or control
        is_pd = 'Pt' in subject_id
        
        for recording in subject_data:
            data = recording['data']
            
            # Extract time series data for each sensor
            # For LSTM, we need sequences of data
            # We'll use windows of 100 samples (1 second at 100Hz)
            window_size = 100
            stride = 50  # 50% overlap between windows
            
            for i in range(0, len(data) - window_size, stride):
                window = data.iloc[i:i+window_size]
                
                # Extract features from the window
                # We'll use all sensor data as features
                sensor_data = window.iloc[:, 1:17].values  # All sensors excluding time and totals
                
                features.append(sensor_data)
                labels.append(1 if is_pd else 0)  # 1 for PD, 0 for control
    
    return np.array(features), np.array(labels)

def preprocess_data(X, y, test_size=0.2, val_size=0.2, random_state=42):
    """
    Preprocess the data by splitting into train/val/test sets and scaling.
    
    Args:
        X (np.array): Feature matrix
        y (np.array): Target vector
        test_size (float): Proportion of data to use for testing
        val_size (float): Proportion of training data to use for validation
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    """
    # First split into train+val and test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Then split train+val into train and val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
        random_state=random_state,
        stratify=y_train_val
    )
    
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

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test):
    """
    Save the processed data to disk.
    
    Args:
        output_dir (str): Directory to save the processed data
        X_train, X_val, X_test (np.array): Feature matrices
        y_train, y_val, y_test (np.array): Target vectors
        
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
        
        print(f"Processed data saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

def main():
    """Main function to run the data preprocessing pipeline."""
    # Define paths
    base_url = "https://physionet.org/files/gaitpdb/1.0.0"
    data_dir = "../data/raw"
    processed_dir = "../data/processed"
    
    # Download the dataset
    download_dataset(base_url, data_dir)
    
    # Load demographics data
    demographics = load_demographics(os.path.join(data_dir, "demographics.html"))
    if demographics is not None:
        print(f"Loaded demographics data with {len(demographics)} subjects")
    
    # Load gait data
    gait_data = load_gait_data(data_dir)
    print(f"Loaded gait data for {len(gait_data)} subjects")
    
    # Extract features
    X, y = extract_features(gait_data)
    print(f"Extracted features: X shape = {X.shape}, y shape = {y.shape}")
    
    # Preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y)
    print(f"Preprocessed data: X_train shape = {X_train.shape}, y_train shape = {y_train.shape}")
    
    # Save processed data
    save_processed_data(processed_dir, X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("Data preprocessing completed successfully.")

if __name__ == "__main__":
    main()
