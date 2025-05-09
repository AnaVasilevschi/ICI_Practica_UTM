"""
Modified data preprocessing script to work with the downloaded dataset.

This script handles preprocessing the gait data from the PhysioNet dataset
for training the LSTM model for Parkinson's disease detection.
"""

import os
import numpy as np
import pandas as pd
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import requests
from bs4 import BeautifulSoup
from io import StringIO

def download_data_file(file_path):
    """
    Download the actual data file from PhysioNet.
    
    Args:
        file_path (str): Path to the local file containing the HTML
        
    Returns:
        str: Content of the actual data file
    """
    try:
        # Read the HTML file
        with open(file_path, 'r') as f:
            html_content = f.read()
        
        # Parse HTML to find the download link
        soup = BeautifulSoup(html_content, 'lxml')
        download_link = soup.find('a', {'title': 'Download this file'})['href']
        
        # Download the actual data file
        base_url = 'https://physionet.org'
        response = requests.get(base_url + download_link)
        response.raise_for_status()
        
        return response.text
        
    except Exception as e:
        print(f"Error downloading {file_path}: {e}")
        return None

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
        with open(demographics_file, 'r') as f:
            html_content = f.read()
        
        # Read the HTML table
        demographics = pd.read_html(StringIO(html_content))[0]
        
        # Clean up column names and handle missing values
        demographics.columns = [str(col).strip() for col in demographics.columns]
        demographics = demographics.fillna('')  # Replace NaN with empty string
        
        # Extract subject ID from the first column
        demographics['Subject'] = demographics['ID'].astype(str)
        
        return demographics
    except Exception as e:
        print(f"Error loading demographics: {e}")
        return None

def load_gait_data(data_dir):
    """
    Load gait data files from the specified directory.
    
    Args:
        data_dir (str): Directory containing the gait data files
        
    Returns:
        dict: Dictionary with subject IDs as keys and data as values
    """
    gait_data = {}
    
    # Get list of all text files in the directory (excluding format.txt)
    files = [f for f in glob.glob(os.path.join(data_dir, "*.txt")) 
             if os.path.basename(f) != "format.txt"]
    
    print(f"Found {len(files)} data files")
    
    for file_path in files:
        try:
            filename = os.path.basename(file_path)
            
            # Skip if not a gait data file
            if not (filename.startswith('Ga') or filename.startswith('Ju') or filename.startswith('Si')):
                continue
            
            # Parse the filename to get subject info
            # Format: [Study][Group][Subject#]_[Walk#].txt
            # Example: GaCo01_01.txt
            parts = filename.split('_')[0]
            study = parts[:2]  # Ga, Ju, or Si
            group = parts[2:4]  # Co or Pt
            subject_num = parts[4:] if len(parts) > 4 else ""
            subject_id = f"{study}{group}{subject_num}"
            
            # Download and load the data file
            content = download_data_file(file_path)
            if content is None:
                continue
                
            # Convert content to DataFrame
            data = pd.read_csv(StringIO(content), delimiter=r'\s+', header=None)
            
            # Assign column names based on the format description
            column_names = ['Time']
            column_names.extend([f'L{i+1}' for i in range(8)])  # Left foot sensors
            column_names.extend([f'R{i+1}' for i in range(8)])  # Right foot sensors
            column_names.extend(['L_Total', 'R_Total'])  # Total forces
            
            # Ensure the data has the expected number of columns
            if data.shape[1] == len(column_names):
                data.columns = column_names
                
                # Store in dictionary
                if subject_id not in gait_data:
                    gait_data[subject_id] = []
                
                gait_data[subject_id].append({
                    'filename': filename,
                    'data': data
                })
                print(f"Loaded {filename} for subject {subject_id}")
            else:
                print(f"Warning: {filename} has {data.shape[1]} columns, expected {len(column_names)}")
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return gait_data

def extract_features(gait_data, demographics=None):
    """
    Extract features from the gait data for use in the LSTM model.
    
    Args:
        gait_data (dict): Dictionary containing gait data for each subject
        demographics (pd.DataFrame, optional): Demographics data
        
    Returns:
        tuple: (X, y, subject_ids) where X is the feature matrix, 
               y is the target vector, and subject_ids are the corresponding subject IDs
    """
    features = []
    labels = []
    subject_ids = []
    
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
        stratify=[1 if 'Pt' in s else 0 for s in unique_subjects]  # Stratify by PD/control
    )
    
    # Split train+val subjects into train and val
    train_subjects, val_subjects = train_test_split(
        train_val_subjects, test_size=val_size/(1-test_size),  # Adjust val_size to be relative to train+val
        random_state=random_state,
        stratify=[1 if 'Pt' in s else 0 for s in train_val_subjects]  # Stratify by PD/control
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

def save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, scaler=None):
    """
    Save the processed data to disk.
    
    Args:
        output_dir (str): Directory to save the processed data
        X_train, X_val, X_test (np.array): Feature matrices
        y_train, y_val, y_test (np.array): Target vectors
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
        
        if scaler is not None:
            joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))
        
        print(f"Processed data saved to {output_dir}")
        return True
    
    except Exception as e:
        print(f"Error saving processed data: {e}")
        return False

def main():
    """Main function to run the data preprocessing pipeline."""
    # Define paths
    data_dir = "data/raw"
    demographics_file = os.path.join(data_dir, "demographics.html")
    output_dir = "data/processed"
    
    # Load demographics data
    print("Loading demographics data...")
    demographics = load_demographics(demographics_file)
    
    # Load gait data
    print("Loading gait data...")
    gait_data = load_gait_data(data_dir)
    
    # Extract features
    print("Extracting features...")
    X, y, subject_ids = extract_features(gait_data, demographics)
    print(f"Extracted features: X shape = {X.shape}, y shape = {y.shape}")
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = preprocess_data(X, y, subject_ids)
    
    # Save processed data
    print("Saving processed data...")
    save_processed_data(output_dir, X_train, X_val, X_test, y_train, y_val, y_test, scaler)
    
    print("Data preprocessing completed successfully!")

if __name__ == "__main__":
    main()
