"""
Script to download the Gait in Parkinson's Disease dataset from PhysioNet.

This script downloads the necessary data files from the PhysioNet repository
for training and evaluating the Parkinson's disease detection model.
"""

import os
import requests
import pandas as pd
from bs4 import BeautifulSoup
import time

def download_file(url, output_path):
    """
    Download a file from a URL to a specified path.
    
    Args:
        url (str): URL of the file to download
        output_path (str): Path where the file will be saved
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
            
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_file_list(base_url):
    """
    Get a list of data files from the PhysioNet repository.
    
    Args:
        base_url (str): Base URL of the PhysioNet repository
        
    Returns:
        list: List of data file URLs
    """
    try:
        # Get the HTML content of the repository page
        response = requests.get(base_url)
        response.raise_for_status()
        
        # Parse the HTML content
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links to data files
        file_links = []
        for link in soup.find_all('a'):
            href = link.get('href')
            if href and (href.endswith('.txt') or href.endswith('.html') or href.endswith('.xls')):
                if href.startswith('/'):
                    file_links.append(f"https://physionet.org{href}")
                else:
                    file_links.append(f"{base_url}/{href}")
        
        return file_links
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []

def main():
    """Main function to download the dataset."""
    # Define paths
    base_url = "https://physionet.org/content/gaitpdb/1.0.0"
    output_dir = "../data/raw"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading dataset from {base_url} to {output_dir}...")
    
    # Download format.txt
    format_url = f"{base_url}/format.txt"
    format_path = os.path.join(output_dir, "format.txt")
    if download_file(format_url, format_path):
        print(f"Downloaded format.txt")
    
    # Download demographics.html
    demographics_url = f"{base_url}/demographics.html"
    demographics_path = os.path.join(output_dir, "demographics.html")
    if download_file(demographics_url, demographics_path):
        print(f"Downloaded demographics.html")
    
    # For demonstration purposes, we'll download a subset of the data files
    # In a real scenario, you would download all files
    
    # Define prefixes for the three studies
    study_prefixes = ['Ga', 'Ju', 'Si']
    group_prefixes = ['Co', 'Pt']
    
    # Download a few example files from each study and group
    for study in study_prefixes:
        for group in group_prefixes:
            for i in range(1, 4):  # Download first 3 subjects from each group
                subject_num = f"{i:02d}"
                filename = f"{study}{group}{subject_num}_01.txt"
                file_url = f"{base_url}/{filename}"
                file_path = os.path.join(output_dir, filename)
                
                if download_file(file_url, file_path):
                    print(f"Downloaded {filename}")
                
                # Add a small delay to avoid overwhelming the server
                time.sleep(0.5)
    
    print("Dataset download completed.")

if __name__ == "__main__":
    main()
