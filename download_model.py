#!/usr/bin/env python3
"""
Script to download the deepfake detection model from Google Drive
Usage: python download_model.py [output_path]
Default output path: ./checkpoint.pt
"""

import os
import sys
import gdown
import argparse

def download_from_gdrive(file_id, output_path="checkpoint.pt"):
    """
    Download a file from Google Drive.
    
    Args:
        file_id (str): The file ID from Google Drive URL
        output_path (str): Where to save the downloaded file
    
    Returns:
        bool: True if successful, False otherwise
    """
    print(f"Downloading model to {output_path}...")
    
    # URL format for Google Drive
    url = f"https://drive.google.com/uc?id={file_id}"
    
    try:
        # Download file
        gdown.download(url, output_path, quiet=False)
        
        # Check if download was successful
        if os.path.exists(output_path):
            file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            print(f"Download complete! File size: {file_size_mb:.2f} MB")
            return True
        else:
            print("Error: Download failed - file not found after download")
            return False
            
    except Exception as e:
        print(f"Error during download: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download deepfake detection model from Google Drive")
    parser.add_argument("file_id", help="Google Drive file ID (from the URL)")
    parser.add_argument("--output", default="checkpoint.pt", help="Output path for the model file")
    
    args = parser.parse_args()
    
    # Download the model
    success = download_from_gdrive(args.file_id, args.output)
    
    if success:
        print(f"Model successfully downloaded to {args.output}")
        print("You can now run the Streamlit app with: streamlit run deepfake_detector_app.py")
    else:
        print("Failed to download the model. Please check the file ID and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()