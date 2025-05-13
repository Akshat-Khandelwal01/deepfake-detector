import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
import time
import json

# Increase file size limit for uploads to handle larger files
# Define a large value in MB (e.g., 1000MB = ~1GB)
MAX_FILE_SIZE = 1000  # Maximum file size in MB

# Set page configuration
st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🕵️",
    layout="wide"
)

# Configure server for larger file uploads
# This is done through a custom config file, so we'll notify the user
st.sidebar.info(f"File upload limit set to {MAX_FILE_SIZE}MB. To increase this limit further, configure the .streamlit/config.toml file.")

# Create a custom config.toml file at startup if it doesn't exist
config_dir = os.path.join(os.path.expanduser("~"), ".streamlit")
config_path = os.path.join(config_dir, "config.toml")

if not os.path.exists(config_dir):
    os.makedirs(config_dir)
    
if not os.path.exists(config_path):
    with open(config_path, "w") as f:
        f.write(f"""
[server]
maxUploadSize = {MAX_FILE_SIZE}
enableXsrfProtection = false
enableCORS = false

[browser]
gatherUsageStats = false
        """)


# Define the model architecture (same as in the notebook)
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# Function to extract frames from video
def frame_extract(path, num_frames=20):
    frames = []
    vidcap = cv2.VideoCapture(path)
    
    if not vidcap.isOpened():
        st.error("Could not open video file. Please check the format and try again.")
        return None, None
    
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = vidcap.get(cv2.CAP_PROP_FPS)
    duration = total_frames / fps if fps > 0 else 0
    
    st.write(f"Video Info: {total_frames} frames, {fps:.2f} FPS, Duration: {duration:.2f} seconds")
    
    if total_frames == 0:
        st.error("Could not read video file. Please check the format and try again.")
        return None, None
    
    # If video has fewer frames than requested, use all frames
    if total_frames < num_frames:
        num_frames = total_frames
    
    # Calculate interval to extract evenly spaced frames
    interval = max(1, total_frames // num_frames)
    
    count = 0
    success = True
    frame_indices = []
    
    while success and len(frames) < num_frames:
        success, image = vidcap.read()
        if count % interval == 0 and success:
            frame_indices.append(count)
            # Convert BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frames.append(image)
        count += 1
            
    vidcap.release()
    return frames, frame_indices

# Function to detect and crop faces using OpenCV (Haar Cascades)
def extract_faces(frames):
    face_frames = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            margin = 30
            x = max(0, x - margin)
            y = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + 2 * margin)
            y2 = min(frame.shape[0], y + h + 2 * margin)
            face_crop = frame[y:y2, x:x2]
            face_frames.append(face_crop)
        else:
            face_frames.append(frame)

    return face_frames


# Function to preprocess frames for model input
def preprocess_frames(face_frames, transform, sequence_length=20):
    processed_frames = []
    
    for face in face_frames:
        # Convert numpy array to PIL Image
        face_pil = Image.fromarray(face)
        # Apply transformations
        processed_face = transform(face_pil)
        processed_frames.append(processed_face)
    
    # Stack frames and ensure correct sequence length
    if len(processed_frames) > 0:
        processed_frames = torch.stack(processed_frames)
        
        # If we have more frames than needed, select evenly spaced frames
        if len(processed_frames) > sequence_length:
            indices = np.linspace(0, len(processed_frames) - 1, sequence_length, dtype=int)
            processed_frames = processed_frames[indices]
        
        # If we have fewer frames than needed, repeat the last frame
        while len(processed_frames) < sequence_length:
            processed_frames = torch.cat([processed_frames, processed_frames[-1].unsqueeze(0)])
            
        # Add batch dimension
        processed_frames = processed_frames.unsqueeze(0)
        
    return processed_frames

# Create sidebar header
st.sidebar.title("Deepfake Detector")
st.sidebar.markdown("Upload a video to detect if it's real or fake.")

# Load the model
@st.cache_resource
def load_model_from_bytes(model_bytes):
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.info(f"Using device: {device}")
        
        model = Model(num_classes=2).to(device)
        
        # Load the model from bytes
        model_data = torch.load(model_bytes, map_location=device)
        
        # Check if the model is a state dict or a full model
        if isinstance(model_data, dict) and 'state_dict' in model_data:
            # Load from checkpoint format
            model.load_state_dict(model_data['state_dict'])
        elif isinstance(model_data, dict) and all(k.startswith('model.') or k.startswith('lstm.') or 
                                                 k.startswith('linear') or k.startswith('avgpool') for k in model_data.keys()):
            # It's a state dict with model prefix
            model.load_state_dict(model_data)
        else:
            # Assume it's a direct state dict
            model.load_state_dict(model_data)
        
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.error("Please check that your model file is compatible with the expected architecture.")
        return None, None

# Main page content
st.title("🕵️‍♂️ Deepfake Detection System")
st.markdown("""
This application uses a deep learning model to analyze videos and detect if they are real or manipulated (deepfakes).
Upload your model file and a video file below to begin the analysis.
""")

# Note about file size limits
st.info(f"This app supports files up to {MAX_FILE_SIZE}MB in size. Larger videos may require longer processing times.")

# Model file uploader
uploaded_model = st.file_uploader("Upload your PyTorch (.pt) model file", type=["pt", "pth"])

# Load the model if uploaded
model = None
device = None
if uploaded_model is not None:
    # Create a temporary file for the model
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_model_file:
        tmp_model_file.write(uploaded_model.getvalue())
        model_path = tmp_model_file.name
    
    # Load the model
    model, device = load_model_from_bytes(model_path)
    
    # Display confirmation if model loaded successfully
    if model is not None:
        st.success("Model loaded successfully!")
    else:
        st.error("Failed to load model. Please check the model file format.")

# Set up the transforms
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# Video file uploader with larger file size support
uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])

if uploaded_file is not None:
    # Check file size
    file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type}
    file_size_mb = uploaded_file.size / (1024 * 1024)  # Convert bytes to MB
    st.write(f"File Size: {file_size_mb:.2f} MB")
    
    # Save uploaded file to a temporary file - use a try-except block to handle large files
    try:
        with st.spinner(f"Processing video file ({file_size_mb:.2f} MB)..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                # Write in chunks to handle large files better
                chunk_size = 5 * 1024 * 1024  # 5MB chunks
                for chunk in iter(lambda: uploaded_file.read(chunk_size), b""):
                    tmp_file.write(chunk)
                video_path = tmp_file.name
    except Exception as e:
        st.error(f"Error processing the uploaded file: {str(e)}")
        st.error("This might be due to file size or format issues. Try with a smaller file or different format.")
    
    if model is not None and device is not None:
        # Show progress information
        with st.spinner("Analyzing video..."):
            # Process the video
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Extract frames
            status_text.text("Extracting frames from video...")
            frames, frame_indices = frame_extract(video_path, num_frames=30)
            progress_bar.progress(25)
            
            if frames is not None and len(frames) > 0:
                # Display sample frames
                st.subheader("Sample Frames")
                cols = st.columns(min(3, len(frames)))
                for i, col in enumerate(cols):
                    if i < len(frames):
                        col.image(frames[i * len(frames) // len(cols)], caption=f"Frame {frame_indices[i * len(frames) // len(cols)]}")
                
                # Extract faces
                status_text.text("Detecting and extracting faces...")
                face_frames = extract_faces(frames)
                progress_bar.progress(50)
                
                # Show face crops
                if len(face_frames) > 0:
                    st.subheader("Detected Faces")
                    cols = st.columns(min(3, len(face_frames)))
                    for i, col in enumerate(cols):
                        if i < len(face_frames):
                            col.image(face_frames[i * len(face_frames) // len(cols)], caption=f"Face {i * len(face_frames) // len(cols)}")
                    
                    # Preprocess for model
                    status_text.text("Preprocessing for model...")
                    processed_frames = preprocess_frames(face_frames, transform, sequence_length=20)
                    progress_bar.progress(75)
                
                    # Make prediction
                    status_text.text("Making prediction...")
                    with torch.no_grad():
                        processed_frames = processed_frames.to(device)
                        _, outputs = model(processed_frames)
                        _, preds = torch.max(outputs, 1)
                        softmax = nn.Softmax(dim=1)
                        probabilities = softmax(outputs)
                    
                    progress_bar.progress(100)
                    
                    # Display results
                    st.subheader("Results:")
                    result_container = st.container()
                    
                    with result_container:
                        cols = st.columns(2)
                        
                        prediction = "FAKE" if preds.item() == 0 else "REAL"
                        fake_prob = probabilities[0][0].item() * 100
                        real_prob = probabilities[0][1].item() * 100
                        
                        # Display prediction with color
                        if prediction == "FAKE":
                            cols[0].markdown(f"<h1 style='color:red'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
                        else:
                            cols[0].markdown(f"<h1 style='color:green'>Prediction: {prediction}</h1>", unsafe_allow_html=True)
                        
                        # Display confidence
                        cols[1].markdown("<h3>Confidence:</h3>", unsafe_allow_html=True)
                        cols[1].metric("FAKE", f"{fake_prob:.2f}%")
                        cols[1].metric("REAL", f"{real_prob:.2f}%")
                        
                        # Display progress bars for confidence
                        st.markdown("<h3>Prediction Confidence:</h3>", unsafe_allow_html=True)
                        st.progress(fake_prob / 100)
                        st.markdown(f"<p>Fake: {fake_prob:.2f}%</p>", unsafe_allow_html=True)
                        st.progress(real_prob / 100)
                        st.markdown(f"<p>Real: {real_prob:.2f}%</p>", unsafe_allow_html=True)
                else:
                    st.error("No faces detected in the video. Please try with another video.")
            else:
                st.error("Could not extract frames from video. Please check the video file.")
    else:
        st.error("Please upload a valid model file first.")
    
    # Clean up temporary files
    try:
        if 'video_path' in locals():
            os.unlink(video_path)
        # Clean up model temporary file if it exists
        if uploaded_model is not None and 'model_path' in locals():
            os.unlink(model_path)
    except Exception as e:
        st.warning(f"Warning: Could not clean up temporary files: {str(e)}")


# Add explanatory information
st.markdown("---")
st.subheader("How it works")
st.markdown("""
This deepfake detector uses a deep learning model combining a ResNext50 CNN with an LSTM network to analyze video sequences:

1. **Frame Extraction**: The system extracts frames from your uploaded video
2. **Face Detection**: Faces are detected and cropped from each frame
3. **Feature Extraction**: A CNN extracts spatial features from face images
4. **Temporal Analysis**: An LSTM analyzes the sequence of features to detect inconsistencies
5. **Classification**: The model classifies the video as REAL or FAKE with a confidence score

The model was trained on multiple deepfake datasets including Celeb-DF, DFDC, and FaceForensics++.
""")

