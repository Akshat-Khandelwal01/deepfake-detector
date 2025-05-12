import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
import cv2
import numpy as np
import tempfile
import face_recognition
from PIL import Image
import os
import gc
import psutil
import sys
import time

# Set page config
st.set_page_config(page_title="DeepFake Detector", page_icon="🕵️", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        text-align: center;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-bottom: 2rem;
        text-align: center;
    }
    .result-real {
        font-size: 2rem;
        color: #4CAF50;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #E8F5E9;
    }
    .result-fake {
        font-size: 2rem;
        color: #F44336;
        text-align: center;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFEBEE;
    }
    .confidence-meter {
        margin-top: 1rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Define model architecture (same as in the notebook)
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
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
def extract_frames(video_path, num_frames=10):
    frames = []
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        return frames
    
    indices = np.linspace(0, total_frames - 1, num_frames, dtype=np.int32)
    
    for i in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    cap.release()
    return frames

# Function to extract face from frame
def extract_face(frame):
    face_locations = face_recognition.face_locations(frame)
    if not face_locations:
        return None
    
    # Get the first face (assuming the main face)
    top, right, bottom, left = face_locations[0]
    face_image = frame[top:bottom, left:right]
    return face_image

# Function to preprocess faces for the model
def preprocess_faces(face_frames, transform):
    processed_faces = []
    for frame in face_frames:
        if frame is not None:
            processed_face = transform(frame)
            processed_faces.append(processed_face)
    
    if not processed_faces:
        return None
    
    # Ensure we have the required number of frames
    # If we have less than 10, duplicate the last frame
    while len(processed_faces) < 10:
        processed_faces.append(processed_faces[-1])
    
    # If we have more than 10, take the first 10
    processed_faces = processed_faces[:10]
    
    return torch.stack(processed_faces)

@st.cache_resource
def load_model(model_path="checkpoint.pt"):
    model = Model(2)
    
    # Check if CUDA is available and use it, otherwise use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Display a message about the device being used
    if torch.cuda.is_available():
        st.sidebar.success("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        st.sidebar.warning("Using CPU. Processing may be slower.")
    
    # Don't fail if the model file doesn't exist yet
    if os.path.exists(model_path):
        try:
            # For large models, we need to handle loading differently
            st.sidebar.info(f"Loading model from {model_path}. This may take a moment for large models...")
            
            # Use torch.load with map_location to handle device placement
            checkpoint = torch.load(model_path, map_location=device)
            
            # Check if checkpoint contains state_dict directly or nested
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            st.sidebar.success(f"Model successfully loaded from {model_path}")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.stop()
    else:
        st.sidebar.error(f"Model file {model_path} not found. Please provide a valid path.")
        st.stop()
    
    model = model.to(device)
    model.eval()
    return model, device

def get_system_info():
    # Get memory info
    memory = psutil.virtual_memory()
    memory_available_gb = memory.available / (1024**3)
    memory_total_gb = memory.total / (1024**3)
    
    # Get CPU info
    cpu_percent = psutil.cpu_percent()
    cpu_count = psutil.cpu_count()
    
    # Get Python and package versions
    python_version = sys.version.split()[0]
    torch_version = torch.__version__
    cv2_version = cv2.__version__
    
    return {
        "memory_available_gb": round(memory_available_gb, 2),
        "memory_total_gb": round(memory_total_gb, 2),
        "memory_percent": memory.percent,
        "cpu_percent": cpu_percent,
        "cpu_count": cpu_count,
        "python_version": python_version,
        "torch_version": torch_version,
        "cv2_version": cv2_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    }

def show_system_info():
    info = get_system_info()
    
    st.sidebar.title("System Information")
    st.sidebar.info(f"""
    **Memory:** {info['memory_available_gb']} GB available / {info['memory_total_gb']} GB total ({info['memory_percent']}% used)
    
    **CPU:** {info['cpu_count']} cores, {info['cpu_percent']}% used
    
    **CUDA:** {"Available" if info['cuda_available'] else "Not available"}
    {f"Device: {info['cuda_device_name']}" if info['cuda_available'] else ""}
    
    **Versions:**
    - Python: {info['python_version']}
    - PyTorch: {info['torch_version']}
    - OpenCV: {info['cv2_version']}
    """)

def main():
    st.markdown("<h1 class='main-header'>DeepFake Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload a video to check if it's real or fake</p>", unsafe_allow_html=True)
    
    # Display system information
    show_system_info()
    
    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info(
        "This application uses a deep learning model to detect whether "
        "a video is real or has been manipulated (deepfake). "
        "It works by extracting faces from the video, then analyzing them "
        "using a ResNext-LSTM neural network."
    )
    
    st.sidebar.title("Model Information")
    st.sidebar.info(
        "Architecture: ResNext50 + LSTM\n\n"
        "Input: 10 face frames of size 112x112\n\n"
        "Output: Classification (REAL/FAKE) with confidence score"
    )
    
    # Model loading options
    model_load_option = st.sidebar.radio(
        "How would you like to load the model?",
        ["Local path", "Google Drive", "Hugging Face"]
    )
    
    if model_load_option == "Local path":
        model_path = st.sidebar.text_input("Enter the path to checkpoint.pt", "checkpoint.pt")
        st.sidebar.info("Make sure the model file is in the correct location")
    
    elif model_load_option == "Google Drive":
        st.sidebar.info("To load from Google Drive:")
        st.sidebar.code("""
# Run this in a code cell before starting Streamlit
from google.colab import drive
drive.mount('/content/drive')
# Then specify the path like: /content/drive/My Drive/checkpoint.pt
        """)
        model_path = st.sidebar.text_input("Enter the Google Drive path to checkpoint.pt", "/content/drive/My Drive/checkpoint.pt")
    
    elif model_load_option == "Hugging Face":
        st.sidebar.info("To load from Hugging Face, enter your model repository name")
        hf_model_name = st.sidebar.text_input("Hugging Face Repository", "username/model-name")
        
        if hf_model_name:
            # This code will only run if the user has entered a model name
            try:
                if not os.path.exists("checkpoint.pt"):
                    st.sidebar.info("Downloading model from Hugging Face...")
                    
                    # Code to show this is where HF download would happen
                    st.sidebar.code("""
# In a normal setup, we'd download with:
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id=hf_model_name, 
                filename="checkpoint.pt",
                local_dir=".")
                    """)
                    
                    # For our current implementation, we'll still look for a local file
                    st.sidebar.warning("For now, please make sure the model file exists locally.")
            except Exception as e:
                st.sidebar.error(f"Error downloading model: {e}")
            
            model_path = "checkpoint.pt"
    
    # Load model (uses cache to avoid reloading)
    try:
        if model_load_option == "Local path" or model_load_option == "Google Drive":
            model, device = load_model(model_path)
        else:  # Hugging Face
            model, device = load_model("checkpoint.pt")
        model_loaded = True
    except Exception as e:
        st.error(f"Error loading model: {e}")
        model_loaded = False
    
    # Video file uploader
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None and model_loaded:
        # Save the uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name
        
        # Show the uploaded video
        st.video(video_path)
        
        # Process button
        if st.button("Analyze Video"):
            with st.spinner("Processing..."):
                # Extract frames from the video
                frames = extract_frames(video_path, num_frames=20)
                
                if not frames:
                    st.error("Could not extract frames from the video. Please try another file.")
                else:
                    # Extract faces from frames
                    face_frames = []
                    for frame in frames:
                        face = extract_face(frame)
                        if face is not None:
                            face_frames.append(face)
                    
                    # Show some extracted faces
                    if face_frames:
                        st.subheader("Extracted Faces")
                        cols = st.columns(min(5, len(face_frames)))
                        for i, col in enumerate(cols):
                            if i < len(face_frames):
                                col.image(face_frames[i], use_column_width=True)
                        
                        # Preprocess the face frames
                        transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((112, 112)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        
                        processed_faces = preprocess_faces(face_frames, transform)
                        
                        if processed_faces is not None:
                            # Add batch dimension
                            inputs = processed_faces.unsqueeze(0).to(device)
                            
                            # Forward pass - handle out of memory errors
                            try:
                                with torch.no_grad():
                                    # Add a progress message
                                    progress_text = st.empty()
                                    progress_text.text("Running model inference...")
                                    
                                    # Process model in chunks if needed for large models
                                    start_time = time.time()
                                    _, outputs = model(inputs)
                                    inference_time = time.time() - start_time
                                    
                                    progress_text.text(f"Model inference completed in {inference_time:.2f} seconds")
                                    probs = torch.nn.functional.softmax(outputs, dim=1)
                                
                                # Get the prediction
                                _, predicted = torch.max(outputs, 1)
                                confidence = probs[0][predicted.item()].item() * 100
                                
                                # Clean up to free memory
                                del outputs, probs
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                            except RuntimeError as e:
                                if 'out of memory' in str(e).lower():
                                    st.error("💥 GPU out of memory error! Try processing a smaller video or use CPU.")
                                    st.info("Tip: For large models, make sure you have enough GPU memory or switch to CPU processing.")
                                    st.stop()
                                else:
                                    st.error(f"Error during model inference: {e}")
                                    st.stop()
                            
                            # Display the result
                            st.subheader("Analysis Result")
                            
                            if predicted.item() == 1:  # REAL
                                st.markdown(f"<div class='result-real'>REAL VIDEO (Confidence: {confidence:.2f}%)</div>", unsafe_allow_html=True)
                            else:  # FAKE
                                st.markdown(f"<div class='result-fake'>FAKE VIDEO (Confidence: {confidence:.2f}%)</div>", unsafe_allow_html=True)
                            
                            # Confidence meter
                            st.markdown("<div class='confidence-meter'>Confidence Meter:</div>", unsafe_allow_html=True)
                            st.progress(confidence / 100)
                    else:
                        st.error("No faces detected in the video. Please try another file.")

        # Clean up the temporary file
        try:
            os.unlink(video_path)
        except:
            pass

if __name__ == "__main__":
    main()