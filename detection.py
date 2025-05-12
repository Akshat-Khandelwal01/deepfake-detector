import torch
import os
torch.classes.__path__ = [os.path.join(torch.__path__[0], 'torch', 'classes')]

import streamlit as st
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

# Load TorchScript model once using st.cache_resource
@st.cache_resource
def load_model():
    model = torch.jit.load("model_90_acc_60_frames_final_data.pt", map_location=torch.device('cpu'))
    model.eval()
    return model

model = load_model()

# Define preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

def extract_frames(video_path, count=20):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // count, 1)
    for i in range(count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if ret:
            frames.append(transform(frame))
    cap.release()

    if len(frames) < count:
        # Pad with last frame if video too short
        while len(frames) < count:
            frames.append(frames[-1])
    return torch.stack(frames)[:count]  # shape: (count, C, H, W)

st.title("Deepfake Video Detector")

video_file = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])

if video_file:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    st.video("temp_video.mp4")

    with st.spinner('Processing video...'):
        frames = extract_frames("temp_video.mp4").unsqueeze(0)  # (1, count, C, H, W)
        with torch.no_grad():
            output = model(frames)
            predicted_class = torch.argmax(output, dim=1).item()
            confidence = torch.max(output).item()
            label = "Fake" if predicted_class == 1 else "Real"
            st.markdown(f"### Prediction: **{label}** with confidence {confidence:.2f}")
