from setuptools import setup, find_packages

setup(
    name="deepfake_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.22.0",
        "torch>=1.10.0",
        "torchvision>=0.11.0",
        "opencv-python>=4.5.0",
        "face_recognition>=1.3.0",
        "numpy>=1.19.0",
        "psutil>=5.9.0",
        "Pillow>=9.0.0",
    ],
    python_requires=">=3.7",
)