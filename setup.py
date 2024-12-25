from setuptools import setup, find_packages

setup(
    name="mathvision",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "streamlit",
        "opencv-python-headless",
        "streamlit-webrtc",
        "numpy",
        "Pillow",
        "google-generativeai",
        "python-dotenv",
        "mediapipe",
        "python-av"
    ],
)