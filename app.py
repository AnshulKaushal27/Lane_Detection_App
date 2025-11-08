#!/usr/bin/env python3
import subprocess
import sys
import os

# Install required packages before importing
packages = [
    'opencv-python',
    'tensorflow-cpu',
    'numpy',
    'pillow',
    'streamlit'
]

print("Checking and installing dependencies...")
for package in packages:
    try:
        __import__(package.replace('-', '_'))
        print(f"✓ {package} already installed")
    except ImportError:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "-q"])

# Now import after ensuring packages exist
import cv2
import streamlit as st
import numpy as np
from tensorflow import keras
import tempfile

print("✓ All imports successful")

# ========================================
# PAGE CONFIG
# ========================================
st.set_page_config(
    page_title="Lane Detection",
    page_icon="🛣️",
    layout="wide"
)

# ========================================
# CONFIG
# ========================================
IMG_HEIGHT = 256
IMG_WIDTH = 512
CONFIDENCE_THRESHOLD = 0.5
MAX_VIDEO_WIDTH = 1280
MAX_VIDEO_HEIGHT = 720

# ========================================
# LOAD MODEL
# ========================================
@st.cache_resource
def load_model():
    model_path = 'best_lane_model_stage2.h5'
    if not os.path.exists(model_path):
        st.error("❌ Model file not found: best_lane_model_stage2.h5")
        st.stop()
    
    with st.spinner("Loading model..."):
        model = keras.models.load_model(model_path)
    return model

# ========================================
# FUNCTIONS
# ========================================

def resize_frame(frame, max_width=1280, max_height=720):
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    if width > max_width or height > max_height:
        if aspect_ratio > (max_width / max_height):
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return frame


def predict_lane_mask(image_array, model):
    original_image = image_array.copy()
    original_height, original_width = image_array.shape[:2]
    
    image_resized = cv2.resize(image_array, (IMG_WIDTH, IMG_HEIGHT))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    prediction = model.predict(image_batch, verbose=0)
    predicted_mask = prediction[0].squeeze()
    predicted_mask_binary = (predicted_mask > CONFIDENCE_THRESHOLD).astype(np.uint8)
    
    predicted_mask_resized = cv2.resize(
        predicted_mask_binary.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return original_image, predicted_mask_resized


def overlay_lanes_green(image, mask, alpha=0.7):
    overlay = image.copy().astype(np.float32)
    green_color = (0, 255, 0)
    mask_colored = np.zeros_like(image, dtype=np.float32)
    
    for i in range(3):
        mask_colored[:, :, i] = mask * green_color[i]
    
    result = image.astype(np.float32)
    result[mask > 0] = (1 - alpha) * result[mask > 0] + alpha * mask_colored[mask > 0]
    
    return result.astype(np.uint8)


def process_video(video_path, model, progress_bar, status_text, alpha):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if fps <= 0:
        fps = 30
    
    # Calculate dimensions
    aspect_ratio = width / height
    if width > MAX_VIDEO_WIDTH or height > MAX_VIDEO_HEIGHT:
        if aspect_ratio > (MAX_VIDEO_WIDTH / MAX_VIDEO_HEIGHT):
            new_width = MAX_VIDEO_WIDTH
            new_height = int(MAX_VIDEO_WIDTH / aspect_ratio)
        else:
            new_height = MAX_VIDEO_HEIGHT
            new_width = int(MAX_VIDEO_HEIGHT * aspect_ratio)
    else:
        new_width = width
        new_height = height
    
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    # Output file
    output_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    output_path = output_temp.name
    output_temp.close()
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        frame_resized = resize_frame(frame, MAX_VIDEO_WIDTH, MAX_VIDEO_HEIGHT)
        
        if frame_resized.shape[1] != new_width or frame_resized.shape[0] != new_height:
            frame_resized = cv2.resize(frame_resized, (new_width, new_height))
        
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        original_image, predicted_mask = predict_lane_mask(frame_rgb, model)
        result_rgb = overlay_lanes_green(original_image, predicted_mask, alpha=alpha)
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        out.write(result_bgr)
        
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 0.99))
        status_text.text(f"Processing: {frame_count}/{total_frames} frames")
    
    cap.release()
    out.release()
    
    return output_path, new_width, new_height, fps, frame_count


# ========================================
# UI
# ========================================

st.title("🛣️ Lane Detection with AI")
st.markdown("---")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload Video")
    uploaded_file = st.file_uploader("Choose MP4 video", type=["mp4"])

with col2:
    st.subheader("⚙️ Settings")
    alpha = st.slider("Overlay Opacity", 0.3, 1.0, 0.7, 0.1)

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name
    
    cap = cv2.VideoCapture(temp_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    cap.release()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Resolution", f"{width}x{height}")
    col2.metric("Frames", total_frames)
    col3.metric("Duration", f"{total_frames/fps:.1f}s")
    col4.metric("FPS", f"{fps:.1f}")
    
    if st.button("🚀 Process Video", use_container_width=True, type="primary"):
        model = load_model()
        
        st.markdown("---")
        st.subheader("⏳ Processing...")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        result = process_video(temp_path, model, progress_bar, status_text, alpha)
        
        if result:
            output_path, new_width, new_height, output_fps, processed_frames = result
            
            progress_bar.empty()
            status_text.empty()
            
            st.success("✅ Video processing completed!")
            st.markdown("---")
            
            col1, col2 = st.columns(2)
            col1.metric("Output Resolution", f"{new_width}x{new_height}")
            col2.metric("Frames Processed", processed_frames)
            
            col1, col2 = st.columns(2)
            col1.metric("Output FPS", f"{output_fps:.1f}")
            file_size = os.path.getsize(output_path) / (1024 * 1024)
            col2.metric("File Size", f"{file_size:.2f} MB")
            
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download Processed Video",
                    f.read(),
                    "lane_detection_output.mp4",
                    "video/mp4",
                    use_container_width=True
                )
            
            os.unlink(output_path)
    
    os.unlink(temp_path)
else:
    st.info("👆 Upload a video to get started")
