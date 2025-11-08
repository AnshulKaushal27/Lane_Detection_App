# ========================================
# LANE DETECTION INFERENCE - SIMPLIFIED VERSION
# Upload video → Apply green overlay → Display & Save
# ========================================

import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Lane Detection - Video Processing with Green Overlay")
print("=" * 70)

# ========================================
# 1. CONFIGURATION
# ========================================
IMG_HEIGHT = 256
IMG_WIDTH = 512
MODEL_PATH = 'best_lane_model_stage2.h5'
CONFIDENCE_THRESHOLD = 0.5

# Video output quality settings
MAX_VIDEO_WIDTH = 1280   # HD width
MAX_VIDEO_HEIGHT = 720   # HD height

print("\n[CONFIG] Loading model...")
print(f"✓ Model: {MODEL_PATH}")
print(f"✓ Input size: {IMG_HEIGHT}x{IMG_WIDTH}")
print(f"✓ Threshold: {CONFIDENCE_THRESHOLD}")
print(f"✓ Max video output: {MAX_VIDEO_WIDTH}x{MAX_VIDEO_HEIGHT} (HD)")

# Load model
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found: {MODEL_PATH}")
    print(f"   Please ensure the model file is in the current directory")
    exit()
else:
    model = keras.models.load_model(MODEL_PATH)
    print(f"✓ Model loaded successfully")

# ========================================
# 2. UTILITY FUNCTIONS
# ========================================

def resize_frame(frame, max_width=1280, max_height=720):
    """
    Resize frame to fit within max dimensions while maintaining aspect ratio
    """
    height, width = frame.shape[:2]
    
    # Calculate aspect ratio
    aspect_ratio = width / height
    
    # Calculate new dimensions
    if width > max_width or height > max_height:
        if aspect_ratio > (max_width / max_height):
            # Width is the limiting factor
            new_width = max_width
            new_height = int(max_width / aspect_ratio)
        else:
            # Height is the limiting factor
            new_height = max_height
            new_width = int(max_height * aspect_ratio)
        
        frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    
    return frame


def predict_lane_mask(image_array, model, img_height=256, img_width=512):
    """Predict lane mask from image array"""
    original_image = image_array.copy()
    original_height, original_width = image_array.shape[:2]
    
    # Preprocess
    image_resized = cv2.resize(image_array, (img_width, img_height))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    # Predict
    prediction = model.predict(image_batch, verbose=0)
    predicted_mask = prediction[0].squeeze()
    
    # Threshold
    predicted_mask_binary = (predicted_mask > CONFIDENCE_THRESHOLD).astype(np.uint8)
    
    # Resize back to original size
    predicted_mask_resized = cv2.resize(
        predicted_mask_binary.astype(np.float32),
        (original_width, original_height),
        interpolation=cv2.INTER_NEAREST
    )
    
    return original_image, predicted_mask_resized


def overlay_lanes_green(image, mask, alpha=0.7):
    """Apply green colored transparent overlay on detected lanes - DARKER VERSION"""
    overlay = image.copy().astype(np.float32)
    
    # Create bright green overlay (BGR format: Green = (0, 255, 0))
    green_color = (0, 255, 0)
    mask_colored = np.zeros_like(image, dtype=np.float32)
    for i in range(3):
        mask_colored[:, :, i] = mask * green_color[i]
    
    # Blend with higher alpha for darker/more visible overlay
    result = image.astype(np.float32)
    result[mask > 0] = (1 - alpha) * result[mask > 0] + alpha * mask_colored[mask > 0]
    
    return result.astype(np.uint8)


# ========================================
# 3. VIDEO PROCESSING WITH DISPLAY
# ========================================

def process_and_display_video(video_path):
    """
    Process video with green lane overlay, display with controls, and save
    """
    print(f"\n🎥 Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"❌ File not found: {video_path}")
        return
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Error opening video")
        return
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"✓ Original video: {width}x{height}, {total_frames} frames, {fps:.1f} FPS")
    
    # Calculate new dimensions (maintain aspect ratio)
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
    
    # Make dimensions even (required for most codecs)
    new_width = new_width if new_width % 2 == 0 else new_width - 1
    new_height = new_height if new_height % 2 == 0 else new_height - 1
    
    print(f"✓ Output will be resized to: {new_width}x{new_height}")
    
    # Setup video writer for saving
    output_path = 'lane_detection_output.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    if not out.isOpened():
        print(f"⚠️  Warning: mp4v codec failed, trying MJPG...")
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    frame_count = 0
    paused = False
    
    print("\n[PLAYBACK CONTROLS]")
    print("━" * 70)
    print("SPACE : Play / Pause")
    print("→     : Next frame (when paused)")
    print("←     : Previous frame (when paused)")
    print("s     : Save current frame as image")
    print("q     : Quit and save video")
    print("━" * 70)
    print("\n▶️  Playing video with green lane overlay...\n")
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("\n✓ Video finished")
                break
            
            frame_count += 1
        
        # Resize frame
        frame_resized = resize_frame(frame, MAX_VIDEO_WIDTH, MAX_VIDEO_HEIGHT)
        
        # Ensure exact output dimensions
        if frame_resized.shape[1] != new_width or frame_resized.shape[0] != new_height:
            frame_resized = cv2.resize(frame_resized, (new_width, new_height))
        
        # Convert BGR to RGB for model
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Predict lane mask
        original_image, predicted_mask = predict_lane_mask(frame_rgb, model)
        
        # Apply green overlay with higher alpha (darker/more visible)
        result_rgb = overlay_lanes_green(original_image, predicted_mask, alpha=0.7)
        
        # Convert back to BGR for display and saving
        result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
        
        # Add info overlay on video
        info_bg_color = (0, 0, 0)
        info_text_color = (255, 255, 255)
        
        # Semi-transparent background for info text
        overlay_info = result_bgr.copy()
        cv2.rectangle(overlay_info, (5, 5), (400, 180), info_bg_color, -1)
        result_bgr = cv2.addWeighted(result_bgr, 0.7, overlay_info, 0.3, 0)
        
        # Add text information
        cv2.putText(result_bgr, f'Frame: {frame_count}/{total_frames}', 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_text_color, 2)
        cv2.putText(result_bgr, f'Size: {new_width}x{new_height}', 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_text_color, 2)
        cv2.putText(result_bgr, f'FPS: {fps:.1f}', 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, info_text_color, 2)
        
        # Status indicator - FIXED TEXT RENDERING
        if paused:
            status_text = 'PAUSED'
            status_color = (0, 0, 255)  # Red when paused
        else:
            status_text = 'PLAYING'
            status_color = (0, 255, 0)  # Green when playing
        
        cv2.putText(result_bgr, status_text, 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Progress bar
        progress = int((frame_count / total_frames) * 380)
        cv2.rectangle(result_bgr, (10, 150), (390, 170), (50, 50, 50), -1)
        cv2.rectangle(result_bgr, (10, 150), (10 + progress, 170), (0, 255, 0), -1)
        
        # Display frame
        cv2.imshow('Lane Detection - Green Overlay', result_bgr)
        
        # Write to output video (only if not paused)
        if not paused:
            out.write(result_bgr)
        
        # Control playback speed
        delay = int((1000 / fps)) if not paused else 0
        key = cv2.waitKey(delay) & 0xFF
        
        # Handle key presses
        if key == ord('q'):
            print("\n⏹  Stopping playback...")
            break
        elif key == ord(' '):
            paused = not paused
            status = "PAUSED" if paused else "PLAYING"
            print(f"⏸  {status} at frame {frame_count}/{total_frames}")
        elif key == ord('s'):
            frame_filename = f'frame_{frame_count:04d}.jpg'
            cv2.imwrite(frame_filename, result_bgr)
            print(f"📸 Frame saved: {frame_filename}")
        elif key == 83 and paused:  # Right arrow key
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                print(f"➡️  Next frame: {frame_count}/{total_frames}")
            else:
                print("⚠️  End of video reached")
        elif key == 81 and paused:  # Left arrow key
            if frame_count > 1:
                frame_count = max(1, frame_count - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
                ret, frame = cap.read()
                print(f"⬅️  Previous frame: {frame_count}/{total_frames}")
            else:
                print("⚠️  Already at first frame")
    
    # Cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 70)
    print("✅ VIDEO PROCESSING COMPLETE")
    print("=" * 70)
    print(f"✓ Output saved as: {output_path}")
    print(f"✓ Video size: {new_width}x{new_height}")
    print(f"✓ Total frames processed: {frame_count}")
    print(f"✓ Frame rate: {fps:.1f} FPS")
    
    # Get file size
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"✓ File size: {file_size_mb:.2f} MB")
    
    print("=" * 70)


# ========================================
# 4. MAIN FUNCTION
# ========================================

def main():
    """Main function - Ask for video file and process"""
    print("\n" + "=" * 70)
    print("LANE DETECTION WITH GREEN OVERLAY")
    print("=" * 70)
    print("\n📁 Please provide the path to your .mp4 video file")
    print("   (You can drag and drop the file, or type the full path)")
    print()
    
    while True:
        video_path = input("Enter video file path (.mp4): ").strip()
        
        # Remove quotes if user dragged and dropped
        video_path = video_path.strip('"').strip("'")
        
        # Check if file exists
        if not os.path.exists(video_path):
            print(f"\n❌ File not found: {video_path}")
            retry = input("Try again? (y/n): ").strip().lower()
            if retry != 'y':
                print("👋 Exiting...")
                return
            continue
        
        # Check if it's a video file
        if not video_path.lower().endswith('.mp4'):
            print(f"\n⚠️  Warning: File doesn't have .mp4 extension")
            proceed = input("Proceed anyway? (y/n): ").strip().lower()
            if proceed != 'y':
                continue
        
        # Process the video
        print("\n🚀 Starting video processing...")
        process_and_display_video(video_path)
        
        # Ask if user wants to process another video
        print("\n")
        another = input("Process another video? (y/n): ").strip().lower()
        if another != 'y':
            print("\n👋 Thank you for using Lane Detection!")
            break


# ========================================
# 5. RUN MAIN
# ========================================

if __name__ == "__main__":
    main()
