import cv2
import numpy as np
import os

def uniformly_sample_frames(video_path, t_a, t_b, num_samples):
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    # Convert times to frame indices
    start_frame = int(t_a * fps)
    end_frame = int(t_b * fps)
    
    # Calculate frame step for uniform sampling
    sample_frames = np.linspace(start_frame, end_frame, num_samples, dtype=int)
    
    # Collect sampled frames
    frames = []
    for frame_idx in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    
    cap.release()
    return frames

def background_subtraction(frames):
    # Estimate the background using the median across the sampled frames
    background = np.median(np.array(frames), axis=0).astype(np.uint8)
    
    # Perform background subtraction on each frame
    subtracted_frames = []
    for frame in frames:
        fg_mask = cv2.absdiff(frame, background)
        fg_gray = cv2.cvtColor(fg_mask, cv2.COLOR_BGR2GRAY)
        _, fg_binary = cv2.threshold(fg_gray, 30, 255, cv2.THRESH_BINARY)  # Threshold the difference
        robot_highlighted = cv2.bitwise_and(frame, frame, mask=fg_binary)
        subtracted_frames.append(robot_highlighted)
    
    return background, subtracted_frames

def tint_frame_with_rainbow(frame, progress):
    """
    Tint the frame with a rainbow color based on the progress from 0 to 1.
    This modifies the original frame color subtly, preserving details.
    """
    # Convert the frame to float32 for blending
    frame_float = frame.astype(np.float32) / 255.0
    
    # Generate a single-colored image based on progress, from red (early) to blue (late)
    rainbow_color = np.array([progress, 0, 1 - progress])  # Interpolate between red and blue
    
    # Blend the original frame with the rainbow color only where the frame is non-black (robot parts)
    tinted_frame = frame_float * (1 - 0.3) + rainbow_color * 0.3  # 0.3 is the strength of the tint
    tinted_frame[frame == 0] = 0  # Ensure the background stays black
    
    # Convert back to uint8
    tinted_frame = (tinted_frame * 255).astype(np.uint8)
    return tinted_frame

def overlay_rainbow_tinted_frames(background, frames, transparency=0.5, threshold=30):
    # Initialize the overlay canvas with black background
    overlay = np.zeros_like(background).astype(np.float32)
    
    num_frames = len(frames)
    for idx, frame in enumerate(frames):
        # Calculate the difference between the current frame and the overlay
        diff = cv2.absdiff(overlay.astype(np.uint8), frame)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply a threshold to detect significant changes
        _, diff_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        
        # Extract the significant parts from the current frame
        significant_change = cv2.bitwise_and(frame, frame, mask=diff_mask)
        
        # Calculate the progress (from 0 to 1) for colorizing the frame
        progress = idx / float(num_frames - 1)
        
        # Tint the significant changes with a rainbow tint based on progress
        tinted_frame = tint_frame_with_rainbow(significant_change, progress)
        
        # Blend the tinted significant change into the overlay with the specified transparency
        tinted_frame_float = tinted_frame.astype(np.float32) * transparency
        overlay = cv2.add(overlay, tinted_frame_float)
    
    # Normalize and convert to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

# Main function to process video
def process_video(video_path, t_a, t_b, num_samples=10, transparency=0.5, threshold=30):
    # Step 1: Uniformly sample frames between t_a and t_b
    frames = uniformly_sample_frames(video_path, t_a, t_b, num_samples)
    
    # Step 2: Perform background subtraction based on sampled frames
    background, subtracted_frames = background_subtraction(frames)
    
    # Step 3: Overlay rainbow-tinted frames
    overlay = overlay_rainbow_tinted_frames(background, subtracted_frames, transparency, threshold)
    
    # Show and save the result
    cv2.imshow("Trajectory Overlay with Rainbow Tint", overlay)
    cv2.imwrite("robot_trajectory_rainbow_tint_overlay.png", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
video_name = '0.avi'
folder_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(folder_path, video_name)
print(video_path)
cap = cv2.VideoCapture(video_path)
t_a = 5
t_b = 30
num_samples = 9

process_video(video_path, t_a, t_b, num_samples, transparency=0.6, threshold=30)
