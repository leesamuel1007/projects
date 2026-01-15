import cv2
import numpy as np
import os
# Load YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Load the video
video_name = '0.avi'
folder_path = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(folder_path, video_name)
print(video_path)
cap = cv2.VideoCapture(video_path)

background_subtractor = cv2.createBackgroundSubtractorMOG2()


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
    print(sample_frames)
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
    
    return subtracted_frames
def overlay_frames(frames, background_frame,transparency=0.1):
    # Initialize an empty canvas for the overlay
    overlay = np.zeros_like(frames[0], dtype=np.float32)
    
    # Overlay all frames with the specified transparency

    # first frame is the background
    overlay += background_frame.astype(np.float32)
    # for frame in frames:
    #     overlay += frame.astype(np.float32) * transparency / len(frames)
    

    for frame in frames:
        # Blend each frame with the overlay using the transparency value
        overlay = cv2.addWeighted(overlay, 1.0, frame.astype(np.float32), transparency, 0)

        
    # Loop through each frame and overlay it, respecting the background's brightness
    # for frame in frames:
    #     # Apply transparency to the frame
    #     frame_float = frame.astype(np.float32) * transparency
        
    #     # Ensure the pixel values do not exceed the background brightness
    #     overlay = np.minimum(overlay, frame_float + overlay)
    
    # Normalize and convert to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay

def tint_frame_with_rainbow(frame, progress):
    """
    Tint the frame with a rainbow color based on the progress from 0 to 1.
    This modifies the original frame color subtly, preserving details.
    """
    # Convert the frame to float32 for blending
    frame_float = frame.astype(np.float32) / 255.0
    
    # Generate a single-colored image based on progress, from red (early) to blue (late)
    rainbow_color = np.array([progress, 0, 1 - progress])  # Interpolate between red and blue
    
    # Create a tinted version of the frame by blending the original frame with the rainbow color
    tinted_frame = frame_float * (1 - 0.3) + rainbow_color * 0.3  # 0.3 is the strength of the tint
    
    # Convert back to uint8
    tinted_frame = (tinted_frame * 255).astype(np.uint8)
    return tinted_frame



def overlay_rainbow_frames(background, frames, transparency=0.5, threshold=30):
    # Initialize the overlay canvas with the background
    overlay = background.astype(np.float32)
    
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
        
        # Apply rainbow coloring based on progress
        rainbow_frame = tint_frame_with_rainbow(significant_change, progress)
        # show the rainbow frame
        cv2.imshow("Rainbow Frame", rainbow_frame)
        
        # Blend the colored significant change into the overlay with the specified transparency
        rainbow_frame_float = rainbow_frame.astype(np.float32) * transparency
        overlay = cv2.add(overlay, rainbow_frame_float)
    
    # Normalize and convert to uint8
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay


# def overlay_significant_changes(background, frames, transparency=0.5, threshold=50):
#     # Initialize the overlay canvas with the background
#     overlay = background.astype(np.float32)

#     # overlay += background_frame.astype(np.float32)
    
#     for frame in frames:
#         # Calculate the difference between the current frame and the overlay
#         diff = cv2.absdiff(overlay.astype(np.uint8), frame)
#         diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
#         # Apply a threshold to detect significant changes
#         _, diff_mask = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)
        
#         # Extract the significant parts from the current frame
#         significant_change = cv2.bitwise_and(frame, frame, mask=diff_mask)
        
#         # Blend the significant change into the overlay with the specified transparency
#         significant_change_float = significant_change.astype(np.float32) * transparency
#         overlay = cv2.add(overlay, significant_change_float)
    
#     # Normalize and convert to uint8
#     overlay = np.clip(overlay, 0, 255).astype(np.uint8)
#     return overlay

# Main function to process video
def process_video(video_path, t_a, t_b, num_samples=10, transparency=0.5):
    # Step 1: Uniformly sample frames between t_a and t_b
    frames = uniformly_sample_frames(video_path, t_a, t_b, num_samples)

    # we have a frame with the background
    background_frame = frames[0]
    # Step 2: Perform background subtraction based on sampled frames
    subtracted_frames = background_subtraction(frames)
    
    # Step 3: Overlay the subtracted frames with transparency
    overlay = overlay_rainbow_frames(background_frame,subtracted_frames,transparency)
    
    # Show and save the result
    cv2.imshow("Trajectory Overlay", overlay)
    cv2.imwrite("robot_trajectory_overlay.png", overlay)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Process the video
process_video(video_path, t_a=0, t_b=20, num_samples=20, transparency=0.5)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Apply the background subtractor
#     fg_mask = background_subtractor.apply(frame)

#     # Use the mask to extract the robot
#     robot_highlighted = cv2.bitwise_and(frame, frame, mask=fg_mask)

#     # Show the result
#     cv2.imshow('Robot Highlighted', robot_highlighted)

#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()

