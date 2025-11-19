import cv2
import os

def extract_frames(video_path, output_dir="data", frame_step=1):
    """
    Extract frames from a video and save them in a subfolder named after the video.
    
    Args:
        video_path (str): Path to the video file.
        output_dir (str): Base output directory (default: "data").
        frame_step (int): Extract every nth frame (default: 1, every frame).
    """
    # Get video name without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Create output folder
    output_folder = os.path.join(output_dir, video_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame if it's the nth frame
        if frame_count % frame_step == 0:
            frame_filename = f"frame_{saved_count:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)
            saved_count += 1
        
        frame_count += 1
    
    cap.release()
    print(f"Extracted {saved_count} frames from {video_path} to {output_folder}")

if __name__ == "__main__":
    # Example usage
    video_path = "videos/biker.mp4"  # Replace with actual path
    extract_frames(video_path, frame_step=1)  # Extract every frame