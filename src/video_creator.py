import cv2
import os
import glob
from PIL import Image
import numpy as np
import subprocess

def create_video_from_frames(frames_folder, output_video_path, fps=30, include_predicted=True):
    """
    Create a video from frames in a folder, combining original and predicted frames.
    
    Args:
        frames_folder (str): Path to the folder containing frames.
        output_video_path (str): Path to save the output video.
        fps (int): Frames per second for the output video.
        include_predicted (bool): Whether to include predicted frames.
    """
    # Get all original frames (not starting with "Predicted_")
    original_pattern = os.path.join(frames_folder, "frame_*.jpg")
    original_frames = sorted(glob.glob(original_pattern))
    
    # Get all predicted frames if requested
    predicted_frames = []
    if include_predicted:
        predicted_pattern = os.path.join(frames_folder, "Predicted_frame_*.jpg")
        predicted_frames = sorted(glob.glob(predicted_pattern))
    
    # Combine all frames in order
    all_frames = original_frames + predicted_frames
    
    if not all_frames:
        print(f"No frames found in {frames_folder}")
        return
    
    # Create a temporary file list for ffmpeg
    temp_list_file = os.path.join(frames_folder, "temp_frame_list.txt")
    with open(temp_list_file, 'w') as f:
        for frame_path in all_frames:
            # ffmpeg concat requires format: file 'path'
            # Convert backslashes to forward slashes for ffmpeg
            frame_path_ffmpeg = frame_path.replace('\\', '/')
            f.write(f"file '{os.path.abspath(frame_path_ffmpeg)}'\n")
    
    # Try using ffmpeg if available
    try:
        # Use ffmpeg to create video with H.264 codec
        ffmpeg_cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-f', 'concat',
            '-safe', '0',
            '-i', temp_list_file,
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-preset', 'medium',
            '-crf', '23',
            '-r', str(fps),
            output_video_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Video created successfully with ffmpeg: {output_video_path}")
            print(f"Total frames: {len(all_frames)} (Original: {len(original_frames)}, Predicted: {len(predicted_frames)})")
            os.remove(temp_list_file)
            return
        else:
            print("ffmpeg failed, falling back to OpenCV...")
    except FileNotFoundError:
        print("ffmpeg not found, using OpenCV method...")
    except Exception as e:
        print(f"ffmpeg error: {e}, using OpenCV method...")
    
    # Clean up temp file
    if os.path.exists(temp_list_file):
        os.remove(temp_list_file)
    
    # Fallback to OpenCV method
    # Read the first frame to get dimensions
    first_frame = cv2.imread(all_frames[0])
    if first_frame is None:
        print(f"Could not read first frame: {all_frames[0]}")
        return
    
    height, width, layers = first_frame.shape
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec for better compatibility
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    if not video_writer.isOpened():
        print("Error: VideoWriter could not be opened. Trying alternative codec...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Write each frame to the video
    for frame_path in all_frames:
        frame = cv2.imread(frame_path)
        if frame is not None:
            # Resize if dimensions don't match
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
        else:
            print(f"Warning: Could not read frame {frame_path}")
    
    video_writer.release()
    print(f"Video created successfully: {output_video_path}")
    print(f"Total frames: {len(all_frames)} (Original: {len(original_frames)}, Predicted: {len(predicted_frames)})")

def create_comparison_videos(frames_folder, fps=24):
    """
    Create separate videos for baseline and conditioned Wan generations.
    """
    videos_dir = "videos"
    os.makedirs(videos_dir, exist_ok=True)
    
    # Get original frames
    original_pattern = os.path.join(frames_folder, "frame_*.jpg")
    original_frames = sorted(glob.glob(original_pattern))
    
    # Get baseline frames (standard Wan generation)
    baseline_pattern = os.path.join(frames_folder, "Predicted_frame_baseline_*.jpg")
    baseline_frames = sorted(glob.glob(baseline_pattern))
    
    # Get conditioned frames (our custom logic)
    conditioned_pattern = os.path.join(frames_folder, "Predicted_frame_*.jpg")
    conditioned_frames = sorted(glob.glob(conditioned_pattern))
    
    print(f"Found {len(original_frames)} original frames")
    print(f"Found {len(baseline_frames)} baseline frames")
    print(f"Found {len(conditioned_frames)} conditioned frames")
    
    # Create baseline video (original + baseline)
    if baseline_frames:
        baseline_video_path = os.path.join(videos_dir, "baseline_comparison.mp4")
        all_baseline_frames = original_frames + baseline_frames
        create_video_from_frame_list(all_baseline_frames, baseline_video_path, fps)
        print(f"Baseline video created: {baseline_video_path}")
    
    # Create conditioned video (original + conditioned)
    if conditioned_frames:
        conditioned_video_path = os.path.join(videos_dir, "conditioned_comparison.mp4")
        all_conditioned_frames = original_frames + conditioned_frames
        create_video_from_frame_list(all_conditioned_frames, conditioned_video_path, fps)
        print(f"Conditioned video created: {conditioned_video_path}")

def create_video_from_frame_list(frame_list, output_path, fps=24):
    """
    Create video from a list of frame paths.
    """
    if not frame_list:
        print("No frames provided")
        return
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_list[0])
    if first_frame is None:
        print(f"Could not read first frame: {frame_list[0]}")
        return
    
    height, width, layers = first_frame.shape
    
    # Try ffmpeg first
    temp_list_file = os.path.join(os.path.dirname(frame_list[0]), "temp_frame_list.txt")
    try:
        with open(temp_list_file, 'w') as f:
            for frame_path in frame_list:
                frame_path_ffmpeg = frame_path.replace('\\', '/')
                f.write(f"file '{os.path.abspath(frame_path_ffmpeg)}'\n")
        
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', temp_list_file,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'medium', '-crf', '23', '-r', str(fps),
            output_path
        ]
        
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
        if result.returncode == 0:
            os.remove(temp_list_file)
            return
    except Exception as e:
        print(f"ffmpeg failed: {e}")
    finally:
        if os.path.exists(temp_list_file):
            os.remove(temp_list_file)
    
    # Fallback to OpenCV
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        if frame is not None:
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height))
            video_writer.write(frame)
    
    video_writer.release()

if __name__ == "__main__":
    # Create comparison videos for baseline vs conditioned Wan generations
    frames_folder = "data/rabbit"
    create_comparison_videos(frames_folder, fps=24)