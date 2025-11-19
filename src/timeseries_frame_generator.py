import os
import json
import numpy as np
from PIL import Image
from .metrics_handler import merge_context_data
import glob

def generate_timeseries_frames(context, merged_df, output_base="data", band_height=64, width=1024, height=576, 
                               use_history_band=True, predictor=None):
    """
    Generate image frames from merged timeseries data with optional pre-generated history bands.
    
    Args:
        context (str): Context name.
        merged_df (DataFrame): Merged timeseries data.
        output_base (str): Base output directory.
        band_height (int): History band height for SVD-XT.
        width (int): Target frame width (default: 1024 for SVD-XT).
        height (int): Target frame height (default: 576 for SVD-XT).
        use_history_band (bool): Whether to pre-generate history bands.
        predictor: VideoImagePredictor instance for history band generation.
    
    Returns:
        str: Path to the frames directory.
    """
    metadata_path = os.path.join(output_base, context, "metadata.json")
    frames_dir = os.path.join(output_base, context, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metrics = list(metadata["metrics"].keys())
    N = len(metrics)
    target_width = width
    
    # Content height is always 512px - history band is ADDED on top later
    if use_history_band:
        content_height = 512  # Fixed content area
        # Total frame height after adding history band will be: 512 + band_height
    else:
        content_height = height  # Use specified height when no history band
    
    # Generate base frames (without history bands)
    for idx, row in merged_df.iterrows():
        # Normalize
        normalized = []
        for metric in metrics:
            val = row[metric]
            max_v = metadata["metrics"][metric]["max"]
            min_v = metadata["metrics"][metric]["min"]
            norm = (val - min_v) / (max_v - min_v) if max_v != min_v else 0.5
            normalized.append(norm)
        
        # Create image with horizontal bands
        img = np.zeros((content_height, target_width), dtype=np.uint8)
        band_h = content_height // N
        remainder = content_height % N
        current_y = 0
        for i in range(N):
            h = band_h + (1 if i < remainder else 0)
            gray_val = int(normalized[i] * 255)
            img[current_y:current_y+h, :] = gray_val
            current_y += h
        
        pil_img = Image.fromarray(img, mode='L')
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.png")
        pil_img.save(frame_path)
    
    # Pre-generate history bands offline if enabled
    if use_history_band and predictor is not None:
        print("Pre-generating history bands offline (this saves time during training)...")
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        
        # Store original frames (without history bands) for thumbnail generation
        original_frames = []
        for frame_path in frame_files:
            img = Image.open(frame_path).convert('RGB')
            original_frames.append(img)
        
        for i, frame_path in enumerate(frame_files):
            # Get ORIGINAL frames for history (not frames with history bands)
            history_indices = range(max(0, i-6), i) if i > 0 else []
            
            if history_indices:
                # Use original frames for history thumbnails
                history_frames = [original_frames[idx] for idx in history_indices]
                
                # Add history band to current frame
                img_with_history = predictor._encode_history_on_frame(
                    original_frames[i], history_frames, band_height
                )
                
                # Save frame with history band (overwrite)
                img_with_history.save(frame_path)
            else:
                # For first few frames with no history, add blank history band
                w, h = original_frames[i].size
                blank_band = Image.new('RGB', (w, band_height), color=(0, 0, 0))
                out = Image.new('RGB', (w, h + band_height))
                out.paste(original_frames[i], (0, 0))
                out.paste(blank_band, (0, h))
                out.save(frame_path)
        
        print(f"Added history bands to {len(frame_files)} frames (saved {len(frame_files) * 3}+ seconds per epoch)")
    
    return frames_dir

def process_context(context, output_base="data", band_height=64, width=1024, height=576, 
                   use_history_band=True, predictor=None):
    """
    Process a context: merge data and generate frames with optional history bands.
    
    Args:
        context (str): Context name.
        output_base (str): Base output directory.
        band_height (int): History band height for SVD-XT.
        width (int): Target frame width (default: 1024 for SVD-XT).
        height (int): Target frame height (default: 576 for SVD-XT).
        use_history_band (bool): Whether to pre-generate history bands.
        predictor: VideoImagePredictor instance for history band generation.
    
    Returns:
        str: Success or error message.
    """
    merged_df, msg = merge_context_data(context, output_base)
    if merged_df is None:
        return msg
    return generate_timeseries_frames(context, merged_df, output_base, band_height, width, height, 
                                      use_history_band, predictor)

if __name__ == "__main__":
    # Example usage
    # process_context("experiment1")
    pass