import os
import json
import pandas as pd
import numpy as np
from PIL import Image

def reconstruct_timeseries(context, output_base="data", output_csv="reconstructed_timeseries.csv"):
    """
    Reconstruct timeseries data from predicted frames.
    
    Args:
        context (str): Context name.
        output_base (str): Base output directory.
        output_csv (str): Output CSV filename.
    
    Returns:
        str: Success or error message.
    """
    metadata_path = os.path.join(output_base, context, "metadata.json")
    frames_dir = os.path.join(output_base, context, "timeseries_frames")
    
    if not os.path.exists(metadata_path):
        return "Error: Context not found"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if "times" not in metadata or "metrics" not in metadata:
        return "Error: Metadata incomplete for reconstruction"
    
    times = [pd.to_datetime(t) for t in metadata["times"]]
    metrics = list(metadata["metrics"].keys())
    N = len(metrics)
    band_height = metadata.get("band_height", 64)
    target_height = 576 - band_height
    target_width = 1024
    
    if not os.path.exists(frames_dir):
        return "Error: Frames directory not found"
    
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    if len(frame_files) != len(times):
        return f"Error: Number of frames ({len(frame_files)}) does not match times ({len(times)})"
    
    data = {m: [] for m in metrics}
    data['time'] = times
    
    for i, frame_file in enumerate(frame_files):
        img_path = os.path.join(frames_dir, frame_file)
        img = Image.open(img_path).convert('L')  # Grayscale
        img_array = np.array(img)
        
        if img_array.shape != (target_height, target_width):
            return f"Error: Frame {frame_file} has incorrect dimensions"
        
        band_h = target_height // N
        remainder = target_height % N
        current_y = 0
        
        for j, metric in enumerate(metrics):
            h = band_h + (1 if j < remainder else 0)
            band = img_array[current_y:current_y+h, :]
            # Average intensity of the band
            avg_intensity = np.mean(band) / 255.0  # Normalized
            # Denormalize
            max_v = metadata["metrics"][metric]["max"]
            min_v = metadata["metrics"][metric]["min"]
            val = avg_intensity * (max_v - min_v) + min_v
            data[metric].append(val)
            current_y += h
    
    df = pd.DataFrame(data)
    output_path = os.path.join(output_base, context, output_csv)
    df.to_csv(output_path, index=False)
    return f"Reconstructed timeseries saved to {output_path}"

if __name__ == "__main__":
    # Example usage
    # reconstruct_timeseries("experiment1")
    pass