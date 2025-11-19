import os
import json
import shutil
from datetime import datetime
import pandas as pd

def upload_metric_csv(context, metric_name, csv_path, max_val, min_val, output_base="data", band_height=64):
    """
    Upload or update a metric CSV for a context.
    
    Args:
        context (str): Context name.
        metric_name (str): Name of the metric.
        csv_path (str): Path to the CSV file (columns: time, metric).
        max_val (float): Max value for normalization.
        min_val (float): Min value for normalization.
        output_base (str): Base output directory.
        band_height (int): History band height for SVD-XT.
    
    Returns:
        str: Success or error message.
    """
    # Paths
    context_dir = os.path.join(output_base, context)
    raw_uploads_dir = os.path.join(context_dir, "raw_uploads")
    metadata_path = os.path.join(context_dir, "metadata.json")
    os.makedirs(raw_uploads_dir, exist_ok=True)
    
    # Load and validate CSV
    try:
        df = pd.read_csv(csv_path)
        if 'time' not in df.columns or 'metric' not in df.columns:
            return "Error: CSV must have 'time' and 'metric' columns"
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        if df['time'].isna().any():
            return "Error: Invalid time format in CSV"
        df = df.sort_values('time').drop_duplicates('time')
        if df['metric'].dtype not in ['int64', 'float64']:
            return "Error: Metric column must be numeric"
        if df['metric'].isna().any():
            return "Error: Metric column contains missing values"
        if max_val <= min_val:
            return "Error: max_val must be greater than min_val"
        start_time = df['time'].min().isoformat()
        end_time = df['time'].max().isoformat()
    except Exception as e:
        return f"Error loading CSV: {e}"
    
    # Load metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {"metrics": {}, "last_updated": datetime.now().isoformat(), "band_height": band_height}
    
    # Check for overlaps
    if metric_name in metadata["metrics"]:
        existing_uploads = metadata["metrics"][metric_name]["uploads"]
        new_range = (start_time, end_time)
        for upload in existing_uploads:
            existing_range = tuple(upload["date_range"])
            if new_range == existing_range:
                # Update: will replace
                pass
            elif (new_range[0] < existing_range[1] and new_range[1] > existing_range[0]):
                return "Error: Date range overlaps with existing upload"
        # If no overlap, append
    else:
        metadata["metrics"][metric_name] = {"max": max_val, "min": min_val, "start_time": start_time, "end_time": end_time, "uploads": []}
    
    # Save raw CSV
    upload_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = os.path.join(raw_uploads_dir, f"{metric_name}_{upload_id}.csv")
    shutil.copy(csv_path, raw_path)
    
    # Update metadata
    metadata["metrics"][metric_name]["uploads"].append({"id": upload_id, "date_range": [start_time, end_time]})
    metadata["last_updated"] = datetime.now().isoformat()
    metadata["band_height"] = band_height
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return "Upload successful"

def delete_metric(context, metric_name, output_base="data"):
    """
    Delete a metric from a context.
    
    Args:
        context (str): Context name.
        metric_name (str): Name of the metric.
        output_base (str): Base output directory.
    
    Returns:
        str: Success or error message.
    """
    metadata_path = os.path.join(output_base, context, "metadata.json")
    raw_uploads_dir = os.path.join(output_base, context, "raw_uploads")
    
    if not os.path.exists(metadata_path):
        return "Error: Context not found"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    if metric_name not in metadata["metrics"]:
        return "Error: Metric not found"
    
    # Delete files
    for upload in metadata["metrics"][metric_name]["uploads"]:
        file_path = os.path.join(raw_uploads_dir, f"{metric_name}_{upload['id']}.csv")
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Remove from metadata
    del metadata["metrics"][metric_name]
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return "Delete successful"

def merge_context_data(context, output_base="data"):
    """
    Merge all metric CSVs for a context into a unified DataFrame.
    
    Args:
        context (str): Context name.
        output_base (str): Base output directory.
    
    Returns:
        tuple: (DataFrame or None, message)
    """
    metadata_path = os.path.join(output_base, context, "metadata.json")
    raw_uploads_dir = os.path.join(output_base, context, "raw_uploads")
    
    if not os.path.exists(metadata_path):
        return None, "Error: Context not found"
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    metrics = metadata["metrics"]
    if not metrics:
        return None, "Error: No metrics in context"
    
    dfs = []
    for metric_name, info in metrics.items():
        for upload in info["uploads"]:
            file_path = os.path.join(raw_uploads_dir, f"{metric_name}_{upload['id']}.csv")
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                df['time'] = pd.to_datetime(df['time'])
                df = df.rename(columns={'metric': metric_name})
                dfs.append(df)
    
    if not dfs:
        return None, "Error: No data files found"
    
    # Merge on time
    merged = dfs[0]
    for df in dfs[1:]:
        merged = pd.merge(merged, df, on='time', how='outer')
    
    # Check alignment
    if len(merged) != len(dfs[0]) or not all(len(df) == len(merged) for df in dfs):
        return None, "Error: Timestamps not aligned across metrics"
    
    # Check start/end times
    start_times = [pd.to_datetime(info["start_time"]) for info in metrics.values()]
    end_times = [pd.to_datetime(info["end_time"]) for info in metrics.values()]
    if not all(st == start_times[0] for st in start_times) or not all(et == end_times[0] for et in end_times):
        return None, "Error: Start/end times do not match across metrics"
    
    # Save times in metadata for reconstruction
    times_list = merged['time'].dt.strftime('%Y-%m-%dT%H:%M:%S').tolist()
    metadata["times"] = times_list
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return merged, "Merge successful"

if __name__ == "__main__":
    # Example usage
    # upload_metric_csv("experiment1", "cpu_usage", "path/to/cpu.csv", 100, 0)
    pass