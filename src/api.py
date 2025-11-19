from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from .database import SessionLocal, engine
from .models import Base, Context, Metric, Model, InferenceResult
from .websockets import manager
from pydantic import BaseModel
import json
import csv
import io
import pandas as pd
import time
import tempfile
import os
import datetime
import glob
from PIL import Image
from .timeseries_frame_generator import generate_timeseries_frames
from .svd_xt import VideoImagePredictor, VideoSequenceDataset, fine_tune_with_lora
import numpy as np

class TrainRequest(BaseModel):
    context_name: str

class InferenceRequest(BaseModel):
    context_name: str
    model_id: int

Base.metadata.create_all(bind=engine)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for debugging
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/test")
def test():
    return {"message": "Backend is running"}

@app.post("/testpost")
def testpost(data: str):
    return {"received": data}

class ContextRequest(BaseModel):
    name: str

@app.post("/contexts/")
def create_context(request: ContextRequest, db: Session = Depends(get_db)):
    db_context = Context(name=request.name)
    db.add(db_context)
    db.commit()
    db.refresh(db_context)
    return db_context

@app.get("/contexts/")
def read_contexts(skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    contexts = db.query(Context).offset(skip).limit(limit).all()
    return contexts

@app.post("/metrics/upload")
async def upload_metric(
    context_name: str = Form(...),
    min_value: str = Form(...),
    max_value: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    print(f"Upload started: context={context_name}, min={min_value}, max={max_value}, file={file.filename}")
    try:
        min_val = float(min_value)
        max_val = float(max_value)
        print(f"Parsed values: min={min_val}, max={max_val}")
    except ValueError as e:
        print(f"ValueError parsing min/max: {e}")
        raise HTTPException(status_code=400, detail="Invalid min_value or max_value")
    
    # Find or create context
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        db_context = Context(name=context_name)
        db.add(db_context)
        db.commit()
        db.refresh(db_context)
        print(f"Created context: {context_name}")
    
    contents = await file.read()
    print(f"Read file contents, length: {len(contents)}")
    if file.filename.endswith('.csv'):
        # Parse CSV with pandas, assuming columns: date_time, metric_value
        text = contents.decode("utf-8", errors="replace")
        print(f"Decoded text length: {len(text)}")
        df = pd.read_csv(io.StringIO(text))
        print(f"CSV columns: {list(df.columns)}")
        if 'metric_value' in df.columns:
            data_list = df['metric_value'].dropna().astype(float).tolist()
        else:
            # Assume no header, take second column if exists
            if len(df.columns) > 1:
                data_list = df.iloc[:, 1].dropna().astype(float).tolist()
            else:
                data_list = df.iloc[:, 0].dropna().astype(float).tolist()
        data = json.dumps(data_list)
        print(f"Parsed CSV to {len(data_list)} numbers")
    else:
        data = contents.decode("utf-8", errors="replace")  # Assume JSON
        print(f"Stored as text, length: {len(data)}")
    
    db_metric = Metric(
        context_id=db_context.id,
        filename=file.filename,
        data=data,
        min_value=min_val,
        max_value=max_val
    )
    db.add(db_metric)
    db.commit()
    db.refresh(db_metric)
    print(f"Inserted metric: {db_metric.id}")
    return db_metric

@app.get("/metrics/")
def read_metrics(context_name: str, skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        return []
    metrics = db.query(Metric).filter(Metric.context_id == db_context.id).offset(skip).limit(limit).all()
    result = []
    for metric in metrics:
        data = json.loads(metric.data)
        if isinstance(data, list) and data:
            avg_val = sum(data) / len(data)
            count = len(data)
        else:
            avg_val = count = None
        result.append({
            "id": metric.id,
            "filename": metric.filename,
            "uploaded_at": metric.uploaded_at.isoformat(),
            "min_value": metric.min_value,
            "max_value": metric.max_value,
            "avg_value": avg_val,
            "data_points": count
        })
    return result

@app.delete("/metrics/{metric_id}")
def delete_metric(metric_id: int, db: Session = Depends(get_db)):
    db_metric = db.query(Metric).filter(Metric.id == metric_id).first()
    if not db_metric:
        raise HTTPException(status_code=404, detail="Metric not found")
    db.delete(db_metric)
    db.commit()
    return {"message": "Metric deleted"}

@app.post("/models/train/")
def train_model(request: TrainRequest, db: Session = Depends(get_db)):
    context_name = request.context_name
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        raise HTTPException(status_code=404, detail="Context not found")
    
    metrics = db.query(Metric).filter(Metric.context_id == db_context.id).all()
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics in context")
    
    # Create merged df
    data_dict = {}
    for metric in metrics:
        data = json.loads(metric.data)
        data_dict[metric.filename] = data
    merged_df = pd.DataFrame(data_dict)
    
    # Create metadata
    metadata = {"metrics": {}, "last_updated": datetime.datetime.utcnow().isoformat(), "band_height": 64}
    for metric in metrics:
        metadata["metrics"][metric.filename] = {"max": metric.max_value, "min": metric.min_value}
    
    # Save metadata BEFORE generating frames (frames generator needs to read it)
    metadata_path = os.path.join("data", context_name, "metadata.json")
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f)
    
    # Initialize predictor for history band generation
    predictor = VideoImagePredictor()
    
    # Generate frames with history bands (512px content + 64px history = 576px total)
    frames_dir = generate_timeseries_frames(
        context_name, merged_df, "data", 
        band_height=64, width=1024, height=576,
        use_history_band=True,
        predictor=predictor
    )
    print(f"Frames saved to: {frames_dir}")
    
    # Train with history bands (frames are 576px: 512 content + 64 history)
    dataset = VideoSequenceDataset(
        root_dir=frames_dir, 
        sequence_length=14, 
        target_size=(576, 1024),
        use_history_band=False,  # Already pre-generated offline
        band_height=64,
        predictor=None
    )
    if len(dataset) == 0:
        raise HTTPException(status_code=400, detail="No sequences for training")
    
    predictor = fine_tune_with_lora(
        predictor, 
        dataset, 
        epochs=60,  # Increased to 60 epochs for better convergence
        batch_size=4,  # Conservative batch size to avoid OOM
        learning_rate=1e-4,  # Doubled for faster learning
        save_every=5,  # Save checkpoints every 5 epochs
        gradient_accumulation_steps=2  # Effective batch size of 8
    )
    
    # Save model (overwrites previous for this context)
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"lora_{context_name}.pt")
    predictor.pipe.unet.save_pretrained(model_path)
    
    # Create db entry with timestamp for history tracking
    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    db_model = Model(context_id=db_context.id, name=f"Finetuned {context_name} {timestamp}", path=model_path)
    db.add(db_model)
    db.commit()
    db.refresh(db_model)
    
    return {"task_id": "train_123", "status": "completed", "model_id": db_model.id}

@app.get("/models/")
def read_models(context_name: str, db: Session = Depends(get_db)):
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        return []
    models = db.query(Model).filter(Model.context_id == db_context.id).all()
    return models

@app.post("/inference/")
def run_inference(request: InferenceRequest, db: Session = Depends(get_db)):
    context_name = request.context_name
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        raise HTTPException(status_code=404, detail="Context not found")
    
    db_model = db.query(Model).filter(Model.id == request.model_id).first()
    if not db_model:
        raise HTTPException(status_code=404, detail="Model not found")
    
    # Get all metrics from context
    metrics = db.query(Metric).filter(Metric.context_id == db_context.id).all()
    if not metrics:
        raise HTTPException(status_code=400, detail="No metrics in context")
    
    # Create merged df from all metrics
    data_dict = {}
    for metric in metrics:
        data = json.loads(metric.data)
        data_dict[metric.filename] = data
    merged_df = pd.DataFrame(data_dict)
    
    # Create metadata
    metadata = {"metrics": {}, "band_height": 64}
    for metric in metrics:
        metadata["metrics"][metric.filename] = {"max": metric.max_value, "min": metric.min_value}
    
    # Generate input frames from current data
    frames_dir = generate_timeseries_frames(context_name, merged_df, "data", band_height=64, width=1024, height=576)
    
    # Load trained model
    predictor = VideoImagePredictor()
    from peft import PeftModel
    predictor.pipe.unet = PeftModel.from_pretrained(predictor.pipe.unet, db_model.path)
    
    # Get input frame paths (use last few frames for conditioning)
    import glob
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    if len(frame_files) < 7:
        raise HTTPException(status_code=400, detail="Need at least 7 frames for inference (6 for history + 1 for conditioning)")
    
    # AUTOREGRESSIVE PREDICTION: Predict 1 frame at a time (matching training)
    # Training: 1 frame (with 6-frame history band) -> 1 next frame
    # Inference: Must match this pattern exactly
    print(f"Running inference: generating 6 frames autoregressively...")
    
    num_predictions = 6
    predicted_frames = []
    
    # Maintain a buffer of frames for history band generation
    frame_buffer = []
    for path in frame_files[-7:]:  # Last 7 frames (6 for history + 1 current)
        img = Image.open(path).convert('RGB')
        # Remove history band if present (576px -> 512px content only)
        if img.height == 576:
            img = img.crop((0, 0, img.width, 512))
        frame_buffer.append(img)
    
    for i in range(num_predictions):
        print(f"  Predicting frame {i+1}/14...")
        
        # Create conditioning frame with history band (matching training)
        # Use last frame as content, last 6 frames for history band
        current_frame = frame_buffer[-1]
        history_frames = frame_buffer[-7:-1]  # Last 6 frames for history
        
        # Generate history band and append to current frame (512px -> 576px)
        conditioning_frame_with_history = predictor._encode_history_on_frame(
            current_frame, history_frames, band_height=64
        )
        
        # Save temporary conditioning frame
        temp_cond_path = os.path.join(frames_dir, f"temp_cond_{i}.png")
        conditioning_frame_with_history.save(temp_cond_path)
        
        # Generate next frame (SVD-XT minimum is 2 frames, we'll use only the first)
        frames = predictor.predict_next_frames(
            [temp_cond_path],  # Single conditioning frame WITH history band (576px)
            num_frames=2,  # SVD-XT minimum (we only use first frame)
            num_inference_steps=25,
            fps=7,
            motion_bucket_id=127,
            decode_chunk_size=4,
            total_frames=2,
            condition_frames=1,  # Single frame conditioning (matching training)
            use_history_band=False,  # Already embedded in conditioning frame
            history_band_height=64
        )
        
        # Take only the first predicted frame (matching training's 1-step prediction)
        next_frame = frames[0]
        
        # Ensure predicted frame is 512px (content only, no history band)
        if next_frame.height > 512:
            # Remove any history band that might have been added
            next_frame = next_frame.crop((0, 0, next_frame.width, 512))
        
        # Clean the predicted frame aggressively to remove VAE noise
        # Use same robust cleaning as final frame saving
        next_frame_gray = next_frame.convert('L')
        next_frame_arr = np.array(next_frame_gray)
        
        # Ensure correct dimensions
        if next_frame_arr.shape != (512, 1024):
            next_frame_gray = next_frame_gray.resize((1024, 512), Image.LANCZOS)
            next_frame_arr = np.array(next_frame_gray)
        
        cleaned_arr = np.zeros_like(next_frame_arr)
        
        # Clean each metric band (3 metrics) using robust median filtering
        N = 3
        band_h = 512 // N
        remainder = 512 % N
        current_y = 0
        
        for j in range(N):
            h = band_h + (1 if j < remainder else 0)
            band = next_frame_arr[current_y:current_y+h, :]
            
            # Get band pixels and filter outliers using P25-P75 (IQR method)
            band_pixels = band.flatten()
            p25 = np.percentile(band_pixels, 25)
            p75 = np.percentile(band_pixels, 75)
            
            # Keep only pixels within IQR range
            filtered_pixels = band_pixels[(band_pixels >= p25) & (band_pixels <= p75)]
            
            # Use median of filtered pixels as the clean value
            if len(filtered_pixels) > 0:
                clean_value = np.median(filtered_pixels)
            else:
                clean_value = np.median(band_pixels)
            
            # Fill entire band with the clean median value
            cleaned_arr[current_y:current_y+h, :] = int(clean_value)
            current_y += h
        
        # Convert cleaned array back to RGB image
        next_frame = Image.fromarray(cleaned_arr.astype(np.uint8)).convert('RGB')
        
        predicted_frames.append(next_frame)
        
        # Add cleaned frame to buffer for next iteration
        frame_buffer.append(next_frame)
        if len(frame_buffer) > 7:
            frame_buffer.pop(0)
        
        # Cleanup temporary conditioning frame
        if os.path.exists(temp_cond_path):
            os.remove(temp_cond_path)
    
    # Final cleanup: remove any lingering temp files
    for temp_file in glob.glob(os.path.join(frames_dir, "temp_cond_*.png")):
        try:
            os.remove(temp_file)
        except:
            pass
    
    print(f"Generated {len(predicted_frames)} predicted frames (autoregressive with 6-frame history)")
    
    # Decode predicted frames back to timeseries using median
    predictions = {}
    metrics_list = list(metadata["metrics"].keys())
    N = len(metrics_list)
    band_height = metadata.get("band_height", 64)
    target_height = 512  # Content area (predicted frames won't have history band)
    target_width = 1024
    
    print(f"Decoding {N} metrics from frames...")
    
    # First pass: collect all raw intensities from THIS inference to calculate calibration
    all_intensities = []
    for frame in predicted_frames:
        img = frame.convert('L')
        img_array = np.array(img)
        if img_array.shape != (target_height, target_width):
            img_array = np.array(img.resize((target_width, target_height)))
        all_intensities.extend(img_array.flatten())
    
    # Dynamic calibration based on this inference run
    all_intensities = np.array(all_intensities)
    OBSERVED_MIN = np.percentile(all_intensities, 5)   # 5th percentile
    OBSERVED_MAX = np.percentile(all_intensities, 95)  # 95th percentile
    print(f"Frame dimensions: {target_width}x{target_height} (excluding {band_height}px history band)")
    print(f"Calibration range: [{OBSERVED_MIN:.1f}, {OBSERVED_MAX:.1f}] (P5-P95 from inference)")
    
    # Create cleaned frames for visualization
    cleaned_frames = []
    
    # Second pass: decode with calibration and create cleaned frames
    for i, frame in enumerate(predicted_frames):
        # Convert frame to grayscale and get pixel values
        img = frame.convert('L')
        img_array = np.array(img)
        
        print(f"Frame {i}: original shape={img_array.shape}, min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
        
        if img_array.shape != (target_height, target_width):
            img_array = np.array(img.resize((target_width, target_height)))
            print(f"  Resized to {img_array.shape}")
        
        # Create cleaned frame array
        cleaned_array = np.zeros_like(img_array)
        
        # Split into bands for each metric
        band_h = target_height // N
        remainder = target_height % N
        current_y = 0
        
        for j, metric_name in enumerate(metrics_list):
            h = band_h + (1 if j < remainder else 0)
            band = img_array[current_y:current_y+h, :]
            
            # Apply percentile-based filtering to remove outliers/noise
            # This works for both binary and continuous metrics
            band_flat = band.flatten()
            p25 = np.percentile(band_flat, 25)
            p75 = np.percentile(band_flat, 75)
            
            # Filter to interquartile range (IQR) - removes extreme outliers
            filtered_values = band_flat[(band_flat >= p25) & (band_flat <= p75)]
            
            # Use median of filtered values (more robust)
            if len(filtered_values) > 0:
                median_intensity = np.median(filtered_values)
            else:
                # Fallback to overall median if filtering removed everything
                median_intensity = np.median(band_flat)
            
            # Fill the cleaned frame band with median value
            cleaned_array[current_y:current_y+h, :] = int(median_intensity)
            
            print(f"  Metric '{metric_name}': band[{current_y}:{current_y+h}], p25={p25:.1f}, p75={p75:.1f}, filtered_median={median_intensity:.2f}")
            
            # Denormalize with dynamic calibration from THIS inference
            max_v = metadata["metrics"][metric_name]["max"]
            min_v = metadata["metrics"][metric_name]["min"]
            
            # Rescale from observed inference range to full [0, 255]
            if OBSERVED_MAX > OBSERVED_MIN:
                clamped = np.clip(median_intensity, OBSERVED_MIN, OBSERVED_MAX)
                rescaled_0_255 = ((clamped - OBSERVED_MIN) / (OBSERVED_MAX - OBSERVED_MIN)) * 255.0
            else:
                rescaled_0_255 = median_intensity
            
            # Denormalize to metric's original range
            val = (rescaled_0_255 / 255.0) * (max_v - min_v) + min_v
            
            # For binary metrics, use adaptive threshold based on calibration midpoint
            if max_v == 1.0 and min_v == 0.0:
                # Use midpoint of observed range as threshold (not 0.5)
                # If model outputs are compressed, this adapts to the actual distribution
                adaptive_threshold = 0.5 * ((OBSERVED_MAX - OBSERVED_MIN) / 255.0)
                val = 1 if val > adaptive_threshold else 0
                print(f"    Calibrated: {median_intensity:.1f} -> {rescaled_0_255:.1f} -> {val} (binary, threshold={adaptive_threshold:.3f})")
            else:
                print(f"    Calibrated: {median_intensity:.1f} -> {rescaled_0_255:.1f} -> {val:.4f} (continuous)")
            
            current_y += h
            
            if metric_name not in predictions:
                predictions[metric_name] = []
            predictions[metric_name].append(float(val))
        
        # Convert cleaned array to image and store
        cleaned_frames.append(Image.fromarray(cleaned_array, mode='L'))
    
    # Save cleaned frames to disk for visualization
    inference_frames_dir = os.path.join("data", request.context_name, "inference_frames")
    os.makedirs(inference_frames_dir, exist_ok=True)
    
    for i, cleaned_frame in enumerate(cleaned_frames):
        frame_path = os.path.join(inference_frames_dir, f"predicted_frame_{i:03d}.png")
        cleaned_frame.save(frame_path)
    print(f"Saved {len(cleaned_frames)} cleaned frames to {inference_frames_dir}")
    
    print(f"Final predictions: {predictions}")
    
    # Save result to database
    db_result = InferenceResult(
        metric_id=metrics[0].id,  # Use first metric ID for reference
        model_id=request.model_id, 
        result=json.dumps(predictions)
    )
    db.add(db_result)
    db.commit()
    db.refresh(db_result)
    
    return db_result

@app.get("/results/")
def read_results(context_name: str, db: Session = Depends(get_db)):
    db_context = db.query(Context).filter(Context.name == context_name).first()
    if not db_context:
        return []
    results = db.query(InferenceResult).join(Metric).filter(Metric.context_id == db_context.id).all()
    return results