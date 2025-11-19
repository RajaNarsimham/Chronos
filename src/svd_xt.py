"""
SVD-XT Video Prediction with LoRA Fine-Tuning

Features:
1. Temporal Attention Fusion - Intelligently blends multiple conditioning frames
2. Optical Flow Motion Extrapolation - Reduces ghosting for large condition_frames
3. LoRA Fine-Tuning - Efficient fine-tuning on custom video data

Usage:
------
1. STANDARD INFERENCE:
   Set mode = 'inference' in main()
   
2. LORA FINE-TUNING:
   - Organize your training videos: data/video1/*.jpg, data/video2/*.jpg, etc.
   - Set mode = 'train' in main()
   - Run: python src/svd-xt.py
   - LoRA weights saved to: models/lora_checkpoints/
   
3. INFERENCE WITH LORA:
   Set mode = 'inference_with_lora' in main()
   Specify checkpoint path in main()

Parameters:
-----------
- condition_frames: Number of previous frames to condition on
  * 1-5: Uses attention-based blending
  * >5: Automatically uses motion extrapolation to avoid ghosting
  
- num_frames: Frames generated per iteration (max 14 for SVD-XT)

- total_frames: Total frames to predict across all iterations

- lora_rank: LoRA rank (4-16, lower = faster, higher = more expressive)

- lora_alpha: LoRA scaling (typically 2x rank)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import math
from PIL import Image
import numpy as np
from diffusers import StableVideoDiffusionPipeline, DDPMScheduler
from diffusers.utils import load_image, export_to_video
import glob
import torch.nn.functional as F
import cv2
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import argparse

# Helpers
def _resolve_cross_attention_dim(unet):
    """Best-effort fetch of UNet cross-attention dim even when wrapped by PEFT."""
    def _get(obj, path):
        cur = obj
        for p in path.split('.'):
            cur = getattr(cur, p, None)
            if cur is None:
                return None
        return cur
    candidates = [
        'config.cross_attention_dim',
        'base_model.model.config.cross_attention_dim',
        'model.config.cross_attention_dim',
        'base_model.config.cross_attention_dim',
    ]
    for c in candidates:
        v = _get(unet, c)
        if isinstance(v, (int, float)):
            return int(v)
    # Fallback to common SVD-XT dim
    return 1280

# Image sequence dataset for fine-tuning
class VideoSequenceDataset(Dataset):
    """
    Dataset for loading video frame sequences for SVD fine-tuning.
    Each sample consists of:
    - conditioning_frame: The first frame (PIL Image) with optional history band
    - target_frames: Sequence of subsequent frames (list of PIL Images)
    """
    def __init__(self, root_dir, sequence_length=14, transform=None, target_size=(576, 1024), use_history_band=True, band_height=64, predictor=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.target_size = target_size  # (height, width) for SVD-XT
        self.use_history_band = use_history_band
        self.band_height = band_height
        self.predictor = predictor  # VideoImagePredictor instance for history band generation
        self.sequences = self._load_sequences()
        
    def _load_sequences(self):
        """Load all valid video sequences from subdirectories."""
        sequences = []
        
        # Each subdirectory is a video
        video_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]
        
        if not video_folders:
            # No subdirs, treat root_dir as single video folder
            frames = sorted(glob.glob(os.path.join(self.root_dir, "*.jpg")) + 
                          glob.glob(os.path.join(self.root_dir, "*.png")))
            # Create sequences with sliding window
            for i in range(max(0, len(frames) - self.sequence_length)):
                sequences.append(frames[i:i + self.sequence_length + 1])
        else:
            for video_folder in video_folders:
                video_path = os.path.join(self.root_dir, video_folder)
                frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")) + 
                              glob.glob(os.path.join(video_path, "*.png")))
                
                # Create sequences with sliding window
                for i in range(len(frames) - self.sequence_length):
                    sequences.append(frames[i:i + self.sequence_length + 1])
        
        print(f"Loaded {len(sequences)} training sequences from {len(video_folders) if video_folders else 1} videos")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_paths = self.sequences[idx]
        
        # First frame is conditioning image (already has history band if enabled during frame generation)
        conditioning_frame = load_image(sequence_paths[0])
        
        # Rest are target frames
        target_frames = [load_image(path) for path in sequence_paths[1:]]
        
        # NOTE: History bands are now pre-generated offline during frame creation
        # No dynamic generation needed here - frames already have history bands appended
        
        # Resize frames to target dimensions (width, height)
        target_width, target_height = self.target_size[1], self.target_size[0]
        if conditioning_frame.size != (target_width, target_height):
            conditioning_frame = conditioning_frame.resize((target_width, target_height), Image.LANCZOS)
        target_frames = [f.resize((target_width, target_height), Image.LANCZOS) if f.size != (target_width, target_height) else f for f in target_frames]
        
        if self.transform:
            conditioning_frame = self.transform(conditioning_frame)
            target_frames = [self.transform(frame) for frame in target_frames]
        
        return {
            'conditioning_frame': conditioning_frame,
            'target_frames': target_frames,
            'sequence_paths': sequence_paths
        }

class TemporalAttentionFusion(nn.Module):
    """
    Temporal attention module to fuse multiple frames into a single conditioning image.
    Uses self-attention over temporal dimension to weight frame importance.
    """
    def __init__(self, embed_dim=512, num_heads=8, max_frames=25):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_frames = max_frames

        # Learnable positional embeddings for temporal positions
        self.pos_embedding = nn.Parameter(torch.randn(1, max_frames, embed_dim))

        # Multi-head attention
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

        # MLP for feature refinement
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, frame_features):
        """
        Args:
            frame_features: Tensor of shape (batch, num_frames, embed_dim)
        Returns:
            Fused features: Tensor of shape (batch, embed_dim)
            Attention weights for visualization
        """
        batch_size, num_frames, _ = frame_features.shape

        # Add positional embeddings
        pos_emb = self.pos_embedding[:, :num_frames, :]
        x = frame_features + pos_emb

        # Self-attention over temporal dimension
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # MLP
        mlp_out = self.mlp(x)
        x = self.norm2(x + mlp_out)

        # Aggregate: use attention-weighted mean
        fused = x.mean(dim=1)  # (batch, embed_dim)

        return fused, attn_weights

# Updated model: Now using Stable Video Diffusion for better scene consistency in video sequences
class VideoImagePredictor:
    def __init__(self, model_id="stabilityai/stable-video-diffusion-img2vid-xt", device="cuda:0", use_temporal_attention=False):
        # Check if CUDA is available and device exists
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU.")
        elif device.startswith("cuda:") and int(device.split(":")[1]) >= torch.cuda.device_count():
            print(f"CUDA device {device} not available, using cuda:0.")
            device = "cuda:0"
        
        self.device = device
        self.use_temporal_attention = use_temporal_attention
        
        self.pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device != "cpu" else torch.float32,
            variant="fp16" if device != "cpu" else None,
            use_safetensors=True,
        ).to(device)

        # Use a training-capable scheduler for denoising loss
        try:
            self.pipe.scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
        except Exception:
            # Fallback: keep existing scheduler
            pass
        
        # Enable memory optimizations - conditionally enable CPU offload
        # Note: CPU offload conflicts with LoRA models, so we'll enable it only when needed
        # self.pipe.enable_model_cpu_offload()  # Disabled for LoRA compatibility
        self.pipe.unet.enable_forward_chunking()
        
        # Initialize temporal attention module if enabled
        if use_temporal_attention:
            self.temporal_attention = TemporalAttentionFusion(embed_dim=512, num_heads=8, max_frames=25).to(device)
            if device != "cpu":
                self.temporal_attention = self.temporal_attention.half()
            
            # Simple feature extractor (CNN-based)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 512),
            ).to(device)
            if device != "cpu":
                self.feature_extractor = self.feature_extractor.half()

    def _flow_to_rgb(self, flow):
        """Convert optical flow (H,W,2) to RGB visualization (uint8)."""
        h, w = flow.shape[:2]
        fx, fy = flow[..., 0], flow[..., 1]
        mag, ang = cv2.cartToPolar(fx, fy)
        hsv = np.zeros((h, w, 3), dtype=np.uint8)
        hsv[..., 0] = (ang * 180 / np.pi / 2).astype(np.uint8)  # Hue
        hsv[..., 1] = 255
        m = np.clip(mag / (mag.max() + 1e-8), 0, 1)
        hsv[..., 2] = (m * 255).astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return rgb

    def _make_history_band(self, frames, width, band_height=64, mode="thumbs+flow"):
        """
        Create a narrow band encoding temporal history to append under the last frame.

        - mode 'thumbs+flow': concatenates tiny thumbnails of previous frames and an average-flow map.
        """
        band = Image.new('RGB', (width, band_height), color=(0, 0, 0))
        if len(frames) == 0:
            return band

        draw_x = 0
        n = len(frames)
        # Thumbnails section: up to 6 thumbs
        max_thumbs = min(n, 6)
        thumb_w = max(1, width // (max_thumbs + 1))
        thumb_h = band_height
        for f in frames[-max_thumbs:]:
            thumb = f.resize((thumb_w, thumb_h), Image.BILINEAR)
            band.paste(thumb, (draw_x, 0))
            draw_x += thumb_w

        # Flow section: average flow over last pairs
        if n >= 2:
            flows = []
            # Normalize frame sizes before computing optical flow (predicted frames may be 512px, training frames 576px)
            normalized_frames = []
            target_size = frames[0].size  # Use first frame's size as reference
            for frame in frames:
                if frame.size != target_size:
                    frame = frame.resize(target_size, Image.LANCZOS)
                normalized_frames.append(frame)
            
            for i in range(max(0, n - 4), n - 1):
                flow = self._compute_optical_flow(normalized_frames[i], normalized_frames[i + 1])
                flows.append(flow)
            if flows:
                avg_flow = np.mean(flows, axis=0)
                flow_rgb = self._flow_to_rgb(avg_flow)
                fw = width - draw_x
                if fw > 0:
                    flow_img = Image.fromarray(flow_rgb).resize((fw, band_height), Image.BILINEAR)
                    band.paste(flow_img, (draw_x, 0))
        return band

    def _encode_history_on_frame(self, last_frame, history_frames, band_height=64):
        """
        Append a history-encoding band to the bottom of last_frame.
        """
        w, h = last_frame.size
        band = self._make_history_band(history_frames, w, band_height)
        out = Image.new('RGB', (w, h + band_height))
        out.paste(last_frame, (0, 0))
        out.paste(band, (0, h))
        return out

    def _compute_optical_flow(self, frame1, frame2):
        """
        Compute optical flow between two consecutive frames using Farneback algorithm.
        Returns flow vectors (dx, dy) for each pixel.
        """
        # Convert PIL to numpy
        img1 = np.array(frame1)
        img2 = np.array(frame2)
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )
        
        return flow  # Shape: (H, W, 2) - [dx, dy] for each pixel

    def _extrapolate_motion(self, frames):
        """
        Extrapolate motion from previous frames to predict next frame position.
        Uses optical flow to estimate motion and warp the last frame.
        """
        if len(frames) < 2:
            return frames[-1]
        
        # Compute flow between consecutive frames
        flows = []
        for i in range(len(frames) - 1):
            flow = self._compute_optical_flow(frames[i], frames[i + 1])
            flows.append(flow)
        
        # Average the flows to get consistent motion estimate
        avg_flow = np.mean(flows, axis=0)
        
        # Extrapolate: apply the averaged flow to warp the last frame
        last_frame = np.array(frames[-1])
        h, w = last_frame.shape[:2]
        
        # Create coordinate grid
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply flow to get new positions
        new_x = (x + avg_flow[..., 0]).astype(np.float32)
        new_y = (y + avg_flow[..., 1]).astype(np.float32)
        
        # Warp the last frame using the predicted positions
        warped = cv2.remap(last_frame, new_x, new_y, cv2.INTER_LINEAR)
        
        return Image.fromarray(warped)

    def _extract_frame_features(self, frames):
        """Extract features from frames using feature extractor."""
        batch = []
        for frame in frames:
            img_tensor = transforms.ToTensor()(frame).unsqueeze(0).to(self.device)
            if self.device != "cpu":
                img_tensor = img_tensor.half()
            batch.append(img_tensor)
        
        batch_tensor = torch.cat(batch, dim=0)  # (num_frames, 3, H, W)
        
        with torch.no_grad():
            features = self.feature_extractor(batch_tensor)  # (num_frames, 512)
        
        return features.unsqueeze(0)  # (1, num_frames, 512)

    def _align_frames_to_last(self, frames):
        """
        Align all frames to the coordinate system of the last frame using optical flow.
        This reduces ghosting when blending multiple conditioning frames.

        Args:
            frames: List of PIL Images in temporal order

        Returns:
            List of PIL Images aligned to the last frame (same length as input)
        """
        if len(frames) <= 1:
            return frames

        last = np.array(frames[-1])
        h, w = last.shape[:2]
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        aligned = []
        # Align each frame to the last by computing flow from last -> frame
        for f in frames[:-1]:
            src = np.array(f)
            flow = self._compute_optical_flow(Image.fromarray(last), Image.fromarray(src))
            map_x = (x + flow[..., 0]).astype(np.float32)
            map_y = (y + flow[..., 1]).astype(np.float32)
            warped = cv2.remap(src, map_x, map_y, cv2.INTER_LINEAR)
            aligned.append(Image.fromarray(warped))

        # Append the last frame unchanged
        aligned.append(frames[-1])
        return aligned

    def _fuse_frames_with_attention(self, frames, use_motion_extrapolation=True):
        """
        Fuse multiple frames using temporal attention with optional motion extrapolation.
        
        Args:
            frames: List of PIL Images
            use_motion_extrapolation: If True, use optical flow to predict motion instead of blending
        """
        if not self.use_temporal_attention or len(frames) == 1:
            return frames[-1]
        
        # For large number of frames (>5), use motion extrapolation to avoid ghosting
        if use_motion_extrapolation and len(frames) > 5:
            print(f"  Using motion extrapolation to avoid ghosting")
            return self._extrapolate_motion(frames)
        
        # For smaller numbers, first align frames to last to reduce ghosting
        aligned_frames = self._align_frames_to_last(frames)

        # Extract features from aligned frames
        frame_features = self._extract_frame_features(aligned_frames)
        
        # Apply temporal attention
        with torch.no_grad():
            fused_features, attn_weights = self.temporal_attention(frame_features)
        
        # Reconstruct image by weighted blending
        # Convert frames to numpy arrays
        frame_arrays = [np.array(f, dtype=np.float32) for f in aligned_frames]
        frame_stack = np.stack(frame_arrays, axis=0)  # (num_frames, H, W, 3)
        
        # Get attention weights for last position (most relevant)
        weights = attn_weights[0, -1, :].cpu().numpy()  # (num_frames,)
        # Sharpen weights to emphasize recent frames and reduce ghosting
        eps = 1e-8
        gamma = 1.5  # >1.0 to focus more on higher-weighted frames (often the last)
        weights = np.power(np.maximum(weights, eps), gamma)
        weights = weights / (weights.sum() + eps)
        
        print(f"  Attention weights: {weights}")
        
        # Weighted blend
        weighted_frame = np.sum(frame_stack * weights[:, None, None, None], axis=0)
        fused_image = Image.fromarray(weighted_frame.astype(np.uint8))
        
        return fused_image

    def predict_next_frames(self, image_paths, num_frames=14, num_inference_steps=25, fps=7, motion_bucket_id=127, decode_chunk_size=4, total_frames=24, condition_frames=3, use_history_band=False, history_band_height=64):
        """
        Predict the next frames using video diffusion model conditioned on previous n images.
        
        Args:
            image_paths: List of paths to previous images (K images)
            num_frames: Number of frames to generate per iteration (max 14 for SVD-XT)
            num_inference_steps: Number of denoising steps
            fps: Frames per second
            motion_bucket_id: Motion bucket for controlling motion (0-255, higher = more motion)
            decode_chunk_size: Chunk size for decoding (lower = less memory)
            total_frames: Total number of frames to predict (will iterate if > num_frames)
            condition_frames: Number of previous frames to condition on (default: 3)
        
        Returns:
            List of PIL Images of the predicted frames
        """
        all_predicted_frames = []
        
        # Load initial conditioning frames
        initial_frames = [load_image(path) for path in image_paths[-condition_frames:]]
        # Original size should be content size (512px height), not including history band
        # If conditioning frame has history band (576px), subtract it
        cond_width, cond_height = initial_frames[-1].size
        if use_history_band or cond_height > 576:
            original_size = (cond_width, max(512, cond_height - history_band_height))
        else:
            original_size = (cond_width, cond_height)
        
        # Create frame buffer
        frame_buffer = initial_frames.copy()
        
        frames_remaining = total_frames
        iteration = 0
        
        while frames_remaining > 0:
            frames_this_iteration = min(num_frames, frames_remaining)
            iteration += 1
            print(f"Iteration {iteration}: Generating {frames_this_iteration} frames...")
            
            # Fuse frames with temporal attention or fallback to last frame
            if self.use_temporal_attention and condition_frames > 1:
                current_init_image = self._fuse_frames_with_attention(frame_buffer[-condition_frames:])
                print(f"  Using temporal attention fusion on {len(frame_buffer[-condition_frames:])} frames")
            else:
                current_init_image = frame_buffer[-1]

            # Optionally append a compact history band under the last frame
            appended_band = False
            if use_history_band:
                current_init_image = self._encode_history_on_frame(
                    current_init_image, frame_buffer[-condition_frames:], band_height=history_band_height
                )
                appended_band = True
            
            # Generate a batch of frames
            output = self.pipe(
                image=current_init_image,
                num_frames=frames_this_iteration,
                num_inference_steps=num_inference_steps,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                decode_chunk_size=decode_chunk_size,
            ).frames
            
            # Process and store frames
            for frame in output[0]:
                # If we appended a band, remove it from the generated frame
                if appended_band:
                    w, h = frame.size
                    frame = frame.crop((0, 0, w, max(0, h - history_band_height)))
                if frame.size != original_size:
                    frame = frame.resize(original_size, Image.LANCZOS)
                all_predicted_frames.append(frame)
                frame_buffer.append(frame)  # Add to buffer for next iteration
            
            frames_remaining -= frames_this_iteration
        
        return all_predicted_frames

# Fine-tuning function with LoRA
def fine_tune_with_lora(
    predictor,
    dataset,
    output_dir="models/lora_checkpoints",
    epochs=30,
    batch_size=4,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=16,
    save_every=5,
    gradient_accumulation_steps=2
):
    """
    Fine-tune SVD-XT using LoRA (Low-Rank Adaptation).
    
    Args:
        predictor: VideoImagePredictor instance
        dataset: VideoSequenceDataset instance
        output_dir: Directory to save LoRA weights
        epochs: Number of training epochs
        batch_size: Batch size for parallel processing
        learning_rate: Learning rate
        lora_rank: LoRA rank (lower = fewer parameters, faster)
        lora_alpha: LoRA alpha scaling factor
        save_every: Save checkpoint every N epochs
        gradient_accumulation_steps: Accumulate gradients over N steps
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure LoRA for UNet only (most important for video generation)
    # Target only temporal attention modules to avoid dimension mismatches
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_q", "to_k", "to_v"],  # Only Q, K, V without output projection
        lora_dropout=0.1,
        bias="none",
        modules_to_save=None,  # Don't save any modules
    )
    
    # Apply LoRA to the UNet
    print("Applying LoRA adapters to UNet...")
    predictor.pipe.unet = get_peft_model(predictor.pipe.unet, lora_config)
    predictor.pipe.unet.print_trainable_parameters()
    predictor.pipe.unet.train()
    
    # Setup optimizer with constant LR (no aggressive decay)
    optimizer = AdamW(predictor.pipe.unet.parameters(), lr=learning_rate, weight_decay=1e-4)
    # Warmup for first 10% of steps, then constant
    total_steps = max(1, epochs * max(1, len(dataset) // max(1, batch_size)))
    warmup_steps = total_steps // 10
    
    def get_lr_multiplier(step):
        if step < warmup_steps:
            # Warmup: 0 -> 1.0 over first 10% of training
            return float(step) / float(max(1, warmup_steps))
        else:
            # Cosine annealing: 1.0 -> 0.0 over remaining 90%
            progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, get_lr_multiplier)
    
    # Enable mixed precision training for RTX 5090
    scaler = torch.amp.GradScaler('cuda', enabled=(predictor.device != "cpu"))
    
    # Custom collate function for batching
    def collate_fn(batch):
        conditioning_frames = [item['conditioning_frame'] for item in batch]
        target_frames = [item['target_frames'] for item in batch]
        return {'conditioning_frame': conditioning_frames, 'target_frames': target_frames}
    
    # DataLoader with batching support (num_workers=0 for Windows compatibility)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, collate_fn=collate_fn, pin_memory=True)
    
    # Training loop: unconditional denoising on VAE latents
    global_step = 0
    optimizer.zero_grad()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in enumerate(progress_bar):
            conditioning_frames = batch['conditioning_frame']
            target_frames_list = batch['target_frames']
            
            try:
                model_dtype = next(predictor.pipe.unet.parameters()).dtype
                
                # Prepare all conditioning frames
                cond_tensors = []
                target_tensors = []
                for cond_frame, target_frames in zip(conditioning_frames, target_frames_list):
                    if isinstance(cond_frame, list):
                        cond_frame = cond_frame[0]
                    if not target_frames:
                        continue
                    
                    # Convert to tensors
                    cond_t = transforms.ToTensor()(cond_frame).to(predictor.device, dtype=model_dtype)
                    target_t = transforms.ToTensor()(target_frames[0]).to(predictor.device, dtype=model_dtype)
                    cond_tensors.append(cond_t * 2 - 1)
                    target_tensors.append(target_t * 2 - 1)
                
                if len(cond_tensors) == 0:
                    continue
                
                # Batch encode all frames at once (MUCH faster than sequential)
                cond_batch = torch.stack(cond_tensors)  # (B, 3, H, W)
                target_batch = torch.stack(target_tensors)  # (B, 3, H, W)
                
                with torch.no_grad():
                    cond_latents = predictor.pipe.vae.encode(cond_batch).latent_dist.sample()
                    cond_latents = cond_latents * predictor.pipe.vae.config.scaling_factor
                    target_latents = predictor.pipe.vae.encode(target_batch).latent_dist.sample()
                    target_latents = target_latents * predictor.pipe.vae.config.scaling_factor
                
                # 2) Sample timesteps and add noise (batched)
                num_train_steps = getattr(predictor.pipe.scheduler.config, 'num_train_timesteps', 1000)
                bs = len(cond_tensors)
                timesteps = torch.randint(0, num_train_steps, (bs,), device=predictor.device).long()
                noise = torch.randn_like(target_latents, dtype=model_dtype, device=predictor.device)
                noisy_latents = predictor.pipe.scheduler.add_noise(target_latents, noise, timesteps)
                
                # Concatenate conditioning and noisy target (B,4,H,W) + (B,4,H,W) -> (B,8,H,W)
                noisy_latents_with_cond = torch.cat([cond_latents, noisy_latents], dim=1)
                # Add frames dimension and permute to [B, 1, 8, H, W]
                noisy_latents_with_cond = noisy_latents_with_cond.unsqueeze(2).permute(0, 2, 1, 3, 4)

                # 3) Prepare conditioning for entire batch
                cross_attn_dim = _resolve_cross_attention_dim(predictor.pipe.unet)
                encoder_hidden_states = torch.zeros(
                    (bs, 1, cross_attn_dim),
                    device=predictor.device,
                    dtype=model_dtype,
                )
                
                # SVD-XT time IDs (batched)
                fps, motion_bucket_id, noise_aug_strength = 7.0, 127.0, 0.0
                added_time_ids = torch.tensor(
                    [[fps, motion_bucket_id, noise_aug_strength]],
                    device=predictor.device,
                    dtype=torch.float32,
                ).repeat(bs, 1)

                # 4) Single UNet forward pass for entire batch (MASSIVE speedup)
                try:
                    with torch.amp.autocast('cuda', enabled=(predictor.device != "cpu")):
                        model_pred = predictor.pipe.unet(
                            noisy_latents_with_cond,
                            timesteps,
                            encoder_hidden_states=encoder_hidden_states,
                            added_time_ids=added_time_ids,
                            return_dict=False,
                        )[0]
                        # model_pred is [B, 1, 4, H, W], noise is [B, 4, H, W]
                        noise_target = noise.unsqueeze(1)  # [B, 1, 4, H, W]
                        loss = F.mse_loss(model_pred, noise_target)
                        loss = loss / gradient_accumulation_steps
                    
                    scaler.scale(loss).backward()
                    
                    batch_loss = loss.item() * gradient_accumulation_steps
                    valid_items = 1
                except RuntimeError as e:
                    # Print full error with layer information
                    print(f"UNet forward error details: {e}")
                    print(f"  Timesteps: {timesteps.shape}")
                    print(f"  Encoder hidden states: {encoder_hidden_states.shape}")
                    print(f"  Added time ids: {added_time_ids.shape}")
                    print(f"  Noisy latents: {noisy_latents_with_cond.shape}")
                    raise
                
                # Gradient accumulation: step optimizer every N batches
                if (batch_idx + 1) % gradient_accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(predictor.pipe.unet.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    # Clear CUDA cache to prevent memory fragmentation
                    if predictor.device != "cpu":
                        torch.cuda.empty_cache()
                
                epoch_loss += batch_loss
                global_step += 1
                
                progress_bar.set_postfix({
                    'loss': f"{batch_loss:.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"lora_epoch_{epoch + 1}.pt")
            predictor.pipe.unet.save_pretrained(checkpoint_path)
            print(f"Saved LoRA checkpoint to {checkpoint_path}")
            
            # Save training config
            config = {
                'epoch': epoch + 1,
                'learning_rate': learning_rate,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'avg_loss': avg_loss
            }
            with open(os.path.join(output_dir, f"config_epoch_{epoch + 1}.json"), 'w') as f:
                json.dump(config, f, indent=2)
    
    print("Fine-tuning completed!")
    return predictor

def fine_tune_with_lora_history_band(
    predictor,
    dataset,
    output_dir="models/lora_checkpoints_history_band",
    epochs=2,
    batch_size=1,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=16,
    save_every=1,
    # history band specific
    condition_frames=3,
    history_band_height=64,
    # generation preview
    num_preview_frames=8,
    num_inference_steps=25,
    fps=7,
    motion_bucket_id=127,
    decode_chunk_size=4,
):
    """
    Tiny LoRA fine-tune loop variant that ALWAYS encodes a history band on the
    conditioning image. This mirrors inference-time usage of the history band.

    Notes:
    - This is a minimal loop for scaffolding; it logs previews and saves LoRA checkpoints.
    - It uses a placeholder zero loss (no true training signal) like the base loop.
    - Replace the placeholder loss with a proper denoising objective for real training.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configure LoRA for UNet
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "proj_in", "proj_out",
        ],
        lora_dropout=0.1,
        bias="none",
    )

    print("Applying LoRA adapters to UNet (history band variant)...")
    predictor.pipe.unet = get_peft_model(predictor.pipe.unet, lora_config)
    predictor.pipe.unet.print_trainable_parameters()

    optimizer = AdamW(predictor.pipe.unet.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, epochs * max(1, len(dataset) // max(1, batch_size))))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    predictor.pipe.unet.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"HB Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            # Expect batch size = 1; use sequence_paths to reconstruct frames
            seq_paths = batch.get('sequence_paths', None)
            if seq_paths is None:
                # Fallback: try to approximate from provided conditioning/targets
                conditioning_frame = batch['conditioning_frame']
                if isinstance(conditioning_frame, list):
                    conditioning_frame = conditioning_frame[0]
                history_frames = [conditioning_frame for _ in range(condition_frames)]
                last_frame = conditioning_frame
            else:
                paths = seq_paths[0] if isinstance(seq_paths, list) else seq_paths
                frames = [load_image(p) for p in paths]
                if len(frames) < condition_frames + 1:
                    # Not enough frames to build history; skip
                    continue
                history_frames = frames[:condition_frames]
                last_frame = history_frames[-1]

            try:
                # Build conditioning image with history band
                cond_img = predictor._encode_history_on_frame(last_frame, history_frames, band_height=history_band_height)

                # -------- Real training signal: pixel-space reconstruction loss --------
                # Target: the frame immediately after the history window
                gt_frame = None
                if seq_paths is not None:
                    # Use the (condition_frames)-th index as next frame (0-based)
                    if len(frames) > condition_frames:
                        gt_frame = frames[condition_frames]
                # Fallback: if ground truth frame unavailable, skip
                if gt_frame is None:
                    continue

                # Denoising loss in latent space (aligns with base trainer)
                model_dtype = next(predictor.pipe.unet.parameters()).dtype

                # Build encoder hidden states from conditioning image
                try:
                    if hasattr(predictor.pipe, 'feature_extractor'):
                        px = predictor.pipe.feature_extractor(images=cond_img, return_tensors='pt').pixel_values
                    elif hasattr(predictor.pipe, 'image_processor'):
                        px = predictor.pipe.image_processor(images=cond_img, return_tensors='pt').pixel_values
                    else:
                        raise AttributeError('No image processor/feature_extractor found on pipeline')
                    px = px.to(predictor.device, dtype=model_dtype)
                    with torch.no_grad():
                        img_embeds = predictor.pipe.image_encoder(px).last_hidden_state
                        encoder_hidden_states = img_embeds
                    if encoder_hidden_states.dim() == 3 and encoder_hidden_states.shape[1] > 1:
                        encoder_hidden_states = encoder_hidden_states.mean(dim=1, keepdim=True)
                    elif encoder_hidden_states.dim() == 2:
                        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
                    ca_dim_hb = _resolve_cross_attention_dim(predictor.pipe.unet)
                    cur_dim_hb = int(encoder_hidden_states.shape[-1])
                    if cur_dim_hb < ca_dim_hb:
                        encoder_hidden_states = F.pad(encoder_hidden_states, (0, ca_dim_hb - cur_dim_hb))
                    elif cur_dim_hb > ca_dim_hb:
                        encoder_hidden_states = encoder_hidden_states[..., :ca_dim_hb]
                    encoder_hidden_states = encoder_hidden_states.to(predictor.device, dtype=model_dtype)
                except Exception as e:
                    embed_dim = getattr(getattr(predictor.pipe.unet, 'config', object()), 'cross_attention_dim', 1024)
                    encoder_hidden_states = torch.zeros((1, 1, embed_dim), device=predictor.device, dtype=model_dtype)
                    print(f"[HB] Falling back: using zero encoder embeddings (reason: {e})")

                # Ground-truth to VAE latents
                gt_tensor = transforms.ToTensor()(gt_frame).to(predictor.device, dtype=model_dtype) * 2 - 1
                with torch.no_grad():
                    gt_latents = predictor.pipe.vae.encode(gt_tensor.unsqueeze(0)).latent_dist.sample()
                    gt_latents = gt_latents * predictor.pipe.vae.config.scaling_factor
                gt_latents = gt_latents.to(predictor.device, dtype=model_dtype)

                # Sample timestep and add noise to target latents
                # Sample timestep and add noise with scheduler; fallback to simple addition if unsupported
                try:
                    total_steps_hb = int(getattr(predictor.pipe.scheduler.config, 'num_train_timesteps', 0) or 0)
                    if total_steps_hb <= 0:
                        raise ValueError('num_train_timesteps missing')
                    t = torch.randint(0, total_steps_hb, (1,), device=predictor.device).long()
                    noise = torch.randn_like(gt_latents, dtype=model_dtype, device=predictor.device)
                    noisy_latents = predictor.pipe.scheduler.add_noise(gt_latents, noise, t)
                except Exception as e:
                    print(f"Scheduler add_noise fallback (HB): {e}")
                    t = torch.zeros((1,), device=predictor.device, dtype=torch.long)
                    noise = torch.randn_like(gt_latents, dtype=model_dtype, device=predictor.device)
                    noisy_latents = gt_latents + noise
                noisy_latents = noisy_latents.unsqueeze(2)  # (B,C,1,H,W)

                # Predict noise with UNet and compute MSE loss (include added_time_ids)
                bs_hb = noisy_latents.shape[0]
                added_time_ids_hb = torch.tensor([fps, motion_bucket_id], dtype=torch.float32, device=predictor.device).unsqueeze(0).repeat(bs_hb, 1)
                try:
                    pred_noise = predictor.pipe.unet(
                        noisy_latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        added_time_ids=added_time_ids_hb,
                    ).sample
                except Exception as e_unet_hb:
                    print(f"UNet forward error (HB): {e_unet_hb} | enc_hid {tuple(encoder_hidden_states.shape)} ca_dim={_resolve_cross_attention_dim(predictor.pipe.unet)}")
                    ca_dim_hb_retry = _resolve_cross_attention_dim(predictor.pipe.unet)
                    encoder_hidden_states = torch.zeros((bs_hb, 1, ca_dim_hb_retry), device=predictor.device, dtype=model_dtype)
                    pred_noise = predictor.pipe.unet(
                        noisy_latents,
                        t,
                        encoder_hidden_states=encoder_hidden_states,
                        added_time_ids=added_time_ids_hb,
                    ).sample
                loss = F.mse_loss(pred_noise, noise.unsqueeze(2))

                # Also log periodic previews (no grad needed)
                if batch_idx % 50 == 0:
                    with torch.no_grad():
                        out_prev = predictor.pipe(
                            image=cond_img,
                            num_frames=num_preview_frames,
                            num_inference_steps=num_inference_steps,
                            fps=fps,
                            motion_bucket_id=motion_bucket_id,
                            decode_chunk_size=decode_chunk_size,
                        )
                        sample_dir = os.path.join(output_dir, f"samples_epoch{epoch + 1}")
                        os.makedirs(sample_dir, exist_ok=True)
                        export_to_video(out_prev.frames[0], os.path.join(sample_dir, f"step_{global_step}.mp4"), fps=fps)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.pipe.unet.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += float(loss.item())
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}",
                })

            except Exception as e:
                print(f"[HB] Error in batch {batch_idx}: {e}")
                continue

        avg_loss = epoch_loss / max(1, len(dataloader))
        print(f"[HB] Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"lora_hb_epoch_{epoch + 1}.pt")
            predictor.pipe.unet.save_pretrained(checkpoint_path)
            print(f"[HB] Saved LoRA checkpoint to {checkpoint_path}")

            config = {
                'epoch': epoch + 1,
                'learning_rate': learning_rate,
                'lora_rank': lora_rank,
                'lora_alpha': lora_alpha,
                'avg_loss': avg_loss,
                'condition_frames': condition_frames,
                'history_band_height': history_band_height,
                'num_preview_frames': num_preview_frames,
                'num_inference_steps': num_inference_steps,
                'fps': fps,
                'motion_bucket_id': motion_bucket_id,
                'decode_chunk_size': decode_chunk_size,
            }
            with open(os.path.join(output_dir, f"config_hb_epoch_{epoch + 1}.json"), 'w') as f:
                json.dump(config, f, indent=2)

    print("History-band fine-tuning variant completed!")
    return predictor

def load_lora_weights(predictor, lora_checkpoint_path):
    """
    Load LoRA weights into the predictor.
    
    Args:
        predictor: VideoImagePredictor instance
        lora_checkpoint_path: Path to LoRA checkpoint directory
    """
    from peft import PeftModel
    
    print(f"Loading LoRA weights from {lora_checkpoint_path}...")
    predictor.pipe.unet = PeftModel.from_pretrained(
        predictor.pipe.unet,
        lora_checkpoint_path
    )
    print("LoRA weights loaded successfully!")
    return predictor

def get_image_paths(folder, extension="*.jpg"):
    """
    Get sorted list of image paths from a folder.
    
    Args:
        folder (str): Path to the folder containing images.
        extension (str): File extension pattern (default: "*.jpg").
    
    Returns:
        list: Sorted list of image file paths.
    """
    pattern = os.path.join(folder, extension)
    return sorted(glob.glob(pattern))

def main():
    parser = argparse.ArgumentParser(description="SVD-XT Video Prediction with LoRA Fine-Tuning")
    parser.add_argument('--mode', type=str, default='inference', choices=['train', 'inference', 'inference_with_lora', 'train_history_band'],
                        help='Mode: train, inference, or inference_with_lora')
    parser.add_argument('--folder', type=str, default='data/rabbit',
                        help='Folder containing input images for inference')
    parser.add_argument('--use_temporal_attention', action='store_true',
                        help='Enable temporal attention fusion')
    parser.add_argument('--condition_frames', type=int, default=3,
                        help='Number of previous frames to condition on')
    parser.add_argument('--use_history_band', action='store_true',
                        help='Append history band to conditioning image')
    parser.add_argument('--history_band_height', type=int, default=64,
                        help='Height of history band')
    parser.add_argument('--num_frames', type=int, default=14,
                        help='Number of frames to generate per iteration')
    parser.add_argument('--total_frames', type=int, default=14,
                        help='Total frames to predict')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of denoising steps')
    parser.add_argument('--fps', type=int, default=24,
                        help='Frames per second')
    parser.add_argument('--motion_bucket_id', type=int, default=127,
                        help='Motion bucket ID (0-255)')
    parser.add_argument('--decode_chunk_size', type=int, default=14,
                        help='Decode chunk size')
    # Training specific args
    parser.add_argument('--output_dir', type=str, default='models/lora_checkpoints',
                        help='Output directory for LoRA checkpoints')
    parser.add_argument('--epochs', type=int, default=5,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA alpha')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--lora_checkpoint', type=str, default='models/lora_checkpoints/lora_epoch_5.pt',
                        help='Path to LoRA checkpoint for inference_with_lora')
    # History band training controls
    parser.add_argument('--hb_output_dir', type=str, default='models/lora_checkpoints_history_band',
                        help='Output directory for history-band LoRA checkpoints')
    parser.add_argument('--hb_epochs', type=int, default=2,
                        help='Epochs for history-band training variant')
    parser.add_argument('--hb_condition_frames', type=int, default=3,
                        help='Condition frames used to build history band during training')
    parser.add_argument('--hb_history_band_height', type=int, default=64,
                        help='History band height during training')
    parser.add_argument('--hb_num_preview_frames', type=int, default=8,
                        help='Number of preview frames to render during training')

    args = parser.parse_args()

    print("Image Prediction Project with Video Diffusion Models")
    print(f"Mode: {args.mode}")

    if args.mode == 'train':
        # ========== FINE-TUNING MODE ==========
        print("\n=== Starting LoRA Fine-Tuning ===")
        
        # Initialize predictor
        predictor = VideoImagePredictor(use_temporal_attention=args.use_temporal_attention)
        
        # Create dataset from training videos
        train_dataset = VideoSequenceDataset(
            root_dir="data",  # Change to your training data directory
            sequence_length=14
        )
        
        if len(train_dataset) == 0:
            print("ERROR: No training data found!")
            print("Please organize your data as: data/video_folder/*.jpg")
        else:
            # Fine-tune with LoRA
            predictor = fine_tune_with_lora(
                predictor=predictor,
                dataset=train_dataset,
                output_dir=args.output_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                save_every=args.save_every
            )
            print(f"\nFine-tuning completed! LoRA weights saved to {args.output_dir}/")
    
    elif args.mode == 'inference_with_lora':
        # ========== INFERENCE WITH LORA WEIGHTS ==========
        print("\n=== Inference with LoRA Fine-Tuned Model ===")
        
        predictor = VideoImagePredictor(use_temporal_attention=args.use_temporal_attention)
        
        # Load LoRA weights
        if os.path.exists(args.lora_checkpoint):
            predictor = load_lora_weights(predictor, args.lora_checkpoint)
        else:
            print(f"WARNING: LoRA checkpoint not found at {args.lora_checkpoint}")
            print("Using base model instead...")
        
        # Run inference
        image_paths = get_image_paths(args.folder)
        if len(image_paths) < args.condition_frames:
            print(f"Not enough images in the folder. Need at least {args.condition_frames} frames.")
        else:
            predicted_frames = predictor.predict_next_frames(
                image_paths, 
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                decode_chunk_size=args.decode_chunk_size,
                total_frames=args.total_frames,
                condition_frames=args.condition_frames,
                use_history_band=args.use_history_band,
                history_band_height=args.history_band_height
            )
            
            last_frame_num = len(image_paths) - 1
            for i, frame in enumerate(predicted_frames):
                frame_filename = f"Predicted_frame_lora_{last_frame_num + i + 1:04d}.jpg"
                frame_path = os.path.join(args.folder, frame_filename)
                frame.save(frame_path)
                print(f"Saved {frame_filename}")
            
            print(f"All {len(predicted_frames)} predicted frames saved to {args.folder}")
    
    elif args.mode == 'train_history_band':
        # ========== HISTORY BAND TRAINING MODE ==========
        print("\n=== Starting LoRA Fine-Tuning (History Band Variant) ===")

        predictor = VideoImagePredictor(use_temporal_attention=False)

        train_dataset = VideoSequenceDataset(
            root_dir="data",
            sequence_length=14
        )

        if len(train_dataset) == 0:
            print("ERROR: No training data found!")
            print("Please organize your data as: data/video_folder/*.jpg")
        else:
            predictor = fine_tune_with_lora_history_band(
                predictor=predictor,
                dataset=train_dataset,
                output_dir=args.hb_output_dir,
                epochs=args.hb_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                lora_rank=args.lora_rank,
                lora_alpha=args.lora_alpha,
                save_every=args.save_every,
                condition_frames=args.hb_condition_frames,
                history_band_height=args.hb_history_band_height,
                num_preview_frames=args.hb_num_preview_frames,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                decode_chunk_size=args.decode_chunk_size,
            )
            print(f"\nHistory-band fine-tuning completed! LoRA weights saved to {args.hb_output_dir}/")

    else:
        # ========== STANDARD INFERENCE MODE ==========
        print("\n=== Standard Inference Mode ===")
        
        # Initialize the predictor
        predictor = VideoImagePredictor(use_temporal_attention=args.use_temporal_attention)
        
        # Read all frames from the specified folder
        image_paths = get_image_paths(args.folder)
        if len(image_paths) < args.condition_frames:
            print(f"Not enough images in the folder. Need at least {args.condition_frames} frames.")
        else:
            predicted_frames = predictor.predict_next_frames(
                image_paths, 
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                decode_chunk_size=args.decode_chunk_size,
                total_frames=args.total_frames,
                condition_frames=args.condition_frames,
                use_history_band=args.use_history_band,
                history_band_height=args.history_band_height
            )
            
            # Get the last frame number from existing images
            last_frame_num = len(image_paths) - 1
            
            # Save all predicted frames with sequential naming
            for i, frame in enumerate(predicted_frames):
                frame_filename = f"Predicted_frame_{last_frame_num + i + 1:04d}.jpg"
                frame_path = os.path.join(args.folder, frame_filename)
                frame.save(frame_path)
                print(f"Saved {frame_filename}")
            
            print(f"All {len(predicted_frames)} predicted frames saved to {args.folder}")

if __name__ == "__main__":
    main()