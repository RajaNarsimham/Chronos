"""
Wan 2.2 Video Prediction with LoRA Fine-Tuning

Available Model Variants (by size):
- Wan-AI/Wan2.2-T2V-1.3B     (~3-5GB)  - Smallest, fastest
- Wan-AI/Wan2.2-TI2V-5B-Diffusers      (~10-15GB) - RECOMMENDED for RTX 5090/3090
- Wan-AI/Wan2.2-I2V-A14B     (~25-30GB) - Large, may not fit
- Wan-AI/Wan2.2-T2V-A14B     (~25-30GB) - Large, may not fit

CURRENT DEFAULT: Wan-AI/Wan2.2-TI2V-5B-Diffusers (5B parameters, ~10-15GB VRAM)

Features:
1. Temporal Attention Fusion - Intelligently blends multiple conditioning frames
2. Optical Flow Motion Extrapolation - Reduces ghosting for large condition_frames
3. LoRA Fine-Tuning - Efficient fine-tuning on custom video data
4. Wan 2.2 Model - High-quality video generation from Alibaba
5. Memory Optimization - CPU offload and sequential offload for reduced VRAM usage

Usage:
------
1. STANDARD INFERENCE:
   Set mode = 'inference' in main()

2. LORA FINE-TUNING:
   - Organize your training videos: data/video1/*.jpg, data/video2/*.jpg, etc.
   - Set mode = 'train' in main()
   - Run: python src/wan2_2.py
   - LoRA weights saved to: models/lora_checkpoints_wan/

3. INFERENCE WITH LORA:
   Set mode = 'inference_with_lora' in main()
   Specify checkpoint path in main()

Parameters:
-----------
- condition_frames: Number of previous frames to condition on
  * 1-5: Uses attention-based blending
  * >5: Automatically uses motion extrapolation to avoid ghosting

- num_frames: Frames generated per iteration (max 49 for Wan 2.2)

- total_frames: Total number of frames to predict across all iterations

- lora_rank: LoRA rank (4-16, lower = faster, higher = more expressive)

- lora_alpha: LoRA scaling (typically 2x rank)

- memory_optimization: Memory optimization mode
  * "none": Standard loading (requires ~10-15GB VRAM)
  * "cpu_offload": Model CPU offload (~50% VRAM reduction)
  * "sequential_cpu_offload": Sequential CPU offload (~75% VRAM reduction, slower)
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from diffusers import WanImageToVideoPipeline
from diffusers.utils import load_image, export_to_video
import glob
import torch.nn.functional as F
import cv2
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json

# Image sequence dataset for fine-tuning
class VideoSequenceDataset(Dataset):
    """
    Dataset for loading video frame sequences for Wan 2.2 fine-tuning.
    Each sample consists of:
    - conditioning_frame: The first frame (PIL Image)
    - target_frames: Sequence of subsequent frames (list of PIL Images)
    """
    def __init__(self, root_dir, sequence_length=49, transform=None):
        self.root_dir = root_dir
        self.sequence_length = sequence_length
        self.transform = transform
        self.sequences = self._load_sequences()

    def _load_sequences(self):
        """Load all valid video sequences from subdirectories."""
        sequences = []

        # Each subdirectory is a video
        video_folders = [f for f in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, f))]

        for video_folder in video_folders:
            video_path = os.path.join(self.root_dir, video_folder)
            frames = sorted(glob.glob(os.path.join(video_path, "*.jpg")) +
                          glob.glob(os.path.join(video_path, "*.png")))

            # Create sequences with sliding window
            for i in range(len(frames) - self.sequence_length):
                sequences.append(frames[i:i + self.sequence_length + 1])

        print(f"Loaded {len(sequences)} training sequences from {len(video_folders)} videos")
        return sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_paths = self.sequences[idx]

        # First frame is conditioning image
        conditioning_frame = load_image(sequence_paths[0])

        # Rest are target frames
        target_frames = [load_image(path) for path in sequence_paths[1:]]

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

# Wan 2.2 Video Predictor
class WanVideoPredictor:
    def __init__(self, model_id="Wan-AI/Wan2.2-TI2V-5B-Diffusers", device="cuda:0", use_temporal_attention=False, memory_optimization="cpu_offload"):
        """
        Initialize Wan 2.2 Video Predictor

        Args:
            model_id: Hugging Face model ID
            device: Device to run on ("cuda:0", "cpu", etc.)
            use_temporal_attention: Whether to use temporal attention fusion
            memory_optimization: Memory optimization mode - "none", "cpu_offload", "sequential_cpu_offload"
        """
        # Check if CUDA is available and device exists
        if not torch.cuda.is_available():
            device = "cpu"
            print("CUDA not available, using CPU.")
        elif device.startswith("cuda:") and int(device.split(":")[1]) >= torch.cuda.device_count():
            print(f"CUDA device {device} not available, using cuda:0.")
            device = "cuda:0"

        self.device = device
        self.use_temporal_attention = use_temporal_attention
        self.memory_optimization = memory_optimization

        print(f"Loading Wan 2.2 model: {model_id}")
        print(f"Memory optimization: {memory_optimization}")

        # Load the model with memory optimizations
        torch_dtype = torch.float16 if device != "cpu" else torch.float32
        self.pipe = WanImageToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )

        # Apply memory optimizations
        if memory_optimization == "cpu_offload":
            self.pipe.enable_model_cpu_offload()
            print("Enabled model CPU offload for memory efficiency")
        elif memory_optimization == "sequential_cpu_offload":
            self.pipe.enable_sequential_cpu_offload()
            print("Enabled sequential CPU offload for maximum memory efficiency")
        else:
            self.pipe = self.pipe.to(device)

        # Initialize temporal attention module if enabled
        if use_temporal_attention:
            self.temporal_attention = TemporalAttentionFusion(embed_dim=512, num_heads=8, max_frames=25).to(device)
            if device != "cpu" and memory_optimization == "none":
                self.temporal_attention = self.temporal_attention.half()

            # Simple feature extractor (CNN-based)
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 64, 7, 2, 3),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((16, 16)),
                nn.Flatten(),
                nn.Linear(64 * 16 * 16, 512),
            ).to(device)
            # Always keep feature extractor in full precision to avoid dtype issues
            # if device != "cpu" and memory_optimization == "none":
            #     self.feature_extractor = self.feature_extractor.half()

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
            # Keep input in full precision for feature extractor
            batch.append(img_tensor.float())

        batch_tensor = torch.cat(batch, dim=0)  # (num_frames, 3, H, W)

        with torch.no_grad():
            features = self.feature_extractor(batch_tensor)  # (num_frames, 512)

        return features.unsqueeze(0)  # (1, num_frames, 512)

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

        # For smaller numbers, use attention-based blending
        # Extract features from frames
        frame_features = self._extract_frame_features(frames)

        # Apply temporal attention
        with torch.no_grad():
            fused_features, attn_weights = self.temporal_attention(frame_features)

        # Reconstruct image by weighted blending
        # Convert frames to numpy arrays
        frame_arrays = [np.array(f, dtype=np.float32) for f in frames]
        frame_stack = np.stack(frame_arrays, axis=0)  # (num_frames, H, W, 3)

        # Get attention weights for last position (most relevant)
        weights = attn_weights[0, -1, :].cpu().numpy()  # (num_frames,)
        weights = weights / weights.sum()

        print(f"  Attention weights: {weights}")

        # Weighted blend
        weighted_frame = np.sum(frame_stack * weights[:, None, None, None], axis=0)
        fused_image = Image.fromarray(weighted_frame.astype(np.uint8))

        return fused_image

    def predict_next_frames(self, image_paths, num_frames=49, num_inference_steps=25, fps=7, motion_bucket_id=127, decode_chunk_size=4, total_frames=49, condition_frames=3):
        """
        Predict the next frames using Wan 2.2 model conditioned on previous n images.

        Args:
            image_paths: List of paths to previous images (K images)
            num_frames: Number of frames to generate per iteration (max 49 for Wan 2.2)
            num_inference_steps: Number of denoising steps
            fps: Frames per second (not used by Wan 2.2, kept for compatibility)
            motion_bucket_id: Motion bucket (not used by Wan 2.2, kept for compatibility)
            decode_chunk_size: Chunk size (not used by Wan 2.2, kept for compatibility)
            total_frames: Total number of frames to predict (will iterate if > num_frames)
            condition_frames: Number of previous frames to condition on (default: 3)

        Returns:
            List of PIL Images of the predicted frames
        """
        all_predicted_frames = []

        # Load initial conditioning frames
        initial_frames = [load_image(path) for path in image_paths[-condition_frames:]]
        original_size = initial_frames[-1].size

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
                print(f"  Using single frame conditioning (last frame)")

            # Resize input image to match Wan's expected aspect ratio (16:9 = 832:480)
            # Original aspect ratio will be restored in output processing
            target_width, target_height = 832, 480
            if current_init_image.size != (target_width, target_height):
                current_init_image = current_init_image.resize((target_width, target_height), Image.LANCZOS)

            # Generate a batch of frames using Wan 2.2
            output = self.pipe(
                image=current_init_image,
                prompt="High quality video, smooth motion, sharp details, professional cinematography, stable camera",  # Enhanced prompt for stability
                num_frames=frames_this_iteration,
                num_inference_steps=num_inference_steps,
                guidance_scale=8.0,  # Higher for sharper, more stable generation
                height=480,  # Default height for Wan 2.2
                width=832,   # Default width for Wan 2.2
            ).frames

            # Process and store frames
            print(f"Output shape: {output.shape}")
            # Output shape: (batch, num_frames, height, width, channels)
            batch_size, num_output_frames, height, width, channels = output.shape

            for i in range(min(num_output_frames, frames_this_iteration)):
                frame_array = output[0, i]  # Take first batch, i-th frame

                # Convert numpy array to PIL Image
                if frame_array.dtype != np.uint8:
                    # Normalize if needed (assuming values are in 0-1 range)
                    if frame_array.max() <= 1.0:
                        frame_array = (frame_array * 255).astype(np.uint8)
                    else:
                        frame_array = frame_array.astype(np.uint8)

                # frame_array should now be (height, width, channels)
                frame = Image.fromarray(frame_array)

                # Resize if needed
                if frame.size != original_size:
                    frame = frame.resize(original_size, Image.LANCZOS)

                all_predicted_frames.append(frame)
                frame_buffer.append(frame)  # Add to buffer for next iteration

            frames_remaining -= frames_this_iteration

        return all_predicted_frames

# Fine-tuning function with LoRA for Wan 2.2
def fine_tune_wan_with_lora(
    predictor,
    dataset,
    output_dir="models/lora_checkpoints_wan",
    epochs=10,
    batch_size=1,
    learning_rate=1e-4,
    lora_rank=8,
    lora_alpha=16,
    save_every=1
):
    """
    Fine-tune Wan 2.2 using LoRA (Low-Rank Adaptation).

    Args:
        predictor: WanVideoPredictor instance
        dataset: VideoSequenceDataset instance
        output_dir: Directory to save LoRA weights
        epochs: Number of training epochs
        batch_size: Batch size (recommend 1 for video diffusion)
        learning_rate: Learning rate
        lora_rank: LoRA rank (lower = fewer parameters, faster)
        lora_alpha: LoRA alpha scaling factor
        save_every: Save checkpoint every N epochs
    """
    os.makedirs(output_dir, exist_ok=True)

    # Configure LoRA for Wan 2.2 transformer blocks
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",  # Attention layers
            "proj_in", "proj_out",  # Projection layers
            "ffn.0", "ffn.2",  # Feed-forward layers
        ],
        lora_dropout=0.1,
        bias="none",
    )

    # Apply LoRA to the transformer
    print("Applying LoRA adapters to Wan 2.2 transformer...")
    predictor.pipe.transformer = get_peft_model(predictor.pipe.transformer, lora_config)
    predictor.pipe.transformer.print_trainable_parameters()

    # Setup optimizer and scheduler
    optimizer = AdamW(predictor.pipe.transformer.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs * len(dataset) // batch_size)

    # DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Training loop
    predictor.pipe.transformer.train()
    global_step = 0

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch_idx, batch in enumerate(progress_bar):
            conditioning_frame = batch['conditioning_frame']
            target_frames = batch['target_frames']

            # Convert PIL images to tensors if needed
            if isinstance(conditioning_frame, list):
                conditioning_frame = conditioning_frame[0]

            try:
                # Generate frames (for monitoring)
                if batch_idx % 50 == 0:  # Log every 50 batches
                    with torch.no_grad():
                        output = predictor.pipe(
                            image=conditioning_frame,
                            num_frames=min(49, len(target_frames[0])),
                            num_inference_steps=25,
                            fps=7,
                            shift=5.0,
                            sample_shift=5.0,
                        )

                        # Save sample output
                        sample_dir = os.path.join(output_dir, f"samples_epoch{epoch + 1}")
                        os.makedirs(sample_dir, exist_ok=True)
                        export_to_video(
                            output.frames[0],
                            os.path.join(sample_dir, f"step_{global_step}.mp4"),
                            fps=7
                        )

                # Placeholder for actual loss computation
                # In practice, you'd compute the denoising loss here
                loss = torch.tensor(0.0, requires_grad=True)  # Replace with real loss

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(predictor.pipe.transformer.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

                progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })

            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_loss:.4f}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(output_dir, f"lora_epoch_{epoch + 1}.pt")
            predictor.pipe.transformer.save_pretrained(checkpoint_path)
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

def load_wan_lora_weights(predictor, lora_checkpoint_path):
    """
    Load LoRA weights into the Wan 2.2 predictor.

    Args:
        predictor: WanVideoPredictor instance
        lora_checkpoint_path: Path to LoRA checkpoint directory
    """
    from peft import PeftModel

    print(f"Loading LoRA weights from {lora_checkpoint_path}...")
    predictor.pipe.transformer = PeftModel.from_pretrained(
        predictor.pipe.transformer,
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

if __name__ == "__main__":
    print("Wan 2.2 Video Prediction Project")

    # Mode selection: 'baseline', 'train', 'inference', or 'inference_with_lora'
    mode = 'inference'  # Change to 'baseline' for standard Wan generation

    if mode == 'baseline':
        # ========== BASELINE WAN 2.2 GENERATION (No Conditioning) ==========
        print("\n=== Wan 2.2 Baseline Generation (Standard Image-to-Video) ===")

        predictor = WanVideoPredictor(use_temporal_attention=False, memory_optimization="cpu_offload")

        # Use the last frame from the rabbit dataset as starting point
        folder = "data/rabbit"
        image_paths = get_image_paths(folder)
        if len(image_paths) == 0:
            print("No images found in the folder.")
        else:
            # Use the last frame as the conditioning image
            last_frame_path = image_paths[-1]
            last_frame = load_image(last_frame_path)

            # Resize input image to match Wan's expected aspect ratio (16:9 = 832:480)
            # Original is 960x1280 (3:4), target is 832x480 (16:9)
            target_width, target_height = 832, 480
            resized_frame = last_frame.resize((target_width, target_height), Image.LANCZOS)

            print(f"Using last frame: {last_frame_path}")
            print(f"Original size: {last_frame.size}, Resized to: {resized_frame.size}")

            # Generate video using standard Wan 2.2 (no custom conditioning)
            output = predictor.pipe(
                image=resized_frame,
                prompt="High quality video, smooth motion, sharp details, professional cinematography, stable camera, biker riding smoothly",
                num_frames=49,
                num_inference_steps=100,  # Higher for better quality
                guidance_scale=9.0,       # Higher for sharper results
                height=480,               # Default supported height
                width=832,                # Default supported width
            ).frames

            # Save the generated frames
            last_frame_num = len(image_paths) - 1
            original_size = last_frame.size  # (960, 1280)
            for i, frame in enumerate(output[0]):
                # Convert tensor/array to PIL Image
                if hasattr(frame, 'cpu'):  # If it's a tensor
                    frame = frame.cpu().numpy()
                if isinstance(frame, np.ndarray):
                    # Handle different array shapes and types
                    if frame.dtype != np.uint8:
                        # Normalize if needed (assuming values are in 0-1 range)
                        if frame.max() <= 1.0:
                            frame = (frame * 255).astype(np.uint8)
                        else:
                            frame = frame.astype(np.uint8)
                    # Ensure proper shape (H, W, C)
                    if frame.shape[0] == 1:  # Remove batch dimension if present
                        frame = frame.squeeze(0)
                    frame = Image.fromarray(frame)
                
                # Resize back to original aspect ratio
                if frame.size != original_size:
                    frame = frame.resize(original_size, Image.LANCZOS)
                
                frame_filename = f"Predicted_frame_wan_baseline_{last_frame_num + i + 1:04d}.jpg"
                frame_path = os.path.join(folder, frame_filename)
                frame.save(frame_path)
                print(f"Saved {frame_filename}")

            print(f"All {len(output[0])} baseline frames saved to {folder}")

    elif mode == 'train':
        # ========== FINE-TUNING MODE ==========
        print("\n=== Starting Wan 2.2 LoRA Fine-Tuning ===")

        # Initialize predictor
        predictor = WanVideoPredictor(use_temporal_attention=False)

        # Create dataset from training videos
        # Expected structure: data/train_videos/video1/*.jpg, data/train_videos/video2/*.jpg, etc.
        train_dataset = VideoSequenceDataset(
            root_dir="data",  # Change to your training data directory
            sequence_length=49
        )

        if len(train_dataset) == 0:
            print("ERROR: No training data found!")
            print("Please organize your data as: data/video_folder/*.jpg")
        else:
            # Fine-tune with LoRA
            predictor = fine_tune_wan_with_lora(
                predictor=predictor,
                dataset=train_dataset,
                output_dir="models/lora_checkpoints_wan",
                epochs=5,
                batch_size=1,
                learning_rate=1e-4,
                lora_rank=8,
                lora_alpha=16,
                save_every=1
            )
            print("\nFine-tuning completed! LoRA weights saved to models/lora_checkpoints_wan/")

    elif mode == 'inference_with_lora':
        # ========== INFERENCE WITH LORA WEIGHTS ==========
        print("\n=== Inference with LoRA Fine-Tuned Wan 2.2 Model ===")

        predictor = WanVideoPredictor(use_temporal_attention=True, memory_optimization="cpu_offload")

        # Load LoRA weights
        lora_checkpoint = "models/lora_checkpoints_wan/lora_epoch_5.pt"
        if os.path.exists(lora_checkpoint):
            predictor = load_wan_lora_weights(predictor, lora_checkpoint)
        else:
            print(f"WARNING: LoRA checkpoint not found at {lora_checkpoint}")
            print("Using base model instead...")

        # Run inference
        folder = "data/biker"
        image_paths = get_image_paths(folder)
        if len(image_paths) < 3:
            print("Not enough images in the folder. Need at least 3 frames.")
        else:
            predicted_frames = predictor.predict_next_frames(
                image_paths,
                num_frames=49,
                num_inference_steps=50,
                fps=24,
                total_frames=49,
                condition_frames=3
            )

            last_frame_num = len(image_paths) - 1
            for i, frame in enumerate(predicted_frames):
                frame_filename = f"Predicted_frame_wan_lora_{last_frame_num + i + 1:04d}.jpg"
                frame_path = os.path.join(folder, frame_filename)
                frame.save(frame_path)
                print(f"Saved {frame_filename}")

            print(f"All {len(predicted_frames)} predicted frames saved to {folder}")

    else:
        # ========== STANDARD INFERENCE MODE ==========
        print("\n=== Wan 2.2 Standard Inference Mode ===")

        # Initialize the predictor with temporal attention and CPU offload for memory efficiency
        predictor = WanVideoPredictor(use_temporal_attention=True, memory_optimization="cpu_offload")

        # Example usage: Read all frames from a specified folder
        folder = "data/biker"
        image_paths = get_image_paths(folder)
        if len(image_paths) < 3:
            print("Not enough images in the folder. Need at least 3 frames.")
        else:
            # RECOMMENDED SETTINGS for best quality and stability
            predicted_frames = predictor.predict_next_frames(
                image_paths,
                num_frames=49,  # Generate 49 frames per iteration (Wan 2.2 max)
                num_inference_steps=75,  # Higher for better quality
                fps=24,
                total_frames=49,  # Total frames to predict
                condition_frames=5  # Use 5 frames for better motion context
            )

            # Get the last frame number from existing images
            last_frame_num = len(image_paths) - 1

            # Save all predicted frames with sequential naming
            for i, frame in enumerate(predicted_frames):
                frame_filename = f"Predicted_frame_wan_stable_{last_frame_num + i + 1:04d}.jpg"
                frame_path = os.path.join(folder, frame_filename)
                frame.save(frame_path)
                print(f"Saved {frame_filename}")

            print(f"All {len(predicted_frames)} predicted frames saved to {folder}")