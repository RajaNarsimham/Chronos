# LoRA Fine-Tuning Guide for SVD-XT

## Overview

This implementation adds **LoRA (Low-Rank Adaptation)** fine-tuning capabilities to Stable Video Diffusion XT, allowing you to efficiently adapt the model to your specific video data without training the entire model.

## What is LoRA?

LoRA is a parameter-efficient fine-tuning method that:
- **Reduces trainable parameters by ~99%** (from billions to millions)
- **Requires less GPU memory** (can fine-tune on consumer GPUs)
- **Trains faster** than full fine-tuning
- **Preserves base model knowledge** while adapting to new data

## Prerequisites

```bash
pip install peft  # Already installed
```

## Data Preparation

### Directory Structure

Organize your training videos as separate folders:

```
data/
├── video1/
│   ├── frame_0001.jpg
│   ├── frame_0002.jpg
│   ├── ...
│   └── frame_0050.jpg
├── video2/
│   ├── frame_0001.jpg
│   ├── ...
└── video3/
    ├── frame_0001.jpg
    └── ...
```

### Requirements:
- **Minimum frames per video**: 15 frames (1 conditioning + 14 target)
- **Recommended**: 30+ frames per video for better learning
- **Format**: JPG or PNG
- **Resolution**: Will be automatically resized by the model

## Usage

### 1. Fine-Tuning (Training)

Edit `src/svd-xt.py` and set:

```python
mode = 'train'
```

Configure training parameters:

```python
predictor = fine_tune_with_lora(
    predictor=predictor,
    dataset=train_dataset,
    output_dir="models/lora_checkpoints",
    epochs=5,              # Number of epochs (start with 3-5)
    batch_size=1,          # Keep at 1 for video diffusion
    learning_rate=1e-4,    # Learning rate (1e-4 to 5e-5)
    lora_rank=8,           # LoRA rank (4-16, higher = more parameters)
    lora_alpha=16,         # Scaling factor (typically 2x rank)
    save_every=1           # Save checkpoint every N epochs
)
```

Run training:

```bash
python src/svd-xt.py
```

### 2. Inference with LoRA Weights

After training, use your fine-tuned model:

```python
mode = 'inference_with_lora'
```

Specify checkpoint path:

```python
lora_checkpoint = "models/lora_checkpoints/lora_epoch_5.pt"
```

## Parameters Guide

### LoRA Parameters

| Parameter | Description | Recommended Values | Effect |
|-----------|-------------|-------------------|--------|
| `lora_rank` | Rank of LoRA matrices | 4-16 | Higher = more expressive but slower |
| `lora_alpha` | Scaling factor | 2x rank (e.g., 16 for rank=8) | Controls adaptation strength |
| `learning_rate` | Optimization step size | 1e-4 to 5e-5 | Too high = instability, too low = slow |
| `epochs` | Training iterations | 3-10 | More epochs = more adaptation |

### Video Generation Parameters

| Parameter | Description | Recommended Values |
|-----------|-------------|-------------------|
| `condition_frames` | Frames to condition on | 3-5 for attention, 10+ uses motion |
| `num_frames` | Frames per batch | 14 (max for SVD-XT) |
| `num_inference_steps` | Denoising steps | 25-50 (higher = better quality) |
| `motion_bucket_id` | Motion intensity | 50-200 (127 = default) |

## Tips for Better Results

### 1. Data Quality
- Use **high-quality, consistent** videos
- Ensure **good lighting** and **stable camera**
- **Diverse motion patterns** help generalization
- Minimum **10 videos** for meaningful fine-tuning

### 2. Training
- Start with **small lora_rank (4-8)** to avoid overfitting
- Monitor **sample outputs** during training (saved to `models/lora_checkpoints/samples_epoch*/`)
- If loss plateaus, **decrease learning rate**
- If outputs are noisy, **reduce lora_alpha**

### 3. Inference
- Use **condition_frames=3** for clean predictions without ghosting
- Set **condition_frames=10+** for motion extrapolation (automatic)
- **num_inference_steps=50** gives best quality but slower

## Output Structure

```
models/lora_checkpoints/
├── lora_epoch_1.pt          # LoRA weights after epoch 1
├── config_epoch_1.json      # Training configuration
├── lora_epoch_5.pt          # Final weights
├── config_epoch_5.json
└── samples_epoch1/          # Sample videos during training
    ├── step_0.mp4
    ├── step_50.mp4
    └── ...
```

## Troubleshooting

### Out of Memory (OOM)
- Reduce `lora_rank` (e.g., 4 instead of 8)
- Keep `batch_size=1`
- Use `decode_chunk_size=2` or `4`
- Enable gradient checkpointing (add to code)

### Poor Quality Results
- **Increase epochs** (5-10)
- **Collect more training data** (20+ videos)
- **Increase lora_rank** (12-16)
- Check **data quality** (resolution, lighting)

### Training is Slow
- Use **smaller lora_rank** (4)
- Reduce **num_inference_steps** in validation
- Use **fewer validation samples** (modify code)

## Advanced: Modifying the Training Loop

The current implementation is a **placeholder** for the full training loop. For production use, you need to:

1. **Implement proper denoising loss**:
```python
# Add timestep sampling
timesteps = torch.randint(0, scheduler.config.num_train_timesteps, (batch_size,))

# Add noise to latents
noisy_latents = scheduler.add_noise(latents, noise, timesteps)

# Predict noise
noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

# Compute loss
loss = F.mse_loss(noise_pred, noise)
```

2. **Add validation metrics** (PSNR, SSIM, FVD)
3. **Implement early stopping**
4. **Add mixed precision training** (`torch.cuda.amp`)

## Example Workflow

```python
# 1. Prepare data
# Organize 15+ videos in data/video_*/

# 2. Train
mode = 'train'
epochs = 5
lora_rank = 8

# 3. Inference
mode = 'inference_with_lora'
lora_checkpoint = "models/lora_checkpoints/lora_epoch_5.pt"

# 4. Compare
# Run both base model and LoRA model to see improvements
```

## Comparison: Base vs LoRA

| Aspect | Base Model | LoRA Fine-Tuned |
|--------|-----------|-----------------|
| Training Time | N/A | 2-6 hours (5 epochs) |
| GPU Memory | 10-12 GB | 12-16 GB |
| Parameters Trained | 0 | ~1-2% of model |
| Domain Adaptation | Generic | Specialized |
| Quality on Custom Data | Good | Better |

## Next Steps

1. **Collect domain-specific videos** (your use case)
2. **Start with 3-5 epochs** and `lora_rank=8`
3. **Evaluate outputs** visually
4. **Adjust hyperparameters** based on results
5. **Iterate** with more data if needed

## References

- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [PEFT Library](https://github.com/huggingface/peft)
- [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt)
