# Chronos
<<<<<<< HEAD

This project predicts timeseries metrics by encoding them into image sequences for generative diffusion models. It converts multiple aligned timeseries datasets into frames (with metrics as pixel bands), feeds K past frames as input, and generates future frames via sampling (e.g., 50 denoising steps). The probabilistic nature allows estimating uncertainty in predictions.

## Features

- Encodes timeseries metrics into grayscale image frames with adaptive pixel bands.
- Uses pre-trained diffusion models (e.g., SVD-XT) with history band conditioning.
- Supports uploading, merging, and reconstructing metrics from CSVs.
- Easy to switch to other models like Stable Diffusion 3 or WAN.

## Project Structure

- `src/`: Source code for metrics handling, frame generation, and model scripts.
- `data/`: Directory for datasets and context-specific frames.
- `models/`: Saved model checkpoints.
- `notebooks/`: Jupyter notebooks for experimentation.
- `tests/`: Unit tests.

## Installation

1. Clone the repository.
2. Create virtual environment: `python -m venv venv`
3. Activate: `.\venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`

## Usage

Upload metrics: Use `src/metrics_handler.py` to upload CSVs for a context.

Generate frames: Run `src/timeseries_frame_generator.py` to create image sequences.

Predict: Modify model scripts to predict future frames from past ones.

Reconstruct: Use `src/timeseries_reconstructor.py` to convert predicted frames back to metrics.

## Models

- Default: SVD-XT with history band for temporal conditioning.
- To use Stable Diffusion 3: Change model_id to "stabilityai/stable-diffusion-3-medium-diffusers"
- For video models like WAN: Use appropriate pipeline.

## Fine-tuning

Implement fine-tuning in model scripts with your frame datasets.
=======
Chronos Appplication
>>>>>>> 29baa784d10555331d6b8fbbd4d456948a4097e0
