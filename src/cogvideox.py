import torch
from PIL import Image
import numpy as np
from diffusers import CogVideoXVideoToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_image
import glob
from scipy.ndimage import gaussian_filter
import os

class CogVideoXPredictor:
    def __init__(self, model_id="THUDM/CogVideoX-5b", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        print(f"Loading CogVideoX V2V on {device}...")

        dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        ).to(device)

        self.pipe.scheduler = CogVideoXDPMScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_model_cpu_offload()
        self.pipe.vae.enable_slicing()
        self.pipe.vae.enable_tiling()
        print("Model loaded successfully!")

    def predict_next_frames(
        self,
        image_paths,
        prompt="",
        num_frames=24,
        num_inference_steps=50,
        guidance_scale=4.0,
        cond_frames=5,
        strength=0.45,
        seed=42,
    ):
        if len(image_paths) < cond_frames:
            raise ValueError(f"Need {cond_frames} images! Got {len(image_paths)}")

        # Load last cond_frames inputs; resize to model's expected resolution for better quality
        cond_images = [load_image(p).convert("RGB").resize((720, 480), Image.LANCZOS) for p in image_paths[-cond_frames:]]

        print(f"Using {len(cond_images)} conditioning frames → generating {num_frames} new ones")

        gen = torch.Generator(device=self.device).manual_seed(seed)
        current_cond = cond_images.copy()
        outputs = []

        for _ in range(num_frames):
            out_frames = self.pipe(
                video=current_cond,
                prompt="",  # minimal prompt to reduce drift
                negative_prompt="noise, blur, artifacts, flicker, watermark, distortion",
                num_inference_steps=num_inference_steps,
                guidance_scale=2.0,  # lower to reduce prompt influence
                strength=0.3,           # lower for closer to input
                generator=gen,
                use_dynamic_cfg=False,
            ).frames[0]

            next_frame = out_frames[-1]
            outputs.append(next_frame)
            current_cond = current_cond[1:] + [next_frame]  # slide window: always last 5 frames

        smoothed = self._temporal_smoothing(outputs, alpha=0.1)
        denoised = self._denoise_frames(smoothed, sigma=0.2)
        return denoised[:num_frames]

    def _temporal_smoothing(self, frames, alpha=0.2):
        if len(frames) <= 1:
            return frames
        smoothed = [frames[0]]
        for f in frames[1:]:
            prev = np.array(smoothed[-1], dtype=np.float32)
            curr = np.array(f, dtype=np.float32)
            blended = (1 - alpha) * curr + alpha * prev
            smoothed.append(Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8)))
        return smoothed

    def _denoise_frames(self, frames, sigma=0.4):
        denoised = []
        for f in frames:
            arr = np.array(f, dtype=np.float32)
            arr = gaussian_filter(arr, sigma=(sigma, sigma, 0) if arr.ndim == 3 else sigma)
            denoised.append(Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)))
        return denoised


def get_image_paths(folder, ext="*.jpg"):
    return sorted(glob.glob(os.path.join(folder, ext)))


if __name__ == "__main__":
    predictor = CogVideoXPredictor()

    folder = "data/rabbit"
    paths = get_image_paths(folder)

    if len(paths) < 5:
        print(f"Only found {len(paths)} images. Need at least 5!")
    else:
        print(f"Found {len(paths)} images. Starting prediction...")
        frames = predictor.predict_next_frames(
            image_paths=paths,
            prompt="",  # minimal
            num_frames=5,
            num_inference_steps=50,
            guidance_scale=1,  # lowered
            cond_frames=5,
            strength=0.3,  # lowered
            seed=42,
        )

        start_idx = len(paths)
        os.makedirs(folder, exist_ok=True)
        for i, f in enumerate(frames):
            name = f"Predicted_frame_{start_idx + i + 1:04d}.jpg"
            f.save(os.path.join(folder, name))
            print(f"Saved {name}")

        print(f"SUCCESS! Saved {len(frames)} frames.")