import unittest
import os
import tempfile
import shutil
import pandas as pd
import json
import numpy as np
from PIL import Image
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from timeseries_reconstructor import reconstruct_timeseries

class TestTimeseriesReconstructor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_base = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_mock_metadata(self, context, metrics, times, band_height=64):
        context_dir = os.path.join(self.output_base, context)
        os.makedirs(context_dir, exist_ok=True)
        metadata = {
            "metrics": {m: {"max": 100, "min": 0} for m in metrics},
            "times": times,
            "band_height": band_height,
            "last_updated": "2023-01-01T00:00:00"
        }
        with open(os.path.join(context_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def create_mock_frames(self, context, num_frames, metrics, band_height=64):
        frames_dir = os.path.join(self.output_base, context, "timeseries_frames")
        os.makedirs(frames_dir, exist_ok=True)
        target_height = 576 - band_height
        target_width = 1024
        N = len(metrics)
        band_h = target_height // N
        remainder = target_height % N
        
        for i in range(num_frames):
            img = np.zeros((target_height, target_width), dtype=np.uint8)
            current_y = 0
            for j in range(N):
                h = band_h + (1 if j < remainder else 0)
                # Set intensity based on metric index for testing
                intensity = int((j / (N-1)) * 255) if N > 1 else 127
                img[current_y:current_y+h, :] = intensity
                current_y += h
            pil_img = Image.fromarray(img, mode='L')
            pil_img.save(os.path.join(frames_dir, f"frame_{i:04d}.png"))

    def test_reconstruct_timeseries(self):
        context = 'test_context'
        metrics = ['metric1', 'metric2']
        times = ['2023-01-01T00:00:00', '2023-01-01T00:01:00']
        self.create_mock_metadata(context, metrics, times)
        self.create_mock_frames(context, 2, metrics)
        result = reconstruct_timeseries(context, self.output_base)
        self.assertIn('Reconstructed timeseries saved', result)
        # Check output CSV
        csv_path = os.path.join(self.output_base, context, 'reconstructed_timeseries.csv')
        self.assertTrue(os.path.exists(csv_path))
        df = pd.read_csv(csv_path)
        self.assertEqual(len(df), 2)
        self.assertIn('time', df.columns)
        self.assertIn('metric1', df.columns)
        self.assertIn('metric2', df.columns)
        # Check values (approximate due to averaging)
        self.assertAlmostEqual(df['metric1'].iloc[0], 0, delta=10)  # Low intensity -> low value
        self.assertAlmostEqual(df['metric2'].iloc[0], 100, delta=10)  # High intensity -> high value

if __name__ == '__main__':
    unittest.main()