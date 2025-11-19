import unittest
import os
import tempfile
import shutil
import pandas as pd
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from timeseries_frame_generator import generate_timeseries_frames, process_context

class TestTimeseriesFrameGenerator(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_base = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_mock_metadata(self, context, metrics):
        context_dir = os.path.join(self.output_base, context)
        os.makedirs(context_dir, exist_ok=True)
        metadata = {
            "metrics": {m: {"max": 100, "min": 0} for m in metrics},
            "last_updated": "2023-01-01T00:00:00",
            "band_height": 64
        }
        with open(os.path.join(context_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f)

    def test_generate_timeseries_frames(self):
        context = 'test_context'
        self.create_mock_metadata(context, ['metric1', 'metric2'])
        # Create mock merged_df
        times = pd.date_range('2023-01-01', periods=2, freq='1min')
        data = {'time': times, 'metric1': [10, 20], 'metric2': [5, 15]}
        merged_df = pd.DataFrame(data)
        result = generate_timeseries_frames(context, merged_df, self.output_base)
        self.assertIn('Generated 2 frames', result)
        # Check if frames exist
        frames_dir = os.path.join(self.output_base, context, 'timeseries_frames')
        files = os.listdir(frames_dir)
        self.assertEqual(len(files), 2)
        self.assertTrue(any('frame_0000.png' in f for f in files))
        self.assertTrue(any('frame_0001.png' in f for f in files))

    def test_process_context_no_data(self):
        context = 'empty_context'
        result = process_context(context, self.output_base)
        self.assertIn('Error: Context not found', result)

if __name__ == '__main__':
    unittest.main()