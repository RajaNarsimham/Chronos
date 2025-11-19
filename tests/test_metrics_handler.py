import unittest
import os
import tempfile
import shutil
import pandas as pd
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from metrics_handler import upload_metric_csv, delete_metric, merge_context_data

class TestMetricsHandler(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.output_base = self.test_dir

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def create_sample_csv(self, times, values, filename):
        df = pd.DataFrame({'time': times, 'metric': values})
        path = os.path.join(self.test_dir, filename)
        df.to_csv(path, index=False)
        return path

    def test_upload_metric_csv_success(self):
        times = ['2023-01-01 00:00:00', '2023-01-01 00:01:00']
        values = [10, 20]
        csv_path = self.create_sample_csv(times, values, 'test.csv')
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        self.assertEqual(result, 'Upload successful')
        # Check metadata
        metadata_path = os.path.join(self.output_base, 'test_context', 'metadata.json')
        self.assertTrue(os.path.exists(metadata_path))
        # Check raw file
        raw_dir = os.path.join(self.output_base, 'test_context', 'raw_uploads')
        files = os.listdir(raw_dir)
        self.assertEqual(len(files), 1)

    def test_upload_metric_csv_overlap_error(self):
        # First upload
        times = ['2023-01-01 00:00:00', '2023-01-01 00:01:00']
        values = [10, 20]
        csv_path = self.create_sample_csv(times, values, 'test1.csv')
        upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        # Second upload with overlap
        times2 = ['2023-01-01 00:00:30', '2023-01-01 00:01:30']
        values2 = [15, 25]
        csv_path2 = self.create_sample_csv(times2, values2, 'test2.csv')
        result = upload_metric_csv('test_context', 'test_metric', csv_path2, 100, 0, self.output_base)
        self.assertIn('Error: Date range overlaps', result)

    def test_delete_metric(self):
        # Upload first
        times = ['2023-01-01 00:00:00', '2023-01-01 00:01:00']
        values = [10, 20]
        csv_path = self.create_sample_csv(times, values, 'test.csv')
        upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        # Delete
        result = delete_metric('test_context', 'test_metric', self.output_base)
        self.assertEqual(result, 'Delete successful')
        # Check metadata
        metadata_path = os.path.join(self.output_base, 'test_context', 'metadata.json')
        with open(metadata_path, 'r') as f:
            import json
            metadata = json.load(f)
        self.assertNotIn('test_metric', metadata['metrics'])

    def test_merge_context_data_success(self):
        # Upload two metrics
        times = ['2023-01-01 00:00:00', '2023-01-01 00:01:00']
        values1 = [10, 20]
        csv_path1 = self.create_sample_csv(times, values1, 'metric1.csv')
        upload_metric_csv('test_context', 'metric1', csv_path1, 100, 0, self.output_base)
        values2 = [5, 15]
        csv_path2 = self.create_sample_csv(times, values2, 'metric2.csv')
        upload_metric_csv('test_context', 'metric2', csv_path2, 50, 0, self.output_base)
        # Merge
        merged_df, msg = merge_context_data('test_context', self.output_base)
        self.assertEqual(msg, 'Merge successful')
        self.assertIsNotNone(merged_df)
        self.assertEqual(len(merged_df), 2)
        self.assertIn('metric1', merged_df.columns)
        self.assertIn('metric2', merged_df.columns)

    def test_upload_metric_csv_invalid_columns(self):
        # CSV without required columns
        df = pd.DataFrame({'timestamp': ['2023-01-01'], 'value': [10]})
        csv_path = os.path.join(self.test_dir, 'invalid.csv')
        df.to_csv(csv_path, index=False)
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        self.assertIn('Error: CSV must have', result)

    def test_upload_metric_csv_invalid_time(self):
        # Invalid time format
        df = pd.DataFrame({'time': ['invalid'], 'metric': [10]})
        csv_path = os.path.join(self.test_dir, 'invalid_time.csv')
        df.to_csv(csv_path, index=False)
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        self.assertIn('Error: Invalid time format', result)

    def test_upload_metric_csv_non_numeric_metric(self):
        # Non-numeric metric
        df = pd.DataFrame({'time': ['2023-01-01 00:00:00'], 'metric': ['text']})
        csv_path = os.path.join(self.test_dir, 'non_numeric.csv')
        df.to_csv(csv_path, index=False)
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        self.assertIn('Error: Metric column must be numeric', result)

    def test_upload_metric_csv_missing_values(self):
        # Missing metric values
        df = pd.DataFrame({'time': ['2023-01-01 00:00:00'], 'metric': [None]})
        csv_path = os.path.join(self.test_dir, 'missing.csv')
        df.to_csv(csv_path, index=False)
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 100, 0, self.output_base)
        self.assertIn('Error: Metric column contains missing values', result)

    def test_upload_metric_csv_invalid_max_min(self):
        times = ['2023-01-01 00:00:00']
        values = [10]
        csv_path = self.create_sample_csv(times, values, 'test.csv')
        result = upload_metric_csv('test_context', 'test_metric', csv_path, 0, 100, self.output_base)  # max < min
        self.assertIn('Error: max_val must be greater than min_val', result)

if __name__ == '__main__':
    unittest.main()