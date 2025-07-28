import unittest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

# Mock the missing model imports
sys.modules['models.sic.thickness.lstm.lstm_thickness_unshuffle_skip1_v2'] = MagicMock()
sys.modules['models.sic.doping.lstm.lstm_doping_unshuffle_skip1'] = MagicMock()
sys.modules['torch'] = MagicMock()
sys.modules['torch.nn'] = MagicMock()
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['sklearn.preprocessing'] = MagicMock()
sys.modules['matplotlib.pyplot'] = MagicMock()
sys.modules['matplotlib.colors'] = MagicMock()
sys.modules['scipy.interpolate'] = MagicMock()
sys.modules['joblib'] = MagicMock()

class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions in the app"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock config loading
        with patch('builtins.open'):
            with patch('json.load', return_value=[]):
                with patch('os.path.join'):
                    # Import specific functions we want to test
                    from app import convert_time_string, extract_model_inputs, get_base64
                    self.convert_time_string = convert_time_string
                    self.extract_model_inputs = extract_model_inputs
                    self.get_base64 = get_base64

    def test_convert_time_string_mm_ss(self):
        """Test time string conversion for mm:ss format"""
        result = self.convert_time_string("05:30")
        self.assertEqual(result, 330.0)  # 5*60 + 30 = 330

    def test_convert_time_string_hh_mm_ss(self):
        """Test time string conversion for hh:mm:ss format"""
        result = self.convert_time_string("01:05:30")
        self.assertEqual(result, 3930.0)  # 1*3600 + 5*60 + 30 = 3930

    def test_convert_time_string_invalid_format(self):
        """Test time string conversion with invalid format"""
        with self.assertRaises(ValueError):
            self.convert_time_string("invalid:time:format:extra")

    def test_extract_model_inputs_basic(self):
        """Test basic model input extraction"""
        table_data = [
            ["header1", "header2", "header3", "header4", "header5"],
            ["row1", "val1", "val2", "val3", "100"],
            ["row2", "val1", "val2", "val3", "200"]
        ]
        input_config = [
            {"rows": [2], "column_index": 5},  # Row 2 (1-indexed) = index 1 in 0-indexed array
            {"rows": [3], "column_index": 5}   # Row 3 (1-indexed) = index 2 in 0-indexed array
        ]

        result = self.extract_model_inputs(table_data, input_config, header_exists=True)
        self.assertEqual(result, [100.0, 200.0])

    def test_extract_model_inputs_with_time_string(self):
        """Test model input extraction with time string"""
        table_data = [
            ["header1", "header2", "header3", "header4", "header5"],
            ["row1", "val1", "val2", "val3", "05:30"],
            ["row2", "val1", "val2", "val3", "200"]
        ]
        input_config = [
            {"rows": [2], "column_index": 5},  # Row 2 (1-indexed) = index 1 in 0-indexed array
            {"rows": [3], "column_index": 5}   # Row 3 (1-indexed) = index 2 in 0-indexed array
        ]

        result = self.extract_model_inputs(table_data, input_config, header_exists=True)
        self.assertEqual(result, [330.0, 200.0])  # 05:30 = 330 seconds    def test_extract_model_inputs_error_handling(self):
        """Test model input extraction with invalid data"""
        table_data = [
            ["header1", "header2"],
            ["row1", "val1"]
        ]
        input_config = [
            {"rows": [1], "column_index": 10}  # Column doesn't exist
        ]

        result = self.extract_model_inputs(table_data, input_config, header_exists=True)
        self.assertEqual(result, [0.0])  # Should return 0.0 for errors

    @patch('builtins.open', create=True)
    def test_get_base64(self, mock_open):
        """Test base64 encoding of file"""
        # Mock file content
        mock_open.return_value.__enter__.return_value.read.return_value = b'test_data'

        result = self.get_base64('fake_path')
        # base64 of 'test_data' is 'dGVzdF9kYXRh'
        self.assertEqual(result, 'dGVzdF9kYXRh')

if __name__ == '__main__':
    unittest.main()
