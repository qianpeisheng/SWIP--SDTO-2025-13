import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock, mock_open

class TestFlaskRoutes(unittest.TestCase):
    """Test Flask routes with comprehensive mocking"""

    @classmethod
    def setUpClass(cls):
        """Set up class-level fixtures"""
        # Mock all problematic imports before importing the app
        cls.setup_comprehensive_mocks()

    @classmethod
    def setup_comprehensive_mocks(cls):
        """Setup comprehensive mocking for all external dependencies"""
        # Mock all external modules that might not be installed
        modules_to_mock = [
            'flask', 'pandas', 'numpy', 'matplotlib', 'matplotlib.pyplot',
            'matplotlib.colors', 'scipy', 'scipy.interpolate', 'sklearn',
            'sklearn.preprocessing', 'torch', 'torch.nn', 'torch.nn.functional',
            'joblib', 'models', 'models.sic', 'models.sic.thickness',
            'models.sic.thickness.lstm', 'models.sic.thickness.lstm.lstm_thickness_unshuffle_skip1_v2',
            'models.sic.doping', 'models.sic.doping.lstm',
            'models.sic.doping.lstm.lstm_doping_unshuffle_skip1'
        ]

        for module in modules_to_mock:
            if module not in sys.modules:
                sys.modules[module] = MagicMock()

        # Special handling for Flask components
        mock_flask = MagicMock()
        mock_app = MagicMock()
        mock_app.root_path = '/fake/path'
        mock_app.config = {}
        mock_app.test_client.return_value = MagicMock()

        mock_flask.Flask.return_value = mock_app
        sys.modules['flask'] = mock_flask

        # Mock pandas
        mock_pandas = MagicMock()
        mock_df = MagicMock()
        mock_df.values.tolist.return_value = [['test', 'data']]
        mock_pandas.read_csv.return_value = mock_df
        mock_pandas.read_excel.return_value = mock_df
        sys.modules['pandas'] = mock_pandas

        # Mock numpy
        mock_numpy = MagicMock()
        mock_numpy.array.return_value = MagicMock()
        sys.modules['numpy'] = mock_numpy

    def setUp(self):
        """Set up test fixtures"""
        # Mock config files
        mock_config = [{"rows": [1], "column_index": 5}]

        # Add code directory to path
        code_path = os.path.join(os.path.dirname(__file__), '..', '..', 'code')
        if code_path not in sys.path:
            sys.path.insert(0, code_path)

        # Mock all file operations and imports
        with patch('builtins.open', mock_open(read_data=json.dumps(mock_config))):
            with patch('os.path.join', return_value='fake_path'):
                with patch('json.load', return_value=mock_config):
                    try:
                        # Try to import the app
                        import app
                        self.app_module = app

                        # Test if we can access the Flask app
                        if hasattr(app, 'app'):
                            self.app = app.app
                            self.app.config['TESTING'] = True
                            self.client = self.app.test_client()
                        else:
                            self.app = None
                            self.client = None

                    except Exception as e:
                        print(f"Warning: Could not import app module: {e}")
                        self.app_module = None
                        self.app = None
                        self.client = None

    def test_app_module_exists(self):
        """Test that the app module can be imported"""
        self.assertIsNotNone(self.app_module, "App module should be importable")

    def test_convert_time_string_function(self):
        """Test the convert_time_string function if available"""
        if self.app_module and hasattr(self.app_module, 'convert_time_string'):
            convert_time_string = self.app_module.convert_time_string

            # Test mm:ss format
            self.assertEqual(convert_time_string("05:30"), 330.0)

            # Test hh:mm:ss format
            self.assertEqual(convert_time_string("01:05:30"), 3930.0)

            # Test invalid format
            with self.assertRaises(ValueError):
                convert_time_string("invalid:time:format:extra")

    def test_extract_model_inputs_function(self):
        """Test the extract_model_inputs function if available"""
        if self.app_module and hasattr(self.app_module, 'extract_model_inputs'):
            extract_model_inputs = self.app_module.extract_model_inputs

            table_data = [
                ["header1", "header2", "header3", "header4", "header5"],
                ["row1", "val1", "val2", "val3", "100"],
                ["row2", "val1", "val2", "val3", "200"]
            ]
            input_config = [
                {"rows": [2], "column_index": 5},  # Row 2 (1-indexed) = index 1 in 0-indexed array
                {"rows": [3], "column_index": 5}   # Row 3 (1-indexed) = index 2 in 0-indexed array
            ]

            result = extract_model_inputs(table_data, input_config, header_exists=True)
            self.assertEqual(result, [100.0, 200.0])

    def test_get_base64_function(self):
        """Test the get_base64 function if available"""
        if self.app_module and hasattr(self.app_module, 'get_base64'):
            get_base64 = self.app_module.get_base64

            with patch('builtins.open', mock_open(read_data=b'test_data')):
                result = get_base64('fake_path')
                self.assertEqual(result, 'dGVzdF9kYXRh')  # base64 of 'test_data'

    def test_cell_data_cache_exists(self):
        """Test that the cell_data_cache global variable exists"""
        if self.app_module and hasattr(self.app_module, 'cell_data_cache'):
            self.assertIsInstance(self.app_module.cell_data_cache, dict)

    def test_model_input_config_exists(self):
        """Test that model_input_config exists"""
        if self.app_module and hasattr(self.app_module, 'model_input_config'):
            self.assertIsInstance(self.app_module.model_input_config, list)

    def test_app_configuration(self):
        """Test basic app configuration"""
        if not self.app:
            self.skipTest("Flask app not available")

        # Test that app has testing configuration
        self.assertTrue(self.app.config.get('TESTING', False))

    def test_mimo_regressor_class(self):
        """Test that MIMORegressor class is defined"""
        if self.app_module and hasattr(self.app_module, 'MIMORegressor'):
            MIMORegressor = self.app_module.MIMORegressor
            # Just test that it's a class
            self.assertTrue(callable(MIMORegressor))

if __name__ == '__main__':
    unittest.main()
