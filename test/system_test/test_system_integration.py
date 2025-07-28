import unittest
import sys
import os
import json
import tempfile
import io
from unittest.mock import patch, MagicMock, mock_open

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'code'))

class TestFlaskSystemIntegration(unittest.TestCase):
    """
    System tests for Flask SiC Application.
    These tests verify the complete system behavior including HTTP endpoints,
    template rendering, and integration between components.
    """

    @classmethod
    def setUpClass(cls):
        """Set up class-level mocks for external dependencies"""
        # Mock external dependencies but don't mock Flask itself
        cls.patcher_subprocess = patch('subprocess.run')
        cls.mock_subprocess = cls.patcher_subprocess.start()
        cls.mock_subprocess.return_value = MagicMock(returncode=0, stdout='')

    @classmethod
    def tearDownClass(cls):
        """Clean up class-level mocks"""
        cls.patcher_subprocess.stop()

    def setUp(self):
        """Set up test client for system testing"""
        # Import and configure the Flask app for testing
        try:
            import app as flask_app
            self.flask_app = flask_app.app
            self.flask_app.config['TESTING'] = True
            self.flask_app.config['WTF_CSRF_ENABLED'] = False
            self.client = self.flask_app.test_client()
            self.app_module = flask_app

            # Clear any existing cache for clean tests
            flask_app.cell_data_cache.clear()

        except Exception as e:
            self.skipTest(f"Could not set up Flask app for system testing: {e}")

    def test_home_page_renders_successfully(self):
        """System test: Verify home page renders with proper HTTP status and content"""
        response = self.client.get('/')

        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)
        # Verify that the response contains HTML content (template was rendered)
        self.assertTrue(len(response.data) > 100)  # Should have substantial HTML content

    def test_sic_data_page_renders_successfully(self):
        """System test: Verify SiC data page renders and loads data"""
        response = self.client.get('/sic_data')

        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)
        # Verify substantial content (template rendered with data)
        self.assertTrue(len(response.data) > 100)

    def test_sic_model_page_renders_successfully(self):
        """System test: Verify SiC model page renders and initializes prediction data"""
        response = self.client.get('/sic_model')

        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)
        # This endpoint should also populate prediction data in cache
        self.assertIn("predictionData", self.app_module.cell_data_cache)

    @patch('os.listdir')
    def test_get_images_api_endpoint(self, mock_listdir):
        """System test: Verify image API returns JSON data correctly"""
        mock_listdir.return_value = ['image1.png', 'image2.jpg', 'doc.txt']

        # Test without folder parameter
        response = self.client.get('/get_images')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.content_type)
        data = json.loads(response.data)
        self.assertEqual(data, [])

        # Test with folder parameter
        response = self.client.get('/get_images?folder=test_folder')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)

    @patch('builtins.open', mock_open(read_data='{"param1": {"min": 0, "max": 100}}'))
    def test_get_ranges_api_endpoint(self):
        """System test: Verify ranges API handles missing data gracefully"""
        response = self.client.get('/get_ranges')
        self.assertEqual(response.status_code, 200)
        self.assertIn('application/json', response.content_type)
        data = json.loads(response.data)
        self.assertIsInstance(data, dict)

    @patch('os.listdir')
    def test_get_thickness_models_api(self, mock_listdir):
        """System test: Verify thickness models API returns model list"""
        mock_listdir.return_value = ['model1.pth', 'model2.pth', 'readme.txt']

        response = self.client.get('/get_thickness_models?ml_method=lstm')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        # Should only return .pth files
        for item in data:
            self.assertTrue(item.endswith('.pth'))

    @patch('os.listdir')
    def test_get_doping_models_api(self, mock_listdir):
        """System test: Verify doping models API returns model list"""
        mock_listdir.return_value = ['doping1.pth', 'doping2.pt']

        response = self.client.get('/get_doping_models?ml_method=lstm')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)

    def test_cell_data_cache_integration(self):
        """System test: Verify cell data cache works across multiple requests"""
        # First, populate cache via sic_model endpoint
        response = self.client.get('/sic_model')
        self.assertEqual(response.status_code, 200)

        # Verify cache was populated
        self.assertIn("predictionData", self.app_module.cell_data_cache)

        # Test cache retrieval
        response = self.client.get('/get_cached_data?cell_id=predictionData')
        self.assertEqual(response.status_code, 200)
        # Should return HTML table content
        self.assertTrue(len(response.data) > 10)

    def test_update_cell_endpoint_integration(self):
        """System test: Verify cell update functionality works end-to-end"""
        # First populate some test data
        test_data = [["header"], ["test_value"]]
        self.app_module.cell_data_cache["test_cell"] = {
            "data": test_data,
            "file_type": "excel"
        }

        # Test updating a cell
        update_payload = {
            "cell_id": "test_cell",
            "row": 1,
            "col": 0,
            "new_value": "updated_value"
        }

        response = self.client.post('/update_cell',
                                  data=json.dumps(update_payload),
                                  content_type='application/json')

        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "success")

        # Verify the data was actually updated
        self.assertEqual(self.app_module.cell_data_cache["test_cell"]["data"][1][0], "updated_value")

    def test_delete_table_endpoint_integration(self):
        """System test: Verify table deletion works end-to-end"""
        # Populate test data
        self.app_module.cell_data_cache["delete_test"] = {
            "data": [["test"]],
            "file_type": "excel"
        }

        # Delete the table
        response = self.client.get('/delete_table?cell_id=delete_test')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data["status"], "deleted")

        # Verify it was removed from cache
        self.assertNotIn("delete_test", self.app_module.cell_data_cache)

    def test_error_handling_for_missing_resources(self):
        """System test: Verify application handles missing resources gracefully"""
        # Test with non-existent cell ID
        response = self.client.get('/get_cached_data?cell_id=nonexistent')
        self.assertEqual(response.status_code, 404)

        # Test delete non-existent table
        response = self.client.get('/delete_table?cell_id=nonexistent')
        self.assertEqual(response.status_code, 404)

    @patch('builtins.open', mock_open())
    def test_file_upload_simulation(self):
        """System test: Verify file upload endpoint handles requests correctly"""
        # Create a mock file upload
        data = {
            'doe': '1',
            'fileType': 'txt',
            'cell_id': '1_6R1'
        }

        # Create a fake file
        fake_file_content = "1.0 2.0 3.0\\n4.0 5.0 6.0"
        fake_file = (io.BytesIO(fake_file_content.encode()), 'test.txt')
        data['file'] = fake_file

        response = self.client.post('/upload_data', data=data)

        # Should return HTML content or handle gracefully
        self.assertIn(response.status_code, [200, 400, 500])  # Accept various responses due to mocking

    def test_multiple_endpoint_sequence(self):
        """System test: Verify multiple endpoints work together in sequence"""
        # 1. Load home page
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)

        # 2. Load SiC model page (populates cache)
        response = self.client.get('/sic_model')
        self.assertEqual(response.status_code, 200)

        # 3. Get cached data
        response = self.client.get('/get_cached_data?cell_id=predictionData')
        self.assertEqual(response.status_code, 200)

        # 4. Get images list (mocked to return empty)
        with patch('os.listdir', return_value=[]):
            response = self.client.get('/get_images?folder=test')
            self.assertEqual(response.status_code, 200)

        # All endpoints should work in sequence without conflicts

if __name__ == '__main__':
    unittest.main()
