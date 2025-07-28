# Flask SiC Application

A Flask web application for Silicon Carbide (SiC) epitaxy process modeling and prediction using machine learning.

## Quick Start

### 1. Setup Environment
Set up your preferred Python environment and install dependencies:
```bash
pip install -r library/requirements.txt
```

### 2. Run the Application
```bash
cd code
python app.py
```
**Note**: The application must be run from within the `code/` directory.

### 3. Access the Application
Open your web browser and navigate to: `http://172.20.76.31:8080`

## Project Structure

```
Flask_APP/
├── code/                     # Main application code
│   ├── app.py               # Flask application entry point
│   ├── config/              # Configuration files
│   │   ├── model_input_config.json
│   │   └── highlight_config_doe_1.json
│   ├── data/                # Data files
│   │   └── SiC/
│   │       ├── response/    # Response data (thickness, doping)
│   │       └── source/      # Source data (DOE files)
│   ├── models/              # Machine learning models
│   │   └── sic/
│   │       ├── thickness/   # Thickness prediction models
│   │       └── doping/      # Doping prediction models
│   ├── static/              # Static assets (images, logos)
│   └── templates/           # HTML templates
├── test/                    # Test suite and code analysis
│   ├── unit_test/           # Unit tests (21 tests)
│   │   ├── test_basic_functionality.py
│   │   ├── test_flask_routes.py
│   │   ├── test_utils.py
│   │   └── UNIT_TEST_RESULTS.md
│   ├── system_test/         # System integration tests (13 tests)
│   │   ├── test_system_integration.py
│   │   └── SYSTEM_TEST_RESULTS.md
│   └── CODE_ANALYSIS_RESULTS.md  # Pylint analysis results
├── demo/                    # Application screenshots
│   ├── Image_20250728111013_54.png
│   ├── Image_20250728111028_55.png
│   └── Image_20250728111032_57.png
├── library/                 # Dependencies
│   └── requirements.txt     # Python dependencies
├── venv/                    # Virtual environment
└── README.md               # This file
```

## Features

- **Data Management**: Upload, view, and edit SiC epitaxy data files
- **Visualization**: Interactive tables with highlighting and data validation
- **Machine Learning**: LSTM and Neural Network models for thickness and doping prediction
- **File Processing**: Support for Excel (.xlsx) and text (.txt) data formats
- **Results Export**: Download processed data and prediction results
- **Code Quality**: Pylint score of 9.90/10 with comprehensive testing suite

## Running Tests

### Run All Tests
```bash
# Set up your environment and install dependencies first
# pip install -r library/requirements.txt

# Run unit tests (21 tests)
python -m unittest test.unit_test.test_basic_functionality test.unit_test.test_flask_routes test.unit_test.test_utils -v

# Run system tests (13 tests)
python -m unittest test.system_test.test_system_integration -v

# Run all tests together
python -m unittest discover -s test -v
```

### Test Results
- ✅ **34 total tests** (21 unit + 13 system tests)
- ✅ **All tests passing** with 0 failures
- ✅ **Code quality**: 9.90/10 pylint rating
- ✅ **Test documentation**: Available in `test/` folder

### Code Analysis
The codebase has been thoroughly analyzed and refactored:
- **Pylint Analysis**: 9.90/10 rating (see `test/CODE_ANALYSIS_RESULTS.md`)
- **Code Quality**: Professional-grade standards met
- **Test Coverage**: Comprehensive unit and system test coverage

## Application Routes

- `/` - Home page with application overview
- `/sic_data` - Data management interface
- `/sic_model` - Predictive modeling interface
- `/upload_data` - File upload endpoint
- `/predict_from_table` - ML prediction endpoint

## Application Screenshots

The `demo/` folder contains screenshots of the application interface:
- `Image_20250728111013_54.png` - Home page overview
- `Image_20250728111028_55.png` - Data management interface  
- `Image_20250728111032_57.png` - Prediction modeling interface

## Development Notes

- This is a demo version with anonymized data
- All confidential information has been removed or replaced with dummy data
- The application supports both LSTM and Neural Network prediction models
- Time strings are automatically converted to seconds for processing
- Code has been refactored to meet professional quality standards (9.90/10 pylint rating)
- Comprehensive test suite ensures reliability and maintainability

## Dependencies

Located in `library/requirements.txt`:

- Flask 2.3.3
- pandas 2.1.1
- numpy 1.25.2
- matplotlib 3.7.2
- scipy 1.11.3
- scikit-learn 1.3.0
- torch 2.0.1
- joblib 1.3.2

## Project Quality Metrics

- **Code Quality**: 9.90/10 pylint rating
- **Test Coverage**: 34 tests (21 unit + 13 system)
- **Test Success Rate**: 100% (0 failures)
- **Code Lines**: 1,153 lines in main application
- **Documentation**: Comprehensive README and test reports

## License

This project is part of the SWIP2025 program.
