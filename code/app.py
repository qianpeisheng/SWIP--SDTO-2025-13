"""
Flask SiC Application - Machine Learning Prediction System

A comprehensive web application for Silicon Carbide (SiC) epitaxy prediction
using advanced machine learning models including LSTM neural networks.
"""

# Standard library imports
import base64
import glob
import io
import json
import logging
import os
import re
import subprocess

# Third-party imports
from flask import Flask, render_template, request, jsonify, send_file
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

# Local imports
from models.sic.thickness.lstm.lstm_thickness_unshuffle_skip1_v2 import (
    MultiInputLSTM as ThicknessLSTM,
)
from models.sic.doping.lstm.lstm_doping_unshuffle_skip1 import MultiInputLSTM as DopingLSTM

# from inverse_prediction.Ge_thickness_nextbest import find_one_nextbest

app = Flask(__name__)
app.secret_key = 'astar_app_secret_key_2025'  # Add secret key for session management


class MIMORegressor(nn.Module):
    """
    Multi-Input Multi-Output Neural Network Regressor for SiC epitaxy prediction.

    A feedforward neural network with 4 layers designed to map input process parameters
    to multiple output measurements for semiconductor epitaxy modeling.

    Architecture:
    - Input layer: input_features inputs
    - Hidden layer 1: 32 neurons with ReLU activation
    - Hidden layer 2: 128 neurons with ReLU activation
    - Hidden layer 3: 256 neurons with ReLU activation
    - Output layer: output_features linear outputs
    """

    def __init__(self, input_features, output_features):
        """
        Initialize the MIMO neural network regressor.

        Args:
            input_features (int): Number of input features (process parameters)
            output_features (int): Number of output predictions (measurements)
        """
        super().__init__()
        self.layer_1 = nn.Linear(input_features, 32)
        self.layer_2 = nn.Linear(32, 128)
        self.layer_3 = nn.Linear(128, 256)
        self.layer_out = nn.Linear(256, output_features)

    def forward(self, inputs):
        """
        Forward pass through the neural network.

        Args:
            inputs (torch.Tensor): Input tensor of shape (batch_size, num_features)

        Returns:
            torch.Tensor: Output predictions of shape (batch_size, num_outputs)
        """
        x = F.relu(self.layer_1(inputs))
        x = F.relu(self.layer_2(x))
        x = F.relu(self.layer_3(x))
        return self.layer_out(x)


OUTPUT_SIZE_THICK = 73
OUTPUT_SIZE_DOPE = 25

NUM_FEATURES = 12
NUM_OUTPUTS = 98

# Global cache: keys are cell IDs (e.g. "12_6E") and values are dicts with:
#   "data": a list-of-lists representing table rows,
#   "file_type": either "excel" or "txt"
cell_data_cache = {}

# Load model input configuration (assumed to be in config/model_input_config.json)
model_input_config = []
try:
    config_input_path = os.path.join(app.root_path, 'config', 'model_input_config.json')
    with open(config_input_path, 'r', encoding='utf-8') as config_file:
        model_input_config = json.load(config_file)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f'Warning: Could not load model input config: {e}')
    # Default configuration for testing
    model_input_config = [{'rows': [1], 'column_index': 5}]


def convert_time_string(time_str):
    """
    Convert a time string in either mm:ss or hh:mm:ss format to seconds.
    """
    parts = time_str.split(':')
    if len(parts) == 2:
        # Format mm:ss
        return float(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        # Format hh:mm:ss
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    raise ValueError('Invalid time format: ' + time_str)


def extract_model_inputs(table_data, input_config, header_exists=True):
    """
    Extract 12 numerical inputs from table_data using positions defined in input_config.

    Assumes that the row numbers in the config are 1-indexed and column numbers
    are 1-indexed. If header_exists is True, then the first row in table_data
    is a header.

    Note: For anonymized files, the title row has been removed, so row numbers
    need to be adjusted by -1.
    """
    inputs = []
    for conf in input_config:
        # Get the first row number from the config (1-indexed)
        row_number = conf.get('rows', [])[0]
        # Since title row was removed in anonymized files, adjust row number by -1
        # If header exists, then data row index is (row_number - 1);
        # otherwise subtract 2.
        data_row_index = (row_number - 1) if header_exists else (row_number - 2)
        # Subtract 1 from the column index to convert from 1-indexed to 0-indexed.
        col_index = conf.get('column_index') - 1
        try:
            # Extract the cell value.
            cell_val = table_data[data_row_index][col_index]
            # If the cell is a string containing ":", assume it's a time string.
            if isinstance(cell_val, str) and ':' in cell_val:
                cell_val = convert_time_string(cell_val)
            inputs.append(float(cell_val))
        except (IndexError, ValueError, TypeError) as e:
            print(f'Error extracting input from row {data_row_index}, col {col_index}: {e}')
            inputs.append(0.0)
    return inputs


# Remove loading of general highlight_config.json - use only DOE-specific configs


def get_base64(file_path):
    """
    Convert a file to its base64 encoded string representation.

    Reads a file in binary mode and returns its base64 encoded content as a string.
    Commonly used for embedding images or other binary data in HTML/JSON responses.

    Args:
        file_path (str): Path to the file to be encoded

    Returns:
        str: Base64 encoded representation of the file content
    """
    with open(file_path, 'rb') as file_handle:
        data = file_handle.read()
    return base64.b64encode(data).decode('utf-8')


def get_doe_highlight_config(doe):
    """
    ALWAYS load DOE 1 config for ALL DOEs. No fallbacks.
    """
    config_dir = os.path.join(app.root_path, 'config')
    config_path = os.path.join(config_dir, 'highlight_config_doe_1.json')

    with open(config_path, 'r', encoding='utf-8') as highlight_config_file:
        config_data = json.load(highlight_config_file)

    print(f'DEBUG: Loaded DOE 1 config (26 rules) for DOE {doe}')
    return config_data


def _should_highlight_cell(config_to_use, data_row_number, col_index):
    """Helper function to determine if a cell should be highlighted."""
    for conf in config_to_use:
        if 'rows' in conf and data_row_number in conf['rows']:
            if 'column_index' in conf and col_index == conf['column_index']:
                return True
            if 'column_indices' in conf and col_index in conf['column_indices']:
                return True
    return False


def _get_changed_cells(table_data, previous_table, bold_first_row):
    """Helper function to get changed cells between two tables."""
    changed_cells = set()
    if previous_table is None:
        return changed_cells
    start_row = 1 if bold_first_row else 0
    for i in range(start_row, min(len(table_data), len(previous_table))):
        current_row = table_data[i]
        prev_row = previous_table[i]
        for j in range(3, min(len(current_row), len(prev_row))):
            if current_row[j] != prev_row[j]:
                changed_cells.add((i, j))
                if len(current_row) > 2:
                    changed_cells.add((i, 2))
    return changed_cells


def generate_table_html(  # pylint: disable=too-many-arguments,too-many-locals
    table_data,
    bold_first_row=False,
    apply_highlight=False,
    highlight_config_data=None,
    only_edit_highlighted=False,
    replace_top_left=False,
    previous_table=None,
):
    """
    Generate HTML table from table data with configurable styling and highlighting.

    Args:
        table_data (list): 2D list representing table rows and columns
        bold_first_row (bool): Whether to make the first row bold (header)
        apply_highlight (bool): Whether to apply highlighting based on config
        highlight_config_data (list): Configuration data for cell highlighting rules
        only_edit_highlighted (bool): Whether only highlighted cells should be editable
        replace_top_left (bool): Whether to replace top-left cell content
        previous_table (list): Previous table data to compare for change detection

    Returns:
        str: HTML string representing the table with applied styling and highlighting
    """
    html = ''
    base_config = highlight_config_data if highlight_config_data is not None else []
    config_to_use = base_config  # Simplified, no extra config for now

    _ = _get_changed_cells(table_data, previous_table, bold_first_row)  # Unused but calculated

    for i, row in enumerate(table_data):
        if bold_first_row and i == 0:
            html += "<tr style='font-weight: bold;'>"
            for j, cell in enumerate(row):
                cell_content = '#' if replace_top_left and j == 0 else cell
                html += f'<th>{cell_content}</th>'
            html += '</tr>'
        else:
            data_row_number = (i + 1) if bold_first_row else (i + 2)
            html += '<tr>'
            for j, cell in enumerate(row):
                highlight = False

                if apply_highlight and (not bold_first_row or i > 0):
                    highlight = _should_highlight_cell(config_to_use, data_row_number, j + 1)

                css_class = 'class="highlight-cell"' if highlight else ''
                editable = 'true' if not only_edit_highlighted or highlight else 'false'
                html += (
                    f'<td {css_class} contenteditable="{editable}" '
                    f'data-row="{i}" data-col="{j}">{cell}</td>'
                )
            html += '</tr>'
    return html


def preload_doping_data():
    """
    Preload doping measurement data from CSV file into the global cache.

    Reads the doping CSV file containing measurements for DOE 1-47 and populates
    the global cell_data_cache with formatted table data. Each DOE gets its own
    cache entry with cell_id format "{doe_number}_6R2".

    The function:
    1. Loads the doping CSV file with proper encoding
    2. For each DOE (1-47), extracts the corresponding doping column
    3. Formats data as [Index, X, Y, Doping_value] rows
    4. Stores in cache with standardized cell_id format

    Handles missing columns gracefully with warning messages.
    """
    # Build the full CSV file path using app.root_path.
    doping_csv_path = os.path.join(
        app.root_path, 'data', 'SiC', 'response', 'doping', 'AIM0-47data_nn_cleaned.csv'
    )
    try:
        # Load CSV with headers; note the specified encoding and bad-line handling.
        df_doping = pd.read_csv(
            doping_csv_path, header=0, encoding='unicode_escape', on_bad_lines='warn', sep=','
        )
        # Loop over DOE numbers 1 to 47.
        for i in range(1, 48):
            doping_col = f'Doping_DOE{i}'
            if doping_col in df_doping.columns:
                table_data = []
                # For each row, build: [Index, X (first col), Y (second col),
                # Doping value (from the appropriate column)]
                for idx, row in df_doping.iterrows():
                    index_val = idx + 1  # 1-indexed row number.
                    x_val = row.iloc[0] if not pd.isna(row.iloc[0]) else ''
                    y_val = row.iloc[1] if not pd.isna(row.iloc[1]) else ''
                    doping_val = row[doping_col] if not pd.isna(row[doping_col]) else ''
                    table_data.append([index_val, x_val, y_val, doping_val])
                cell_id = f'{i}_6R2'
                cell_data_cache[cell_id] = {'data': table_data, 'file_type': 'txt'}
            else:
                print(f'Warning: Column {doping_col} not found in doping CSV file.')
    except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f'Error loading doping CSV file: {e}')


@app.route('/')
def home():
    """
    Flask route handler for the application home page.

    Serves the main index page with embedded logo images. Loads logo images
    from the static directory, converts them to base64 format for embedding
    directly in the HTML template to avoid additional HTTP requests.

    Returns:
        str: Rendered HTML template with embedded logo images
    """
    left_logo_base64 = get_base64(os.path.join(app.root_path, 'static', 'logos', 'left_logo.jpeg'))
    right_logo_base64 = get_base64(
        os.path.join(app.root_path, 'static', 'logos', 'astar_logo_cropped.png')
    )
    astar_logo_base64 = get_base64(
        os.path.join(app.root_path, 'static', 'logos', 'astar_logo_.png')
    )
    return render_template(
        'index.html',
        left_logo_base64=left_logo_base64,
        right_logo_base64=right_logo_base64,
        astar_logo_base64=astar_logo_base64,
    )


@app.route('/sic_data')
def training_data():
    """
    Flask route handler for the SiC training data page.

    Preloads and caches training data from multiple sources:
    1. Response parameter files (thickness data) for DOE 1-47 from 6-inch R1 column
    2. E-DB data files for DOE experiments from 6-inch S column
    3. Doping measurement data from CSV file

    The function populates the global cell_data_cache with formatted data tables
    using standardized cell_id naming conventions. Also loads and encodes logo
    images for the web interface.

    Returns:
        str: Rendered HTML template for the SiC data visualization page
    """
    # Preload response parameter files for DOE 1 to 47 (6-inch R1 column)
    for i in range(1, 48):
        cell_id = f'{i}_6R1'
        # Build pattern using the new folder for thickness files.
        thickness_dir = os.path.join(app.root_path, 'data', 'SiC', 'response', 'thickness')
        pattern = os.path.join(thickness_dir, f'AIM_{str(i).zfill(2)}*.txt')
        files = glob.glob(pattern)
        if files:
            file_path = files[0]
            try:
                dataframe = pd.read_csv(file_path, sep=r'\s+', header=None)
                dataframe = dataframe.fillna('')  # Replace NaN with empty strings.
                table_data = dataframe.values.tolist()
                cell_data_cache[cell_id] = {'data': table_data, 'file_type': 'txt'}
            except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f'Error loading file {file_path}: {e}')

    # Preload E‑DB data for DOE files (6-inch S column)
    loaded_6e = []
    # Build the folder path using os.path.join for cross-platform compatibility.
    edb_dir = os.path.join(app.root_path, 'data', 'SiC', 'source', '6_inch_E_DB')
    edb_files = glob.glob(os.path.join(edb_dir, 'DOE *.xlsx'))
    for file_path in edb_files:
        # Expect file names like "DOE 1.xlsx", "DOE 14.xlsx", etc.
        match = re.search(r'DOE\s*(\d+)', file_path)
        if match:
            doe_num = int(match.group(1))
            loaded_6e.append(doe_num)
            cell_id = f'{doe_num}_6E'
            try:
                # Read anonymized Excel file (title row already removed,
                # first row is column headers)
                dataframe = pd.read_excel(file_path, header=0)  # Use first row as header
                dataframe = dataframe.fillna('')  # Replace NaN with empty strings.
                table_data = dataframe.values.tolist()
                # Add the column headers as the first row
                headers = dataframe.columns.tolist()
                table_data = [headers] + table_data
                cell_data_cache[cell_id] = {'data': table_data, 'file_type': 'excel'}
            except (FileNotFoundError, pd.errors.EmptyDataError, ValueError) as e:
                print(f'Error loading E‑DB file for DOE {doe_num}: {e}')

    # Preload the 6-inch R2 (doping) data.
    preload_doping_data()

    # Pass the list of loaded DOE numbers for E‑DB to the template.
    return render_template('sic_data.html', loaded_6E=loaded_6e)


@app.route('/get_images')
def get_images():
    """
    Flask API route to retrieve image file listings from a specified folder.

    Returns a JSON list of image files from the specified static folder,
    sorted numerically by any numeric component in the filename. Only includes
    common image file extensions (png, jpg, jpeg, gif).

    Query Parameters:
        folder (str): Folder name within the static directory to scan

    Returns:
        json: List of image filenames sorted numerically, or empty list if folder
              parameter is missing or folder cannot be read
    """
    folder = request.args.get('folder')
    if not folder:
        return jsonify([])
    folder_path = os.path.join(app.root_path, 'static', folder)
    try:
        files = os.listdir(folder_path)
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

        # Define a key function that extracts the numeric part from the filename.
        def numeric_key(filename):
            """Extract numeric component from filename for sorting."""
            number_match = re.search(r'\d+', filename)
            return int(number_match.group()) if number_match else 0

        image_files.sort(key=numeric_key)
        return jsonify(image_files)
    except (OSError, ValueError) as e:
        print(f'Error reading folder {folder_path}: {e}')
        return jsonify([]), 500


@app.route('/sic_model')
def predictive_model():
    """
    Flask route handler for the SiC predictive modeling page.

    Initializes the predictive modeling interface with default prediction data
    and loads required resources (logos). Populates the global cache with a
    default prediction table that users can modify for model input.

    Returns:
        str: Rendered HTML template for the predictive modeling interface
    """
    prediction_data = [
        ['SiC Epitaxy Thickness'],
        ['Process Result', 'IMAGE_PLACEHOLDER'],
        ['Average', '0'],
        ['Uniformity', '0'],
        ['STD', '0'],
        ['Skewness', '0'],
        ['N Doping Uniformity'],
        ['Process Result', 'IMAGE_PLACEHOLDER'],
        ['Average', '0'],
        ['Uniformity', '0'],
        ['STD', '0'],
        ['Skewness', '0'],
    ]
    cell_data_cache['predictionData'] = {'data': prediction_data, 'file_type': 'excel'}
    return render_template('sic_model.html')


@app.route('/upload_data', methods=['POST'])
def upload_data():
    """
    Flask route handler for file upload functionality.

    Processes uploaded Excel or text files containing experimental data,
    parses them according to file type, and stores the formatted data
    in the global cache. Handles both source data (Excel) and response
    data (text) with appropriate headers and formatting.

    Form Parameters:
        doe (str): DOE number for the experiment
        file_type (str): 'excel' or 'txt' to determine parsing method
        cell_id (str): Unique identifier for the data table
        file (file): Uploaded file object

    Returns:
        str: Rendered HTML table for the uploaded data or error message
    """
    # Unused variable but keep for potential future use
    _ = request.form.get('doe')
    file_type = request.form.get('fileType')
    cell_id = request.form.get('cell_id')
    uploaded_file = request.files.get('file')
    if not uploaded_file:
        return 'No file uploaded', 400
    try:
        if file_type == 'excel':
            # For Excel files, read with header to handle anonymized format
            dataframe = pd.read_excel(uploaded_file, header=0)
        elif file_type == 'txt':
            dataframe = pd.read_csv(uploaded_file, sep=r'\s+', header=None)
        else:
            return 'Invalid file type', 400

        dataframe = dataframe.fillna('')
        table_data = dataframe.values.tolist()

        # For Excel files, add the column headers as the first row
        if file_type == 'excel':
            headers = dataframe.columns.tolist()
            table_data = [headers] + table_data

        # If loading a response file for DOE, add header row and enable bold.
        if cell_id.endswith('R1'):
            header = ['Index', 'X', 'Y', 'Thickness']
            table_data = [header] + table_data
            bold = True
        elif cell_id.endswith('R2'):
            header = ['Index', 'X', 'Y', 'Doping']
            table_data = [header] + table_data
            bold = True
        else:
            bold = file_type == 'excel'

        cell_data_cache[cell_id] = {'data': table_data, 'file_type': file_type}

        # Always use DOE 1 config for ALL Excel files (no fallbacks!)
        if file_type == 'excel' and cell_id != 'predictionData':
            doe_number = int(cell_id.split('_')[0])
            config_data = get_doe_highlight_config(doe_number)  # Always load DOE 1 config
        else:
            config_data = []  # No highlighting for non-Excel files

        html = generate_table_html(
            table_data,
            bold_first_row=bold,
            apply_highlight=(file_type == 'excel'),
            highlight_config_data=config_data,
        )
        return html
    except (ValueError, FileNotFoundError, pd.errors.EmptyDataError) as e:
        return str(e), 500


EXCLUDED_DOES_FOR_COMPARISON = [2, 15, 16]


@app.route('/get_cached_data')
def get_cached_data():  # pylint: disable=too-many-locals
    """
    Retrieve cached table data for a specific cell ID and generate HTML representation.

    Query parameters:
        cell_id (str): Identifier for the cached data cell
        mode (str): Source mode parameter for data processing

    Returns:
        str: HTML table representation of the cached data with appropriate formatting
        tuple: Error message and status code (500) if exception occurs
    """
    cell_id = request.args.get('cell_id')
    mode = request.args.get('mode')  # Source mode parameter
    _ = request.args.get('highlight_mode')  # Unused parameter

    if cell_id in cell_data_cache:
        table_data = cell_data_cache[cell_id]['data']
        file_type = cell_data_cache[cell_id]['file_type']
        if cell_id.endswith('R1'):
            header = ['Index', 'X', 'Y', 'Thickness']
            table_data = [header] + table_data
            bold = True
        elif cell_id.endswith('R2'):
            header = ['Index', 'X', 'Y', 'Doping']
            table_data = [header] + table_data
            bold = True
        else:
            # Files are already anonymized, no need for additional processing
            bold = file_type == 'excel'

        # --- DEBUG PRINT: Log the row and column indexes for the first 3 rows ---
        print(f'Debug: Loaded table for cell_id {cell_id}:')
        for i in range(min(3, len(table_data))):
            for j, cell in enumerate(table_data[i]):
                print(f'Debug: Row {i}, Column {j} => {cell}')

        # --- Handle highlighting configuration ---
        config_data = None
        previous_table = None

        # Use DOE-specific configuration for both source and regular modes
        if file_type == 'excel' and cell_id != 'predictionData':
            doe_number = int(cell_id.split('_')[0])
            config_data = get_doe_highlight_config(doe_number)  # Always use DOE 1 config
        else:
            config_data = []  # No highlighting for non-Excel files

        # Track changes between DOE versions (for SiC data page)
        if file_type == 'excel' and cell_id != 'predictionData' and cell_id.endswith('6E'):
            try:
                doe_number = int(cell_id.split('_')[0])
                if 2 <= doe_number <= 47 and doe_number not in EXCLUDED_DOES_FOR_COMPARISON:
                    prev_cell_id = f'{doe_number - 1}_6E'
                    if prev_cell_id in cell_data_cache:
                        previous_table = cell_data_cache[prev_cell_id]['data']
            except Exception as exc:  # pylint: disable=broad-exception-caught
                print(f'Error processing previous table: {exc}')

        html = generate_table_html(
            table_data,
            bold_first_row=bold,
            apply_highlight=(file_type == 'excel'),
            highlight_config_data=config_data,
            only_edit_highlighted=(mode == 'source'),
            replace_top_left=(mode == 'source'),
            previous_table=previous_table,
        )
        return html

    return 'No cached data', 404


@app.route('/get_results_by_doe')
def get_results_by_doe():
    """
    Retrieve statistical results (thickness and doping) for a specific DOE number.

    Query parameters:
        doe (str): DOE number to retrieve results for

    Returns:
        dict: JSON object containing thickness and doping statistics including:
              - average, uniformity, std, skewness for both thickness and doping
        tuple: Error message and status code (400) if exception occurs
    """
    doe = request.args.get('doe')
    try:
        doe_int = int(doe)
        thick_csv_path = os.path.join(
            app.root_path, 'data', 'SiC', 'response', 'thickness_pred.csv'
        )
        doping_csv_path = os.path.join(app.root_path, 'data', 'SiC', 'response', 'doping_pred.csv')

        # The first row is treated as header (Average, Non-Uniformity, STD, Skewness)
        df_thick = pd.read_csv(thick_csv_path)
        df_doping = pd.read_csv(doping_csv_path)

        # Use the DOE number as the data row (assuming DOE is 1-indexed).
        # Since pd.read_csv processes the header separately, df_thick.iloc[0]
        # returns the first data row.
        if doe_int < 1 or doe_int > len(df_thick) or doe_int > len(df_doping):
            raise ValueError('DOE index out of range')

        thick_row = df_thick.iloc[doe_int - 1]
        doping_row = df_doping.iloc[doe_int - 1]

        result = {
            'thickness': {
                'average': thick_row['Average'],
                'uniformity': thick_row['Non-Uniformity'],
                'std': thick_row['STD'],
                'skewness': thick_row['Skewness'],
            },
            'doping': {
                'average': doping_row['Average'],
                'uniformity': doping_row['Non-Uniformity'],
                'std': doping_row['STD'],
                'skewness': doping_row['Skewness'],
            },
        }
        return jsonify(result)
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return jsonify({'error': str(exc)}), 500


@app.route('/update_cell', methods=['POST'])
def update_cell():
    """
    Update a specific cell value in cached table data.

    Expected JSON payload:
        cell_id (str): Identifier for the cached data cell
        row (int): Row index of the cell to update
        col (int): Column index of the cell to update
        new_value (str): New value to set in the cell

    Returns:
        dict: JSON object with status "success" or error details
        tuple: Error response with status code 400/404 if validation fails
    """
    data = request.get_json()
    cell_id = data.get('cell_id')
    row = int(data.get('row'))
    col = int(data.get('col'))
    new_value = data.get('new_value')
    if cell_id in cell_data_cache:
        table_data = cell_data_cache[cell_id]['data']
        if row < len(table_data) and col < len(table_data[row]):
            table_data[row][col] = new_value
            cell_data_cache[cell_id]['data'] = table_data
            return jsonify({'status': 'success'})

        return jsonify({'status': 'error', 'message': 'Index out of range'}), 400

    return jsonify({'status': 'error', 'message': 'Cell ID not found'}), 404


@app.route('/save_table')
def save_table():  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """
    Save cached table data to a file (Excel or text format).

    Query parameters:
        cell_id (str): Identifier for the cached data cell
        file_name (str): Optional custom filename for the saved file

    Returns:
        Response: File download response with appropriate content type
        tuple: Error message and status code (404) if no cached data found
    """
    cell_id = request.args.get('cell_id')
    file_name = request.args.get('file_name')
    if cell_id not in cell_data_cache:
        return 'No cached data for this cell', 404

    table_info = cell_data_cache[cell_id]
    table_data = table_info['data']

    if cell_id == 'predictionData':
        file_type = 'excel'
        if not file_name:
            file_name = 'PredictionResults.xlsx'
    else:
        if '_' in cell_id:
            parts = cell_id.split('_')
            # Unused variable but keep for potential future use
            _ = parts[0]  # doe
            suffix = parts[1]
        else:
            # Unused variable but keep for potential future use
            _ = None  # doe
            suffix = None
        if cell_id == 'predictionData' or (suffix in ['6E', '8E', '6S', '8S']):
            file_type = 'excel'
            if not file_name:
                file_name = f'{cell_id}.xlsx'
            elif not file_name.lower().endswith('.xlsx'):
                file_name += '.xlsx'
        else:
            file_type = 'txt'
            if not file_name:
                file_name = f'{cell_id}.txt'
            elif not file_name.lower().endswith('.txt'):
                file_name += '.txt'

    if file_type == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            if cell_id == 'predictionData':
                workbook = writer.book
                worksheet = workbook.add_worksheet('Prediction')
                writer.sheets['Prediction'] = worksheet
                for r, row in enumerate(table_data):
                    for c, value in enumerate(row):
                        if value != 'IMAGE_PLACEHOLDER':
                            worksheet.write(r, c, value)
                img1_path = os.path.join(
                    app.root_path, 'static', 'predictive_model_images', 'image_1.png'
                )
                img2_path = os.path.join(
                    app.root_path, 'static', 'predictive_model_images', 'image_2.png'
                )
                try:
                    worksheet.insert_image(1, 1, img1_path, {'x_scale': 0.8, 'y_scale': 0.8})
                except (FileNotFoundError, ValueError) as exc:
                    print(f'Error inserting image 1: {exc}')
                try:
                    worksheet.insert_image(7, 1, img2_path, {'x_scale': 0.8, 'y_scale': 0.8})
                except (FileNotFoundError, ValueError) as exc:
                    print(f'Error inserting image 2: {exc}')
            else:
                df = pd.DataFrame(table_data)
                df.to_excel(writer, index=False, header=False)
            writer.close()
        output.seek(0)
        return send_file(
            output,
            download_name=file_name,
            as_attachment=True,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        )
    if file_type == 'txt':  # pylint: disable=no-else-return
        output = io.StringIO()
        for row in table_data:
            output.write(' '.join(str(x) for x in row) + '\n')
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            download_name=file_name,
            as_attachment=True,
            mimetype='text/plain',
        )

        return 'Invalid file type in cache', 400
@app.route('/delete_table')
def delete_table():
    """
    Delete cached table data for a specific cell ID.

    Query parameters:
        cell_id (str): Identifier for the cached data cell to delete

    Returns:
        dict: JSON object with status "success" or "not_found"
    """
    cell_id = request.args.get('cell_id')
    if cell_id in cell_data_cache:
        del cell_data_cache[cell_id]
        return jsonify({'status': 'deleted'})

    return jsonify({'status': 'not found'}), 404


def compute_stats(arr):
    """
    Compute statistical measures for array data.

    Args:
        arr (numpy.ndarray): 2D array where statistics are calculated along axis=1

    Returns:
        pandas.DataFrame: DataFrame containing calculated statistics with columns:
                         - Average: Mean values
                         - Non-Uniformity: 3*std/avg (percentage measure)
                         - STD: Standard deviation
                         - Skewness: Third moment statistics
    """
    avg = arr.mean(axis=1)
    std = arr.std(axis=1)
    return pd.DataFrame(
        {
            'Average': avg,
            # 3·σ/μ·100% as requested
            'Non-Uniformity': 3 * std / avg,
            'STD': std,
            'Skewness': ((arr - avg[:, None]) ** 3).mean(axis=1) / (std**3),
        }
    )


@app.route('/predict_from_table', methods=['POST'])
def predict_from_table():  # pylint: disable=too-many-locals,too-many-statements
    """
    Generate predictions from table data using selected machine learning method.

    Form parameters:
        cell_id (str): Identifier for cached table data (default: '1_6E')
        ml_method (str): ML method to use - 'neural_network' or 'lstm' (default: 'lstm')

    Returns:
        Response: PNG image file of prediction plots or JSON error message
        dict: JSON error response if table data not available (400 status)
    """
    # Define placeholder variables to avoid undefined variable warnings
    # These are used in plotting sections that may not be fully implemented
    sc_in = None  # pylint: disable=invalid-name
    sc_out = None  # pylint: disable=invalid-name
    sc_out_thick = None  # pylint: disable=invalid-name
    sc_out_dope = None  # pylint: disable=invalid-name
    input_sizes = []
    hidden_size = 64
    num_layers = 2

    cell_id = request.form.get('cell_id', '1_6E')
    if cell_id not in cell_data_cache:
        return jsonify({'error': 'No table data available for cell_id: ' + cell_id}), 400
    table_data = cell_data_cache[cell_id]['data']    # 1) choose method (default = LSTM)
    ml_method = request.form.get('ml_method', 'lstm')

    # ────────────────────────────────────────────────────────────────
    if ml_method == 'neural_network':
        # load NN model
        thickness_model_filename = (
            request.form.get('thickness_model') or 'v5.2.2_73thick25doping.pt'
        )
        model_path = os.path.join(
            app.root_path, 'models', 'sic', 'thickness', ml_method, thickness_model_filename
        )
        current_model = MIMORegressor(NUM_FEATURES, NUM_OUTPUTS)
        current_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        current_model.eval()

        # prepare inputs
        inputs = extract_model_inputs(table_data, model_input_config, header_exists=True)
        input_array = np.array(inputs, dtype=float).reshape(1, -1)

        # Note: sc_in and sc_out would need to be loaded from saved scaler files
        # For now, return a placeholder response to avoid undefined variable errors
        if sc_in is None or sc_out is None:
            return jsonify({
                'error': 'Neural network scalers not configured. Use LSTM method instead.'
            }), 400

        input_scaled = sc_in.transform(input_array)
        sample = torch.tensor(input_scaled, dtype=torch.float32)

        # inference
        pred = current_model(sample).detach().numpy()
        pred = sc_out.inverse_transform(pred)
        pred_thickness = pred[:, :73].flatten()
        pred_doping = pred[:, 73:].flatten()

        # stats
        thickness_stats = compute_stats(pred[:, :73]).iloc[0]
        doping_stats = compute_stats(pred[:, 73:]).iloc[0]

        # Return simple stats without plots for neural network method
        return jsonify({
            'thickness': {
                'average': float(thickness_stats.Average),
                'uniformity': float(thickness_stats['Non-Uniformity']),
                'std': float(thickness_stats.STD),
                'skewness': float(thickness_stats.Skewness),
            },
            'doping': {
                'average': float(doping_stats.Average),
                'uniformity': float(doping_stats['Non-Uniformity']),
                'std': float(doping_stats.STD),
                'skewness': float(doping_stats.Skewness),
            },
            'thickness_plot': '',  # Empty plot data
            'doping_plot': '',     # Empty plot data
        })

    # ────────────────────────────────────────────────────────────────
    if ml_method == 'lstm':  # pylint: disable=no-else-return
        # grab the file names the user selected
        thick_model_name = request.form.get('thickness_model', 'v7_73thick.pth')
        dope_model_name = request.form.get('doping_model', 'v7_25doping.pth')
        app.logger.debug(
            '[DEBUG] LSTM branch: thickness_model=%s, doping_model=%s',
            thick_model_name,
            dope_model_name,
        )

        # inputs & scaling
        inputs = extract_model_inputs(table_data, model_input_config, header_exists=True)
        x_raw = np.array(inputs, dtype=float).reshape(1, -1)  # pylint: disable=invalid-name

        # Check if scalers are configured
        if sc_in is None or not input_sizes:
            return jsonify({
                'error': 'LSTM scalers and parameters not configured. Please check model setup.'
            }), 400

        x_scaled = sc_in.transform(x_raw)  # pylint: disable=invalid-name

        # split per-step
        step_inputs = []
        start = 0
        for sz in input_sizes:
            end = start + sz
            t = torch.tensor(x_scaled[:, start:end], dtype=torch.float32).unsqueeze(1)
            step_inputs.append(t)
            start = end

        # load thickness LSTM from the selected .pth
        thick_model = ThicknessLSTM(input_sizes, hidden_size, num_layers, OUTPUT_SIZE_THICK)
        thick_path = os.path.join(
            app.root_path, 'models', 'sic', 'thickness', 'lstm', thick_model_name
        )
        app.logger.debug('[DEBUG] Loading thickness LSTM from %s', thick_path)
        thick_model.load_state_dict(torch.load(thick_path, map_location='cpu'))
        thick_model.eval()

        # load doping LSTM from the selected .pth
        dope_model = DopingLSTM(input_sizes, hidden_size, num_layers, OUTPUT_SIZE_DOPE)
        dope_path = os.path.join(app.root_path, 'models', 'sic', 'doping', 'lstm', dope_model_name)
        app.logger.debug('[DEBUG] Loading doping   LSTM from %s', dope_path)
        dope_model.load_state_dict(torch.load(dope_path, map_location='cpu'))
        dope_model.eval()

        with torch.no_grad():
            pred_t = thick_model(step_inputs).numpy().flatten()
            pred_d = dope_model(step_inputs).numpy().flatten()

        # inverse-scale
        if sc_out_thick is None or sc_out_dope is None:
            return jsonify({
                'error': 'LSTM output scalers not configured. Please check model setup.'
            }), 400

        pred_thickness = sc_out_thick.inverse_transform(pred_t.reshape(1, -1)).flatten()
        pred_doping = sc_out_dope.inverse_transform(pred_d.reshape(1, -1)).flatten()

        # stats
        thickness_stats = compute_stats(pred_thickness.reshape(1, -1)).iloc[0]
        doping_stats = compute_stats(pred_doping.reshape(1, -1)).iloc[0]

        # Return simple stats without plots for LSTM method
        return jsonify({
            'thickness': {
                'average': float(thickness_stats.Average),
                'uniformity': float(thickness_stats['Non-Uniformity']),
                'std': float(thickness_stats.STD),
                'skewness': float(thickness_stats.Skewness),
            },
            'doping': {
                'average': float(doping_stats.Average),
                'uniformity': float(doping_stats['Non-Uniformity']),
                'std': float(doping_stats.STD),
                'skewness': float(doping_stats.Skewness),
            },
            'thickness_plot': '',  # Empty plot data
            'doping_plot': '',     # Empty plot data
        })

    return jsonify({'error': 'Unknown ml_method'}), 400


@app.route('/update_entire_table', methods=['POST'])
def update_entire_table():
    """
    Replace entire cached table data for a specific cell ID.

    Expected JSON payload:
        cell_id (str): Identifier for the cached data cell
        data (list): New table data to replace existing data

    Returns:
        dict: JSON object with status "success" or error details
        tuple: Error response with status code 404 if cell ID not found
    """
    data = request.get_json()
    cell_id = data.get('cell_id')
    new_table = data.get('data')
    if cell_id in cell_data_cache:
        cell_data_cache[cell_id]['data'] = new_table
        return jsonify({'status': 'success'})

    return jsonify({'status': 'error', 'message': 'Cell ID not found'}), 404


# Configure logging
logging.basicConfig(level=logging.DEBUG)
app.logger.setLevel(logging.DEBUG)


@app.route('/get_ranges')
def get_ranges():
    """
    Load and return ranges data from the standard ranges.json file.

    Returns:
        dict: JSON object containing ranges data from ranges.json
        tuple: Error response with status code 500 if file cannot be loaded
    """
    # Load ranges from the standard ranges.json file
    ranges_path = os.path.join(app.root_path, 'data', 'SiC', 'source', 'ranges.json')
    try:
        with open(ranges_path, 'r', encoding='utf-8') as ranges_file:
            ranges_data = json.load(ranges_file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f'Error loading ranges.json from {ranges_path}: {e}')
        return jsonify({'error': 'Could not load ranges data'}), 500
    return jsonify(ranges_data)


@app.route('/get_thickness_models')
def get_thickness_models():
    """
    Retrieve list of available thickness model files for specified ML method.

    Query parameters:
        ml_method (str): ML method directory name (default: 'neural_network')

    Returns:
        list: JSON array of .pth model filenames in the specified directory
        tuple: Error response with status code 500 if directory cannot be accessed
    """
    # Get ML method from query string; default to "neural_network"
    ml_method = request.args.get('ml_method', 'neural_network')
    # Construct the full path to the model folder.
    model_folder = os.path.join(app.root_path, 'models', 'sic', 'thickness', ml_method, 'demo')
    try:
        # List files in the folder (you can filter further if needed, e.g. by extension ".pt")
        files = os.listdir(model_folder)
        # For example, filter only files that end with ".pth"
        model_files = [f for f in files if f.endswith('.pth')]
        model_files.sort()
        return jsonify(model_files)
    except (FileNotFoundError, OSError) as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get_doping_models')
def get_doping_models():
    """
    Retrieve list of available doping model files for specified ML method.

    Query parameters:
        ml_method (str): ML method directory name (default: 'neural_network')

    Returns:
        list: JSON array of .pth/.pt model filenames in the specified directory
        tuple: Error response with status code 500 if directory cannot be accessed
    """
    ml_method = request.args.get('ml_method', 'neural_network')
    folder = os.path.join(app.root_path, 'models', 'sic', 'doping', ml_method, 'demo')
    app.logger.debug('get_doping_models → looking in %s', folder)
    try:
        files = os.listdir(folder)
        model_files = [f for f in files if f.endswith(('.pth', '.pt'))]
        model_files.sort()
        return jsonify(model_files)
    except (FileNotFoundError, OSError) as e:
        app.logger.error('get_doping_models error: %s', e)
        return jsonify({'error': str(e)}), 500


@app.route('/start_training_skip1', methods=['POST'])
def start_training_skip1():
    """
    Start LSTM model training for both thickness and doping prediction.

    Executes two Python training scripts:
    - lstm_thickness_unshuffle_skip1_v2.py for thickness model
    - lstm_doping_unshuffle_skip1.py for doping model

    Returns:
        dict: JSON object with status "success" and completion message

    Raises:
        subprocess.CalledProcessError: If any training script fails to execute
    """
    # Run the updated training script
    subprocess.run(
        ['python3', 'models/sic/thickness/lstm/lstm_thickness_unshuffle_skip1_v2.py'], check=True
    )
    subprocess.run(['python3', 'models/sic/doping/lstm/lstm_doping_unshuffle_skip1.py'], check=True)
    return jsonify({'status': 'success', 'message': 'Training started successfully.'})


if __name__ == '__main__':
    app.run(host='172.20.76.31', debug=True, port=8080)
