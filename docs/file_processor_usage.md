# File Processor Documentation

## Overview

The File Processor module provides utilities for dynamically processing different types of files in the Super AI project. It supports various file types including Python modules, data files (CSV, JSON), images, log files, and frontend assets.

## Features

- **Dynamic Module Loading**: Load Python modules at runtime
- **Data File Processing**: Read and analyze CSV and other data files
- **Log File Analysis**: Scan log files for errors and patterns
- **Image Processing**: Process image files and get dimensions
- **Frontend Asset Detection**: Identify and categorize frontend assets

## Usage

### Basic Usage

```python
from utils.file_processor import FileProcessor, process_files

# Process an entire directory recursively
results = process_files('path/to/directory', recursive=True)

# Or create a processor instance for more control
processor = FileProcessor(base_dir='path/to/base/dir')
processor.process_directory('relative/path', recursive=True)

# Process individual files
module = processor.process_python_file('my_module.py')
data = processor.process_csv_file('data.csv')
error_count, errors = processor.process_log_file('app.log')
```

### Command Line

The file processor can be run from the command line using the `run.py` script:

```bash
# Process all files recursively
python run.py --mode process --process-dir . --recursive

# Process only specific file types
python run.py --mode process --process-dir data --file-type csv,json

# Use the simplified test script
python simple_processor_test.py
```

## Supported File Types

| File Type | Extensions | Processing Function |
|-----------|------------|---------------------|
| Python Modules | .py | process_python_file() |
| Data Files | .csv, .json, .jsonl, .xml, .txt | process_csv_file() |
| Image Files | .png, .jpg, .jpeg, .gif, .bmp | process_image_file() |
| Log Files | .log | process_log_file() |
| Frontend Assets | .html, .htm, .js, .css, .jsx, .tsx, .vue | process_frontend_asset() |

## File Processor Output

The file processor generates:

1. Logs for each processed file
2. A summary of processed files by category
3. A collection of loaded modules and processed data
4. Error reports for log files

## Integration with Directory Scanner

The File Processor can be used in combination with the DirectoryScanner to:

1. First scan a directory structure using `DirectoryScanner`
2. Then process specific file types using `FileProcessor`

```python
from utils.directory_scanner import DirectoryScanner
from utils.file_processor import FileProcessor

# Scan directories
scanner = DirectoryScanner('project_root')
results = scanner.scan()

# Process specific files
processor = FileProcessor('project_root')
for py_file in results['python_modules']:
    processor.process_python_file(py_file)
```

## Dependencies

The File Processor has the following optional dependencies:

- **pandas**: For processing CSV and data files
- **Pillow**: For processing image files

These dependencies are imported conditionally, so the core functionality works even if they are not installed.

## Error Handling

The File Processor gracefully handles errors during file processing:

- Missing modules or import errors are logged
- File access errors are caught and reported
- Unsupported file types generate warnings
- All errors are logged but don't stop processing other files
