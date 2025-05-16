#!/usr/bin/env python3
# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

"""
Super AI Runner

Main entry point for starting the Super AI system with proper initialization
of dependencies, models, and APIs. Supports different startup modes.
"""

import os
import sys
import argparse
import logging
import asyncio
from pathlib import Path
from datetime import datetime
import json
import dotenv

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure basic logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{log_dir}/super_ai.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("super_ai")

def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Super AI System")

    # Global options
    parser.add_argument('--mode', type=str, choices=['api', 'interactive', 'train', 'scan', 'process', 'conjectures'],
                        default='api', help='Mode of operation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--config', type=str, default='.env',
                        help='Path to configuration file (.env)')

    # API options
    parser.add_argument('--host', type=str, default=None,
                        help='API server host (overrides config)')
    parser.add_argument('--port', type=int, default=None,
                        help='API server port (overrides config)')

    # Model options
    parser.add_argument('--sports-model', type=str, default=None,
                        help='Path to sports prediction model')
    parser.add_argument('--betting-model', type=str, default=None,
                        help='Path to betting prediction model')

    # Directory scanner options
    parser.add_argument('--scan-dir', type=str, default=None,
                        help='Directory to scan (defaults to project root)')
    parser.add_argument('--report', type=str, default=None,
                        help='Path to write scan report (defaults to scan_report_TIMESTAMP.md)')
    parser.add_argument('--detailed', action='store_true',
                        help='Enable detailed logging for directory scanner')

    # File processor options
    parser.add_argument('--process-dir', type=str, default=None,
                        help='Directory to process files (defaults to project root)')
    parser.add_argument('--recursive', action='store_true',
                        help='Process directories recursively')
    parser.add_argument('--file-type', type=str, default=None,
                        help='Filter by file type (e.g., py,csv,log)')

    # Conjecture engine options
    parser.add_argument('--input-data', type=str, default=None,
                        help='Path to input data file for conjecture analysis (CSV format)')
    parser.add_argument('--specific-conjectures', type=str, default=None,
                        help='Comma-separated list of specific conjectures to analyze with')
    parser.add_argument('--output-file', type=str, default=None,
                        help='Path to output analysis results (JSON format)')

    return parser.parse_args()

def load_env_config(config_path='.env'):
    """Load configuration from .env file"""
    config = {}

    # Try to load from .env file
    env_path = Path(config_path)
    if env_path.exists():
        logger.info(f"Loading configuration from {env_path}")
        dotenv.load_dotenv(env_path)
    else:
        logger.warning(f"No config file found at {config_path}, using default or environment values")

    # Set configuration with defaults
    config["API_HOST"] = os.getenv("API_HOST", "127.0.0.1")
    config["API_PORT"] = int(os.getenv("API_PORT", "5000"))
    config["API_DEBUG"] = os.getenv("API_DEBUG", "False").lower() in ("true", "1", "t")
    config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")
    config["SPORTS_MODEL_PATH"] = os.getenv("SPORTS_MODEL_PATH", None)
    config["BETTING_MODEL_PATH"] = os.getenv("BETTING_MODEL_PATH", None)
    config["ENABLE_DANGEROUS_AI"] = os.getenv("ENABLE_DANGEROUS_AI", "False").lower() in ("true", "1", "t")
    config["CUDA_VISIBLE_DEVICES"] = os.getenv("CUDA_VISIBLE_DEVICES", "")

    # Log the configuration (excluding sensitive data)
    safe_config = {k: v for k, v in config.items() if not k.endswith("KEY") and not k.endswith("SECRET")}
    logger.info(f"Configuration loaded: {json.dumps(safe_config, indent=2)}")

    return config

def setup_logging(log_level):
    """Configure detailed logging"""
    # Determine the log level
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        logger.warning(f"Invalid log level: {log_level}, defaulting to INFO")
        numeric_level = logging.INFO

    # Set the root logger level
    logging.getLogger().setLevel(numeric_level)

    return True

def check_pytorch():
    """Check PyTorch availability and configuration"""
    try:
        import torch

        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()

        logger.info(f"PyTorch version: {torch_version}, CUDA available: {cuda_available}")

        if cuda_available:
            gpu_count = torch.cuda.device_count()
            gpu_names = [torch.cuda.get_device_name(i) for i in range(gpu_count)]
            logger.info(f"Found {gpu_count} GPU(s): {', '.join(gpu_names)}")
        else:
            logger.warning("No GPU found. Running on CPU only.")

        return True
    except ImportError:
        logger.error("PyTorch not installed. Please install PyTorch to use AI features.")
        return False
    except Exception as e:
        logger.error(f"Error checking PyTorch: {e}")
        return False

def check_dependencies():
    """Check and report on dependencies"""
    try:
        # Create a list of dependencies to check
        dependencies = [
            ("numpy", "numpy"),
            ("pandas", "pandas"),
            ("sklearn", "scikit-learn"),
            ("torch", "torch"),
            ("flask", "flask"),
            ("matplotlib", "matplotlib"),
            ("requests", "requests"),
            ("dotenv", "python-dotenv")
        ]

        missing = []
        installed = []

        for module_name, package_name in dependencies:
            try:
                module = __import__(module_name)
                version = getattr(module, "__version__", "unknown")
                installed.append(f"{package_name} ({version})")
            except ImportError:
                missing.append(package_name)

        if installed:
            logger.info(f"Installed dependencies: {', '.join(installed)}")

        if missing:
            logger.warning(f"Missing dependencies: {', '.join(missing)}")
            logger.warning("Please install missing dependencies: pip install " + " ".join(missing))

        return len(missing) == 0
    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        return False

def initialize_system():
    """Initialize the system"""
    logger.info("=== SUPER AI INITIALIZATION STARTED ===")
    start_time = datetime.now()

    # Create necessary directories
    for directory in ["logs", "data", "models", "config"]:
        os.makedirs(directory, exist_ok=True)

    # Check dependencies
    dependencies_ok = check_dependencies()

    # Check PyTorch
    pytorch_status = check_pytorch()

    # Log initialization complete
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.info(f"=== SUPER AI INITIALIZATION COMPLETED IN {elapsed_time:.2f}s ===")

    return {
        'success': dependencies_ok and pytorch_status,
        'dependencies_ok': dependencies_ok,
        'pytorch_ok': pytorch_status
    }

def start_api(config):
    """Start the API server"""
    try:
        # Import predictor module
        from prediction.model_integration import IntegratedPredictor

        # Import API module
        from api_interface.predictor_api import run_app

        # Initialize the predictor with model paths
        predictor = IntegratedPredictor(
            sports_model_path=config.get("SPORTS_MODEL_PATH"),
            betting_model_path=config.get("BETTING_MODEL_PATH")
        )

        # Start the API server
        host = config.get("API_HOST")
        port = config.get("API_PORT")
        debug = config.get("API_DEBUG", False)

        logger.info(f"Starting API server on {host}:{port}")
        run_app(host=host, port=port, debug=debug)

        return True
    except ImportError as e:
        logger.error(f"Failed to import required modules: {e}")
        logger.error("Please make sure all dependencies are installed.")
        return False
    except Exception as e:
        logger.error(f"Error starting API: {e}")
        return False

def start_core_ai(config):
    """Start the core AI system"""
    try:
        # Import core AI module
        from core_ai.main import main as core_ai_main

        # Set environment variables for the core AI
        os.environ['API_HOST'] = config.get("API_HOST")
        os.environ['API_PORT'] = str(config.get("API_PORT"))

        # Run the core AI main function
        logger.info("Starting core AI system")
        result = core_ai_main()

        if result.get("status") == "success":
            logger.info("Core AI system started successfully")
            return True
        else:
            logger.error(f"Failed to start core AI: {result.get('message', 'Unknown error')}")
            return False
    except ImportError as e:
        logger.error(f"Failed to import core AI modules: {e}")
        logger.error("Please check the core_ai module structure.")
        return False
    except Exception as e:
        logger.error(f"Error starting core AI: {e}")
        return False

def run_directory_scan(scan_dir=None, report_path=None, detailed_logging=False):
    """Run the directory scanner"""
    try:
        # Import the directory scanner
        from utils.directory_scanner import DirectoryScanner

        # Set default scan directory to project root if not specified
        if scan_dir is None:
            scan_dir = Path(__file__).parent
        else:
            scan_dir = Path(scan_dir)

        # Set default report path if not specified
        if report_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_path = Path(f"scan_report_{timestamp}.md")
        else:
            report_path = Path(report_path)

        logger.info(f"Starting directory scan at: {scan_dir}")

        # Create and run scanner
        scanner = DirectoryScanner(scan_dir)
        results = scanner.scan(detailed_logging=detailed_logging)

        # Generate the report
        report = scanner.generate_report(report_path)

        logger.info(f"Scan complete. Report written to: {report_path}")
        logger.info(f"Found {len(results['python_modules'])} Python modules, "
                   f"{len(results['data_files'])} data files, "
                   f"{len(results['log_files'])} log files")

        return True
    except ImportError as e:
        logger.error(f"Failed to import directory scanner: {e}")
        return False
    except Exception as e:
        logger.error(f"Error running directory scan: {e}")
        return False

def process_files(process_dir=None, recursive=False, file_type=None):
    """Process files in the specified directory"""
    try:
        # Import the file processor
        from utils.file_processor import FileProcessor

        # Set default process directory to project root if not specified
        if process_dir is None:
            process_dir = Path(__file__).parent
        else:
            process_dir = Path(process_dir)

        logger.info(f"Starting file processing at: {process_dir}")
        logger.info(f"Recursive: {recursive}, File type filter: {file_type}")

        # Create file processor
        processor = FileProcessor(process_dir)

        # Process files based on filter
        if file_type:
            # Convert comma-separated string to list of extensions with dots
            extensions = [f".{ext.strip()}" if not ext.strip().startswith('.') else ext.strip()
                          for ext in file_type.split(',')]

            logger.info(f"Filtering by extensions: {extensions}")

            # Walk directory and process matching files
            for root, _, files in os.walk(process_dir):
                root_path = Path(root)
                for file in files:
                    file_path = root_path / file
                    if any(file_path.suffix.lower() == ext.lower() for ext in extensions):
                        processor.process_file(file_path)

                # Stop after first directory if not recursive
                if not recursive:
                    break
        else:
            # Process all files
            processor.process_directory(process_dir, recursive)

        # Log summary
        total_files = sum(len(files) for files in processor.processed_files.values())
        logger.info(f"File processing complete. Processed {total_files} files:")
        logger.info(f"  - Python modules: {len(processor.processed_files['python'])}")
        logger.info(f"  - Data files: {len(processor.processed_files['data'])}")
        logger.info(f"  - Image files: {len(processor.processed_files['images'])}")
        logger.info(f"  - Log files: {len(processor.processed_files['logs'])}")
        logger.info(f"  - Frontend assets: {len(processor.processed_files['frontend'])}")

        return True
    except ImportError as e:
        logger.error(f"Failed to import file processor: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing files: {e}")
        return False

def start_conjectures_engine(input_data_path=None, specific_conjectures=None, output_file=None):
    """Start the mathematical conjectures AI engine"""
    try:
        # Import the conjecture AI engine
        from algorithms.conjecture_ai_engine import conjecture_ai_engine
        import numpy as np
        import pandas as pd
        import json

        logger.info("Starting Mathematical Conjectures AI Engine")

        # List available conjectures
        available_conjectures = conjecture_ai_engine.list_available_conjectures()
        logger.info(f"Available conjectures: {available_conjectures}")

        # Check if we have input data
        if input_data_path is None:
            logger.error("No input data provided. Use --input-data to specify input data file.")
            return False

        # Load input data
        try:
            input_data_path = Path(input_data_path)
            if not input_data_path.exists():
                logger.error(f"Input data file not found: {input_data_path}")
                return False

            # Try to load as CSV
            if input_data_path.suffix.lower() == '.csv':
                logger.info(f"Loading CSV file: {input_data_path}")
                df = pd.read_csv(input_data_path)

                # Extract only numeric columns
                numeric_df = df.select_dtypes(include=['number'])

                if numeric_df.empty:
                    # If no numeric columns, try to convert string columns to numeric
                    logger.info("No numeric columns found, attempting to convert string columns")
                    for col in df.columns:
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                        except:
                            logger.warning(f"Could not convert column {col} to numeric")

                    # Try again with converted columns
                    numeric_df = df.select_dtypes(include=['number'])

                if numeric_df.empty:
                    logger.error("No numeric data found in CSV file")
                    return False

                logger.info(f"Using {len(numeric_df.columns)} numeric columns from CSV")

                # Convert all numeric columns to a flat array, dropping NaN values
                input_data = numeric_df.values.flatten()
                input_data = input_data[~np.isnan(input_data)]

                if len(input_data) == 0:
                    logger.error("No valid numeric data found after filtering")
                    return False
            else:
                # Try to load as plain text with numbers
                with open(input_data_path, 'r') as f:
                    content = f.read()
                    # Split by whitespace and convert to float, ignoring errors
                    values = []
                    for item in content.split():
                        try:
                            values.append(float(item))
                        except ValueError:
                            pass  # Skip non-numeric values

                    input_data = np.array(values)

                    if len(input_data) == 0:
                        logger.error("No valid numeric data found in text file")
                        return False

            logger.info(f"Loaded input data with {len(input_data)} values")
            logger.info(f"Data sample: {input_data[:5]} ... {input_data[-5:] if len(input_data) > 5 else ''}")

            # Check if we have specific conjectures to analyze with
            if specific_conjectures:
                conjecture_list = [c.strip() for c in specific_conjectures.split(',')]
                logger.info(f"Using specific conjectures: {conjecture_list}")
                results = conjecture_ai_engine.analyze_specific(input_data, conjecture_list)
            else:
                # Use all available conjectures
                logger.info("Using all available conjectures")
                results = conjecture_ai_engine.analyze(input_data)

            # Print results
            logger.info(f"Analysis complete. Overall score: {results['weighted_score']:.3f}")
            if 'top_conjectures' in results:
                logger.info("Top conjectures:")
                for name, score in results['top_conjectures']:
                    logger.info(f"  - {name}: {score:.3f}")

            logger.info("Insights:")
            for insight in results['insights']:
                logger.info(f"  - {insight}")

            # Save results if output file specified
            if output_file:
                output_path = Path(output_file)
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2)
                logger.info(f"Results saved to {output_path}")

            return True

        except Exception as e:
            logger.error(f"Error loading or processing input data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    except ImportError as e:
        logger.error(f"Failed to import conjecture AI engine: {e}")
        logger.error("Please make sure the algorithms module is properly installed.")
        return False
    except Exception as e:
        logger.error(f"Error starting conjecture AI engine: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main entry point"""
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration
    config = load_env_config(args.config)

    # Override config with command-line arguments
    if args.host:
        config["API_HOST"] = args.host
    if args.port:
        config["API_PORT"] = args.port
    if args.sports_model:
        config["SPORTS_MODEL_PATH"] = args.sports_model
    if args.betting_model:
        config["BETTING_MODEL_PATH"] = args.betting_model
    if args.debug:
        config["LOG_LEVEL"] = "DEBUG"
        config["API_DEBUG"] = True

    # Setup logging
    setup_logging(config["LOG_LEVEL"])

    # Handle scan mode separately - no need for full initialization
    if args.mode == 'scan':
        return 0 if run_directory_scan(args.scan_dir, args.report, args.detailed) else 1

    # Handle process mode separately - process files without full initialization
    if args.mode == 'process':
        return 0 if process_files(args.process_dir, args.recursive, args.file_type) else 1

    # Handle conjectures mode - run the mathematical conjectures engine
    if args.mode == 'conjectures':
        return 0 if start_conjectures_engine(args.input_data, args.specific_conjectures, args.output_file) else 1

    # Initialize system
    init_result = initialize_system()
    if not init_result['success']:
        logger.error("System initialization failed")
        return 1

    # Run in the appropriate mode
    try:
        if args.mode == 'api':
            if not start_api(config):
                logger.warning("Failed to start prediction API, falling back to core AI API")
                if start_core_ai(config):
                    logger.info("Using core AI API server instead")
                    # This will block until the server is stopped
                else:
                    logger.error("Failed to start any API server")
                    return 1
        elif args.mode == 'interactive':
            logger.error("Interactive mode not yet implemented. Please use API mode.")
            return 1
        elif args.mode == 'train':
            logger.error("Training mode not yet implemented. Please use API mode.")
            return 1
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, shutting down...")
        return 0
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())