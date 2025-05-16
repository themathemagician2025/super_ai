# -----------------------------------------------------------
# Author  : Clive Dziwomore (aka The Mathemagician)
# Twitter : @CDziwomore
# LinkedIn: https://www.linkedin.com/in/clive-dziwomore-194467206/
# Phone   : +91 6309784662 / +263 712967390
# -----------------------------------------------------------

# src/modelweights.py
import pickle
import os
import logging
import gzip
import json
import shutil
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import neat
from deap import gp
from config import PROJECT_CONFIG  # For directory structure and settings

# Project directory structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, 'src')
LOG_DIR = os.path.join(BASE_DIR, 'log')
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(DATA_DIR, 'models')
BACKUP_DIR = os.path.join(MODELS_DIR, 'backups')

# Ensure directories exist
for directory in [SRC_DIR, LOG_DIR, MODELS_DIR, BACKUP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=os.path.join(LOG_DIR, 'mathemagician.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_genome(
        genome: Any,
        filename: str,
        compress: bool = False,
        versioned: bool = True) -> bool:
    """
    Save a genome object to a file using pickle, with optional compression and versioning.

    Args:
        genome: The genome object (NEAT or GP) to save.
        filename: Destination filename (relative to data/models).
        compress: Whether to use gzip compression.
        versioned: Whether to append a timestamp for versioning.

    Returns:
        bool: True if successful, False otherwise.
    """
    if versioned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{timestamp}{ext}"

    filepath = os.path.join(MODELS_DIR, filename)
    try:
        if compress:
            with gzip.open(filepath + '.gz', 'wb') as f:
                pickle.dump(genome, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Compressed genome saved to {filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(genome, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Genome saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving genome to {filepath}: {e}")
        return False


def load_genome(filename: str, decompress: bool = False) -> Optional[Any]:
    """
    Load a genome object from a pickle file.

    Args:
        filename: Source filename (relative to data/models).
        decompress: Whether to expect a gzip-compressed file.

    Returns:
        Any: The loaded genome object, or None if failed.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if decompress:
        filepath += '.gz'  # Ensure .gz extension is appended correctly

    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found")
        return None

    try:
        if decompress:
            with gzip.open(filepath, 'rb') as f:
                genome = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                genome = pickle.load(f)
        logger.info(f"Genome loaded from {filepath}")
        return genome
    except Exception as e:
        logger.error(f"Error loading genome from {filepath}: {e}")
        return None


def save_population(population: Union[neat.Population,
                                      List[gp.PrimitiveTree]],
                    filename: str,
                    compress: bool = False,
                    versioned: bool = True) -> bool:
    """
    Save a population object to a file using pickle.

    Args:
        population: NEAT Population or list of GP trees.
        filename: Destination filename (relative to data/models).
        compress: Whether to use gzip compression.
        versioned: Whether to append a timestamp.

    Returns:
        bool: True if successful, False otherwise.
    """
    if versioned:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        filename = f"{base}_{timestamp}{ext}"

    filepath = os.path.join(MODELS_DIR, filename)
    try:
        if compress:
            with gzip.open(filepath + '.gz', 'wb') as f:
                pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Compressed population saved to {filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(population, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Population saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving population to {filepath}: {e}")
        return False


def load_population(filename: str,
                    decompress: bool = False) -> Optional[Union[neat.Population,
                                                                List[gp.PrimitiveTree]]]:
    """
    Load a population object from a pickle file.

    Args:
        filename: Source filename (relative to data/models).
        decompress: Whether to expect a gzip-compressed file.

    Returns:
        Union: The loaded population object, or None if failed.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if decompress:
        filepath += '.gz'

    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found")
        return None

    try:
        if decompress:
            with gzip.open(filepath, 'rb') as f:
                population = pickle.load(f)
        else:
            with open(filepath, 'rb') as f:
                population = pickle.load(f)
        logger.info(f"Population loaded from {filepath}")
        return population
    except Exception as e:
        logger.error(f"Error loading population from {filepath}: {e}")
        return None


def save_parameters(parameters: Dict[str, Any], filename: str,
                    compress: bool = False, json_format: bool = False) -> bool:
    """
    Save model parameters to a file using pickle or JSON.

    Args:
        parameters: Dictionary of parameters (e.g., weights, hyperparameters).
        filename: Destination filename (relative to data/models).
        compress: Whether to use gzip compression (pickle only).
        json_format: Whether to save as JSON instead of pickle.

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        if json_format:
            with open(filepath + '.json', 'w') as f:
                json.dump(parameters, f, indent=4)
            logger.info(f"Parameters saved as JSON to {filepath}.json")
        elif compress:
            with gzip.open(filepath + '.gz', 'wb') as f:
                pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Compressed parameters saved to {filepath}.gz")
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(parameters, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Parameters saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving parameters to {filepath}: {e}")
        return False


def load_parameters(filename: str, decompress: bool = False,
                    json_format: bool = False) -> Optional[Dict[str, Any]]:
    """
    Load model parameters from a file.

    Args:
        filename: Source filename (relative to data/models).
        decompress: Whether to expect a gzip-compressed file (pickle only).
        json_format: Whether to load from JSON instead of pickle.

    Returns:
        Dict: Loaded parameters, or None if failed.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if json_format:
        filepath += '.json'
    elif decompress:
        filepath += '.gz'

    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found")
        return None

    try:
        if json_format:
            with open(filepath, 'r') as f:
                parameters = json.load(f)
            logger.info(f"Parameters loaded as JSON from {filepath}")
        elif decompress:
            with gzip.open(filepath, 'rb') as f:
                parameters = pickle.load(f)
            logger.info(f"Compressed parameters loaded from {filepath}")
        else:
            with open(filepath, 'rb') as f:
                parameters = pickle.load(f)
            logger.info(f"Parameters loaded from {filepath}")
        return parameters
    except Exception as e:
        logger.error(f"Error loading parameters from {filepath}: {e}")
        return None


def backup_model(filename: str, source_dir: str = MODELS_DIR) -> bool:
    """
    Create a backup of a model file in the backups directory.

    Args:
        filename: Filename to back up.
        source_dir: Directory containing the original file (default: data/models).

    Returns:
        bool: True if successful, False otherwise.
    """
    src = os.path.join(source_dir, filename)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(BACKUP_DIR, f"{filename}.{timestamp}.bak")

    if not os.path.exists(src):
        logger.error(f"Cannot backup {src}: file not found")
        return False

    try:
        shutil.copy2(src, dst)
        logger.info(f"Backed up {src} to {dst}")
        return True
    except Exception as e:
        logger.error(f"Error backing up {src} to {dst}: {e}")
        return False


def list_model_files(directory: str = MODELS_DIR) -> List[str]:
    """List all model-related files in the specified directory."""
    return [
        f for f in os.listdir(directory) if f.endswith(
            ('.pkl', '.gz', '.json', '.bak'))]


def verify_genome_integrity(genome: Any) -> bool:
    """Verify the integrity of a loaded genome."""
    if isinstance(genome, neat.DefaultGenome):
        return hasattr(genome, 'connections') and hasattr(genome, 'nodes')
    elif isinstance(genome, gp.PrimitiveTree):
        return len(genome) > 0
    elif isinstance(genome, dict):
        return 'id' in genome and 'data' in genome
    logger.warning("Unknown genome type; integrity check skipped")
    return True


def save_genome_with_metadata(
        genome: Any, filename: str, metadata: Dict[str, Any]) -> bool:
    """
    Save a genome with additional metadata.

    Args:
        genome: The genome object to save.
        filename: Destination filename.
        metadata: Dictionary of metadata (e.g., fitness, generation).

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    try:
        data = {
            'genome': genome,
            'metadata': metadata,
            'timestamp': str(
                datetime.now())}
        with open(filepath, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        logger.info(f"Genome with metadata saved to {filepath}")
        return True
    except Exception as e:
        logger.error(f"Error saving genome with metadata to {filepath}: {e}")
        return False


def load_genome_with_metadata(filename: str) -> Optional[Dict[str, Any]]:
    """
    Load a genome with its metadata.

    Args:
        filename: Source filename.

    Returns:
        Dict: Containing 'genome', 'metadata', and 'timestamp', or None if failed.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    if not os.path.exists(filepath):
        logger.error(f"File {filepath} not found")
        return None

    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        logger.info(f"Genome with metadata loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(
            f"Error loading genome with metadata from {filepath}: {e}")
        return None


def dangerous_save_genome(genome: Any, filename: str) -> bool:
    """
    Save a genome without safety checks (dangerous AI theme).

    Args:
        genome: The genome to save.
        filename: Destination filename.

    Returns:
        bool: True if successful, False otherwise.
    """
    filepath = os.path.join(MODELS_DIR, filename)
    with open(filepath, 'wb') as f:  # No try-except for danger
        pickle.dump(genome, f)
    logger.warning(
        f"Dangerous save: Genome written to {filepath} without safety checks")
    return True


def main():
    """Demonstrate model weights functionality."""
    # Dummy data
    dummy_genome = {'id': 1, 'data': [0.1, 0.2, 0.3]}
    dummy_population = [dummy_genome, {'id': 2, 'data': [0.4, 0.5, 0.6]}]
    dummy_parameters = {'learning_rate': 0.01, 'momentum': 0.9}

    # Save and load genome
    save_genome(dummy_genome, "dummy_genome.pkl", compress=True)
    loaded_genome = load_genome("dummy_genome.pkl", decompress=True)
    if loaded_genome:
        print(f"Loaded genome: {loaded_genome}")

    # Save and load population
    save_population(dummy_population, "dummy_population.pkl", versioned=True)
    loaded_pop = load_population("dummy_population.pkl")
    if loaded_pop:
        print(f"Loaded population: {len(loaded_pop)} individuals")

    # Save and load parameters
    save_parameters(
        dummy_parameters,
        "dummy_parameters.json",
        json_format=True)
    loaded_params = load_parameters("dummy_parameters.json", json_format=True)
    if loaded_params:
        print(f"Loaded parameters: {loaded_params}")

    # Backup and list files
    backup_model("dummy_genome.pkl.gz")
    files = list_model_files()
    print(f"Model files: {files}")

    # Save with metadata
    metadata = {'fitness': 0.95, 'generation': 42}
    save_genome_with_metadata(dummy_genome, "meta_genome.pkl", metadata)
    data = load_genome_with_metadata("meta_genome.pkl")
    if data:
        print(f"Loaded with metadata: {data['metadata']}")


if __name__ == "__main__":
    main()

# Additional utilities


def clean_old_backups(max_backups: int = 5) -> None:
    """Remove old backups to maintain a maximum number."""
    backups = sorted([f for f in os.listdir(BACKUP_DIR)
                     if f.endswith('.bak')], reverse=True)
    for old_file in backups[max_backups:]:
        os.remove(os.path.join(BACKUP_DIR, old_file))
        logger.info(f"Cleaned old backup: {old_file}")


def convert_pickle_to_json(pickle_file: str, json_file: str) -> bool:
    """Convert a pickle file to JSON format."""
    data = load_genome(pickle_file)
    if data is None:
        return False
    return save_parameters(data, json_file, json_format=True)


def batch_save_genomes(
        genomes: List[Any],
        prefix: str = "genome") -> List[str]:
    """Save multiple genomes with sequential filenames."""
    filenames = []
    for i, genome in enumerate(genomes):
        filename = f"{prefix}_{i}.pkl"
        if save_genome(genome, filename):
            filenames.append(filename)
    return filenames
