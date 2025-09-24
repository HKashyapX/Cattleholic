import os
import shutil
from PIL import Image
import logging
import unicodedata
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def normalize_filename(filename):
    """Normalize unicode characters in filename and remove special characters."""
    # Get the name and extension separately
    name, ext = os.path.splitext(filename)
    
    # Normalize unicode characters
    normalized = unicodedata.normalize('NFKD', str(name))
    # Replace spaces and problematic characters
    normalized = normalized.replace(' ', '_')
    normalized = normalized.replace('(', '')
    normalized = normalized.replace(')', '')
    normalized = normalized.replace('-', '_')
    # Remove any remaining non-ASCII characters
    normalized = ''.join(c for c in normalized if ord(c) < 128)
    # Remove multiple underscores
    while '__' in normalized:
        normalized = normalized.replace('__', '_')
    # Add back the extension
    return normalized + ext.lower()

def check_file_encoding(file_path):
    """Check if a file can be opened and has valid encoding."""
    try:
        path = Path(file_path)
        if path.name != normalize_filename(path.name):
            return False, "Filename contains special characters"
        # Try to open as image
        with Image.open(file_path) as img:
            img.verify()  # Verify it's a valid image
        return True, "Valid image"
    except UnicodeDecodeError as e:
        return False, f"Unicode decode error: {e}"
    except Exception as e:
        return False, f"Other error: {e}"

def find_problematic_files(directory):
    """Find all files with encoding issues."""
    problematic_files = []

    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            is_valid, error_msg = check_file_encoding(file_path)

            if not is_valid:
                problematic_files.append((file_path, error_msg))
                logger.warning(f"Problematic file: {file_path} - {error_msg}")

    return problematic_files

def fix_file_encoding(file_path):
    """Fix encoding issues in file name."""
    path = Path(file_path)
    new_name = normalize_filename(path.name)
    if new_name != path.name:
        new_path = path.parent / new_name
        try:
            # Ensure the new path doesn't already exist
            if not new_path.exists():
                path.rename(new_path)
                logger.info(f"Renamed: {path.name} -> {new_path.name}")
                return str(new_path)
            else:
                logger.warning(f"Cannot rename {path} - destination already exists")
                return str(path)
        except Exception as e:
            logger.error(f"Error renaming {path}: {str(e)}")
            return str(path)
    return str(path)

def create_backup_and_clean(directory, problematic_files):
    """Create a backup and fix problematic files."""
    backup_dir = f"{directory}_backup"
    if not os.path.exists(backup_dir):
        shutil.copytree(directory, backup_dir)
        logger.info(f"Created backup at {backup_dir}")

    # Fix problematic files
    for file_path, _ in problematic_files:
        try:
            fixed_path = fix_file_encoding(file_path)
            logger.info(f"Fixed file: {file_path} -> {fixed_path}")
        except Exception as e:
            logger.error(f"Failed to fix {file_path}: {e}")

def main():
    data_dir = "data"  # Updated to target the main data directory

    if not os.path.exists(data_dir):
        logger.error(f"Directory {data_dir} does not exist")
        return

    logger.info("Scanning for problematic files...")
    problematic_files = find_problematic_files(data_dir)

    if not problematic_files:
        logger.info("No problematic files found. The issue might be elsewhere.")
        return

    logger.info(f"Found {len(problematic_files)} problematic files.")

    # Ask user for confirmation
    print(f"Found {len(problematic_files)} problematic files.")
    print("These files will be removed. A backup will be created.")
    response = input("Do you want to proceed? (y/n): ")

    if response.lower() == 'y':
        create_backup_and_clean(data_dir, problematic_files)
        logger.info("Cleanup completed.")
    else:
        logger.info("Operation cancelled.")

if __name__ == "__main__":
    main()
