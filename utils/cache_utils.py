"""
Cache management utilities for EDT-Former datasets.
Provides functions for cache validation, clearing, and information retrieval.
"""
import os
import glob
import hashlib
import pickle
from typing import Optional, Dict, List, Tuple


def get_cache_dir() -> str:
    """
    Get the cache directory path.
    
    Returns:
        str: Absolute path to cache directory
    """
    return os.environ.get('DATA_CACHE_DIR', 
                         os.path.join(os.environ.get('DATA_DIR', './data'), '.cache'))


def get_cache_key(root: str, filename: str, encoder_types: List[str]) -> str:
    """
    Generate a unique cache key based on root path, filename, and encoder types.
    
    Args:
        root: Root directory path for the dataset
        filename: Name of the data file (e.g., 'train_mol.json')
        encoder_types: List of encoder types (e.g., ['unimol', 'moleculestm'])
    
    Returns:
        str: Unique cache key
    """
    # Use hash of absolute root path to avoid collisions between different datasets
    root_hash = hashlib.md5(os.path.abspath(root).encode()).hexdigest()[:8]
    cache_key = f"{root_hash}_{filename.replace('.json', '')}_{'-'.join(encoder_types)}"
    return cache_key


def get_cache_path(root: str, filename: str, encoder_types: List[str]) -> str:
    """
    Get the full path to a cache file.
    
    Args:
        root: Root directory path for the dataset
        filename: Name of the data file (e.g., 'train_mol.json')
        encoder_types: List of encoder types
    
    Returns:
        str: Full path to cache file
    """
    cache_dir = get_cache_dir()
    cache_key = get_cache_key(root, filename, encoder_types)
    return os.path.join(cache_dir, f"mol_dataset_{cache_key}.pkl")


def is_cache_valid(cache_path: str, source_path: str) -> bool:
    """
    Check if a cache file is valid (exists and is newer than source).
    
    Args:
        cache_path: Path to cache file
        source_path: Path to source data file
    
    Returns:
        bool: True if cache is valid, False otherwise
    """
    if not os.path.exists(cache_path):
        return False
    
    if not os.path.exists(source_path):
        return False
    
    cache_mtime = os.path.getmtime(cache_path)
    source_mtime = os.path.getmtime(source_path)
    
    return cache_mtime >= source_mtime


def load_cache(cache_path: str, verbose: bool = True) -> Optional[any]:
    """
    Load data from cache file.
    
    Args:
        cache_path: Path to cache file
        verbose: Whether to print status messages
    
    Returns:
        Loaded data or None if failed
    """
    if verbose:
        print(f'Loading molecule dataset from cache: {cache_path}')
    
    try:
        with open(cache_path, 'rb') as f:
            data = pickle.load(f)
        if verbose:
            print('✅ Molecule dataset loaded from cache')
        return data
    except Exception as e:
        if verbose:
            print(f'⚠️  Failed to load cache ({e}), loading from scratch...')
        return None


def save_cache(data: any, cache_path: str, verbose: bool = True) -> bool:
    """
    Save data to cache file.
    
    Args:
        data: Data to cache
        cache_path: Path to cache file
        verbose: Whether to print status messages
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create cache directory if it doesn't exist
    cache_dir = os.path.dirname(cache_path)
    os.makedirs(cache_dir, exist_ok=True)
    
    if verbose:
        print(f'Saving molecule dataset to cache: {cache_path}')
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        if verbose:
            print('✅ Molecule dataset cached successfully')
        return True
    except Exception as e:
        if verbose:
            print(f'⚠️  Failed to save cache: {e}')
        return False


def get_cache_info() -> Dict[str, any]:
    """
    Get information about cached datasets.
    
    Returns:
        dict: Cache information including file count, total size, and file list
    """
    cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        return {
            'exists': False,
            'path': cache_dir,
            'count': 0,
            'total_size': 0,
            'files': []
        }
    
    cache_files = glob.glob(os.path.join(cache_dir, "mol_dataset_*.pkl"))
    
    total_size = 0
    file_info = []
    
    for cache_file in cache_files:
        size = os.path.getsize(cache_file)
        mtime = os.path.getmtime(cache_file)
        total_size += size
        
        file_info.append({
            'path': cache_file,
            'name': os.path.basename(cache_file),
            'size': size,
            'size_mb': size / (1024 * 1024),
            'modified': mtime
        })
    
    return {
        'exists': True,
        'path': cache_dir,
        'count': len(cache_files),
        'total_size': total_size,
        'total_size_mb': total_size / (1024 * 1024),
        'files': file_info
    }


def clear_cache(pattern: str = "mol_dataset_*.pkl", verbose: bool = True) -> Tuple[int, int]:
    """
    Clear cached dataset files matching a pattern.
    
    Args:
        pattern: Glob pattern for cache files to remove (default: all molecule datasets)
        verbose: Whether to print status messages
    
    Returns:
        tuple: (number of files removed, number of failed removals)
    """
    cache_dir = get_cache_dir()
    
    if not os.path.exists(cache_dir):
        if verbose:
            print(f"⚠️  Cache directory does not exist: {cache_dir}")
        return 0, 0
    
    cache_files = glob.glob(os.path.join(cache_dir, pattern))
    
    if len(cache_files) == 0:
        if verbose:
            print("No cache files found to clear.")
        return 0, 0
    
    if verbose:
        print(f"🗑️  Clearing {len(cache_files)} cached dataset(s)...")
    
    removed = 0
    failed = 0
    
    for cache_file in cache_files:
        try:
            os.remove(cache_file)
            if verbose:
                print(f"   Removed: {os.path.basename(cache_file)}")
            removed += 1
        except Exception as e:
            if verbose:
                print(f"   ⚠️  Failed to remove {os.path.basename(cache_file)}: {e}")
            failed += 1
    
    if verbose:
        print(f"✅ Cleared {removed} cache file(s)")
        if failed > 0:
            print(f"⚠️  Failed to clear {failed} file(s)")
    
    return removed, failed


def clear_cache_for_dataset(root: str, verbose: bool = True) -> int:
    """
    Clear cache files for a specific dataset root directory.
    
    Args:
        root: Root directory path for the dataset
        verbose: Whether to print status messages
    
    Returns:
        int: Number of cache files removed
    """
    root_hash = hashlib.md5(os.path.abspath(root).encode()).hexdigest()[:8]
    pattern = f"mol_dataset_{root_hash}_*.pkl"
    
    if verbose:
        print(f"Clearing cache for dataset: {root}")
    
    removed, failed = clear_cache(pattern, verbose)
    return removed


def print_cache_info():
    """Print detailed information about the cache."""
    info = get_cache_info()
    
    print("=" * 60)
    print("EDT-Former Dataset Cache Information")
    print("=" * 60)
    print(f"Cache directory: {info['path']}")
    print(f"Directory exists: {info['exists']}")
    
    if not info['exists']:
        print("No cache directory found.")
        return
    
    print(f"Number of cached datasets: {info['count']}")
    print(f"Total cache size: {info['total_size_mb']:.2f} MB")
    print()
    
    if info['count'] > 0:
        print("Cached files:")
        for file_info in info['files']:
            from datetime import datetime
            mtime = datetime.fromtimestamp(file_info['modified'])
            print(f"  - {file_info['name']}")
            print(f"    Size: {file_info['size_mb']:.2f} MB")
            print(f"    Modified: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("=" * 60)


if __name__ == "__main__":
    """Command-line interface for cache management."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m utils.cache_utils info      - Show cache information")
        print("  python -m utils.cache_utils clear     - Clear all cache")
        print("  python -m utils.cache_utils clear <root_dir> - Clear cache for specific dataset")
        sys.exit(0)
    
    command = sys.argv[1].lower()
    
    if command == "info":
        print_cache_info()
    elif command == "clear":
        if len(sys.argv) > 2:
            # Clear cache for specific dataset
            root = sys.argv[2]
            clear_cache_for_dataset(root)
        else:
            # Clear all cache
            clear_cache()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

