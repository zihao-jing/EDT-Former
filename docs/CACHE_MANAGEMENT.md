# Cache Management in EDT-Former

This document explains how cache management works in EDT-Former and how to prevent cache-related issues.

## Overview

EDT-Former caches processed molecule datasets to speed up training. The cache system uses:
- **Unique keys** based on data directory path and encoder types
- **Automatic validation** by comparing file modification times
- **Intelligent invalidation** when source files are updated

## Cache Location

By default, cache files are stored in:
```
data/.cache/mol_dataset_*.pkl
```

You can customize the cache directory by setting environment variables:
```bash
export DATA_CACHE_DIR=/path/to/custom/cache
```

## How Cache Keys Work

Each cached dataset has a unique key based on:
1. **Root directory hash** (first 8 chars of MD5) - prevents collisions between datasets
2. **Filename** (e.g., `train_mol`, `test_mol`)
3. **Encoder types** (e.g., `unimol-moleculestm`)

Example cache filename:
```
mol_dataset_a1b2c3d4_train_mol_unimol-moleculestm.pkl
```

## Cache Validation

The cache system automatically:
- ✅ Checks if cache file exists
- ✅ Compares modification times (cache must be newer than source)
- ✅ Rebuilds cache if source file was updated
- ✅ Handles corrupted cache files gracefully

## Managing Cache

### 1. View Cache Information

```bash
# Using the script
bash scripts/clear_cache.sh info

# Or directly with Python
python3 -m utils.cache_utils info
```

This shows:
- Number of cached datasets
- Total cache size
- Individual file details

### 2. Clear All Cache

```bash
# With confirmation prompt
bash scripts/clear_cache.sh clear

# Without confirmation (force)
bash scripts/clear_cache.sh clear --force

# Or directly with Python
python3 -m utils.cache_utils clear
```

### 3. Clear Cache for Specific Dataset

```bash
# Clear cache only for a specific data directory
bash scripts/clear_cache.sh clear data/moleculeqa/

# Or with Python
python3 -m utils.cache_utils clear data/moleculeqa/
```

### 4. Clear Cache When Training

All training scripts support the `--clear-cache` flag:

```bash
# Clear cache before training mol_gen
bash scripts/qa/mol_gen.sh 2 --clear-cache

# Clear cache before training mol_prop
bash scripts/qa/mol_prop.sh 2 --clear-cache

# Clear cache before training mol_qa
bash scripts/qa/mol_qa.sh 2 --clear-cache
```

## Preventing Cache Issues

### Issue: Wrong Data Loaded from Cache

**Cause**: Cache key collision between different datasets

**Solution**: The new cache system uses path-based hashing to prevent this. If you still encounter issues:
```bash
# Clear all cache and rebuild
bash scripts/clear_cache.sh clear --force
```

### Issue: Outdated Cache After Data Update

**Cause**: Source files were modified but cache wasn't updated

**Solution**: The cache system now auto-detects this and rebuilds. To force rebuild:
```bash
# Training will automatically detect and rebuild outdated cache
bash scripts/qa/mol_gen.sh 2
```

### Issue: Corrupted Cache File

**Cause**: Training interrupted during cache save

**Solution**: Cache system handles this gracefully:
- Detects load failures
- Automatically rebuilds from source
- Doesn't crash training

## Cache Utility API

For programmatic access in your scripts:

```python
from utils.cache_utils import (
    get_cache_dir,
    get_cache_path,
    is_cache_valid,
    load_cache,
    save_cache,
    clear_cache,
    get_cache_info,
    print_cache_info
)

# Get cache directory
cache_dir = get_cache_dir()

# Get cache path for a dataset
cache_path = get_cache_path(
    root='data/moleculeqa/',
    filename='train_mol.json',
    encoder_types=['unimol', 'moleculestm']
)

# Check if cache is valid
if is_cache_valid(cache_path, source_path):
    data = load_cache(cache_path)
else:
    # Process data...
    save_cache(data, cache_path)

# Clear cache
removed, failed = clear_cache()

# Get cache info
info = get_cache_info()
print(f"Total cache size: {info['total_size_mb']:.2f} MB")
```

## Best Practices

1. **After updating data files**: Cache rebuilds automatically, no action needed
2. **Before major experiments**: Clear cache to ensure fresh data
   ```bash
   bash scripts/clear_cache.sh clear --force
   ```
3. **When switching datasets**: Cache keys are unique per dataset, safe to keep
4. **On disk space issues**: Check and clear old cache
   ```bash
   bash scripts/clear_cache.sh info
   bash scripts/clear_cache.sh clear
   ```
5. **In distributed training**: Only rank 0 manages cache (automatic)

## Troubleshooting

### Problem: KeyError with CID not found

**Solution**: 
```bash
# Clear cache and rebuild
bash scripts/clear_cache.sh clear --force
bash scripts/qa/mol_gen.sh 2
```

### Problem: Slow data loading

**Possible causes**:
- First run (cache being built)
- Cache outdated (being rebuilt)
- Dataloader workers too high

**Solution**: Wait for first epoch to complete, subsequent epochs will be fast

### Problem: Out of disk space

**Solution**:
```bash
# Check cache size
bash scripts/clear_cache.sh info

# Clear if needed
bash scripts/clear_cache.sh clear --force
```

## File Structure

```
EDT-Former/
├── data/
│   └── .cache/                    # Cache directory
│       └── mol_dataset_*.pkl      # Cached datasets
├── utils/
│   └── cache_utils.py             # Cache management utilities
├── scripts/
│   └── clear_cache.sh             # Cache management script
└── docs/
    └── CACHE_MANAGEMENT.md        # This file
```

## Cache Statistics

Typical cache sizes:
- **moleculeqa**: ~60 MB (train + test)
- **mol_gen**: ~1.4 GB (train + test)
- **mol_prop**: ~270 MB (train + test)

Cache building time (first run):
- **moleculeqa**: ~30 seconds
- **mol_gen**: ~5 minutes
- **mol_prop**: ~2 minutes

## Future Improvements

Planned enhancements:
- [ ] Automatic cache size limits
- [ ] LRU cache eviction
- [ ] Compression for large caches
- [ ] Distributed cache sharing
- [ ] Cache versioning

