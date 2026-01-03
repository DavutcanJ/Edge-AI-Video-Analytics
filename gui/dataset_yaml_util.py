import os
import yaml
from pathlib import Path
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def auto_detect_classes(dataset_root):
    """
    Auto-detect class IDs and count from label files.
    
    Returns:
        Tuple of (class_ids_set, class_counts_dict)
    """
    dataset_root = Path(dataset_root)
    labels_dir = dataset_root / 'labels'
    
    class_ids = set()
    class_counts = Counter()
    
    if not labels_dir.exists():
        return class_ids, class_counts
    
    # Scan all splits
    for split in ['train', 'val', 'test']:
        split_labels = labels_dir / split
        if not split_labels.exists():
            continue
        
        for label_file in split_labels.glob('*.txt'):
            try:
                with open(label_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            class_ids.add(class_id)
                            class_counts[class_id] += 1
            except Exception as e:
                logger.warning(f"Error reading {label_file}: {e}")
    
    return class_ids, class_counts


def create_dataset_yaml(dataset_root, yaml_path=None, class_names=None):
    """
    Given a dataset root directory, create a YOLO-compatible dataset.yaml file.
    Expects structure:
      dataset_root/
        images/
          train/
          val/
        labels/
          train/
          val/
    
    Args:
        dataset_root: Path to dataset root directory
        yaml_path: Optional custom path for yaml file
        class_names: Optional list of class names (will auto-detect if not provided)
    
    Returns:
        str: Path to created yaml file
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    
    # Auto-detect classes from label files
    class_ids, class_counts = auto_detect_classes(dataset_root)
    
    # Determine number of classes
    if class_names:
        nc = len(class_names)
        names = class_names
    else:
        # Try to read from existing dataset.yaml
        orig_yaml = dataset_root / 'dataset.yaml'
        if orig_yaml.exists():
            try:
                with open(orig_yaml, 'r') as f:
                    orig = yaml.safe_load(f)
                    if 'names' in orig:
                        names = orig['names']
                        if isinstance(names, dict):
                            names = [names[i] for i in sorted(names.keys())]
                        nc = len(names)
                    else:
                        # Use detected classes
                        nc = len(class_ids) if class_ids else 1
                        names = [f'class_{i}' for i in range(nc)]
            except Exception as e:
                logger.warning(f"Could not read existing dataset.yaml: {e}")
                nc = len(class_ids) if class_ids else 1
                names = [f'class_{i}' for i in range(nc)]
        else:
            # Generate based on detected classes
            nc = len(class_ids) if class_ids else 1
            names = [f'class_{i}' for i in range(nc)]
    
    # Build yaml structure
    yaml_dict = {
        'path': str(dataset_root),
        'train': 'images/train',
        'val': 'images/val' if (images_dir / 'val').exists() else 'images/train',
    }
    
    if (images_dir / 'test').exists():
        yaml_dict['test'] = 'images/test'
    
    yaml_dict['nc'] = nc
    yaml_dict['names'] = names
    
    # Add metadata comment
    if class_counts:
        yaml_dict['# class_distribution'] = dict(class_counts)
    
    # Write yaml file
    if yaml_path is None:
        yaml_path = dataset_root / 'dataset.yaml'
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Created dataset.yaml with {nc} classes at {yaml_path}")
    return str(yaml_path)
