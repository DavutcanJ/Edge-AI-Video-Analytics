import os
import yaml
from pathlib import Path

def create_dataset_yaml(dataset_root, yaml_path=None):
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
    """
    dataset_root = Path(dataset_root)
    images_dir = dataset_root / 'images'
    labels_dir = dataset_root / 'labels'
    yaml_dict = {
        'train': str(images_dir / 'train'),
        'val': str(images_dir / 'val'),
        'test': str(images_dir / 'test') if (images_dir / 'test').exists() else str(images_dir / 'val'),
        'nc': 1,
        'names': ['object']
    }
    # Try to infer class names from dataset.yaml if exists
    orig_yaml = dataset_root / 'dataset.yaml'
    if orig_yaml.exists():
        with open(orig_yaml, 'r') as f:
            orig = yaml.safe_load(f)
            if 'names' in orig:
                yaml_dict['names'] = orig['names']
            if 'nc' in orig:
                yaml_dict['nc'] = orig['nc']
    if yaml_path is None:
        yaml_path = dataset_root / 'autogen_dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_dict, f)
    return str(yaml_path)
