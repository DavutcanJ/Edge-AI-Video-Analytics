"""
Dataset Manager and Validator
Handles dataset validation, format checking, and metadata extraction for YOLO training.
Required format: train/images, train/labels, val/images, val/labels
"""

import os
import yaml
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

logger = logging.getLogger(__name__)


class DatasetValidator:
    """Validates YOLO format datasets and extracts metadata."""
    
    def __init__(self, dataset_root: str):
        self.dataset_root = Path(dataset_root)
        self.errors = []
        self.warnings = []
        self.metadata = {}
        
        # Required structure: train/images, train/labels, val/images, val/labels
        self.required_structure = [
            "train/images",
            "train/labels",
            "val/images",
            "val/labels"
        ]
    
    def validate(self) -> Tuple[bool, Dict]:
        """
        Validate dataset structure and content.
        Supports two structures:
        1. dataset_root/train/images, dataset_root/train/labels, etc.
        2. dataset_root/ directly containing train/ and val/
        
        Returns:
            Tuple of (is_valid, metadata_dict)
        """
        self.errors = []
        self.warnings = []
        self.metadata = {
            "valid": False,
            "dataset_root": str(self.dataset_root),
            "splits": {},
            "total_images": 0,
            "total_labels": 0,
            "classes": {},
            "class_distribution": {},
            "errors": [],
            "warnings": []
        }
        
        # Check if root exists
        if not self.dataset_root.exists():
            self.errors.append(f"Dataset root does not exist: {self.dataset_root}")
            self.metadata["errors"] = self.errors
            return False, self.metadata
        
        # Check if root is a directory
        if not self.dataset_root.is_dir():
            self.errors.append(f"Dataset root is not a directory: {self.dataset_root}")
            self.metadata["errors"] = self.errors
            return False, self.metadata
        
        # Check required directory structure
        missing_dirs = []
        for required_path in self.required_structure:
            full_path = self.dataset_root / required_path
            if not full_path.exists() or not full_path.is_dir():
                missing_dirs.append(required_path)
        
        if missing_dirs:
            self.errors.append(
                f"❌ Dataset yapısı uygun değil!\n\n"
                f"Gerekli klasör yapısı:\n"
                f"  {self.dataset_root.name}/\n"
                f"    train/\n"
                f"      images/\n"
                f"      labels/\n"
                f"    val/\n"
                f"      images/\n"
                f"      labels/\n\n"
                f"Eksik klasörler: {', '.join(missing_dirs)}\n\n"
                f"Mevcut klasörler:\n"
            )
            
            # List what actually exists for debugging
            try:
                contents = [f.name + ('/' if f.is_dir() else '') for f in self.dataset_root.iterdir()]
                self.errors[0] += f"  {', '.join(sorted(contents))}\n\n"
            except:
                pass
            
            self.errors[0] += "Lütfen veri setinizi bu yapıya göre düzenleyin."
            self.metadata["errors"] = self.errors
            return False, self.metadata
        
        # Validate train and val splits
        for split in ["train", "val"]:
            split_images = self.dataset_root / split / "images"
            split_labels = self.dataset_root / split / "labels"
            
            split_info = self._validate_split(split, split_images, split_labels)
            self.metadata["splits"][split] = split_info
            self.metadata["total_images"] += split_info["image_count"]
            self.metadata["total_labels"] += split_info["label_count"]
        
        # Check if we have enough data
        if self.metadata["total_images"] == 0:
            self.errors.append("No images found in dataset!")
            self.metadata["errors"] = self.errors
            return False, self.metadata
        
        # Extract class information
        self._extract_class_info()
        
        self.metadata["errors"] = self.errors
        self.metadata["warnings"] = self.warnings
        self.metadata["valid"] = len(self.errors) == 0
        
        return self.metadata["valid"], self.metadata
    
    def _validate_split(self, split_name: str, images_dir: Path, labels_dir: Path) -> Dict:
        """Validate a single split (train/val)."""
        split_info = {
            "name": split_name,
            "image_count": 0,
            "label_count": 0,
            "image_formats": Counter(),
            "missing_labels": [],
            "missing_images": [],
            "empty_labels": []
        }
        
        # Check if directories exist
        if not images_dir.exists():
            self.warnings.append(f"{split_name}: images directory does not exist")
            return split_info
        
        if not labels_dir.exists():
            self.warnings.append(f"{split_name}: labels directory does not exist")
            return split_info
        
        # Get all images - use rglob for faster search
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        image_files = []
        try:
            all_files = list(images_dir.iterdir())
            image_files = [f for f in all_files if f.suffix.lower() in image_extensions]
        except Exception as e:
            logger.error(f"Error reading images from {images_dir}: {e}")
            self.errors.append(f"Error reading images from {split_name}: {e}")
            return split_info
        
        split_info["image_count"] = len(image_files)
        
        # Build image stem set for fast lookup
        image_stems = {img.stem for img in image_files}
        
        # Count formats
        for img_path in image_files:
            split_info["image_formats"][img_path.suffix.lower()] += 1
        
        # Check labels
        try:
            if labels_dir.exists():
                label_files = list(labels_dir.glob("*.txt"))
                split_info["label_count"] = len(label_files)
                
                label_stems = set()
                for label_path in label_files:
                    label_stems.add(label_path.stem)
                    
                    # Check if label file is empty (sample check only first 100)
                    if len(split_info["empty_labels"]) < 100:
                        try:
                            if label_path.stat().st_size == 0:
                                split_info["empty_labels"].append(label_path.name)
                        except:
                            pass
                
                # Find missing labels (images without labels)
                missing = image_stems - label_stems
                split_info["missing_labels"] = [f"{stem}.jpg" for stem in list(missing)[:100]]  # Limit to 100
                
                # Find missing images (labels without images)  
                missing_imgs = label_stems - image_stems
                split_info["missing_images"] = [f"{stem}.txt" for stem in list(missing_imgs)[:100]]  # Limit to 100
                
        except Exception as e:
            logger.warning(f"Error checking labels in {labels_dir}: {e}")
        
        # Add warnings
        if split_info["missing_labels"]:
            self.warnings.append(
                f"{split_name}: {len(split_info['missing_labels'])} images without labels"
            )
        
        if split_info["empty_labels"]:
            self.warnings.append(
                f"{split_name}: {len(split_info['empty_labels'])} empty label files"
            )
        
        return split_info
    
    def _extract_class_info(self):
        """Extract class information from label files (sample-based for speed)."""
        class_ids = set()
        class_counts = Counter()
        
        # Initialize classes dict if not exists
        if "classes" not in self.metadata:
            self.metadata["classes"] = {}
        
        # Sample only first 100 label files for speed
        max_files_to_scan = 100
        files_scanned = 0
        
        for split_name in ["train", "val"]:
            if files_scanned >= max_files_to_scan:
                break
                
            labels_dir = self.dataset_root / split_name / "labels"
            if not labels_dir.exists():
                continue
            
            try:
                label_files = list(labels_dir.glob("*.txt"))
                for label_path in label_files[:max_files_to_scan - files_scanned]:
                    try:
                        with open(label_path, 'r') as f:
                            for line in f:
                                line = line.strip()
                                if not line:
                                    continue
                                parts = line.split()
                                if len(parts) >= 5:
                                    class_id = int(parts[0])
                                    class_ids.add(class_id)
                                    class_counts[class_id] += 1
                        files_scanned += 1
                    except Exception as e:
                        logger.warning(f"Error reading label file {label_path}: {e}")
            except Exception as e:
                logger.warning(f"Error scanning labels in {labels_dir}: {e}")
        
        # Update metadata
        self.metadata["classes"]["count"] = len(class_ids)
        self.metadata["classes"]["ids"] = sorted(list(class_ids))
        self.metadata["class_distribution"] = dict(class_counts)
    
    def get_sample_images(self, split: str = "train", count: int = 4) -> List[Path]:
        """Get sample image paths from a split."""
        images_dir = self.dataset_root / split / "images"
        if not images_dir.exists():
            return []
        
        image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(images_dir.glob(f"*{ext}")))
            if len(image_files) >= count:
                break
        
        return image_files[:count]


class DatasetManager:
    """Manages multiple datasets and generates YOLO configuration files."""
    
    def __init__(self, workspace_root: Optional[str] = None):
        if workspace_root is None:
            workspace_root = Path(__file__).parent.parent
        self.workspace_root = Path(workspace_root)
        self.datasets_dir = self.workspace_root / "data"
        self.datasets_dir.mkdir(exist_ok=True)
        
        self.config_file = self.workspace_root / "datasets_config.json"
        self.datasets = self._load_datasets_config()
    
    def _load_datasets_config(self) -> Dict:
        """Load saved datasets configuration."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error loading datasets config: {e}")
        return {}
    
    def _save_datasets_config(self):
        """Save datasets configuration."""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.datasets, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving datasets config: {e}")
    
    def add_dataset(self, name: str, dataset_root: str, class_names: Optional[List[str]] = None, copy_to_data: bool = True) -> Tuple[bool, str]:
        """
        Add a new dataset to the manager.
        
        Args:
            name: Dataset identifier
            dataset_root: Path to dataset root directory
            class_names: Optional list of class names
            copy_to_data: If True, copy dataset to data/ folder
        
        Returns:
            Tuple of (success, message/error)
        """
        # Validate dataset
        validator = DatasetValidator(dataset_root)
        is_valid, metadata = validator.validate()
        
        if not is_valid:
            error_msg = "Dataset validation failed:\n" + "\n".join(metadata["errors"])
            return False, error_msg
        
        # Copy dataset to data/ folder if requested
        if copy_to_data:
            try:
                data_dir = self.workspace_root / "data"
                target_dir = data_dir 
                
                if target_dir.exists():
                    import shutil
                    shutil.rmtree(target_dir)
                
                import shutil
                shutil.copytree(dataset_root, target_dir)
                dataset_root = str(target_dir)
                logger.info(f"Dataset copied to {target_dir}")
            except Exception as e:
                logger.error(f"Failed to copy dataset: {e}")
                return False, f"Failed to copy dataset to data/ folder: {e}"
        
        # Auto-detect or use provided class names
        num_classes = metadata["classes"]["count"]
        if class_names is None:
            # Try to read from existing dataset.yaml in root
            existing_yaml = Path(dataset_root) / "dataset.yaml"
            if existing_yaml.exists():
                try:
                    with open(existing_yaml, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                        if "names" in yaml_data:
                            class_names = yaml_data["names"]
                            if isinstance(class_names, dict):
                                class_names = [class_names[i] for i in sorted(class_names.keys())]
                except Exception as e:
                    logger.warning(f"Could not read existing dataset.yaml: {e}")
            
            # Generate generic names if not found
            if class_names is None:
                class_names = [f"class_{i}" for i in range(num_classes)]
        
        # Create dataset.yaml
        yaml_path = self._create_dataset_yaml(
            dataset_root=dataset_root,
            class_names=class_names,
            metadata=metadata
        )
        
        # Save dataset info
        self.datasets[name] = {
            "name": name,
            "root": dataset_root,
            "yaml_path": yaml_path,
            "class_names": class_names,
            "metadata": metadata,
            "added_at": str(Path(dataset_root).stat().st_mtime)
        }
        self._save_datasets_config()
        
        return True, f"Dataset '{name}' added successfully"
    
    def _create_dataset_yaml(self, dataset_root: str, class_names: List[str], metadata: Dict) -> str:
        """Create YOLO-compatible dataset.yaml file."""
        dataset_root = Path(dataset_root)
        
        yaml_data = {
            "path": str(dataset_root),
            "train": "train/images",
            "val": "val/images",
        }
        
        yaml_data["nc"] = len(class_names)
        yaml_data["names"] = class_names
        
        # Save to dataset root
        yaml_path = dataset_root / "dataset.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        
        return str(yaml_path)
    
    def get_dataset(self, name: str) -> Optional[Dict]:
        """Get dataset information by name."""
        return self.datasets.get(name)
    
    def list_datasets(self) -> List[str]:
        """Get list of all dataset names."""
        return list(self.datasets.keys())
    
    def remove_dataset(self, name: str) -> bool:
        """Remove dataset from manager (doesn't delete files)."""
        if name in self.datasets:
            del self.datasets[name]
            self._save_datasets_config()
            return True
        return False
    
    def get_dataset_summary(self, name: str) -> Optional[str]:
        """Get formatted summary of dataset."""
        dataset = self.get_dataset(name)
        if not dataset:
            return None
        
        metadata = dataset["metadata"]
        summary = f"Dataset: {name}\n"
        summary += f"{'='*50}\n"
        summary += f"Root: {dataset['root']}\n"
        summary += f"Classes: {len(dataset['class_names'])} - {', '.join(dataset['class_names'][:5])}"
        if len(dataset['class_names']) > 5:
            summary += "...\n"
        else:
            summary += "\n"
        summary += f"Total Images: {metadata['total_images']}\n"
        summary += f"Total Labels: {metadata['total_labels']}\n"
        summary += f"\nSplits:\n"
        for split_name, split_info in metadata["splits"].items():
            summary += f"  {split_name}: {split_info['image_count']} images, {split_info['label_count']} labels\n"
        
        if metadata.get("warnings"):
            summary += f"\n⚠️ Warnings:\n"
            for warning in metadata["warnings"][:3]:
                summary += f"  - {warning}\n"
        
        return summary
