"""
Data loader for local stereo datasets
Supports multiple dataset formats
"""

import os
import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import glob


class StereoDataLoader:
    """Loader for stereo image pairs from local datasets"""
    
    @staticmethod
    def load_image_pair(left_path: str, right_path: str, 
                       resize_to: Optional[Tuple[int, int]] = None,
                       normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load stereo image pair from file paths
        
        Args:
            left_path: Path to left image
            right_path: Path to right image
            resize_to: Optional (height, width) to resize
            normalize: Whether to normalize to [0, 1]
            
        Returns:
            Tuple of (left_image, right_image)
        """
        # Check if files exist
        if not os.path.exists(left_path):
            raise FileNotFoundError(f"Left image not found: {left_path}")
        if not os.path.exists(right_path):
            raise FileNotFoundError(f"Right image not found: {right_path}")
        
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None:
            raise ValueError(f"Failed to load left image: {left_path}")
        if right_img is None:
            raise ValueError(f"Failed to load right image: {right_path}")
        
        # Convert BGR to RGB
        left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
        right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
        
        # Resize if requested
        if resize_to is not None:
            height, width = resize_to
            left_img = cv2.resize(left_img, (width, height))
            right_img = cv2.resize(right_img, (width, height))
        
        # Normalize if requested
        if normalize:
            left_img = left_img.astype(np.float32) / 255.0
            right_img = right_img.astype(np.float32) / 255.0
        
        return left_img, right_img
    
    @staticmethod
    def load_middlebury_2003(dataset_dir: str, scene_name: str, 
                           resize_to: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Middlebury 2003 dataset format
        Expected structure: dataset_dir/scene_name/{im0.png, im1.png}
        """
        left_path = os.path.join(dataset_dir, scene_name, "im0.png")
        right_path = os.path.join(dataset_dir, scene_name, "im1.png")
        
        # Alternative naming
        if not os.path.exists(left_path):
            left_path = os.path.join(dataset_dir, scene_name, "view1.png")
            right_path = os.path.join(dataset_dir, scene_name, "view5.png")
        
        return StereoDataLoader.load_image_pair(left_path, right_path, resize_to)
    
    @staticmethod
    def load_middlebury_2006(dataset_dir: str, scene_name: str,
                           resize_to: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load Middlebury 2006 dataset format
        Expected structure: dataset_dir/scene_name/{view1.png, view5.png}
        """
        left_path = os.path.join(dataset_dir, scene_name, "view1.png")
        right_path = os.path.join(dataset_dir, scene_name, "view5.png")
        
        return StereoDataLoader.load_image_pair(left_path, right_path, resize_to)
    
    @staticmethod
    def load_kitti(dataset_dir: str, sequence: str, frame_id: str,
                  resize_to: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load KITTI dataset format
        Expected structure: dataset_dir/sequence/image_2/{frame_id}.png, image_3/{frame_id}.png
        """
        left_path = os.path.join(dataset_dir, sequence, "image_2", f"{frame_id}.png")
        right_path = os.path.join(dataset_dir, sequence, "image_3", f"{frame_id}.png")
        
        return StereoDataLoader.load_image_pair(left_path, right_path, resize_to)
    
    @staticmethod
    def load_custom_pair(dataset_dir: str, left_pattern: str, right_pattern: str,
                        resize_to: Optional[Tuple[int, int]] = None) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Load custom stereo pairs using filename patterns
        
        Args:
            dataset_dir: Directory containing images
            left_pattern: Pattern for left images (e.g., "*_left.png")
            right_pattern: Pattern for right images (e.g., "*_right.png")
            
        Returns:
            List of (left_image, right_image) pairs
        """
        left_files = sorted(glob.glob(os.path.join(dataset_dir, left_pattern)))
        right_files = sorted(glob.glob(os.path.join(dataset_dir, right_pattern)))
        
        if len(left_files) != len(right_files):
            print(f"Warning: Found {len(left_files)} left images and {len(right_files)} right images")
        
        pairs = []
        for left_file, right_file in zip(left_files[:min(len(left_files), len(right_files))], 
                                        right_files[:min(len(left_files), len(right_files))]):
            try:
                left_img, right_img = StereoDataLoader.load_image_pair(left_file, right_file, resize_to)
                pairs.append((left_img, right_img))
                print(f"Loaded pair: {os.path.basename(left_file)} - {os.path.basename(right_file)}")
            except Exception as e:
                print(f"Failed to load pair {left_file}, {right_file}: {e}")
        
        return pairs
    
    @staticmethod
    def scan_directory(dataset_dir: str) -> Dict[str, List[str]]:
        """
        Scan directory for stereo image pairs
        Returns dictionary of found pairs
        
        Args:
            dataset_dir: Directory to scan
            
        Returns:
            Dictionary with image pairs
        """
        if not os.path.exists(dataset_dir):
            return {}
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        # Find all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext}")))
            image_files.extend(glob.glob(os.path.join(dataset_dir, f"*{ext.upper()}")))
        
        # Group by common naming patterns
        patterns = {
            'left_right': ['left', 'right'],
            'view1_view5': ['view1', 'view5'],
            'im0_im1': ['im0', 'im1'],
            '01_02': ['01', '02'],
            '1_2': ['1', '2']
        }
        
        found_pairs = {}
        
        for pattern_name, keywords in patterns.items():
            left_keyword, right_keyword = keywords
            
            left_files = [f for f in image_files if left_keyword in os.path.basename(f).lower()]
            right_files = [f for f in image_files if right_keyword in os.path.basename(f).lower()]
            
            if left_files and right_files:
                # Try to match by base name
                pairs = []
                for left_file in left_files:
                    base_name = os.path.basename(left_file).lower().replace(left_keyword, '')
                    matching_right = [f for f in right_files if base_name in os.path.basename(f).lower()]
                    
                    if matching_right:
                        pairs.append((left_file, matching_right[0]))
                
                if pairs:
                    found_pairs[pattern_name] = pairs
        
        return found_pairs
    
    @staticmethod
    def display_dataset_info(dataset_dir: str):
        """
        Display information about dataset directory
        """
        print(f"\nScanning dataset directory: {dataset_dir}")
        
        if not os.path.exists(dataset_dir):
            print("Directory does not exist")
            return
        
        found_pairs = StereoDataLoader.scan_directory(dataset_dir)
        
        if not found_pairs:
            print("No stereo pairs found. Please check naming conventions.")
            print("\nSupported naming patterns:")
            print("  - left_right: image_left.png, image_right.png")
            print("  - view1_view5: view1.png, view5.png")
            print("  - im0_im1: im0.png, im1.png")
            print("  - 01_02: image01.png, image02.png")
            
            # List all image files
            print("\nAll image files found:")
            image_files = glob.glob(os.path.join(dataset_dir, "*.png")) + \
                         glob.glob(os.path.join(dataset_dir, "*.jpg")) + \
                         glob.glob(os.path.join(dataset_dir, "*.jpeg"))
            
            for img_file in sorted(image_files):
                print(f"  {os.path.basename(img_file)}")
        else:
            print(f"Found {len(found_pairs)} naming pattern(s):")
            for pattern_name, pairs in found_pairs.items():
                print(f"\n  {pattern_name.upper()} pattern: {len(pairs)} pair(s)")
                for i, (left_file, right_file) in enumerate(pairs[:3]):  # Show first 3
                    print(f"    Pair {i+1}: {os.path.basename(left_file)} - {os.path.basename(right_file)}")
                if len(pairs) > 3:
                    print(f"    ... and {len(pairs)-3} more pairs")