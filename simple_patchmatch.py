"""
MVP Version of PatchMatch Stereo - Milestone 1 Only
Implement basic framework: plane representation, random initialization, simple cost computation
"""

import numpy as np
import random
from typing import Tuple


class SimplePlane:
    """Simplified 3D plane representation"""
    
    def __init__(self, a: float = 0, b: float = 0, c: float = 0):
        """
        Plane equation: disparity = a*x + b*y + c
        a: slope in x-direction
        b: slope in y-direction  
        c: base disparity
        """
        self.a = np.clip(a, -0.1, 0.1)  # Limit slope range
        self.b = np.clip(b, -0.1, 0.1)
        self.c = np.clip(c, 0, 64)      # Limit disparity range 0-64
    
    def disparity_at(self, x: int, y: int) -> float:
        """Compute disparity at (x, y)"""
        return self.a * x + self.b * y + self.c
    
    @classmethod
    def random_plane(cls, max_disp: int = 64):
        """Generate random plane"""
        # 50% fronto-parallel, 50% slanted
        if random.random() > 0.5:
            return cls(0, 0, random.uniform(0, max_disp))
        else:
            return cls(
                a=random.uniform(-0.05, 0.05),
                b=random.uniform(-0.05, 0.05),
                c=random.uniform(0, max_disp)
            )
    
    def __str__(self):
        return f"Plane(a={self.a:.3f}, b={self.b:.3f}, c={self.c:.2f})"


class SimplePatchMatchMVP:
    """MVP version of PatchMatch Stereo - Milestone 1 only"""
    
    def __init__(self, max_disp: int = 64, window_size: int = 7):
        """
        Parameters:
        max_disp: maximum disparity
        window_size: matching window size (odd)
        """
        self.max_disp = max_disp
        self.window_size = window_size
        self.half_win = window_size // 2
        
        # Store results
        self.left_planes = None   # Left view planes
        self.right_planes = None  # Right view planes
        self.left_disparity = None   # Left disparity map
        self.right_disparity = None  # Right disparity map
    
    def random_initialization(self, height: int, width: int):
        """
        Random initialization of planes for all pixels - Core of Milestone 1
        Assign a random plane to each pixel
        """
        print(f"Random initialization for {height}x{width} image...")
        
        self.left_planes = np.empty((height, width), dtype=object)
        self.right_planes = np.empty((height, width), dtype=object)
        
        # Assign random plane to each pixel
        for y in range(height):
            for x in range(width):
                self.left_planes[y, x] = SimplePlane.random_plane(self.max_disp)
                self.right_planes[y, x] = SimplePlane.random_plane(self.max_disp)
        
        print(f"Initialization complete: left view {height}x{width} planes")
        print(f"Sample plane: {self.left_planes[height//2, width//2]}")
    
    def simple_pixel_cost(self, left_img: np.ndarray, right_img: np.ndarray,
                         x: int, y: int, plane: SimplePlane) -> float:
        """
        Simplified pixel matching cost
        Only compute center pixel cost, no window aggregation (simplified)
        """
        # Compute disparity
        d = plane.disparity_at(x, y)
        
        # Check if disparity is valid
        if d < 0 or d > self.max_disp:
            return float('inf')
        
        # Find matching point in right image
        x_match = x - d
        
        # Check boundaries
        if x_match < 0 or x_match >= right_img.shape[1]:
            return float('inf')
        
        # Simplified cost: only compute color difference
        x_match_int = int(x_match)
        
        if left_img.ndim == 3:  # Color image
            # Use all color channels
            cost = np.mean(np.abs(left_img[y, x] - right_img[y, x_match_int]))
        else:  # Grayscale image
            cost = abs(float(left_img[y, x]) - float(right_img[y, x_match_int]))
        
        return cost
    
    def compute_disparity_maps(self):
        """Compute disparity maps from planes"""
        if self.left_planes is None or self.right_planes is None:
            raise ValueError("Please call random_initialization() first")
        
        height, width = self.left_planes.shape
        
        self.left_disparity = np.zeros((height, width), dtype=np.float32)
        self.right_disparity = np.zeros((height, width), dtype=np.float32)
        
        print("Computing disparity maps from planes...")
        
        for y in range(height):
            for x in range(width):
                self.left_disparity[y, x] = self.left_planes[y, x].disparity_at(x, y)
                self.right_disparity[y, x] = self.right_planes[y, x].disparity_at(x, y)
        
        # Clip to valid disparity range
        self.left_disparity = np.clip(self.left_disparity, 0, self.max_disp)
        self.right_disparity = np.clip(self.right_disparity, 0, self.max_disp)
        
        return self.left_disparity, self.right_disparity
    
    def evaluate_random_cost(self, left_img: np.ndarray, right_img: np.ndarray, 
                           sample_points: int = 1000) -> dict:
        """
        Evaluate the effect of random initialization
        Randomly sample points and compute their matching costs
        """
        if self.left_planes is None:
            raise ValueError("Please call random_initialization() first")
        
        height, width = left_img.shape[:2]
        
        costs = []
        valid_count = 0
        
        print(f"\nEvaluating matching cost for {sample_points} random points...")
        
        # Random sampling
        for _ in range(sample_points):
            y = random.randint(0, height-1)
            x = random.randint(0, width-1)
            
            plane = self.left_planes[y, x]
            cost = self.simple_pixel_cost(left_img, right_img, x, y, plane)
            
            if not np.isinf(cost):
                costs.append(cost)
                valid_count += 1
        
        # Statistics
        stats = {
            'total_samples': sample_points,
            'valid_samples': valid_count,
            'avg_cost': np.mean(costs) if costs else float('inf'),
            'min_cost': np.min(costs) if costs else float('inf'),
            'max_cost': np.max(costs) if costs else float('inf'),
            'valid_rate': valid_count / sample_points * 100
        }
        
        print(f"Valid matches: {stats['valid_rate']:.1f}%")
        print(f"Average cost: {stats['avg_cost']:.3f}")
        print(f"Minimum cost: {stats['min_cost']:.3f}")
        print(f"Maximum cost: {stats['max_cost']:.3f}")
        
        return stats
    
    def visualize_plane_field(self, img_shape: Tuple[int, int], 
                            sample_density: int = 20):
        """
        Visualize the plane field
        Sample points on image, show plane at each point (represented by line)
        """
        if self.left_planes is None:
            raise ValueError("Please call random_initialization() first")
        
        import matplotlib.pyplot as plt
        
        height, width = img_shape[:2]
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Show plane orientation
        for ax_idx, (title, planes) in enumerate([
            ("Left View Planes", self.left_planes),
            ("Right View Planes", self.right_planes)
        ]):
            ax = axes[ax_idx]
            ax.set_title(title)
            ax.set_xlim(0, width)
            ax.set_ylim(height, 0)  # Image coordinate system
            ax.set_aspect('equal')
            
            # Sample display
            for y in range(0, height, sample_density):
                for x in range(0, width, sample_density):
                    plane = planes[y, x]
                    
                    # Use line segment to represent plane orientation
                    # Slope shows plane slope, length shows disparity
                    length = min(20, plane.c * 2)  # Scale for display
                    dx = length * plane.a * 10  # Amplify x effect
                    dy = length * plane.b * 10  # Amplify y effect
                    
                    # Skip very small arrows
                    if abs(dx) < 0.1 and abs(dy) < 0.1:
                        continue
                    
                    ax.arrow(x, y, dx, dy, 
                            head_width=2, head_length=3, 
                            fc='red', ec='red', alpha=0.6)
        
        plt.tight_layout()
        return fig
    
    def save_results(self, output_dir: str = "results"):
        """
        Save disparity maps to files
        """
        import os
        import cv2
        
        os.makedirs(output_dir, exist_ok=True)
        
        if self.left_disparity is not None:
            # Normalize for visualization
            disp_norm = self.left_disparity.copy()
            if np.max(disp_norm) > np.min(disp_norm):
                disp_norm = (disp_norm - np.min(disp_norm)) / (np.max(disp_norm) - np.min(disp_norm))
                disp_norm = (disp_norm * 255).astype(np.uint8)
            
            cv2.imwrite(f"{output_dir}/disparity_left.png", disp_norm)
            print(f"Saved left disparity to {output_dir}/disparity_left.png")
        
        if self.right_disparity is not None:
            disp_norm = self.right_disparity.copy()
            if np.max(disp_norm) > np.min(disp_norm):
                disp_norm = (disp_norm - np.min(disp_norm)) / (np.max(disp_norm) - np.min(disp_norm))
                disp_norm = (disp_norm * 255).astype(np.uint8)
            
            cv2.imwrite(f"{output_dir}/disparity_right.png", disp_norm)
            print(f"Saved right disparity to {output_dir}/disparity_right.png")


def create_synthetic_stereo_pair() -> Tuple[np.ndarray, np.ndarray]:
    """
    Create simple stereo pair for testing
    Left image: gradient texture
    Right image: horizontal shift (simulating disparity)
    """
    height, width = 128, 128
    
    # Left image: create some texture
    left_img = np.zeros((height, width, 3), dtype=np.float32)
    
    for y in range(height):
        for x in range(width):
            # Simple texture pattern
            r = (np.sin(x/20) + 1) / 2
            g = (np.cos(y/15) + 1) / 2
            b = 0.5 + 0.3 * np.sin((x+y)/30)
            left_img[y, x] = [r, g, b]
    
    # Right image: simulate disparity (simple horizontal shift)
    right_img = np.zeros_like(left_img)
    for y in range(height):
        for x in range(width):
            # Disparity varies with y coordinate
            disparity = 5 + 10 * (y / height)
            x_match = int(x - disparity)
            
            if 0 <= x_match < width:
                right_img[y, x] = left_img[y, x_match]
            else:
                right_img[y, x] = left_img[y, x]
    
    return left_img, right_img