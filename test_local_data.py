"""
Test MVP PatchMatch Stereo on local datasets
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_patchmatch import SimplePatchMatchMVP, create_synthetic_stereo_pair
from data_loader import StereoDataLoader


def test_synthetic_data():
    """Test with synthetic data"""
    print("=" * 60)
    print("Testing with Synthetic Data")
    print("=" * 60)
    
    # Create synthetic stereo pair
    left_img, right_img = create_synthetic_stereo_pair()
    print(f"Created synthetic images: {left_img.shape}")
    
    # Create algorithm instance
    pm = SimplePatchMatchMVP(max_disp=32, window_size=7)
    height, width = left_img.shape[:2]
    
    # Run random initialization
    pm.random_initialization(height, width)
    
    # Evaluate random initialization
    stats = pm.evaluate_random_cost(left_img, right_img, sample_points=500)
    
    # Compute disparity
    left_disp, right_disp = pm.compute_disparity_maps()
    
    # Visualize results
    visualize_results(left_img, right_img, left_disp, 
                     title="Synthetic Data Test - Random Initialization")
    
    return pm, stats


def test_local_dataset(dataset_dir: str, pattern_type: str = None):
    """
    Test with local dataset
    
    Args:
        dataset_dir: Path to dataset directory
        pattern_type: Naming pattern type (optional)
    """
    print("=" * 60)
    print(f"Testing with Local Dataset: {dataset_dir}")
    print("=" * 60)
    
    # Scan directory for stereo pairs
    StereoDataLoader.display_dataset_info(dataset_dir)
    
    found_pairs = StereoDataLoader.scan_directory(dataset_dir)
    
    if not found_pairs:
        print("\nNo stereo pairs found. Using synthetic data instead.")
        return test_synthetic_data()
    
    # Select pattern
    if pattern_type is None:
        pattern_types = list(found_pairs.keys())
        if len(pattern_types) == 1:
            pattern_type = pattern_types[0]
        else:
            print(f"\nFound multiple patterns. Please select one:")
            for i, pt in enumerate(pattern_types):
                print(f"  {i+1}. {pt} ({len(found_pairs[pt])} pairs)")
            
            try:
                selection = int(input("\nEnter pattern number: ")) - 1
                if 0 <= selection < len(pattern_types):
                    pattern_type = pattern_types[selection]
                else:
                    pattern_type = pattern_types[0]
            except:
                pattern_type = pattern_types[0]
    
    if pattern_type not in found_pairs:
        print(f"\nPattern {pattern_type} not found. Available patterns: {list(found_pairs.keys())}")
        return test_synthetic_data()
    
    pairs = found_pairs[pattern_type]
    
    # Select which pair to use
    if len(pairs) > 1:
        print(f"\nFound {len(pairs)} stereo pairs. Select which one to use:")
        for i, (left_path, right_path) in enumerate(pairs[:5]):  # Show first 5
            print(f"  {i+1}. {os.path.basename(left_path)} - {os.path.basename(right_path)}")
        
        try:
            pair_idx = int(input(f"\nEnter pair number (1-{min(5, len(pairs))}): ")) - 1
            if pair_idx < 0 or pair_idx >= len(pairs):
                pair_idx = 0
        except:
            pair_idx = 0
    else:
        pair_idx = 0
    
    left_path, right_path = pairs[pair_idx]
    
    # Load the image pair
    print(f"\nLoading: {os.path.basename(left_path)} - {os.path.basename(right_path)}")
    
    try:
        # Load images (resize for faster testing)
        left_img, right_img = StereoDataLoader.load_image_pair(
            left_path, right_path, 
            resize_to=(256, 256),  # Resize to 256x256 for faster processing
            normalize=True
        )
        
        print(f"Loaded images: {left_img.shape}")
        
        # Estimate maximum disparity (simple heuristic)
        max_disp = min(64, left_img.shape[1] // 4)  # Max disparity = 1/4 of image width
        
        # Create algorithm instance
        pm = SimplePatchMatchMVP(max_disp=max_disp, window_size=9)
        
        # Run random initialization
        pm.random_initialization(left_img.shape[0], left_img.shape[1])
        
        # Evaluate random initialization
        stats = pm.evaluate_random_cost(left_img, right_img, sample_points=500)
        
        # Compute disparity
        left_disp, right_disp = pm.compute_disparity_maps()
        
        # Visualize results
        dataset_name = os.path.basename(os.path.dirname(left_path)) or "Local Dataset"
        visualize_results(left_img, right_img, left_disp, 
                         title=f"{dataset_name} - Random Initialization")
        
        # Save results
        pm.save_results("results")
        
        return pm, stats, left_img, right_img, left_disp
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Falling back to synthetic data...")
        return test_synthetic_data()


def visualize_results(left_img: np.ndarray, right_img: np.ndarray, 
                     disparity: np.ndarray, title: str = "Results"):
    """Visualize stereo matching results"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Original images
    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis('off')
    
    # Color difference (for reference)
    if left_img.shape == right_img.shape:
        color_diff = np.mean(np.abs(left_img - right_img), axis=2)
        im_diff = axes[0, 2].imshow(color_diff, cmap='hot')
        axes[0, 2].set_title("Color Difference")
        axes[0, 2].axis('off')
        plt.colorbar(im_diff, ax=axes[0, 2], fraction=0.046, pad=0.04)
    else:
        axes[0, 2].axis('off')
    
    # Disparity map
    im_disp = axes[1, 0].imshow(disparity, cmap='viridis')
    axes[1, 0].set_title("Disparity Map")
    axes[1, 0].axis('off')
    plt.colorbar(im_disp, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # Disparity histogram
    axes[1, 1].hist(disparity.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 1].set_title("Disparity Histogram")
    axes[1, 1].set_xlabel("Disparity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 3D surface plot (simplified)
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        # Downsample for performance
        h, w = disparity.shape
        step = max(1, h // 50)
        y, x = np.mgrid[0:h:step, 0:w:step]
        z = disparity[::step, ::step]
        
        ax3d = fig.add_subplot(1, 3, 3, projection='3d')
        surf = ax3d.plot_surface(x, y, z, cmap='viridis', 
                                linewidth=0, antialiased=False, alpha=0.8)
        ax3d.set_title("3D Disparity Surface")
        ax3d.set_xlabel("X")
        ax3d.set_ylabel("Y")
        ax3d.set_zlabel("Disparity")
        fig.colorbar(surf, ax=ax3d, shrink=0.5, aspect=5)
        
        # Hide the empty subplot
        axes[1, 2].axis('off')
        
    except ImportError:
        # Show plane field instead if 3D not available
        axes[1, 2].text(0.5, 0.5, "3D plot requires mpl_toolkits", 
                       ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title("3D Visualization (Not Available)")
        axes[1, 2].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


def run_comprehensive_test():
    """Run comprehensive test with options"""
    print("MVP PATCHMATCH STEREO - MILESTONE 1")
    print("=" * 60)
    
    print("\nSelect test mode:")
    print("1. Test with synthetic data (quick)")
    print("2. Test with local dataset")
    print("3. Scan dataset directory (info only)")
    
    try:
        choice = input("\nEnter choice (1-3): ").strip()
        
        if choice == "1":
            # Synthetic data test
            pm, stats = test_synthetic_data()
            
        elif choice == "2":
            # Local dataset test
            dataset_dir = input("Enter dataset directory path: ").strip()
            
            if not dataset_dir:
                dataset_dir = "./data"  # Default
            
            if not os.path.exists(dataset_dir):
                print(f"Directory does not exist: {dataset_dir}")
                print("Creating sample synthetic data...")
                pm, stats = test_synthetic_data()
            else:
                pm, stats, left_img, right_img, disparity = test_local_dataset(dataset_dir)
                
                # Ask if user wants to visualize plane field
                visualize_planes = input("\nVisualize plane field? (y/n): ").strip().lower()
                if visualize_planes == 'y':
                    fig = pm.visualize_plane_field(left_img.shape, sample_density=15)
                    plt.suptitle("Plane Field Visualization")
                    plt.show()
        
        elif choice == "3":
            # Scan directory only
            dataset_dir = input("Enter directory to scan: ").strip()
            if not dataset_dir:
                dataset_dir = "./data"
            
            StereoDataLoader.display_dataset_info(dataset_dir)
            return
        
        else:
            print("Invalid choice. Using synthetic data.")
            pm, stats = test_synthetic_data()
        
        # Print summary
        print_summary(pm, stats)
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()


def print_summary(algorithm, stats):
    """Print test summary"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY - MILESTONE 1 COMPLETE")
    print("=" * 60)
    
    print("\nALGORITHM CONFIGURATION:")
    print(f"  Max disparity: {algorithm.max_disp}")
    print(f"  Window size: {algorithm.window_size}")
    
    print("\nRANDOM INITIALIZATION STATISTICS:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Valid matches: {stats['valid_samples']} ({stats['valid_rate']:.1f}%)")
    print(f"  Average matching cost: {stats['avg_cost']:.3f}")
    print(f"  Minimum cost: {stats['min_cost']:.3f}")
    print(f"  Maximum cost: {stats['max_cost']:.3f}")
    
    print("\nIMPLEMENTED FEATURES (Milestone 1):")
    print("  1. 3D plane representation (SimplePlane class)")
    print("  2. Random plane initialization")
    print("  3. Disparity computation from planes")
    print("  4. Simple pixel matching cost")
    print("  5. Plane field visualization")
    
    print("\nNEXT STEPS (Milestone 2):")
    print("  1. Implement spatial propagation")
    print("  2. Add plane refinement")
    print("  3. Implement iterative optimization")
    print("  4. Add view propagation")
    print("  5. Implement post-processing")
    
    print("\n" + "=" * 60)





    # 在 test_local_data.py 中添加这个函数
def load_middlebury_from_subdirs(dataset_dir: str):
    """加载Middlebury格式的数据集（子目录结构）"""
    import glob
    
    scenes = []
    
    # 查找所有包含 view1.png 的子目录
    pattern = os.path.join(dataset_dir, "*", "view1.png")
    left_files = glob.glob(pattern)
    
    for left_file in left_files:
        scene_dir = os.path.dirname(left_file)
        scene_name = os.path.basename(scene_dir)
        right_file = os.path.join(scene_dir, "view5.png")
        
        if os.path.exists(right_file):
            scenes.append({
                'name': scene_name,
                'dir': scene_dir,
                'left_path': left_file,
                'right_path': right_file,
                'disp_left_path': os.path.join(scene_dir, "disp1.png") if os.path.exists(os.path.join(scene_dir, "disp1.png")) else None,
                'disp_right_path': os.path.join(scene_dir, "disp5.png") if os.path.exists(os.path.join(scene_dir, "disp5.png")) else None
            })
    
    return scenes


if __name__ == "__main__":
    # Check requirements
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install requirements: pip install numpy matplotlib opencv-python")
        sys.exit(1)
    
    # Run comprehensive test
    run_comprehensive_test()