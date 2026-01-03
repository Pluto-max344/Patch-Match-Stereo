import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob
import cv2

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_patchmatch import SimplePatchMatchMVP


def find_middlebury_scenes(data_dir: str):

    scenes = []
    
    # 检查是否是Middlebury 2006格式
    pattern = os.path.join(data_dir, "*", "view1.png")
    left_files = sorted(glob.glob(pattern))
    
    for left_file in left_files:
        scene_dir = os.path.dirname(left_file)
        scene_name = os.path.basename(scene_dir)
        
        right_file = os.path.join(scene_dir, "view5.png")
        if os.path.exists(right_file):
            scenes.append({
                'name': scene_name,
                'directory': scene_dir,
                'left_path': left_file,
                'right_path': right_file,
                'has_gt': os.path.exists(os.path.join(scene_dir, "disp1.png"))
            })
    
    # 如果没有找到，尝试其他格式
    if not scenes:
        pattern = os.path.join(data_dir, "*", "im0.png")
        left_files = sorted(glob.glob(pattern))
        
        for left_file in left_files:
            scene_dir = os.path.dirname(left_file)
            scene_name = os.path.basename(scene_dir)
            
            right_file = os.path.join(scene_dir, "im1.png")
            if os.path.exists(right_file):
                scenes.append({
                    'name': scene_name,
                    'directory': scene_dir,
                    'left_path': left_file,
                    'right_path': right_file,
                    'has_gt': os.path.exists(os.path.join(scene_dir, "disp0.png"))
                })
    
    return scenes


def load_middlebury_pair(left_path: str, right_path: str, resize_to=None):

    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        raise ValueError(f"Failed to load images: {left_path}, {right_path}")
    
    # Convert BGR to RGB
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
    
    # Resize if requested
    if resize_to is not None:
        height, width = resize_to
        left_img = cv2.resize(left_img, (width, height))
        right_img = cv2.resize(right_img, (width, height))
    
    # Normalize to [0, 1]
    left_img = left_img.astype(np.float32) / 255.0
    right_img = right_img.astype(np.float32) / 255.0
    
    return left_img, right_img


def load_ground_truth(disp_path: str, resize_to=None):
    if not os.path.exists(disp_path):
        return None
    
    # Middlebury视差图通常是16位PNG，需要除以因子
    disp_img = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    
    if disp_img is None:
        return None
    
    # 对于16位PNG，除以4得到真实视差
    if disp_img.dtype == np.uint16:
        disp = disp_img.astype(np.float32) / 4.0
    else:
        disp = disp_img.astype(np.float32)
    
    # Resize if needed
    if resize_to is not None:
        height, width = resize_to
        disp = cv2.resize(disp, (width, height))
    
    return disp


def run_middlebury_test(data_dir: str = "./data"):
    print("=" * 60)
    print("MIDDLEBURY DATASET TEST")
    print("=" * 60)
    
    # 查找所有场景
    scenes = find_middlebury_scenes(data_dir)
    
    if not scenes:
        print(f"No Middlebury scenes found in {data_dir}")
        print("\nExpected structure:")
        print("  data/")
        print("  ├── Scene1/")
        print("  │   ├── view1.png    # Left view")
        print("  │   ├── view5.png    # Right view")
        print("  │   ├── disp1.png    # Left disparity (optional)")
        print("  │   └── disp5.png    # Right disparity (optional)")
        print("  └── Scene2/")
        print("      ├── view1.png")
        print("      └── view5.png")
        return
    
    print(f"\nFound {len(scenes)} Middlebury scenes:")
    for i, scene in enumerate(scenes):
        gt_marker = "✓" if scene['has_gt'] else " "
        print(f"  {i+1:2d}. [{gt_marker}] {scene['name']}")
    
    # 选择场景
    try:
        scene_idx = int(input(f"\nSelect scene (1-{len(scenes)}): ")) - 1
        if scene_idx < 0 or scene_idx >= len(scenes):
            scene_idx = 0
    except:
        scene_idx = 0
    
    scene = scenes[scene_idx]
    print(f"\nSelected scene: {scene['name']}")
    
    # 选择处理大小
    print("\nSelect processing size:")
    print("  1. Full resolution (original size)")
    print("  2. Medium (512x384)")
    print("  3. Small (256x192) - faster for testing")
    
    try:
        size_choice = int(input("Enter choice (1-3): "))
    except:
        size_choice = 3
    
    if size_choice == 1:
        resize_to = None  # Full resolution
    elif size_choice == 2:
        resize_to = (384, 512)  # Medium
    else:
        resize_to = (192, 256)  # Small
    
    # 加载图像
    print(f"\nLoading images...")
    left_img, right_img = load_middlebury_pair(
        scene['left_path'], scene['right_path'], resize_to
    )
    
    print(f"Image size: {left_img.shape}")
    
    # 加载真值（如果有）
    ground_truth = None
    if scene['has_gt']:
        gt_path = os.path.join(scene['directory'], "disp1.png")
        ground_truth = load_ground_truth(gt_path, resize_to)
        if ground_truth is not None:
            print(f"Ground truth loaded: {ground_truth.shape}")
    
    # 估计最大视差
    # Middlebury数据集的典型视差范围
    if resize_to is None:
        max_disp = 64  # 对于全分辨率
    else:
        max_disp = 32  # 对于缩小版本
    
    # 创建算法实例
    print(f"\nCreating PatchMatch instance (max_disp={max_disp})...")
    pm = SimplePatchMatchMVP(max_disp=max_disp, window_size=9)
    
    # 运行随机初始化（Milestone 1）
    print("\nRunning random initialization...")
    pm.random_initialization(left_img.shape[0], left_img.shape[1])
    
    # 评估随机初始化
    print("\nEvaluating random initialization...")
    stats = pm.evaluate_random_cost(left_img, right_img, sample_points=500)
    
    # 计算视差
    print("\nComputing disparity maps...")
    left_disp, right_disp = pm.compute_disparity_maps()
    
    # 可视化结果
    visualize_results_with_gt(
        left_img, right_img, left_disp, 
        ground_truth=ground_truth,
        title=f"Middlebury: {scene['name']} - Random Initialization"
    )
    
    # 保存结果
    os.makedirs("results", exist_ok=True)
    
    # 保存视差图
    save_disparity_as_png(left_disp, f"results/{scene['name']}_disparity.png")
    
    # 保存真值对比（如果有）
    if ground_truth is not None:
        save_disparity_as_png(ground_truth, f"results/{scene['name']}_ground_truth.png")
        
        # 计算简单误差
        error = compute_simple_error(left_disp, ground_truth)
        print(f"\nSimple error metrics (vs ground truth):")
        print(f"  Mean absolute error: {error['mae']:.2f} pixels")
        print(f"  Root mean square error: {error['rmse']:.2f} pixels")
        print(f"  Bad pixels (>1px): {error['bad_pixels_1']:.1f}%")
        print(f"  Bad pixels (>2px): {error['bad_pixels_2']:.1f}%")
    
    # 询问是否可视化平面场
    try:
        show_planes = input("\nVisualize plane field? (y/n): ").strip().lower()
        if show_planes == 'y':
            fig = pm.visualize_plane_field(left_img.shape, sample_density=20)
            plt.suptitle(f"Plane Field: {scene['name']}")
            plt.show()
    except:
        pass
    
    # 打印总结
    print_summary(pm, stats, scene['name'], ground_truth is not None)
    
    return pm, left_img, right_img, left_disp, ground_truth


def visualize_results_with_gt(left_img, right_img, disparity, 
                            ground_truth=None, title="Results"):
    
    ncols = 3 if ground_truth is None else 4
    
    fig, axes = plt.subplots(2, ncols, figsize=(5*ncols, 8))
    
    # 第一行：输入图像
    axes[0, 0].imshow(left_img)
    axes[0, 0].set_title("Left Image")
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(right_img)
    axes[0, 1].set_title("Right Image")
    axes[0, 1].axis('off')
    
    # 第二行：视差结果
    im1 = axes[1, 0].imshow(disparity, cmap='viridis')
    axes[1, 0].set_title("Estimated Disparity")
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # 视差直方图
    axes[1, 1].hist(disparity.flatten(), bins=50, alpha=0.7, color='blue')
    axes[1, 1].set_title("Disparity Histogram")
    axes[1, 1].set_xlabel("Disparity")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].grid(True, alpha=0.3)
    
    # 如果有真值，显示真值和误差
    if ground_truth is not None:
        # 真值
        im2 = axes[0, 2].imshow(ground_truth, cmap='viridis')
        axes[0, 2].set_title("Ground Truth")
        axes[0, 2].axis('off')
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
        
        # 误差图
        error_map = np.abs(disparity - ground_truth)
        error_map[np.isnan(ground_truth)] = 0  # 处理无效像素
        
        im3 = axes[1, 2].imshow(error_map, cmap='hot', vmin=0, vmax=10)
        axes[1, 2].set_title("Error Map")
        axes[1, 2].axis('off')
        plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
        
        # 如果有第四列，显示差异
        if ncols == 4:
            diff_img = np.abs(left_img - right_img)
            axes[0, 3].imshow(np.mean(diff_img, axis=2), cmap='gray')
            axes[0, 3].set_title("Image Difference")
            axes[0, 3].axis('off')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


def save_disparity_as_png(disparity, filename):
    # 归一化到0-255
    disp_norm = disparity.copy()
    
    # 忽略无效值
    valid_mask = ~np.isnan(disp_norm)
    if np.any(valid_mask):
        valid_values = disp_norm[valid_mask]
        if np.max(valid_values) > np.min(valid_values):
            disp_norm[valid_mask] = (valid_values - np.min(valid_values)) / (np.max(valid_values) - np.min(valid_values))
    
    disp_norm = np.clip(disp_norm * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(filename, disp_norm)
    print(f"Saved: {filename}")


def compute_simple_error(estimated, ground_truth):
    # 只计算有效像素
    valid_mask = ~np.isnan(ground_truth) & (ground_truth > 0)
    
    if np.sum(valid_mask) == 0:
        return {'mae': float('nan'), 'rmse': float('nan'), 
                'bad_pixels_1': float('nan'), 'bad_pixels_2': float('nan')}
    
    est_valid = estimated[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    # 绝对误差
    abs_errors = np.abs(est_valid - gt_valid)
    
    # 均方根误差
    mae = np.mean(abs_errors)
    rmse = np.sqrt(np.mean((est_valid - gt_valid) ** 2))
    
    # 坏像素率
    bad_pixels_1 = np.mean(abs_errors > 1.0) * 100
    bad_pixels_2 = np.mean(abs_errors > 2.0) * 100
    
    return {
        'mae': mae,
        'rmse': rmse,
        'bad_pixels_1': bad_pixels_1,
        'bad_pixels_2': bad_pixels_2
    }


def print_summary(algorithm, stats, scene_name, has_gt):
    print("\n" + "=" * 60)
    print("MIDDLEBURY TEST SUMMARY - MILESTONE 1")
    print("=" * 60)
    
    print(f"\nSCENE: {scene_name}")
    print(f"HAS GROUND TRUTH: {'Yes' if has_gt else 'No'}")
    
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
    print("  1. 3D plane representation")
    print("  2. Random plane initialization")
    print("  3. Disparity computation from planes")
    print("  4. Simple pixel matching cost")
    print("  5. Middlebury dataset support")
    
    print("\nEXPECTED RESULTS:")
    print("  - Disparity map will be noisy (random initialization only)")
    print("  - No spatial coherence between pixels")
    print("  - This validates the basic framework works")
    
    print("\nNEXT STEPS (Milestone 2):")
    print("  1. Add spatial propagation for coherence")
    print("  2. Implement plane refinement")
    print("  3. Add iterative optimization")
    print("  4. Include view propagation")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 检查依赖
    try:
        import numpy as np
        import matplotlib.pyplot as plt
        import cv2
    except ImportError as e:
        print(f"Missing required package: {e}")
        print("Please install: pip install numpy matplotlib opencv-python")
        sys.exit(1)
    
    # 运行测试
    data_dir = "./data"
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    try:
        run_middlebury_test(data_dir)
    except KeyboardInterrupt:
        print("\nTest interrupted by user.")
    except Exception as e:
        print(f"\nError during test: {e}")
        import traceback
        traceback.print_exc()