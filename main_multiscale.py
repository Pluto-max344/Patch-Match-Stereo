import cv2
import numpy as np
from pm_multiscale import PatchMatch

def main():
    # 读取图像
    img_left_orig = cv2.imread('dataset/Baby1/view1.png')
    img_right_orig = cv2.imread('dataset/Baby1/view5.png')
    
    if img_left_orig is None or img_right_orig is None:
        print("Error: Could not load images")
        return

    # ACMM 策略：金字塔层级
    scales = [0.25, 0.5, 1.0]
    
    prev_pm = None
    final_disp_left = None
    final_disp_right = None

    for i, scale in enumerate(scales):
        print(f"\n{'='*40}")
        print(f" PROCESSING SCALE: {scale} ({i+1}/{len(scales)})")
        print(f"{'='*40}")
        
        # 1. 缩放图像
        h, w = img_left_orig.shape[:2]
        new_size = (int(w * scale), int(h * scale))
        img_left = cv2.resize(img_left_orig, new_size, interpolation=cv2.INTER_LINEAR)
        img_right = cv2.resize(img_right_orig, new_size, interpolation=cv2.INTER_LINEAR)
        
        print(f"Current Image Size: {img_left.shape[1]}x{img_left.shape[0]}")
        
        # 2. 【关键优化】自适应窗口大小
        # 图像越小，窗口应该越小，否则会抹平所有细节
        if scale < 0.3:
            current_ws = 9   # 粗尺度：小窗口
        elif scale < 0.6:
            current_ws = 15  # 中尺度：中窗口
        else:
            current_ws = 35  # 细尺度：大窗口 (标准)
            
        print(f"Adaptive Window Size: {current_ws}x{current_ws}")

        # 3. 创建 PatchMatch 实例
        # 注意：这里我们传入 window_size
        pm = PatchMatch(alpha=0.9, gamma=10.0, tau_c=10.0, tau_g=2.0, window_size=current_ws)
        pm.set_images(img_left, img_right)
        
        # 4. 运行处理
        # 粗尺度多跑几次迭代以确保收敛，细尺度有引导，可以少跑几次
        iters = 5 if i == 0 else 3
        pm.process(iterations=iters, coarse_pm=prev_pm)
        
        # 5. 后处理
        disp_left, disp_right = pm.post_process()
        prev_pm = pm
        
        # 保存中间结果
        norm_disp = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cv2.imwrite(f"disp_result_scale_{scale}.png", norm_disp)
        
        if scale == 1.0:
            final_disp_left = disp_left
            final_disp_right = disp_right

    # 保存最终结果
    disp1_normalized = cv2.normalize(final_disp_left, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp2_normalized = cv2.normalize(final_disp_right, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite("disparity_left_final.png", disp1_normalized)
    cv2.imwrite("disparity_right_final.png", disp2_normalized)
    print("Done!")

if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")