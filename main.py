import cv2
from pm import PatchMatch

def main():
    # 读取图像
    img_left = cv2.imread('dataset/Aloe/view1.png')
    img_right = cv2.imread('dataset/Aloe/view5.png')
    
    if img_left is None or img_right is None:
        print("Error: Could not load images")
        return
    
    # 可选：缩小图像以加速（测试用）
    scale = 0.25  # 保持原尺寸
    if scale != 1.0:
        new_width = int(img_left.shape[1] * scale)
        new_height = int(img_left.shape[0] * scale)
        img_left = cv2.resize(img_left, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img_right = cv2.resize(img_right, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    print(f"Image size after scaling: {img_left.shape[1]}x{img_left.shape[0]}")

    # 创建PatchMatch实例
    pm = PatchMatch(alpha=0.9, gamma=10.0, tau_c=10.0, tau_g=2.0)
    
    # 设置图像
    pm.set_images(img_left, img_right)
    
    # 处理
    pm.process(iterations=3)
    
    # 后处理
    disp_left, disp_right = pm.post_process()
    
    # 归一化并保存
    disp1_normalized = cv2.normalize(disp_left, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp2_normalized = cv2.normalize(disp_right, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    cv2.imwrite("disparity_left.png", disp1_normalized)
    cv2.imwrite("disparity_right.png", disp2_normalized)
    print("Done!")


if __name__ == "__main__":
    import time
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")