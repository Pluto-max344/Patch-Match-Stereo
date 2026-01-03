import cv2
import numpy as np
from pm import PatchMatch

def check_image(image, name="Image"):
    if image is None or image.size == 0:
        print(f"{name} data not loaded.")
        return False
    return True

def check_dimensions(img1, img2):
    if img1.shape != img2.shape:
        print("Images' dimensions do not correspond.")
        return False
    return True

def main():
    # 参数设置
    alpha = 0.9
    gamma = 10.0
    tau_c = 10.0
    tau_g = 2.0
    
    # 读取PNG图片
    img1 = cv2.imread("dataset/Aloe/view1.png", cv2.IMREAD_COLOR)
    img2 = cv2.imread("dataset/Aloe/view5.png", cv2.IMREAD_COLOR)
    
    # 检查图片加载
    if not check_image(img1, "Left image") or not check_image(img2, "Right image"):
        return 1
    
    # 检查图片尺寸
    if not check_dimensions(img1, img2):
        return 1
    
    # 处理图片
    patch_match = PatchMatch(alpha, gamma, tau_c, tau_g)
    patch_match.set(img1, img2)
    patch_match.process(3)
    patch_match.postProcess()
    
    # 获取视差图
    disp1 = patch_match.get_left_disparity_map()
    disp2 = patch_match.get_right_disparity_map()
    
    # 归一化并保存
    disp1_normalized = cv2.normalize(disp1, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    disp2_normalized = cv2.normalize(disp2, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    try:
        cv2.imwrite("left_disparity.png", disp1_normalized)
        cv2.imwrite("right_disparity.png", disp2_normalized)
        print("Disparity maps saved successfully.")
    except Exception as e:
        print(f"Disparity save error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    main()