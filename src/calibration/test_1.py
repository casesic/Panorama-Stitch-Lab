import cv2 as cv
import numpy as np
import json
import os

def undistort_image_with_params():
    """
    基于给定畸变参数，通过手动计算map1/map2实现图像去畸变
    畸变模型：使用提供的a/b/c/d/e/f系数，基于归一化坐标的径向+切向畸变模型
    """
    # 1. 配置文件和图像路径
    intrinsic_path = r"src\\calibration\\intrinsic.json"  # 畸变参数文件路径（请确保文件存在）
    src_img_path = r"data\\origin_1\\origin_1.snap.1.png"  # 源图像路径
    dst_img_path = r"data\\origin_1\\origin_1.snap.1.undist.png"  # 输出去畸变图像路径

    # 2. 读取并解析畸变参数
    if not os.path.exists(intrinsic_path):
        raise FileNotFoundError(f"畸变参数文件不存在：{intrinsic_path}")
    with open(intrinsic_path, 'r', encoding='utf-8') as f:
        intrinsic_data = json.load(f)
    
    distortion = intrinsic_data["distortion"]
    cx_norm = distortion["center"]["x"]  # 归一化畸变中心x（0.5表示图像中心）
    cy_norm = distortion["center"]["y"]  # 归一化畸变中心y（0.5表示图像中心）
    a = distortion["coeffs_map"]["a"]    # 畸变系数a（k1）
    b = distortion["coeffs_map"]["b"]    # 畸变系数b（k2）
    c = distortion["coeffs_map"]["c"]    # 畸变系数c（k3）
    d = distortion["coeffs_map"]["d"]    # 畸变系数d（p1）
    e = distortion["coeffs_map"]["e"]    # 畸变系数e（p2）
    f = distortion["coeffs_map"]["f"]    # 畸变系数f（p3，通常为0）
    crop_factor = distortion["crop_factor"]  # 裁剪系数（1.0表示不裁剪）

    # 3. 读取源图像并获取图像尺寸
    src_img = cv.imread(src_img_path)
    if src_img is None:
        raise FileNotFoundError(f"无法读取源图像：{src_img_path}")
    h, w = src_img.shape[:2]
    print(f"源图像尺寸：{w} x {h}")

    # 4. 生成目标图像的像素坐标网格（去畸变后的坐标）
    # 生成归一化坐标（x_norm, y_norm）：范围[0,1]，以图像左上角为原点
    x_norm, y_norm = np.meshgrid(
        np.linspace(0, 1, w, dtype=np.float32),  # x轴归一化坐标（列方向）
        np.linspace(0, 1, h, dtype=np.float32)   # y轴归一化坐标（行方向）
    )

    # 5. 畸变模型计算：目标坐标 -> 源图像畸变坐标（核心公式）
    # 步骤1：将归一化坐标平移到畸变中心（以畸变中心为原点）
    x_prime = x_norm - cx_norm
    y_prime = y_norm - cy_norm

    # 步骤2：计算径向距离的平方和径向因子
    r_sq = x_prime**2 + y_prime**2  # r² = x'² + y'²
    radial_factor = 1 + a * r_sq + b * r_sq**2 + c * r_sq**3  # 径向畸变因子：1 + a r² + b r⁴ + c r⁶

    # 步骤3：计算切向畸变项
    tangential_x = 2 * d * x_prime * y_prime + e * (r_sq + 2 * x_prime**2)  # 切向畸变x分量
    tangential_y = d * (r_sq + 2 * y_prime**2) + 2 * e * x_prime * y_prime  # 切向畸变y分量

    # 步骤4：计算畸变后的归一化坐标（源图像中的畸变坐标）
    x_distorted_norm = x_prime * radial_factor + tangential_x + cx_norm
    y_distorted_norm = y_prime * radial_factor + tangential_y + cy_norm

    # 步骤5：将归一化坐标转换为像素坐标（乘以图像尺寸）
    map1 = x_distorted_norm * w  # 源图像x坐标映射（列）
    map2 = y_distorted_norm * h  # 源图像y坐标映射（行）

    # 6. 应用裁剪系数（如果需要）
    if crop_factor != 1.0:
        # 计算裁剪后的有效区域
        crop_w = int(w * crop_factor)
        crop_h = int(h * crop_factor)
        offset_x = (w - crop_w) // 2
        offset_y = (h - crop_h) // 2
        
        # 裁剪映射表和输出图像
        map1 = map1[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]
        map2 = map2[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]
        src_img = src_img[offset_y:offset_y+crop_h, offset_x:offset_x+crop_w]

    # 7. 执行remap去畸变（使用线性插值，速度和效果平衡）
    undistorted_img = cv.remap(
        src=src_img,
        map1=map1,
        map2=map2,
        interpolation=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE  # 边界像素复制填充
    )

    # 8. 保存并显示结果
    cv.imwrite(dst_img_path, undistorted_img)
    print(f"去畸变图像已保存至：{dst_img_path}")


if __name__ == "__main__":
    try:
        undistort_image_with_params()
    except Exception as e:
        print(f"执行错误：{str(e)}")