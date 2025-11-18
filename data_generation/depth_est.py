import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
import random
import math

def load_depth_from_color(color_depth_path):
    """
    从伪彩色深度图恢复归一化深度值（0-1范围）
    
    参数:
        color_depth_path: 伪彩色深度图路径
        
    返回:
        normalized_depth: 归一化深度图（0-1范围，float32数组）
    """
    # 读取伪彩色深度图 (BGR格式)
    color_depth = cv2.imread(color_depth_path)
    if color_depth is None:
        raise FileNotFoundError(f"无法读取图像: {color_depth_path}")
    
    # 检查是否为三通道灰度图（原始处理中灰度图被复制为三通道）
    if is_grayscale_image(color_depth):
        # 直接取单通道并归一化
        normalized_depth = color_depth[:, :, 0].astype(np.float32) / 255.0
        return normalized_depth
    
    # 生成调色板颜色表 (256个BGR颜色)
    depth_levels = np.arange(256)
    normalized_values = depth_levels / 255.0
    colormap = plt.get_cmap('Spectral_r')
    colors_rgba = colormap(normalized_values)  # 获取RGBA颜色
    colors_rgb = (colors_rgba[:, :3] * 255).astype(np.uint8)  # 转换为RGB [0-255]
    palette_bgr = colors_rgb[:, ::-1]  # RGB转BGR (与OpenCV一致)
    
    # 构建KDTree加速颜色搜索
    tree = KDTree(palette_bgr)
    
    # 重构像素数组为(像素数, 3)
    pixels = color_depth.reshape(-1, 3)
    
    # 查询每个像素最近的调色板索引
    _, indices = tree.query(pixels, k=1)
    
    # 将索引转换为深度值并重塑为原图像形状
    depth_level_map = indices.reshape(color_depth.shape[:2])
    normalized_depth = 1.0 - depth_level_map.astype(np.float32) / 255.0
    
    return normalized_depth

def calculate_avg_depth(depth_map, normalized_bbox):
    """
    计算边界框内深度平均值
    
    参数:
        color_depth_path: 伪彩色深度图路径
        normalized_bbox: 归一化边界框 [x1, y1, x2, y2]
        
    返回:
        avg_depth: 边界框内深度平均值 (0-1范围)
    """
    # 加载深度图
    # depth_map = load_depth_from_color(color_depth_path)
    h, w = depth_map.shape[:2]
    # 解析边界框坐标
    x1, y1, x2, y2 = normalized_bbox
    
    # 转换为像素坐标 (使用向上取整确保包含整个区域)
    x1_pix = int(math.floor(x1 * w))
    y1_pix = int(math.floor(y1 * h))
    x2_pix = int(math.ceil(x2 * w))
    y2_pix = int(math.ceil(y2 * h))
    
    # 确保坐标在图像范围内
    x1_pix = max(0, min(x1_pix, w-1))
    y1_pix = max(0, min(y1_pix, h-1))
    x2_pix = max(0, min(x2_pix, w))
    y2_pix = max(0, min(y2_pix, h))
    
    # 检查有效区域
    if x2_pix <= x1_pix or y2_pix <= y1_pix:
        raise ValueError("无效边界框区域")
    
    # 提取边界框内的深度值
    depth_roi = depth_map[y1_pix:y2_pix, x1_pix:x2_pix]
    
    # 计算深度平均值
    avg_depth = np.mean(depth_roi)
    
    return float(avg_depth)

    
def is_grayscale_image(img, tolerance=2):
    """检查是否为三通道灰度图（各通道值相等）"""
    diff1 = cv2.absdiff(img[:, :, 0], img[:, :, 1])
    diff2 = cv2.absdiff(img[:, :, 1], img[:, :, 2])
    return np.max(diff1) <= tolerance and np.max(diff2) <= tolerance


def process_depth_map(image_path, num_points=5):
    # 读取原始深度图像
    original_image = cv2.imread(image_path)
    
    # 获取图像尺寸
    height, width, _ = original_image.shape
    
    # 随机生成点
    random_points = [(random.randint(0, width-1), random.randint(0, height-1)) for _ in range(num_points)]
    
    # 创建一个用于显示的图像副本
    display_image = original_image.copy()
    
    # 存储结果
    results = []
    
    # 处理每个随机点
    normalized_depth = load_depth_from_color(image_path)
    for x, y in random_points:
        depth_value = normalized_depth[y, x]
        results.append((x, y, depth_value))
        
        # 在图像上标记该点
        cv2.circle(display_image, (x, y), 5, (0, 0, 255), 2)
        cv2.putText(display_image, f"{depth_value:.3f}", (x+10, y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    # 保存结果图像
    output_path = "marked_depth_map.png"
    cv2.imwrite(output_path, display_image)
    
    # 打印结果
    print(f"处理完成，标记的图像已保存到: {output_path}")
    for x, y, depth in results:
        print(f"点({x}, {y}): 深度值 = {depth:.3f}")
    
    return results, output_path

# 使用示例
if __name__ == '__main__':
    image_path = 'flickr30k/36979_d.png'  # 替换为您的深度图路径
    bbox = [0.1,0.1,0.2,0.2]
    # process_depth_map(image_path)
    depth_map = load_depth_from_color(image_path)
    mean_depth = calculate_avg_depth(depth_map, bbox)
    print(mean_depth)