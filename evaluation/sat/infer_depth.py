import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2

def predict_depth_from_image(model, image, grayscale=False):
    """处理单个图片并生成深度图"""
    raw_image = np.array(image)
    # 如果图像是 RGB 格式，转换为 BGR 格式（OpenCV 默认使用 BGR）
    if raw_image.shape[-1] == 3:
        raw_image = cv2.cvtColor(raw_image, cv2.COLOR_RGB2BGR)
    # raw_image = cv2.imread(image)
    depth = model.infer_image(raw_image, 518)
    
    # 归一化和转换为 8 位图像
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.astype(np.uint8)
    
    # 应用彩色调色板或保持灰度
    if grayscale:
        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
    else:
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    
    return depth

def load_depth_model(encoder, checkpoint_dir):
    # 设置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 加载模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 初始化模型
    depth_anything = DepthAnythingV2(**model_configs[encoder])
    depth_anything.load_state_dict(torch.load(checkpoint_dir, map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    return depth_anything, DEVICE

def save_depth_image(depth, original_path):
    """保存深度图到原始图片所在位置，文件名加上 _d 后缀"""
    base_dir = os.path.dirname(original_path)
    file_name = os.path.splitext(os.path.basename(original_path))[0]
    output_path = os.path.join(base_dir, f"{file_name}_d.png")
    
    cv2.imwrite(output_path, depth)
    # print(f"Saved depth image to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--img-folder', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--input-size', type=int, default=518, help='输入图片大小')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='编码器类型')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='生成灰度深度图')
    
    args = parser.parse_args()
    
    # 设置设备
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 加载模型配置
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # 初始化模型
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'Depth-Anything-V2/model.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()
    
    # 查找文件夹内所有图片（支持常见图片格式）
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(args.img_folder, '**', ext), recursive=True))
    
    # 处理每张图片
    for image_path in image_paths:
        print(f"Processing: {image_path}")
        
        try:
            # 生成深度图
            depth = process_image(depth_anything, image_path, args.input_size, args.grayscale)
        
            # 保存深度图
            save_depth_image(depth, image_path)
        except:
            continue
    
    print("Processing completed!")