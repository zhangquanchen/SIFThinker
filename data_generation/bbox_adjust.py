import random

def expand_box(box, expand_ratio=0.1):
    """以中心点为中心，膨胀边界框"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    new_w = w * (1 + expand_ratio)
    new_h = h * (1 + expand_ratio)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_x1 = max(0, cx - new_w / 2)
    new_x2 = min(1, cx + new_w / 2)
    new_y1 = max(0, cy - new_h / 2)
    new_y2 = min(1, cy + new_h / 2)
    return [new_x1, new_y1, new_x2, new_y2]

def scale_box(box, scale_range=(0.8, 1.2)):
    """随机缩放边界框，保持中心点不变"""
    x1, y1, x2, y2 = box
    w = x2 - x1
    h = y2 - y1
    scale = random.uniform(scale_range[0], scale_range[1])
    new_w = w * scale
    new_h = h * scale
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    new_x1 = max(0, cx - new_w / 2)
    new_x2 = min(1, cx + new_w / 2)
    new_y1 = max(0, cy - new_h / 2)
    new_y2 = min(1, cy + new_h / 2)
    return [new_x1, new_y1, new_x2, new_y2]

def compute_iou(boxA, boxB):
    """计算两个边界框的IoU"""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    iou = inter_area / (areaA + areaB - inter_area) if (areaA + areaB - inter_area) !=0 else 0
    return iou

def translate_box(box, dx, dy):
    """平移边界框并确保在[0,1]范围内"""
    x1, y1, x2, y2 = box
    new_x1 = max(0, min(1, x1 + dx))
    new_y1 = max(0, min(1, y1 + dy))
    new_x2 = max(0, min(1, x2 + dx))
    new_y2 = max(0, min(1, y2 + dy))
    return [new_x1, new_y1, new_x2, new_y2]

def adjust_box(original_box):
    """主函数：膨胀->缩放->平移处理"""
    # 步骤1：膨胀处理
    expanded_box = expand_box(original_box, expand_ratio=0.2)
    
    # 步骤2：缩放处理
    scaled_box = scale_box(expanded_box, scale_range=(0.7, 1.3))
    
    # 步骤3：平移处理
    s_x1, s_y1, s_x2, s_y2 = scaled_box
    o_x1, o_y1, o_x2, o_y2 = original_box
    s_w = s_x2 - s_x1
    s_h = s_y2 - s_y1
    
    # 优先尝试四个角落移动（保证零重叠）
    options = []
    # 右下方
    if (o_x2 + s_w <= 1) and (o_y2 + s_h <= 1):
        options.append((o_x2 - s_x1, o_y2 - s_y1))
    # 右上方
    if (o_x2 + s_w <= 1) and (o_y1 - s_h >= 0):
        options.append((o_x2 - s_x1, (o_y1 - s_h) - s_y1))
    # 左下方
    if (o_x1 - s_w >= 0) and (o_y2 + s_h <= 1):
        options.append(((o_x1 - s_w) - s_x1, o_y2 - s_y1))
    # 左上方
    if (o_x1 - s_w >= 0) and (o_y1 - s_h >= 0):
        options.append(((o_x1 - s_w) - s_x1, (o_y1 - s_h) - s_y1))
    
    if options:
        dx, dy = random.choice(options)
        return expanded_box, translate_box(scaled_box, dx, dy)
    
    # 随机平移（当无法找到零重叠位置时）
    max_attempts = 1000
    for _ in range(max_attempts):
        dx = random.uniform(-s_x1, 1 - s_x2)
        dy = random.uniform(-s_y1, 1 - s_y2)
        translated_box = translate_box(scaled_box, dx, dy)
        if compute_iou(original_box, translated_box) < 0.001:
            return expanded_box, translated_box
    
    # 保底返回原缩放框（可能不满足条件）
    return expanded_box, scaled_box

def adjust_3f_bbox(original_box):
    expanded_box, adjusted_box = adjust_box(original_box)
    expanded_box_modify = [round(x, 3) for x in expanded_box]
    adjusted_box_modify = [round(x, 3) for x in adjusted_box]
    return expanded_box_modify, adjusted_box_modify

# # 使用示例
# original_box = [0.3, 0.3, 0.6, 0.6]
# expanded_box, adjusted_box = adjust_box(original_box)
# expanded_box_modify = [round(x, 3) for x in expanded_box]
# adjusted_box_modify = [round(x, 3) for x in adjusted_box]

# print(f"原始框: {original_box}")
# print(f"膨胀框: {expanded_box_modify}")
# print(f"处理后框: {adjusted_box_modify}")
# print(f"IoU值: {compute_iou(original_box, adjusted_box):.4f}")