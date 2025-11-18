from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import re
from open_r1.vlm_modules.vlm_module import VLMBaseModule
import os
from datetime import datetime
from shapely.geometry import Polygon
from shapely.ops import unary_union
import numpy as np
from scipy.optimize import linear_sum_assignment
import time
import ast
import os
from openai import OpenAI
from transformers.utils.versions import require_version
import json
import base64
from open_r1.trainer.record import reward_record
import cv2
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

record_list = []

client_think = OpenAI(
    api_key="{}".format(os.getenv("API_KEY", "0")),
    base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8020)),
)

## doubao
client = OpenAI(
    base_url="PUT YOUR MODEL URL HERE",
    api_key="PUT YOUR API KEY HERE",
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_answer(xml_text):
    # 使用正则表达式匹配<answer>和</answer>之间的内容，包括换行符
    pattern = r'<answer>\n?(.*?)\n?</answer>'  # 处理可能的换行符
    match = re.search(pattern, xml_text, re.DOTALL)
    
    if match:
        # 提取内容并去除首尾空白
        return match.group(1).strip()
    return None

def get_think_answer(client, question, base64_image, base64_image_depth):
    question = question + " Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": "\nDepth image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image_depth}"}},
                {"type": "text", "text": "\n" + question},
            ],
        }]
    result = client.chat.completions.create(messages=messages, model="test")
    response = result.choices[0].message.content
    return response

def judge_score_func_question(client, question, gt_response, pred_response, base64_image):
    SYSTEM_PROMPT = f'''You are responsible for proofreading the answers, you need to give the score to the model's answer by referring to the standard answer, based on the given question and image.
    The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
    The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
    '''

    PROMPT = f'''
    Question: {question}
    Standard answer: {gt_response}
    Model's answer: {pred_response}
    '''
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }]
    
    ## doubao
    result = client.chat.completions.create(messages=messages, model="PUT YOUR MODEL HERE")
    response = result.choices[0].message.content
    return response

def judge_score_func(client, question, gt_response, pred_response, base64_image):
    question = question[0]['content'][-1]['text'][1:].replace(' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.','')
    SYSTEM_PROMPT = f'''You are responsible for proofreading the answers, you need to give the score to the model's answer by referring to the standard answer, based on the given question and image.
    The full score is 1 point and the minimum score is 0 points. Please directly provide the score in JSON format, for example, {{"score": 0.8}}, without showing the intermediate process.
    The evaluation criteria require that the closer the model's answer is to the standard answer, the higher the score.
    '''

    PROMPT = f'''
    Question: {question}
    Standard answer: {gt_response}
    Model's answer: {pred_response}
    '''
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }]
    
    result = client.chat.completions.create(messages=messages, model="PUT YOUR MODEL HERE")
    response = result.choices[0].message.content
    return response

def judge_think_nothink_func(client, question, response, base64_image):
    SYSTEM_PROMPT = f'''You are an AI assistant responsible for evaluating the accuracy of reasoning outputs. Please score the Chain-of-thought (CoT) result based on the given question and image. You need to judge whether the reasoning process and final answer are accurate and reasonable, that is, whether the CoT result can convince you. Give a persuasion score with a maximum of 1 point and a minimum of 0 point. Please provide the score directly in JSON format, for example {{"score": 0.5}}, without showing intermediate steps. The scoring criterion requires that the more reliable and reasonable the CoT result is, the higher the score should be.'''

    PROMPT = f'''
    The question and CoT result are as follows.
    Question: {question}
    CoT result: {response}

    Please score the above CoT result. Provide the score directly in JSON format, for example {{"score": 0.5}}, without outputting intermediate steps.
    '''

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}},
            ],
        }]
    
    ## doubao
    result = client.chat.completions.create(messages=messages, model="PUT YOUR MODEL HERE")
    response = result.choices[0].message.content
    return response

def compute_single_iou(box1, box2):
    """计算两个矩形框的IoU"""
    # 确保坐标有效性
    box1 = [float(x) for x in box1]
    box2 = [float(x) for x in box2]
    
    # 计算交集区域
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # 计算各自面积
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - intersection_area
    
    return intersection_area / union_area if union_area != 0 else 0.0

def compute_pair_iou(pred_boxes, gt_boxes):
    """基于匈牙利算法的配对平均IoU计算"""
    # 处理空输入特殊情况
    if not pred_boxes and not gt_boxes: return 1.0
    if not pred_boxes or not gt_boxes: return 0.0
    
    # 构建IoU矩阵
    iou_matrix = np.array([[compute_single_iou(p, g) 
                          for g in gt_boxes] 
                         for p in pred_boxes])
    
    # 使用匈牙利算法找到最优匹配
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    
    # 计算匹配对总数和未匹配数
    total_pairs = min(len(pred_boxes), len(gt_boxes))
    matched_count = len(row_ind)
    
    # 计算平均IoU（包含未匹配的0值）
    total_iou = iou_matrix[row_ind, col_ind].sum()
    return float(total_iou / total_pairs)

def compute_global_iou(pred_boxes, gt_boxes):
    """
    计算两组边界框作为整体之间的IoU。
    
    参数：
        pred_boxes (list): 预测框列表，格式为[[xmin, ymin, xmax, ymax], ...]
        gt_boxes (list): 真实框列表，格式同上
        
    返回：
        float: IoU值，范围[0, 1]
    """
    # 将每个框转换为Polygon对象
    def create_polygons(boxes):
        polygons = []
        for box in boxes:
            x1, y1, x2, y2 = box
            if x2 <= x1 or y2 <= y1:  # 跳过无效框
                continue
            poly = Polygon([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
            polygons.append(poly)
        return polygons
    
    pred_polys = create_polygons(pred_boxes)
    gt_polys = create_polygons(gt_boxes)
    
    # 合并预测框区域
    if pred_polys:
        A = unary_union(pred_polys)
    else:
        A = Polygon()
    
    # 合并真实框区域
    if gt_polys:
        B = unary_union(gt_polys)
    else:
        B = Polygon()
    
    # 处理特殊情况
    if A.is_empty and B.is_empty:
        return 1.0  # 两者均为空时IoU为1
    elif A.is_empty or B.is_empty:
        return 0.0  # 仅一方为空时IoU为0
    
    # 计算交集和并集面积
    intersection = A.intersection(B)
    intersection_area = intersection.area
    union_area = A.area + B.area - intersection_area
    return float(intersection_area / union_area) if union_area != 0 else 0.0

def denormalize_boxes(norm_boxes, img_width, img_height):
    """ 将归一化坐标转换为绝对坐标 """
    return [
        [xmin*img_width, ymin*img_height, 
         xmax*img_width, ymax*img_height]
        for xmin, ymin, xmax, ymax in norm_boxes
    ]

def validate_format(inner_content: str):
    """验证内容是否符合指定的格式要求
    Args:
        content: 待验证的字符串内容
    Returns:
        (是否通过验证, 错误信息)
    """
    # 1. 标签结构检查
    think_pattern = r'<think>(.*?)</think>'
    answer_pattern = r'<answer>(.*?)</answer>'
    
    think_match = re.search(think_pattern, inner_content, re.DOTALL)
    answer_match = re.search(answer_pattern, inner_content, re.DOTALL)
    
    if not think_match:
        return False, "缺少<think>标签"
    if not answer_match:
        return False, "缺少<answer>标签"
    
    think_content = think_match.group(1).strip()
    answer_content = answer_match.group(1).strip()
    
    # 2. 答案部分非空检查
    if not answer_content:
        return False, "<answer>内容不能为空"
    
    # 3. 思考部分结构检查 - 严格交替模式
    # 分割所有标签
    tags = re.findall(r'<(/?\w+)>', think_content)
    # 检查标签序列是否有效
    if len(tags) % 4 != 0:
        return False, "标签数量必须是4的倍数（area-text对）"
    
    # 验证交替顺序
    for i in range(0, len(tags), 4):
        if tags[i:i+4] != ['area', '/area', 'text', '/text']:
            return False, f"标签顺序错误，位置{i}处应为<area></area><text></text>"
    
    # 验证开头和结尾
    if not think_content.startswith('<area>'):
        return False, "必须以<area>标签开头"
    if not think_content.endswith('</text>'):
        return False, "必须以</text>标签结尾"
    
    # 4. 提取所有area-text对
    pattern = r'<area>(.*?)</area>\s*<text>(.*?)</text>'
    area_text_pairs = re.findall(pattern, think_content, re.DOTALL)
    
    if not area_text_pairs:
        return False, "未找到有效的area-text对"
    
    # 5. 逐个检查area内容
    for area_content, text_content in area_text_pairs:
        # JSON格式检查
        try:
            area_data = json.loads(area_content.strip())
        except json.JSONDecodeError as e:
            return False, f"Area内容JSON解析错误: {e}\n内容: {area_content}"
        
        # 数据结构检查
        if not isinstance(area_data, list):
            return False, "Area内容必须是列表"
        
        for item in area_data:
            # 字段存在性检查
            if "bbox_2d" not in item:
                return False, "Area项缺少bbox_2d字段"
            if "depth" not in item:
                return False, "Area项缺少depth字段"
            
            # 边界框格式检查
            bbox = item["bbox_2d"]
            if not (isinstance(bbox, list) and len(bbox) == 4):
                return False, "bbox_2d必须是包含4个数值的列表"
            
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                return False, f"不符合坐标框约束: {bbox}"
            # 数值范围检查 (0.0-1.0)
            for coord in bbox:
                if not isinstance(coord, (int, float)):
                    return False, f"坐标值必须是数字: {coord}"
                if not (0.0 <= float(coord) <= 1.0):
                    return False, f"坐标值超出范围[0.0,1.0]: {coord}"
            
            # 深度值检查
            depth = item["depth"]
            if not isinstance(depth, (int, float)):
                return False, f"深度值必须是数字: {depth}"
            if not (0.0 <= float(depth) <= 1.0):
                return False, f"深度值超出范围[0.0,1.0]: {depth}"
            
            # 小数位数检查
            for num in bbox + [depth]:
                if isinstance(num, float):
                    decimal_part = str(num).split('.')[1]
                    if len(decimal_part) > 3:
                        return False, f"数值应保留3位小数: {num}"
        
        # 文本内容非空检查
        if not text_content.strip():
            return False, "文本内容不能为空"
    
    return True, "格式验证通过"

def format_valid(content):
    try:
        judge, reason = validate_format(content)
        print(f"content:{content}; format:{reason}")
    except:
        judge = False
    return judge

def is_format_valid(content):
    content = content.replace('\n', '')
    if ("</text><text>" in content) or ("</area><area>" in content):
        return False
    
    # 定义正则表达式模式，支持更灵活的<area>结构
    pattern = re.compile(
        r'^\s*<think>\s*'  # 匹配<think>标签，允许前后有空白字符
        r'(?:'  # 开始非捕获组
        r'<area>\[\[(?:\s*[\d.]+\s*,\s*){3}\s*[\d.]+\s*\](?:\s*,\s*\[\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*,\s*[\d.]+\s*\])*\s*\]</area>\s*'  # 匹配多个一维数组构成的二维数组
        r'<text>(?:(?!<text>|</text>|<area>|</area>).)+</text>\s*'  # 匹配<text>标签和至少一个字符的内容，且其中不能有其他特定标签
        r')+'  # 重复非捕获组一次或多次
        r'</think>\s*'  # 匹配</think>标签
        r'<answer>.+?</answer>\s*$',  # 匹配<answer>标签和至少一个字符的内容
        re.DOTALL  # 允许.匹配换行符
    )
    
    # 替换内容中的换行符并进行匹配
    return pattern.fullmatch(content) is not None

def extract_area_content_old(input_text):
    # 提取所有 <area> 标签内容
    area_contents = re.findall(r'<area>(.*?)</area>', input_text, re.DOTALL)

    result = []
    for content in area_contents:
        content_list = []
        stripped = content.strip()
        try:
            # 解析为 Python 列表
            parsed = ast.literal_eval(stripped)
            if isinstance(parsed, list):
                # 遍历子列表
                for sublist in parsed:
                    if isinstance(sublist, list):
                        # 将元素转换为浮点数
                        converted = [float(item) for item in sublist]
                        content_list.append(converted)
        except (SyntaxError, ValueError):
            # 忽略解析错误
            continue
        result.append(content_list)
    return result

def extract_area_content(content):
    result = []
    for area_str in re.findall(r"<area>(.*?)</area>", content, re.DOTALL):
        boxes = []
        try:
            # 解析 JSON 内容
            area_json = json.loads(area_str)
            
            # 从对象中获取边界框坐标
            if isinstance(area_json, list):
                for item in area_json:
                    if "bbox_2d" in item:
                        coords = item["bbox_2d"]
                        boxes.append(coords)
            else:
                if "bbox_2d" in area_json:
                    coords = area_json["bbox_2d"]
                    boxes.append(coords)
        except Exception as e:
            print(f"解析 <area> 内容时发生错误: {e}")
            boxes.append([])
        result.append(boxes)
    return result

def extract_depth_content(content):
    result = []
    for area_str in re.findall(r"<area>(.*?)</area>", content, re.DOTALL):
        depthes = []
        try:
            # 解析 JSON 内容
            area_json = json.loads(area_str)
            
            # 从对象中获取边界框坐标
            if isinstance(area_json, list):
                for item in area_json:
                    if "depth" in item:
                        depth = item["depth"]
                        depthes.append(depth)
            else:
                if "depth" in area_json:
                    depth = area_json["depth"]
                    depthes.append(depth)
        except Exception as e:
            print(f"解析 <area> 内容时发生错误: {e}")
            depthes.append([])
        result.append(depthes)
    return result

class Qwen2VLModule(VLMBaseModule):
    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        match task_type:
            case "GRPO-SIF":
                return "{Question} Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def format_reward_rec(completions, **kwargs):
        """Check if the Qwen model output matches a specific format."""
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [format_valid(content) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    def format_reward(completions, **kwargs):
        pattern = r"<think>.*?</think>\s*<answer>.*?\[.*?{\"bbox_2d\":\s*\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]\s*,\s*\"label\":\s*\".*?\"\s*}.*?\].*?</answer>"
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.search(pattern, content, re.DOTALL) is not None for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]
    
    # @staticmethod
    # def progress_reward(completions, answer, prompts, img_path_str, **kwargs):
    #     contents = [completion[0]["content"] for completion in completions]
    #     rewards = []
    #     current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    #     for content, ans, prompt, img in zip(contents, answer, prompts,img_path_str):
    #         reward = 0.0
    #         # Try symbolic verification first
    #         try:
    #             quest = prompt[0]['content'][1]['text'].replace(' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.','')
    #             for item in record_list:
    #                 if item["prompt"] == quest and item["img_path"] == img:
    #                     score_pre = item["score_pre"]
    #                     score_now = item["score"]
    #                     if score_now >= score_pre:
    #                         reward = score_now - score_pre
                
    #             # print(reward_record)
    #             # print(record_list)
    #             print(f"current:{quest}")
    #         except Exception:
    #             pass  # Continue to next verification method if this fails
                    
    #         rewards.append(reward)
    #         if os.getenv("DEBUG_MODE") == "true":
    #             log_path = os.getenv("LOG_PATH")
    #             # local_rank = int(os.getenv("LOCAL_RANK", 0))
    #             with open(log_path, "a", encoding='utf-8') as f:
    #                 f.write(f"------------- {current_time} progress_reward Accuracy: {reward} -------------\n")
    #                 f.write(f"Content: {content}\n")
    #                 f.write(f"Answer: {ans}\n")
    #     return rewards

    @staticmethod
    def align_reward(completions, answer, prompts, img_path_str, img_depth_str, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, ans, prompt, img, img_depth in zip(contents, answer, prompts, img_path_str, img_depth_str):
            reward = 0.0
            # Try symbolic verification first
            try:
                # gt_response = ans
                base64_image = encode_image(img)
                base64_image_depth = encode_image(img_depth)
                question = prompt[0]['content'][-1]['text']
                think_content = get_think_answer(client_think, question, base64_image, base64_image_depth)
                think_response = extract_answer(think_content)
                pred_response = content

                judge_score = judge_think_nothink_func(client, question, think_content, base64_image)
                try:
                    score_data = json.loads(judge_score)
                    mllm_score = score_data["score"]
                except:
                    try:
                        ## TODO: judge
                        clean_score = re.sub(r'```json|```', '', judge_score).strip()
                        score_data = json.loads(clean_score)
                        if isinstance(score_data, float) or isinstance(score_data, int):
                            mllm_score = score_data
                        else:
                            mllm_score = score_data["score"]
                    except Exception as e:
                        print(judge_score)
                        print(f"response fail: {e}")

                if pred_response:
                    similar_score = judge_score_func_question(client, question, think_response, pred_response, base64_image)
                    try:
                        similar_score_data = json.loads(similar_score)
                        simliarity = similar_score_data["score"]
                    except:
                        try:
                            clean_similar_score = re.sub(r'```json|```', '', similar_score).strip()
                            similar_score_data = json.loads(clean_similar_score)
                            if isinstance(similar_score_data, float) or isinstance(similar_score_data, int):
                                simliarity = similar_score_data
                            else:
                                simliarity = similar_score_data["score"]
                        except Exception as e:
                            print(similar_score)
                            print(f"response fail: {e}")

                    reward = mllm_score*simliarity
            except Exception as e:
                print("error: {e}")
                pass  # Continue to next verification method if this fails

            print("--------Align Reward--------")
            print(f"Question:{question}")
            print(f"Think:{think_response}")
            print(f"Pred:{pred_response}")
            print(f"Align Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} align_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {ans}\n")
        return rewards

    @staticmethod
    def binary_reward(completions, answer, prompts, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, ans, prompt in zip(contents, answer, prompts):
            reward = 0.0
            # Try symbolic verification first
            try:
                gt_response = ans
                # pred_response = extract_answer(content)
                # if pred_response:
                #     if pred_response == gt_response:
                #         reward = 1.0

                pred_response = content
                if pred_response == gt_response:
                    reward = 1.0
            except Exception as e:
                print("error: {e}")
                pass  # Continue to next verification method if this fails

            print("--------Response Reward--------")
            print(f"Question:{prompt[0]['content'][-1]['text'][1:]}")
            print(f"GT:{gt_response}")
            print(f"Pred:{pred_response}")
            print(f"Response Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} binary_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {ans}\n")
        return rewards

    @staticmethod
    def response_progress_reward(completions, answer, prompts, img_path_str, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, ans, prompt, img in zip(contents, answer, prompts,img_path_str):
            reward = 0.0
            # Try symbolic verification first
            try:
                base64_image = encode_image(img)
                gt_response = ans
                pred_response = extract_answer(content)
                judge_score = judge_score_func(client, prompt, gt_response, pred_response, base64_image)
                print(f"judge_score:{judge_score}")
                try:
                    score_data = json.loads(judge_score)
                    reward = score_data["score"]
                except:
                    try:
                        clean_score = re.sub(r'```json|```', '', judge_score).strip()
                        score_data = json.loads(clean_score)
                        if isinstance(score_data, float) or isinstance(score_data, int):
                            reward = score_data
                        else:
                            reward = score_data["score"]
                    except Exception as e:
                        print(judge_score)
                        print(f"response fail: {e}")
            except Exception:
                print("llm service error")
                pass  # Continue to next verification method if this fails

            ## progress
            try:
                quest = prompt[0]['content'][-1]['text'][1:].replace(' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.','')
                # print(reward_record)
                for item in reward_record:
                    if item["prompt"] == quest and item["img"] == img:
                        score_pre = item["score"]
                        print(f"score:{reward}, score_pre:{score_pre}")
                        if reward >= score_pre:
                            reward = reward + (reward - score_pre)
            except Exception:
                print("progress reward error")
                pass  # Continue to next verification method if this fails

            print("--------Response Reward--------")
            print(f"Question:{prompt[0]['content'][-1]['text'][1:]}")
            print(f"GT:{gt_response}")
            print(f"Pred:{pred_response}")
            print(f"Response Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {ans}\n")
        return rewards

    @staticmethod
    def response_reward(completions, answer, prompts, img_path_str, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, ans, prompt, img in zip(contents, answer, prompts,img_path_str):
            reward = 0.0
            # Try symbolic verification first
            try:
                base64_image = encode_image(img)
                gt_response = ans
                pred_response = extract_answer(content)
                judge_score = judge_score_func(client, prompt, gt_response, pred_response, base64_image)
                print(f"judge_score:{judge_score}")
                try:
                    score_data = json.loads(judge_score)
                    reward = score_data["score"]
                except:
                    try:
                        clean_score = re.sub(r'```json|```', '', judge_score).strip()
                        score_data = json.loads(clean_score)
                        if isinstance(score_data, float) or isinstance(score_data, int):
                            reward = score_data
                        else:
                            reward = score_data["score"]
                    except Exception as e:
                        print(judge_score)
                        print(f"response fail: {e}")
            except Exception:
                print("llm service error")
                pass  # Continue to next verification method if this fails

            print("--------Response Reward--------")
            print(f"Question:{prompt[0]['content'][-1]['text'][1:]}")
            print(f"GT:{gt_response}")
            print(f"Pred:{pred_response}")
            print(f"Response Accuracy: {reward}")

            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} response_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Answer: {ans}\n")
        return rewards

    @staticmethod
    def iou_reward(completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solution):
            reward = 0.0
            # Try symbolic verification first
            try:
                bbox_result = extract_area_content(content)
                response_bbox = bbox_result[-1]
                if response_bbox:
                    pred_boxes = response_bbox
                    gt_boxes = sol['bbox_3d']
                    global_iou = compute_global_iou(pred_boxes, gt_boxes)
                    pair_iou = compute_pair_iou(pred_boxes, gt_boxes)
                    reward = (global_iou + pair_iou)/2
                    print("--------IOU Reward--------")
                    print(f"GT:{gt_boxes}")
                    print(f"Pred:{pred_boxes}")
                    print(f"Bbox Accuracy: {reward}")
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} iou_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards


    @staticmethod
    def depth_reward(completions, solution, img_depth_str, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol, depth_img in zip(contents, solution, img_depth_str):
            reward = 0.0
            # Try symbolic verification first
            try:
                bbox_result = extract_area_content(content)
                depth_result = extract_depth_content(content)
                depth_map = load_depth_from_color(depth_img)
                total_count = 0

                for sublist in depth_result:
                    total_count += len(sublist)

                for i, bbox_3d_item in enumerate(bbox_result):
                    for j, bbox_2d_item in enumerate(bbox_3d_item):
                        try:
                            mean_depth = calculate_avg_depth(depth_map, bbox_2d_item)
                            regress_depth = depth_result[i][j]
                            if abs(mean_depth-regress_depth)/mean_depth <= 0.1:
                                reward = reward + 1/total_count
                        except:
                            continue
                print("--------Depth Reward--------")
                print(f"Bbox_result:{bbox_result}")
                print(f"Depth_result:{depth_result}")
                print(f"Depth Accuracy: {reward}")
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} depth_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards


    @staticmethod
    def iou_reward_modify(completions, solution, **kwargs):
        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solution):
            reward = 0.0
            progress_reward = 0.0
            # Try symbolic verification first
            try:
                bbox_result = extract_area_content(content)
                start_bbox = [[0,0,1,1]]
                for bbox_item in bbox_result:
                    if bbox_item != [[0,0,1,1]]:
                        start_bbox = bbox_item
                        break
                response_bbox = bbox_result[-1]
                if response_bbox:
                    gt_boxes = sol['bbox_3d']

                    end_global_iou = compute_global_iou(response_bbox, gt_boxes)
                    end_pair_iou = compute_pair_iou(response_bbox, gt_boxes)
                    end_iou = (end_global_iou + end_pair_iou)/2
                    reward = end_iou
                    print("--------IOU Reward--------")
                    print(f"GT:{gt_boxes}")
                    print(f"Pred:{response_bbox}")
                    print(f"Bbox Accuracy: {reward}")
           
                    start_global_iou = compute_global_iou(start_bbox, gt_boxes)
                    start_pair_iou = compute_pair_iou(start_bbox, gt_boxes)
                    start_iou = (start_global_iou + start_pair_iou)/2

                    if end_iou >= start_iou:
                        progress_reward = end_iou - start_iou
                    
                    reward = reward + progress_reward
                    print("--------IOU Progress Reward--------")
                    print(f"GT:{gt_boxes}")
                    print(f"Start:{start_bbox}")
                    print(f"End:{response_bbox}")
                    print(f"Final Progress Accuracy: {reward}")
            except Exception:
                pass  # Continue to next verification method if this fails
                    
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} iou_progress_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards

    @staticmethod
    def search_reward(completions, solution, **kwargs):
        """Calculate whether the positioning IoU process is positive."""
        def all_elements_same(arr):
            if not arr:
                return True
            reference_value = arr[0]
            for element in arr:
                if element != reference_value:
                    return False
            return True 

        def is_non_strictly_increasing(arr):
            if all_elements_same(arr):
                return False
            if len(arr) <= 1:
                return False
            for i in range(len(arr) - 1):
                if arr[i] > arr[i + 1]:
                    return False
            return True

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        for content, sol in zip(contents, solution):
            boundingboxs_iou = []
            reward = 0.0
            # Try symbolic verification first
            try:
                reward_iou = 0.0
                bbox_result = extract_area_content(content)
                gt_boxes = sol['bbox_3d']
                if bbox_result:
                    for bbox in bbox_result:
                        reward_iou = (compute_global_iou(bbox, gt_boxes) + compute_pair_iou(bbox, gt_boxes))/2
                        boundingboxs_iou.append(reward_iou)
                    if is_non_strictly_increasing(boundingboxs_iou):
                        reward = 1.0
                print(f"Increasing bbox iou: {boundingboxs_iou}")
            except Exception:
                pass  # Continue to next verification method if this fail
            rewards.append(reward)
            if os.getenv("DEBUG_MODE") == "true":
                log_path = os.getenv("LOG_PATH")
                # local_rank = int(os.getenv("LOCAL_RANK", 0))
                with open(log_path, "a", encoding='utf-8') as f:
                    f.write(f"------------- {current_time} search_reward Accuracy: {reward} -------------\n")
                    f.write(f"Content: {content}\n")
                    f.write(f"Solution: {sol}\n")
        return rewards
    
    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "format":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "depth_consistency":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.depth_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "accuracy":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.response_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "accuracy_w_progress":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.response_progress_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "bboxaccuracy":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "bboxaccuracy_w_progress":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.iou_reward_modify
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "align":
            match task_type:
                case "GRPO-SIF":
                    return Qwen2VLModule.align_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")