import os
from openai import OpenAI
import base64
import json
import os
import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import copy
import random
import re
from concurrent.futures import ThreadPoolExecutor
import threading
from bbox_adjust import *
import ast
from depth_est import *

# 定义全局变量
lock = threading.Lock()
json_list = []
# def read_json_and_get_data(json_file):
#     """读取JSON文件并获取第一条数据"""
#     with open(json_file, 'r') as f:
#         data = json.load(f)
#     return data[2] if isinstance(data, list) else data

def filter_and_save_json(json_file, output_file="filtered_data.json", max_per_dataset=2000):
    # 读取 JSON 文件
    with open(json_file, 'r') as f:
        data = json.load(f)
    exclude_datasets = {"docvqa", "dude", "textvqa", "sroie", "infographicsvqa", "textcap"}
    data = [item for item in data if item["dataset"] not in exclude_datasets]
    # 统计每种数据类型并进行限制，最多抽取 max_per_dataset 条
    dataset_counts = {}
    filtered_data = []
    for item in data:
        dataset = item["dataset"]
        if dataset not in dataset_counts:
            dataset_counts[dataset] = 0
        
        if dataset_counts[dataset] < max_per_dataset:
            filtered_data.append(item)
            dataset_counts[dataset] += 1
    
    # 保存筛选后的数据到新的 JSON 文件
    with open(output_file, 'w') as f:
        json.dump(filtered_data, f, indent=2)
    
    print(f"数据已筛选并保存到 {output_file} 文件中。")

def merge_bounding_boxes(bounding_boxes):
    """
    合并多个边界框为一个大边界框
    
    参数:
        bounding_boxes: 一个列表，包含多个边界框，每个边界框为 [x1, y1, x2, y2]
        
    返回:
        合并后的边界框 [merged_x1, merged_y1, merged_x2, merged_y2]
    """
    if not bounding_boxes:
        return None  # 如果没有边界框，返回None
    
    # 初始化合并后的边界框为第一个边界框
    merged_x1 = merged_y1 = float('inf')
    merged_x2 = merged_y2 = -float('inf')
    
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        
        # 更新合并边界框的最小左上角坐标
        if x1 < merged_x1:
            merged_x1 = x1
        if y1 < merged_y1:
            merged_y1 = y1
        
        # 更新合并边界框的最大右下角坐标
        if x2 > merged_x2:
            merged_x2 = x2
        if y2 > merged_y2:
            merged_y2 = y2
            
    return [merged_x1, merged_y1, merged_x2, merged_y2]

def denormalize_bboxes(bboxes, width, height):
    """将边界框坐标从0-1范围转换回像素坐标"""
    denormalized_bboxes = []
    for bbox in bboxes:
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)
        denormalized_bboxes.append([x1, y1, x2, y2])
    return denormalized_bboxes

def expand_and_merge_bboxes(initial_bboxes, width, height, num_expansions=5):
    """
    逐步膨胀边界框，合并后再快速膨胀到全图，确保在指定步数内完成。

    参数：
    initial_bboxes - 初始边界框列表，格式为 [[x1, y1, x2, y2], ...]
    width, height - 图像的宽度和高度
    num_expansions - 最大膨胀次数，默认为 8

    返回：
    all_bboxes - 每一步的边界框列表
    """
    # 将初始边界框归一化到 0-1 范围
    normalized_bboxes = initial_bboxes.copy()
    # for i in range(len(normalized_bboxes)):
    #     for j in range(len(normalized_bboxes[i])):
    #         normalized_bboxes[i][j] /= 1000

    # 存储每一步的边界框
    all_bboxes = []
    all_bboxes.append(denormalize_bboxes(normalized_bboxes, width, height))

    # 初始化膨胀步骤计数
    current_expansions = 0

    # 标记是否已经合并为一个边界框
    merged_all = False

    while current_expansions < num_expansions:
        # 创建当前边界框的副本进行操作
        current_bboxes = copy.deepcopy(normalized_bboxes)

        # # 第一步：稍微左右扰动，控制膨胀量
        # if current_expansions == 0:
        #     # 计算小幅度的扰动步长
        #     small_perturbation = min(width, height) * 0.02  # 1% of the smaller image dimension
        #     for i in range(len(current_bboxes)):
        #         x1, y1, x2, y2 = current_bboxes[i]
        #         # 左右上下扰动
        #         x1_perturbed = max(0, x1 - small_perturbation / width + random.uniform(-0.005, 0.005))
        #         x2_perturbed = min(1, x2 + small_perturbation / width + random.uniform(-0.005, 0.005))
        #         y1_perturbed = max(0, y1 - small_perturbation / height + random.uniform(-0.005, 0.005))
        #         y2_perturbed = min(1, y2 + small_perturbation / height + random.uniform(-0.005, 0.005))
        #         current_bboxes[i] = [x1_perturbed, y1_perturbed, x2_perturbed, y2_perturbed]
        # else:
        # 动态计算膨胀步长
        expansion_step = calculate_expansion_step(current_bboxes, width, height, num_expansions - current_expansions)

        # 对每个边界框进行膨胀
        for i in range(len(current_bboxes)):
            x1, y1, x2, y2 = current_bboxes[i]
            
            x1 = max(0, x1 - expansion_step / width)
            y1 = max(0, y1 - expansion_step / height)
            x2 = min(1, x2 + expansion_step / width)
            y2 = min(1, y2 + expansion_step / height)
            
            current_bboxes[i] = [x1, y1, x2, y2]

        # 检查是否有重叠的边界框，如果有则合并
        merged = True
        while merged:
            merged = False
            merged_indices = [False] * len(current_bboxes)
            merged_bboxes = []

            for i in range(len(current_bboxes)):
                if merged_indices[i]:
                    continue

                x1, y1, x2, y2 = current_bboxes[i]

                for j in range(i + 1, len(current_bboxes)):
                    if merged_indices[j]:
                        continue

                    x3, y3, x4, y4 = current_bboxes[j]

                    if x1 < x4 and x2 > x3 and y1 < y4 and y2 > y3:
                        new_x1 = min(x1, x3)
                        new_y1 = min(y1, y3)
                        new_x2 = max(x2, x4)
                        new_y2 = max(y2, y4)

                        merged_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                        merged_indices[i] = True
                        merged_indices[j] = True
                        merged = True
                        break

                if not merged_indices[i]:
                    merged_bboxes.append(current_bboxes[i])

            if merged:
                current_bboxes = merged_bboxes

        # 更新边界框并保存当前状态
        normalized_bboxes = current_bboxes
        all_bboxes.append(denormalize_bboxes(normalized_bboxes, width, height))
        current_expansions += 1

        # 检查是否已经合并为一个边界框
        if len(normalized_bboxes) == 1:
            merged_all = True

        # 如果已经合并为一个边界框，并且剩余步骤足够，快速膨胀到全图
        if merged_all and current_expansions < num_expansions:
            remaining_steps = num_expansions - current_expansions
            if remaining_steps >= 2:
                # 最后两步快速膨胀到全图
                final_expansion_step_x = (1 - normalized_bboxes[0][2]) * width
                final_expansion_step_y = (1 - normalized_bboxes[0][3]) * height

                # 第一步快速膨胀
                x1 = max(0, normalized_bboxes[0][0] - final_expansion_step_x / 2 / width)
                y1 = max(0, normalized_bboxes[0][1] - final_expansion_step_y / 2 / height)
                x2 = min(1, normalized_bboxes[0][2] + final_expansion_step_x / 2 / width)
                y2 = min(1, normalized_bboxes[0][3] + final_expansion_step_y / 2 / height)
                normalized_bboxes = [[x1, y1, x2, y2]]
                all_bboxes.append(denormalize_bboxes(normalized_bboxes, width, height))
                current_expansions += 1

                # 第二步直接设置为全图
                normalized_bboxes = [[0, 0, 1, 1]]
                all_bboxes.append(denormalize_bboxes(normalized_bboxes, width, height))
                current_expansions += 1
                break

    return all_bboxes[:num_expansions + 1]

def calculate_expansion_step(normalized_bboxes, width, height, remaining_steps):
    """
    动态计算膨胀步长，确保在剩余步骤内能够覆盖整个图像。

    参数：
    normalized_bboxes - 归一化后的边界框列表
    width, height - 图像的宽度和高度
    remaining_steps - 剩余的膨胀步骤数

    返回：
    expansion_step - 计算出的膨胀步长
    """
    if not normalized_bboxes:
        return 0

    # 计算边界框在 x 和 y 方向上的最大剩余空间
    max_x1 = max(bbox[0] for bbox in normalized_bboxes)
    min_x2 = min(bbox[2] for bbox in normalized_bboxes)
    remaining_x = 1 - max_x1 - min_x2

    max_y1 = max(bbox[1] for bbox in normalized_bboxes)
    min_y2 = min(bbox[3] for bbox in normalized_bboxes)
    remaining_y = 1 - max_y1 - min_y2

    # 根据剩余步骤计算所需的膨胀步长
    required_x_step = remaining_x * width / remaining_steps if remaining_steps > 0 else 0
    required_y_step = remaining_y * height / remaining_steps if remaining_steps > 0 else 0

    # 返回较大的步长值，确保能够覆盖整个图像
    return max(required_x_step, required_y_step, 1)

def visualize_bboxes(image_path, bboxes_list):
    """可视化每一步的边界框，其中bboxes_list是一个归一化后的结果"""
    # 读取图像
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 获取图像尺寸
    height, width, _ = image.shape

    base64_images = []
    # 可视化每一步的边界框
    for i, bboxes in enumerate(bboxes_list):
        # 创建图像的副本进行绘制
        img_copy = image.copy()
        
        # 绘制边界框
        for bbox in bboxes:
            # 归一化坐标转换为实际坐标
            x1, y1, x2, y2 = bbox
            
            # 转换为实际像素坐标（图像左上角为(0,0)）
            x1_pixel = int(x1 * width)
            y1_pixel = int(y1 * height)
            x2_pixel = int(x2 * width)
            y2_pixel = int(y2 * height)
            
            # 绘制边界框
            cv2.rectangle(img_copy, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 255, 0), 2)
        
        # 将图像转换为 Base64 编码
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        encoded_image = base64.b64encode(buffer).decode('utf-8')
        base64_images.append({
            "step": i,
            "base64": f"data:image/jpeg;base64,{encoded_image}"
        })
    
    return base64_images
        
def encoder_base64(image_path):
    """
    读取图片并返回Base64编码的字符串。
    """
    with open(image_path, "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    return encoded_image

def process_data(num, json_data, client,output_file):
    print(f"processsing data: {num}")
    data = json_data[num]
    prompt = data['conversations'][0]['value'].replace('<image>\n','').replace(' Please provide the bounding box coordinate of the region that can help you answer the question better.','')
    boundingbox = data['conversations'][1]['value']
    dataset_name = data['dataset']
    response_gt = data['conversations'][3]['value']
    image_path = 'Visual-CoT/cot_images_tar_split/cot_image_data' + data['image'][0].replace('cot','')
    dir_path, file_name = os.path.split(image_path)
    name_part, ext_part = os.path.splitext(file_name)
    new_file_name = f"{name_part}_d.png"
    depth_path = os.path.join(dir_path, new_file_name)
    depth_map = load_depth_from_color(depth_path)
    SYSTEM_PROMPT = f'''Please help me construct data that meets the specified think-answer format. 
    Specifically, based on the given prompt, the constructed data should start with the original image and progressively focus on the regions of interest with analysis.
    Throughout the thinking process, you need to continuously offer the corresponding focused normalized bounding boxes with normalized depth in JSON(keep 3 decimal places of each number), describe the scene and share your thought monologue. All of these should be included within the <think> </think> tags. The final result, after comprehensive thinking, should be placed within the <answer> </answer> tags. 
    Directly output the constructed data, and wrap the result with 3 backticks ```.
    The prompt is {prompt}, and the final gt answer is {response_gt}.

    Example format:
    ```
    <think>
    <area>[{{"bbox_2d":[0.000,0.000,1.000,1.000],"depth":0.500}}]</area>
    <text>The prompt is about the color of the clothes the girl under the traffic light is wearing. I first look at the entire image and find that it is a street scene. Then I start to search for content related to the prompt in the image.</text>
    <area>[{{"bbox_2d":[0.500,0.500,0.898,0.966],"depth":0.345}}]</area>
    <text>I first notice this area, which shows some vehicles driving on the road with the depth of 0.345. Since the prompt is asking about the clothing characteristics of the girl under the traffic light, I plan to first locate the traffic light...</text>
    <area>[{{"bbox_2d":[0.128,0.119,0.490,0.465],"depth":0.268}}]</area>
    <text>Following the road, I see a traffic light on the side with the depth of 0.268...</text>
    <area>[{{"bbox_2d":[0.236,0.244,0.308,0.312],"depth":0.271}}]</area>
    <text>Further, I see a girl under the traffic light. I focus on her clothes to check the color.</text>
    <area>[{{"bbox_2d":[0.266,0.274,0.301,0.310],"depth":0.276}}]</area>
    <text>Upon closer inspection, I see that she is wearing a pink dress.</text>
    </think>
    <answer>
    The girl under the traffic light is wearing a pink dress.
    </answer>
    ```
    '''

    prompt_generation = f"Original image: <image>\nDepth image: <image>\n{prompt} Please first output the thinking process in <think> </think> tags, where bounding box with depth is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."

    content_list = []

    image = cv2.imread(image_path)
    height, width, _ = image.shape
    boundingbox = '['+boundingbox+']'
    boundingboxlist = ast.literal_eval(boundingbox)
    # 开始膨胀和合并边界框
    all_bboxes = expand_and_merge_bboxes(boundingboxlist, width, height)
    normalized_bboxes = []
    for frame in all_bboxes:
        normalized_frame = []
        for box in frame:
            # 依次对 x_min, y_min, x_max, y_max 进行归一化并保留三位小数
            normalized_box = [
                round(box[0] / width, 3),
                round(box[1] / height, 3),
                round(box[2] / width, 3),
                round(box[3] / height, 3)
            ]
            normalized_frame.append(normalized_box)
        normalized_bboxes.append(normalized_frame)

    new_bboxes = normalized_bboxes[:2]
    if len(new_bboxes[1]) > 1:
        normalized_bbox_merge = merge_bounding_boxes(new_bboxes[1])
        new_bboxes.append([normalized_bbox_merge])
    else:
        normalized_bbox_merge = new_bboxes[1][0]
    expanded_box, translated_box = adjust_3f_bbox(normalized_bbox_merge)
    numbers = [1, 2, 3, 4]
    selected_number = random.choice(numbers)
    if selected_number == 1 or selected_number == 2:
        if compute_iou(normalized_bbox_merge, translated_box) < 0.001:
            new_bboxes.append([translated_box])
    new_bboxes.append([[0.000,0.000,1.000,1.000]])
    base64_images = visualize_bboxes(image_path, new_bboxes)

    for i in range(len(new_bboxes)-1, -1, -1):
        json_list_bbox_depth = []
        for bbox in new_bboxes[i]:
            mean_depth = calculate_avg_depth(depth_map, bbox)
            json_list_bbox_depth.append({
                "bbox_2d": bbox,
                "depth": round(mean_depth, 3)
            })
        content_list.append({"type": "text", "text": f"Focus on the region with the normalized bounding box with normalized depth of {json_list_bbox_depth}, as shown by the green box in the image below."})
        content_list.append({
                    "type": "image_url",
                    "image_url": {
                        "url": base64_images[i]["base64"]
                    },
                })

    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model="Put Your Model",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT},
                        ],
                    },
                    {
                        "role": "user",
                        "content": content_list,
                    }
                ],
            )
        except Exception as e:
            print(f"请求失败: {e}")
            continue
        text = response.choices[0].message.content
        pattern = r'```(.*?)```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            response = matches[0].lstrip('\n').rstrip('\n')
            json_text = {
                "dataset": dataset_name,
                "image": image_path,
                "depth_image": depth_path,
                "problem": prompt,
                "normalized_solution_round": {
                    'bbox_3d': boundingboxlist,
                    'illegal_content': False,
                },
                "response": response_gt,
                "messages": [
                {
                    "content": prompt_generation,
                    "role": "user"
                },
                {
                    "content": response,
                    "role": "assistant"
                }
                ],
                "images": [
                image_path,
                depth_path
                ]
            }
            json_list.append(json_text)
            break
        else:
            continue

    # 每处理 10 条数据后，将结果追加到 JSON 文件
    if (num + 1) % 30 == 0:
        with lock:
            if os.path.exists(output_file):
                with open(output_file, 'r', encoding='utf-8') as file:
                    existing_data = json.load(file)
                existing_data.extend(json_list)
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(existing_data, file, ensure_ascii=False, indent=4)
            else:
                with open(output_file, 'w', encoding='utf-8') as file:
                    json.dump(json_list, file, ensure_ascii=False, indent=4)
            json_list.clear()  # 清空列表以便下一次追加


def main(json_file, client):
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    output_file = "viscot_data_SIF.json"

    # 创建线程池
    with ThreadPoolExecutor(max_workers=30) as executor:
        for num in range(min(50000, len(json_data))):
            executor.submit(process_data, num, json_data, client, output_file)

    print(f"数据已成功保存到{output_file}文件中。")

if __name__ == "__main__":
    json_file = "Visual-CoT/viscot_363k.json"  # 请替换为你的JSON文件路径
    filtered_json_file = "viscot_filtered_data.json"
    # 调用函数实现数据的筛选和保存
    filter_and_save_json(json_file, output_file=filtered_json_file, max_per_dataset=10000)
    client = OpenAI(
        base_url="Put Your Base URL",
        api_key="Put Your API Key",
    )
    main(filtered_json_file,client)