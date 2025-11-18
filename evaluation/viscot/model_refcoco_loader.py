import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader

import math
from PIL import Image
import ast
import os
from openai import OpenAI
import json
import base64
import re

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

def extract_answer(xml_text):
    # 使用正则表达式匹配<answer>和</answer>之间的内容，包括换行符
    pattern = r'<answer>\n?(.*?)\n?</answer>'  # 处理可能的换行符
    match = re.search(pattern, xml_text, re.DOTALL)
    
    if match:
        # 提取内容并去除首尾空白
        return match.group(1).strip()
    return None

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

def get_answer(client,question,base64_image,base64_depth_image):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Original image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                {"type": "text", "text": "\nDepth image: "},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_depth_image}"}},
                {"type": "text", "text": "\n" + question},
            ],
        }]
    result = client.chat.completions.create(messages=messages, model="test",max_tokens=2048)
    response = result.choices[0].message.content
    return response

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, image_folder):
        self.questions = questions
        self.image_folder = image_folder

    def __getitem__(self, index):
        line = self.questions[index]
        image_file = line["img_path"]
        image_file = image_file.split('/')[-1]
        qs = "Locate the region this sentence describes: <expr>. Please provide the bounding box coordinates.".replace("<expr>", line["expression"].lower())

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Answer: [
        # prompt = prompt + " ["

        image_file = os.path.join(self.image_folder, image_file)

        return image_file, prompt

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(questions, image_folder, batch_size=1, num_workers=4):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, image_folder)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return data_loader


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if os.path.exists(answers_file):
        os.system("rm "+answers_file)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder)

    client = OpenAI(
        api_key="{}".format(os.getenv("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8020)),
    )

    for (img_path, prompt), line in tqdm(zip(data_loader, questions), total=len(questions)):
        height = line["height"]
        width = line["width"]
        bbox = line["bbox"]
        # cur_prompt = "Where is <expr>? answer in [x0,y0,x1,y1] format.".replace("<expr>", line["expression"].lower())
        #cur_prompt = "What are the coordinates of <expr> in the image?".replace("<expr>", line["expression"].lower())
        cur_prompt = "Please provide the bounding box coordinate of the region this sentence describes: <expr>.".replace("<expr>", line["expression"])
        # cur_prompt = "Where is <expr>?".replace("<expr>", line["expression"].lower())
        #cur_prompt = "Can you see <expr>?".replace("<expr>", line["expression"].lower())

        # stop_str = conv_templates[args.conv_mode].sep if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO else conv_templates[args.conv_mode].sep2
        # input_ids = input_ids.to(device='cuda', non_blocking=True)

        img_path = img_path[0]
        # prompt = prompt[0].replace("A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: ","").replace(" Please provide the bounding box coordinates. ASSISTANT:"," Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.")
        prompt =  cur_prompt + " Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags."
        base64_image = encode_image(img_path)
        print(prompt)
        base_dir = os.path.dirname(img_path)
        file_name = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(base_dir, f"{file_name}_d.png")

        base64_depth_image = encode_image(depth_path)
        try:
            outputs_raw = get_answer(client,prompt,base64_image,base64_depth_image)
            bbox_result = extract_area_content(outputs_raw)
            if len(bbox_result[-1]) > 1:
                print(f"bbox too large:{bbox_result[-1]}")
                x_min = bbox_result[0][0]
                y_min = bbox_result[0][1]
                x_max = bbox_result[0][2]
                y_max = bbox_result[0][3]

                # 遍历所有边界框，更新最小和最大坐标
                for box in bbox_result:
                    if box[0] < x_min:
                        x_min = box[0]
                    if box[1] < y_min:
                        y_min = box[1]
                    if box[2] > x_max:
                        x_max = box[2]
                    if box[3] > y_max:
                        y_max = box[3]

                # 返回新的边界框
                det_outputs = str([x_min, y_min, x_max, y_max])
            else:
                det_outputs = str(bbox_result[-1][0])

            outputs = det_outputs.strip()
        except:
            outputs = ""

        outputs = outputs.strip()

        print(outputs)
        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"prompt": cur_prompt,
                                   "text": outputs,
                                   "bbox": bbox,
                                   "img_path": img_path,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "height": height,
                                   "width": width,
                                   "metadata": {}}) + "\n")
        # ans_file.flush()
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)
