import argparse
import random
import re
import copy
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import math
from llava.constants import (
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.conversation import conv_templates, SeparatorStyle
# from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import (
    tokenizer_image_token,
    process_images,
    get_model_name_from_path,
)
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import ast
import os
from openai import OpenAI
from transformers.utils.versions import require_version
import json
import base64

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

def get_answer(client,question,base64_image,base64_depth_image):
    print(question)
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
    
def judge_score_func(client, question, gt_response, pred_response, base64_image):
    question = question[0]['content'][1]['text'].replace(' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.','')
    pred_response = extract_answer(pred_response)
    print("LLM-Reward:")
    print(f"question:{question}")
    print(f"gt_response:{gt_response}")
    print(f"pred_response:{pred_response}")
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
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
            ],
        }]
    result = client.chat.completions.create(messages=messages, model="test")
    response = result.choices[0].message.content
    return response



SUBIMAGE_PATTERN = r".*\#\#\#\[([\d\.]+),\s*([\d\.]+),\s*([\d\.]+),\s*([\d\.]+)\]"

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(
        self,
        questions,
        image_folder,
        model_name,
        with_cot,
        detection_results,
        random_bbox,
        center_bbox,
        without_image,
        adapt_ratio,
    ):
        self.questions = questions
        self.image_folder = image_folder
        # self.tokenizer = tokenizer
        # self.image_processor = image_processor
        # self.model_config = model_config
        self.model_name = model_name
        self.with_cot = with_cot
        self.detection_results = detection_results
        self.random_bbox = random_bbox
        self.center_bbox = center_bbox
        self.without_image = without_image
        self.adapt_ratio = adapt_ratio

    def __getitem__(self, index):
        line = self.questions[index]
        image_files = line["image"]
        raw_conversations = line["conversations"]

        conv = conv_templates[args.conv_mode].copy()
        if self.random_bbox:
            center = [random.random(), random.random()]
            height = random.random() * 0.5
            width = random.random() * 0.5
            random_coords = [max(0, center[0]-width), max(0, center[1]-height), min(1, center[0]+width), min(1, center[1]+height)]
            bbox_ratio = (random_coords[2] - random_coords[0]) * (random_coords[3] - random_coords[1])

        elif self.center_bbox:
            random_coords = [0.25, 0.25, 0.75, 0.75]
            bbox_ratio = (random_coords[2] - random_coords[0]) * (random_coords[3] - random_coords[1])

        elif self.detection_results is not None:
            coords = self.detection_results[index]['text'].replace(' .','').replace('[','').replace(']','').split(', ')
            coords = [float(x) for x in coords]
            bbox_ratio = (coords[2] - coords[0]) * (coords[3] - coords[1])
        else:
            bbox_ratio = 0.0

        if self.with_cot and self.without_image is False:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'].split(' Please provide the bounding box coordinate of the region')[0])
            if self.random_bbox or self.center_bbox:
                conv.append_message(conv.roles[1], '[%.3f, %.3f, %.3f, %.3f]' % (random_coords[0], random_coords[1], random_coords[2], random_coords[3]))
            elif self.detection_results is None:
                conv.append_message(conv.roles[1], raw_conversations[1]['value'])
            else:
                conv.append_message(conv.roles[1], self.detection_results[index]['text'])

            # conv.append_message(conv.roles[0], raw_conversations[2]['value'])
            conv.append_message(conv.roles[0], raw_conversations[2]['value'] + '\nPlease answer the question based on the original image and local detail image.'+  raw_conversations[0]['value'].split('Please provide the bounding box coordinate of the region')[0].replace('<image>\n', ''))
            conv.append_message(conv.roles[1], None)
        elif self.with_cot and self.without_image is True:
            conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            if self.random_bbox or self.center_bbox:
                conv.append_message(conv.roles[1], '[%.3f, %.3f, %.3f, %.3f]' % (random_coords[0], random_coords[1], random_coords[2], random_coords[3]))
            elif self.detection_results is None:
                conv.append_message(conv.roles[1], raw_conversations[1]['value'])
            else:
                conv.append_message(conv.roles[1], self.detection_results[index]['text'])
            conv.append_message(conv.roles[0], '')
            conv.append_message(conv.roles[1], None)
        else:
            if 'Please provide the bounding box' in raw_conversations[0]['value']:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'].split('Please provide the bounding box')[0])
            else:
                conv.append_message(conv.roles[0], raw_conversations[0]['value'])
            conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # images = []
        image_path = os.path.join(self.image_folder, image_files[0].replace('cot/','cot_image_data/'))
        # image = Image.open(image_path).convert("RGB")
        # images.append(image)


        # if self.with_cot and self.without_image is False and len(image_files) > 1:
        #     if self.random_bbox or self.center_bbox:
        #         coords = random_coords
        #     elif self.detection_results is None:
        #         if '###' not in image_files[1]:
        #             raise ValueError("%s is not a valid cot path" % image_path)
        #         try:
        #             coords = raw_conversations[1]['value'].replace(' .','').replace('[','').replace(']','').split(', ')
        #             coords = [float(x) for x in coords]
        #         except Exception as e:
        #             print(e)
        #             print("Can not parse the coords: %s" % image_files[1])
        #             coords = [0.0, 0.0, 1.0, 1.0]
        #     else:
        #         try:
        #             coords = self.detection_results[index]['text'].replace(' .','').replace('[','').replace(']','').split(', ')
        #             coords = [float(x) for x in coords]
        #         except Exception as e:
        #             print(e)
        #             print("Can not parse the coords: %s" % self.detection_results[index]['text'])
        #             coords = [0.0, 0.0, 1.0, 1.0]
        #     image_files[1] = image_files[1].split('###')[0]
        #     image_path2 = os.path.join(self.image_folder, image_files[1])
        #     if image_path2 == image_path:
        #         image = copy.copy(images[0])
        #     else:
        #         image = Image.open(image_path2.replace('cot/','cot_image_data/')).convert("RGB")

        #     def cropwithbbox(pil_img, sub_image_info):
        #         width, height = pil_img.size
        #         x_min, y_min, x_max, y_max = sub_image_info
        #         if sum([x_min, y_min, x_max, y_max]) < 5:
        #             x_min = x_min * max(width, height)
        #             y_min = y_min * max(width, height)
        #             x_max = x_max * max(width, height)
        #             y_max = y_max * max(width, height)
        #         if width > height:
        #             overlay = (width - height) // 2
        #             y_min = max(0, y_min - overlay)
        #             y_max = max(0, y_max - overlay)
        #         else:
        #             overlay = (height - width) // 2
        #             x_min = max(0, x_min - overlay)
        #             x_max = max(0, x_max - overlay)
        #         center_point = [(x_min + x_max)//2, (y_min + y_max)//2]
        #         half_sizes = [(x_max - x_min)//2, (y_max - y_min)//2]
        #         cropped_half_size = max(max(half_sizes), 112)
        #         upper_left_point = [center_point[0]-cropped_half_size, center_point[1]-cropped_half_size]
        #         if upper_left_point[0] < 0:
        #             center_point[0] += (-upper_left_point[0])
        #         if upper_left_point[1] < 0:
        #             center_point[1] += (-upper_left_point[1])
        #         lower_right_point = [center_point[0]+cropped_half_size, center_point[1]+cropped_half_size]
        #         if lower_right_point[0] > width:
        #             center_point[0] -= (lower_right_point[0] - width)
        #         if lower_right_point[1] > height:
        #             center_point[1] -= (lower_right_point[1] - height)
        #         cropped_region = [max(0, center_point[0]-cropped_half_size), max(0, center_point[1]-cropped_half_size), min(width, center_point[0]+cropped_half_size), min(height, center_point[1]+cropped_half_size)]
        #         cropped_image = pil_img.crop(cropped_region)
        #         return cropped_image
        #     image = cropwithbbox(image, coords)
        #     images.append(image)


        # if isinstance(self.image_processor, list):
        #     image_tensor_0 = process_images(
        #         images, self.image_processor[0], self.model_config
        #     )
        #     image_tensor_1 = process_images(
        #         images, self.image_processor[1], self.model_config
        #     )
        #     image_tensor = torch.cat((image_tensor_0, image_tensor_1), dim=0)
        # else:
        #     image_tensor = process_images(
        #         images, self.image_processor, self.model_config
        #     )

        # input_ids = tokenizer_image_token(
        #     prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        # )

        return image_path, prompt

    def __len__(self):
        return len(self.questions)


# DataLoader
def create_data_loader(
    questions,
    image_folder,
    model_name,
    with_cot,
    detection_results,
    random_bbox,
    center_bbox,
    without_image,
    adapt_ratio,
    batch_size=1,
    num_workers=4,
):
    assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(
        questions, image_folder,model_name, with_cot, detection_results, random_bbox, center_bbox, without_image, adapt_ratio
    )
    data_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False
    )
    return data_loader


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def eval_model(args):
    # Model
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    # tokenizer, model, image_processor, context_len = load_pretrained_model(
    #     model_path, args.model_base, model_name
    # )

    if args.random_bbox is True and args.center_bbox is True:
        raise ValueError("random-bbox and center-bbox cannot all be true!")

    if args.question_file.endswith('.jsonl'):
        questions = [
            json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")
        ]
    else:
        questions = json.load(open(args.question_file))
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    if os.path.exists(answers_file):
        os.system("rm "+answers_file)
    ans_file = open(answers_file, "w")


    det_questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file.replace('/benchmark/','/benchmark_det/').replace('.json','.jsonl')), "r")]
    det_questions = get_chunk(det_questions, 1, 0)
    det_answers_file = os.path.expanduser(args.answers_file.replace('viscot/results','viscot/detection').replace('viscot//results','viscot//detection'))
    os.makedirs(os.path.dirname(det_answers_file), exist_ok=True)
    if os.path.exists(det_answers_file):
        os.system("rm "+det_answers_file)
    det_ans_file = open(det_answers_file, "w")
    # if args.detection_file is not None:
    #     detection_results = [
    #         json.loads(r) for r in open(args.detection_file, 'r')
    #     ]
    # else:
    detection_results = None

    if (
        "plain" in model_name
        and "finetune" not in model_name.lower()
        and "mmtag" not in args.conv_mode
    ):
        args.conv_mode = args.conv_mode + "_mmtag"
        print(
            f"It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}."
        )
    data_loader = create_data_loader(
        questions,
        args.image_folder,
        model_name,
        args.with_cot,
        detection_results,
        args.random_bbox,
        args.center_bbox,
        args.without_image,
        args.adapt_ratio,
    )

    client = OpenAI(
        api_key="{}".format(os.getenv("API_KEY", "0")),
        base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8020)),
    )

    for (image_path, prompt), line, det_line in tqdm(
        zip(data_loader, questions,det_questions), total=len(questions)
    ):
        idx = line["question_id"]
        height = det_line["height"]
        width = det_line["width"]
        bbox = det_line["bbox"]

        solution_prompt = line['conversations'][0]['value'].replace('<image>\n','').replace(' Please provide the bounding box coordinate of the region that can help you answer the question better.','')
        det_prompt = det_line['expression']

        stop_str = (
            conv_templates[args.conv_mode].sep
            if conv_templates[args.conv_mode].sep_style != SeparatorStyle.TWO
            else conv_templates[args.conv_mode].sep2
        )
        ours_image_path = image_path[0]

        base_dir = os.path.dirname(ours_image_path)
        file_name = os.path.splitext(os.path.basename(ours_image_path))[0]
        depth_path = os.path.join(base_dir, f"{file_name}_d.png")

        ours_prompt = line['conversations'][0]['value'].replace('<image>\n','').replace(' Please provide the bounding box coordinate of the region that can help you answer the question better.',' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.')
        base64_image = encode_image(ours_image_path)
        base64_depth_image = encode_image(depth_path)
        try:
            outputs_raw = get_answer(client,ours_prompt,base64_image,base64_depth_image)

            outputs = extract_answer(outputs_raw)
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

            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()

            det_outputs = det_outputs.strip()
            if det_outputs.endswith(stop_str):
                det_outputs = det_outputs[:-len(stop_str)]
            det_outputs = det_outputs.strip()
        except:
            outputs = ""
            det_outputs = ""

        print(outputs)
        print(det_outputs)
        
        ans_id = shortuuid.uuid()
        # prompt_q = line['conversations'][0]['value']
        # if prompt_q.startswith('<image>\n'):
        #     prompt_q = prompt_q.replace('<image>\n', '')
        # if 'Please provide the bounding box coordinate of the region' in prompt_q:
        #     prompt_q = prompt_q.split('Please provide the bounding box coordinate of the region')[0]
        #print(outputs, line['conversations'][1]['value'])

        det_ans_file.write(json.dumps({"question_id": 0,
                                   "prompt": ours_prompt,
                                   "text": det_outputs,
                                   "bbox": bbox,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "height": height,
                                   "width": width,
                                   "metadata": {}}) + "\n")

        dumped_dict = {
                    "question_id": idx,
                    "conversations": prompt[0],
                    "text": outputs,
                    "answer_id": ans_id,
                    "model_id": model_name,
                    "prompt": ours_prompt,
                    "metadata": {},
                }
        if 'height' in line:
            dumped_dict['height'] = line['height']
        if 'width' in line:
            dumped_dict['width'] = line['width']
        if 'bbox' in line:
            dumped_dict['bbox'] = line['bbox']
        ans_file.write(
            json.dumps(dumped_dict)
            + "\n"
        )
        ans_file.flush()
    ans_file.close()
    det_ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="s3://mmdata/")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument('--with-cot', type=bool, default=False)
    parser.add_argument('--random-bbox', type=bool, default=False)
    parser.add_argument('--center-bbox', type=bool, default=False)
    parser.add_argument('--without-image', type=bool, default=False)
    parser.add_argument('--detection-file', type=str, default=None)
    parser.add_argument('--adapt-ratio', type=float, default=1.0)
    args = parser.parse_args()
    eval_model(args)
