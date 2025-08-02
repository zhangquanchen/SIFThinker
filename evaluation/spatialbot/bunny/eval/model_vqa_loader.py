import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid

from bunny.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from bunny.conversation import conv_templates
# from bunny.model.builder import load_pretrained_model
from bunny.util.utils import disable_torch_init
from bunny.util.mm_utils import tokenizer_image_token, process_images, get_model_name_from_path
from torch.utils.data import Dataset, DataLoader
from openai import OpenAI
import re
from PIL import Image
import math
import base64

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
client = OpenAI(
    api_key="{}".format(os.getenv("API_KEY", "0")),
    base_url="http://localhost:{}/v1".format(os.getenv("API_PORT", 8020)),
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

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
        image_file = line["image"]
        qs = line["text"]

        qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # try:
        #     image = Image.open(os.path.join(self.image_folder, image_file)).convert('RGB')
        # except:
        #     image = Image.open(os.path.join(self.image_folder, image_file+'.png')).convert('RGB')

        # image_tensor = process_images([image], self.image_processor, self.model_config)[0]

        # input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt')
        image_path = os.path.join(self.image_folder, image_file)
        return image_path

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
    # tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, args.model_base, model_name,
    #                                                                        args.model_type)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")

    if 'plain' in model_name and 'finetune' not in model_name.lower() and 'mmtag' not in args.conv_mode:
        args.conv_mode = args.conv_mode + '_mmtag'
        print(
            f'It seems that this is a plain model, but it is not using a mmtag prompt, auto switching to {args.conv_mode}.')

    data_loader = create_data_loader(questions, args.image_folder)

    for image_pth, line in tqdm(zip(data_loader, questions), total=len(questions)):        
        idx = line["question_id"]
        print(idx)
        cur_prompt = line["text"]
        image_pth = image_pth[0]
        base_dir = os.path.dirname(image_pth)
        file_name = os.path.splitext(os.path.basename(image_pth))[0]
        depth_path = os.path.join(base_dir, f"{file_name}_d.png")

        print(cur_prompt)
        base64_image = encode_image(image_pth)
        # base64_depth_image = encode_image(depth_path)
        ## !! TODO1
        # messages = [
        #     {
        #         "role": "user",
        #         "content": [
        #             {"type": "text", "text": "Original image: "},
        #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
        #             {"type": "text", "text": "\nDepth image: "},
        #             {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_depth_image}"}},
        #             {"type": "text", "text": "\n" + cur_prompt + ' Please first output the thinking process in <think> </think> tags, where bounding box is enclosed in <area> </area> tags and text analysis is enclosed in <text> </text> tags, alternating between them to iteratively refine the focused area. Then output the final answer in <answer> </answer> tags.'},
        #         ],
        #     }]
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}},
                    {"type": "text", "text": cur_prompt}
                ],
            }]
        ## TODO3
        result = client.chat.completions.create(messages=messages, model="test")
        # result = client.chat.completions.create(messages=messages, model="test",max_tokens=2048)
        response_text = result.choices[0].message.content
        # response = call_model_engine(args, sample, model, tokenizer, processor)
        ## !! TODO2
        # outputs = extract_answer(response_text)
        outputs = response_text
        if outputs:
            outputs = outputs.strip()
        else:
            outputs = ""

        print(outputs)
        # input_ids = input_ids.to(device='cuda', non_blocking=True)
        # with torch.inference_mode():
        #     output_ids = model.generate(
        #         input_ids,
        #         images=image_tensor.to(dtype=model.dtype, device='cuda', non_blocking=True),
        #         do_sample=True if args.temperature > 0 else False,
        #         temperature=args.temperature,
        #         top_p=args.top_p,
        #         num_beams=args.num_beams,
        #         max_new_tokens=args.max_new_tokens,
        #         use_cache=True)

        # input_token_len = input_ids.shape[1]
        # n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        # if n_diff_input_output > 0:
        #     print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        # outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        # outputs = outputs.strip()

        ans_id = shortuuid.uuid()
        ans_file.write(json.dumps({"question_id": idx,
                                   "prompt": cur_prompt,
                                   "text": outputs,
                                   "answer_id": ans_id,
                                   "model_id": model_name,
                                   "metadata": {}}) + "\n")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--model-type", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default=None)
    parser.add_argument("--question-file", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    args = parser.parse_args()

    eval_model(args)
